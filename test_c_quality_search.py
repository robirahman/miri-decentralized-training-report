"""Regression test for _min_nodes_for_c_quality (formerly _binary_search_nodes_for_c_quality).

For a fixed model, C_quality rises and then falls as nodes are added (the loss saturates
while the replica penalty keeps growing), so the old bisection, which required the target
to be met at max_nodes, dropped models that reach it only in a middle range of n. The new
search must agree with a linear scan, including targets met at a single node count.

Run: python -m pytest test_c_quality_search.py   (or: python test_c_quality_search.py)
"""
import evasion_calculator as ec

H100_FP8 = ec.CONFIGS_FP8["16x H100 FP8"]
A100 = ec.CONFIGS["50x A100 80GB"]
WAN = dict(bw_up_bps=ec.BW_UP_BPS, bw_down_bps=ec.BW_DOWN_BPS, latency_s=ec.LATENCY_S)


def flat(cfg, pb):
    return lambda n, h_override=None: ec.compute_generic_scenario(
        cfg, n, target_params_b=pb, h_override=h_override, **WAN)


def hier(cfg, pb, npg=16):
    return lambda n: ec.compute_hierarchical_scenario(
        None, n, nodes_per_group=npg, cfg=cfg, target_params_b=pb,
        regional_bw_bps=ec.BW_UP_BPS, regional_latency_s=ec.LATENCY_S, **WAN)


def pp(cfg, pb):
    return lambda n: ec.compute_pp_diloco_scenario(
        None, n, target_params_b=pb, cfg=cfg,
        pp_bw_bps=ec.BW_UP_BPS, pp_latency_s=ec.LATENCY_S, **WAN)


def check_against_linear_scan(fn, n_max, search_h=False, past_peak=True):
    vals = []
    for n in range(2, n_max + 1):
        r = ec._best_over_h(fn, n) if search_h else fn(n)
        vals.append(r["c_quality"] if r is not None else -1.0)
    peak = max(vals)
    if past_peak:
        assert vals[-1] < peak, "pick n_max past the C_quality peak, so the case is non-monotone"
    # Fractions of the peak, the peak itself (feasible at one node count), just above it
    # (infeasible), and the value at n_max (first met on the rising side).
    for target in [peak * f for f in (0.5, 0.9, 0.99, 0.999, 1.0, 1.001)] + [vals[-1]]:
        expected = next((i + 2 for i, v in enumerate(vals) if v >= target), None)
        r = ec._min_nodes_for_c_quality(fn, target, max_nodes=n_max, search_h=search_h)
        got = r["n_nodes"] if r is not None else None
        assert got == expected, (target, got, expected)
        if r is not None:
            assert r["c_quality"] >= target


def test_flat_with_h_search_matches_linear_scan():
    check_against_linear_scan(flat(H100_FP8, 22.86), 300, search_h=True)  # peak near 50 nodes


def test_hierarchical_matches_linear_scan():
    check_against_linear_scan(hier(H100_FP8, 91.4), 600)  # peak near 145 nodes
    check_against_linear_scan(hier(H100_FP8, 22.86), 300)  # peak in the flat fallback (< 32)


def test_pp_matches_linear_scan():
    # None below two groups (8 nodes at 4 stages), then a step per added group. Over a
    # 100 Mbps pipeline it is still rising at thousands of nodes, so no fall to test here.
    check_against_linear_scan(pp(H100_FP8, 274.3), 400, past_peak=False)


def test_finds_model_that_misses_target_at_max_nodes():
    # 250B, hierarchical (16/group) on 50x A100 reaches 1e25 from 300 nodes, but has fallen
    # back to 5.4e24 by 100,000 nodes, where the old search checked first and gave up.
    fn = hier(A100, 250.0)
    assert fn(100000)["c_quality"] < 1e25
    expected = next(n for n in range(2, 1000) if fn(n)["c_quality"] >= 1e25)
    r = ec._min_nodes_for_c_quality(fn, 1e25)
    assert r is not None and r["n_nodes"] == expected == 300


def test_find_optimal_config_uses_it():
    # Old answer: flat, 320 nodes, $225.9M (the hierarchical option above was dropped).
    r = ec.find_optimal_config_for_target(A100, 1e25)
    assert r["n_nodes"] == 300 and r["mode"] == "Hier 16x18"
    assert round(r["cost_usd"] / 1e6, 1) == 211.8


if __name__ == "__main__":
    test_flat_with_h_search_matches_linear_scan()
    test_hierarchical_matches_linear_scan()
    test_pp_matches_linear_scan()
    test_finds_model_that_misses_target_at_max_nodes()
    test_find_optimal_config_uses_it()
    print("ok")
