"""Regression test for find_min_nodes: feasibility is not monotone in node count.

Run: python -m pytest test_find_min_nodes.py   (or: python test_find_min_nodes.py)
"""
import generate_paper_tables as g

CFG = dict(g.ALL_CFGS)["16x H100 FP8"]


def feasible(n, target=1e25, bw=100e6, latency=0.1):
    _, r = g.best_at_nodes(CFG, n, bw, latency, target=target)
    return bool(r and r["c_local"] >= target)


def test_feasible_set_has_a_hole():
    # 91B Flat/Hier reaches 1e25 from 36 nodes until overtraining passes 100x above 156;
    # nothing else reaches it until PP (2 stages, 180B) at 312.
    assert [feasible(n) for n in (35, 36, 156, 157, 311, 312)] == [False, True, True, False, False, True]


def test_find_min_nodes_returns_first_window():
    _, r = g.find_min_nodes(CFG, 1e25, 100e6, 0.1)
    assert r["n_nodes"] == 36


def test_find_min_nodes_finds_window_one_node_wide():
    # 91B Hier(8) peaks at 156 nodes (99.4x overtraining) and passes 100x at 157; with the
    # target set to that peak, 156 is the only feasible node count for hundreds of nodes.
    target = g.best_at_nodes(CFG, 156, 100e6, 0.1)[1]["c_local"]
    assert [feasible(n, target=target) for n in (155, 156, 157)] == [False, True, False]
    _, r = g.find_min_nodes(CFG, target, 100e6, 0.1)
    assert r["n_nodes"] == 156


def test_find_min_nodes_matches_linear_scan_near_overtraining_edge():
    # The answer must be the first feasible node count found by checking every n.
    # 2.70e25 and 2.75e25 have windows several nodes wide (controls); 2.77e25 and 2.785e25
    # have windows of two nodes and one node.
    best = {n: g.best_at_nodes(CFG, n, 100e6, 0.1)[1]["c_local"] for n in range(1, 400)}
    for target in (2.70e25, 2.75e25, 2.77e25, 2.785e25):
        expected = next((n for n in range(1, 400) if best[n] >= target), None)
        assert expected is not None, target
        _, r = g.find_min_nodes(CFG, target, 100e6, 0.1)
        assert r is not None and r["n_nodes"] == expected, target
        assert r["c_local"] >= target
        assert g.OT_MIN <= r["overtraining_ratio"] <= g.OT_MAX


if __name__ == "__main__":
    test_feasible_set_has_a_hole()
    test_find_min_nodes_returns_first_window()
    test_find_min_nodes_finds_window_one_node_wide()
    test_find_min_nodes_matches_linear_scan_near_overtraining_edge()
    print("ok")
