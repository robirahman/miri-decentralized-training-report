"""Regression test: the Chinchilla overtraining penalty (chi) applies at one node.

A single node can over- or under-train like any other run; chi used to be forced to 1
there, so a one-node row reported, e.g., a 10B model at 53x overtraining as full quality.
The replica penalty still needs more than one replica, so C_local is unchanged.

Run: python -m pytest test_single_node_chi.py   (or: python test_single_node_chi.py)
"""
import contextlib
import io
import math

import evasion_calculator as ec

H100_FP8 = ec.CONFIGS_FP8["16x H100 FP8"]
TIME_S = 740 * 86400


def expected_chi(r, params_b):
    params, tokens = params_b * 1e9, r["total_tokens_T"] * 1e12
    return ec.chinchilla_efficiency(params, tokens, 6 * params * tokens)


def test_generic_scenario_single_node_chi():
    r = ec.compute_generic_scenario(H100_FP8, 1, time_seconds=TIME_S, target_params_b=10)
    assert r["overtraining_ratio"] > 50
    assert math.isclose(r["chi"], expected_chi(r, 10), rel_tol=1e-6)
    assert r["chi"] < 0.3
    # No replica penalty on one node: eta and C_local are as before, only C_quality moves.
    assert r["eta"] == 1.0 and r["eta_replica"] == 1.0
    assert math.isclose(r["c_quality"], r["c_local"] * r["chi"], rel_tol=1e-12)


def test_named_scenario_single_node_chi():
    r = ec.compute_scenario("16x H100 FP8", 1, time_seconds=TIME_S, target_params_b=10)
    assert math.isclose(r["chi"], expected_chi(r, 10), rel_tol=1e-6)
    assert r["chi"] < 0.3


def test_moe_single_node_chi():
    r = ec.compute_moe_ep_scenario("16x H100 FP8", 1, 100, 20)
    assert 0 < r["chi"] < 1


def test_ccc_cost_analysis_single_node_chi():
    with contextlib.redirect_stdout(io.StringIO()):  # the module prints its report on import
        import ccc_cost_analysis as ccc
    r = ccc.compute(ccc.HARDWARE["48x A100 80GB"], 1)
    assert r["overtraining_ratio"] < 0.1  # 240B model on one node: heavily undertrained
    assert r["eta_chinchilla"] < 0.5


def test_multi_node_unchanged_path():
    # Above one node chi was already computed; spot-check it still matches the formula.
    r = ec.compute_generic_scenario(H100_FP8, 36, time_seconds=TIME_S, target_params_b=91,
                                    bw_up_bps=100e6, bw_down_bps=100e6, latency_s=0.1)
    assert math.isclose(r["chi"], expected_chi(r, 91), rel_tol=1e-6)


if __name__ == "__main__":
    test_generic_scenario_single_node_chi()
    test_named_scenario_single_node_chi()
    test_moe_single_node_chi()
    test_ccc_cost_analysis_single_node_chi()
    test_multi_node_unchanged_path()
    print("ok")
