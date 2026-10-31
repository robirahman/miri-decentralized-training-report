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


if __name__ == "__main__":
    test_feasible_set_has_a_hole()
    test_find_min_nodes_returns_first_window()
    print("ok")
