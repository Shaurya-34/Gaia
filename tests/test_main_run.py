import json

import main_run as mr


def test_cell_targets_follow_adaptive_allocation():
    t = mr.cells()
    n_dist = len(mr.INJ["distance_bins_pc"]) - 1
    assert len(t) == n_dist * (len(mr.INJ["n_star_bins"]) - 1 + 1)
    for (i, j), target in t.items():
        if j == -1:
            assert target == mr.INJ["below5_per_distance_bin"]
        elif mr.INJ["n_star_bins"][j] >= mr.INJ["rich_n_star"]:
            assert target == mr.INJ["trials_per_cell_trainval_rich"]
        else:
            assert target == mr.INJ["trials_per_cell_trainval"]
    # 2 sparse n_* bins and 3 rich bins per distance bin, plus the N < 5 quota
    inj = mr.INJ
    assert sum(t.values()) == n_dist * (2 * inj["trials_per_cell_trainval"] + 3 * inj["trials_per_cell_trainval_rich"]
                                        + inj["below5_per_distance_bin"])


def test_cell_of_edges():
    assert mr.cell_of(50.0, 5) == (0, 0)
    assert mr.cell_of(139.9, 9) == (0, 0)
    assert mr.cell_of(140.0, 10) == (1, 1)
    assert mr.cell_of(500.0, 2000) == (4, 4)   # upper edges fold into the last bin
    assert mr.cell_of(300.0, 4) == (2, -1)     # n_* below 5
    assert mr.cell_of(300.0, 0) == (2, -1)


def test_shard_targets_add_up_to_full_targets():
    t = mr.cells()
    for n in (1, 2, 3, 4):
        shares = [mr.shard_targets(t, i, n) for i in range(n)]
        assert all(sum(s[c] for s in shares) == t[c] for c in t)


def fake_trial(params, seed):
    """Instant stand-in for a detector run: every member survives (as with twins)."""
    row = {k: params[k] for k in ["injection_id", "field_pixel", "dist_pc", "l_deg", "b_deg",
                                  "n_members", "r_half_pc", "sigma_v_kms"]}
    row.update({"seed": str(seed), "n_surviving": params["n_members"], "matched_jaccard": 1.0,
                "runtime_s": 0.0, "peak_memory_gb": 0.0})
    return {"row": row, "group_ids": None, "vetter": None}


def run_fake(tmp_path, **kw):
    from concurrent.futures import ThreadPoolExecutor

    return mr.run(["train", "val"], tmp_path, tmp_path, workers=2, trial_fn=fake_trial,
                  executor=ThreadPoolExecutor(max_workers=2), **kw)


def test_top_up_fills_every_cell_to_its_target(tmp_path):
    filled, attempts, targets = run_fake(tmp_path)
    assert filled == targets
    assert all(a <= mr.INJ["max_attempts_per_cell"] for a in attempts.values())
    import pandas as pd
    t = pd.read_csv(tmp_path / "trials.csv")
    assert len(t) == sum(targets.values()) and t.injection_id.is_unique


def test_top_up_resumes_without_overfilling(tmp_path):
    run_fake(tmp_path, max_trials=40)
    filled, _, targets = run_fake(tmp_path)  # resume
    assert filled == targets


def test_attempt_cap_stops_a_cell_that_never_fills(tmp_path):
    def unlucky(params, seed):
        res = fake_trial(params, seed)
        if 5 <= params["n_members"] < 10:  # these trials always lose all members
            res["row"]["n_surviving"] = 0
        return res

    from concurrent.futures import ThreadPoolExecutor

    filled, attempts, targets = mr.run(["train", "val"], tmp_path, tmp_path, workers=2, trial_fn=unlucky,
                                       executor=ThreadPoolExecutor(max_workers=2))
    starved = [c for c in targets if c[1] == 0]  # the [5, 10) n_* bin
    assert all(filled[c] == 0 and attempts[c] == mr.INJ["max_attempts_per_cell"] for c in starved)


def test_shards_take_disjoint_trials_with_identical_params(tmp_path):
    from concurrent.futures import ThreadPoolExecutor
    import pandas as pd

    a = tmp_path / "s0"
    b = tmp_path / "s1"
    for i, d in enumerate([a, b]):
        mr.run(["train", "val"], d, d, workers=2, trial_fn=fake_trial,
               executor=ThreadPoolExecutor(max_workers=2), shard=i, n_shards=2)
    ta, tb = pd.read_csv(a / "trials.csv"), pd.read_csv(b / "trials.csv")
    assert set(ta.injection_id).isdisjoint(tb.injection_id)
    assert (ta.injection_id % 2 == 0).all() and (tb.injection_id % 2 == 1).all()
    # same k gives the same parameters as a single-process proposal stream
    single = dict(zip(range(10_000), (p for _, p in mr.proposals(["train", "val"], mr.INJ["main_seed"]))))
    for _, r in pd.concat([ta, tb]).iterrows():
        assert abs(single[r.injection_id]["dist_pc"] - r.dist_pc) < 1e-9


def test_proposals_are_seeded_and_use_only_requested_splits():
    blocks = json.loads(mr.injection.SPLITS_PATH.read_text())["blocks"]
    a = [p for _, p in zip(range(50), mr.proposals(["train", "val"], 7))]
    b = [p for _, p in zip(range(50), mr.proposals(["train", "val"], 7))]
    assert a == b
    assert {p[1]["split"] for p in a} <= {"train", "val"}
    for _, p in a:
        assert p["field_pixel"] // 16 in blocks[p["split"]]
        assert mr.INJ["n_members"][0] <= p["n_members"] <= mr.INJ["n_members"][1]
    assert [k for k, _ in a] == list(range(50))
