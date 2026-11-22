import numpy as np
import pandas as pd

from fraud.data import make_fixture, split_by_time, load_paysim, TRAIN_END, VALID_END
from fraud.features import ALL_FEATURES, HARD_FEATURES, DRAIN_FEATURES, OnlineFeatures, build


def test_batch_and_online_features_are_identical():
    df = make_fixture(3000, seed=1)
    batch = build(df)
    online = OnlineFeatures()
    rows = pd.DataFrame([online.transform(r) for r in df.to_dict("records")])[ALL_FEATURES]
    assert np.allclose(batch.to_numpy(float), rows.to_numpy(float))


def test_features_use_only_earlier_rows():
    """Changing a later transaction must not change the features of an earlier one."""
    df = make_fixture(1500, seed=2)
    before = build(df)
    changed = df.copy()
    changed.loc[1200:, "amount"] *= 7
    changed.loc[1200:, "nameDest"] = "C999999"
    after = build(changed)
    assert before.iloc[:1200].equals(after.iloc[:1200])


def test_first_time_destination_has_zero_history():
    df = pd.DataFrame({"step": [1, 1, 2], "type": ["TRANSFER"] * 3, "amount": [10.0, 20.0, 30.0],
                       "nameOrig": ["C1", "C2", "C3"], "oldbalanceOrg": [10.0, 20.0, 30.0],
                       "newbalanceOrig": [0.0] * 3, "nameDest": ["C9", "C8", "C9"],
                       "oldbalanceDest": [0.0] * 3, "newbalanceDest": [10.0, 20.0, 30.0],
                       "isFraud": [0, 0, 0], "isFlaggedFraud": [0] * 3})
    x = build(df)
    assert x["dest_prior_count"].tolist() == [0, 0, 1]
    assert x["dest_prior_amount_sum"].tolist() == [0.0, 0.0, 10.0]
    assert x["dest_first_seen"].tolist() == [1, 1, 0]


def test_drain_ratio_and_zero_balance_handling():
    df = make_fixture(50, seed=3)
    df.loc[0, ["oldbalanceOrg", "amount", "newbalanceOrig"]] = [0.0, 100.0, 0.0]
    x = build(df)
    assert x.loc[0, "drain_ratio"] == -1.0 and x.loc[0, "orig_zero_balance"] == 1


def test_hard_feature_set_drops_only_the_drain_features():
    assert set(ALL_FEATURES) - set(HARD_FEATURES) == set(DRAIN_FEATURES)


def test_split_is_chronological_and_disjoint():
    df = make_fixture(4000, seed=4)
    s = split_by_time(df)
    assert s["train"].step.max() <= TRAIN_END < s["valid"].step.min()
    assert s["valid"].step.max() <= VALID_END < s["test"].step.min()
    assert sum(len(v) for v in s.values()) == len(df)


def test_loader_skips_stray_title_line(tmp_path):
    df = make_fixture(20, seed=5)
    p = tmp_path / "ps.csv"
    p.write_text("PS_20174392719_1491204439457_log\n" + df.to_csv(index=False))
    assert len(load_paysim(str(p))) == 20
    q = tmp_path / "plain.csv"
    df.to_csv(q, index=False)
    assert len(load_paysim(str(q))) == 20
