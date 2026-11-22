import numpy as np

from fraud import evaluate as ev


def _toy(n=20000, prevalence=0.01, seed=0):
    rng = np.random.default_rng(seed)
    y = (rng.random(n) < prevalence).astype(int)
    return y, rng


def test_random_scores_sit_at_the_prevalence_not_at_99_percent():
    """Negative control: a useless model must look useless on PR-AUC and recall at a budget."""
    y, rng = _toy()
    s = rng.random(len(y))
    assert ev.pr_auc(y, s) < 0.05
    assert ev.alert_budget(y, s, 0.01)["recall"] < 0.05


def test_perfect_scores_are_perfect():
    y, _ = _toy()
    assert ev.pr_auc(y, y.astype(float)) == 1.0
    r = ev.alert_budget(y, y.astype(float), y.mean())
    assert r["recall"] > 0.99


def test_always_legit_has_99_percent_accuracy_but_zero_recall():
    y, _ = _toy()
    assert (y == 0).mean() > 0.98
    assert ev.alert_budget(y, np.zeros(len(y)), 0.001)["recall"] <= 0.001 / y.mean() + 0.05


def test_cost_curve_matches_hand_calculation():
    y = np.array([1, 0, 1, 0]); amount = np.array([100.0, 50.0, 300.0, 10.0]); s = np.array([0.9, 0.8, 0.7, 0.1])
    c = ev.cost_curve(y, s, amount, review_cost=10)
    # review top-1: miss 300, pay 10 -> 310 ; top-2: miss 300, pay 20 ; top-3: miss 0, pay 30
    assert c["cost"].tolist() == [310.0, 320.0, 30.0, 40.0]
    assert ev.best_k(y, s, amount, review_cost=10) == 3


def test_best_k_can_choose_to_review_nothing():
    y = np.array([1, 0, 0]); amount = np.array([5.0, 1.0, 1.0]); s = np.array([0.9, 0.5, 0.1])
    assert ev.best_k(y, s, amount, review_cost=1000) == 0


def test_cost_at_threshold_counts_alerts_and_misses():
    y = np.array([1, 1, 0]); amount = np.array([10.0, 20.0, 5.0]); s = np.array([0.9, 0.2, 0.8])
    r = ev.cost_at_threshold(y, s, amount, 0.5, review_cost=1)
    assert (r["alerts"], r["caught"], r["missed_amount"], r["total_cost"]) == (2, 1, 20.0, 22.0)


def test_psi_is_zero_for_same_distribution_and_large_for_shift():
    rng = np.random.default_rng(0)
    a = rng.normal(0, 1, 20000)
    assert ev.psi(a, rng.normal(0, 1, 20000)) < 0.02
    assert ev.psi(a, rng.normal(2, 1, 20000)) > 0.5


def test_bootstrap_interval_contains_point_estimate():
    y, rng = _toy(5000, 0.05)
    s = y * 0.5 + rng.random(len(y))
    lo, hi = ev.bootstrap_pr_auc(y, s, n=100)
    assert lo <= ev.pr_auc(y, s) <= hi
