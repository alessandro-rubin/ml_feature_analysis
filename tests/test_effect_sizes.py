import numpy as np

from tessa.analysis.effect_sizes import (
    bootstrap_ci,
    cliffs_delta,
    cohens_d,
    hedges_g,
    js_divergence,
    rank_biserial_from_u,
)


def test_cliffs_delta_extremes():
    a = np.array([1.0, 2.0, 3.0])
    b = np.array([10.0, 11.0, 12.0])
    assert cliffs_delta(a, b) == -1.0
    assert cliffs_delta(b, a) == 1.0
    assert abs(cliffs_delta(a, a)) < 1e-9


def test_cohens_d_sign_and_magnitude():
    rng = np.random.default_rng(0)
    a = rng.normal(0, 1, 200)
    b = rng.normal(1, 1, 200)
    d = cohens_d(a, b)
    assert -1.5 < d < -0.5  # ~1 SD shift, negative because mean(a) < mean(b)


def test_hedges_g_close_to_d_for_large_n():
    rng = np.random.default_rng(0)
    a = rng.normal(0, 1, 500)
    b = rng.normal(1, 1, 500)
    assert np.isclose(cohens_d(a, b), hedges_g(a, b), atol=0.01)


def test_rank_biserial_bounds():
    # complete separation
    n1, n2 = 5, 5
    u = 0.0  # all a < all b
    assert rank_biserial_from_u(u, n1, n2) == 1.0
    u = n1 * n2  # all a > all b
    assert rank_biserial_from_u(u, n1, n2) == -1.0


def test_js_divergence_zero_for_identical():
    a = np.linspace(0, 1, 100)
    assert js_divergence(a, a) < 1e-6


def test_js_divergence_positive_for_disjoint():
    a = np.zeros(50)
    b = np.ones(50)
    assert js_divergence(a, b) > 0.5


def test_bootstrap_ci_contains_point():
    rng = np.random.default_rng(0)
    a = rng.normal(0, 1, 50)
    b = rng.normal(1, 1, 50)
    point, lo, hi = bootstrap_ci(cohens_d, a, b, n_resamples=200, rng=rng)
    assert lo <= point <= hi
