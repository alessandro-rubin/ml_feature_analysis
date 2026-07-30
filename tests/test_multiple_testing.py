import numpy as np

from tessa.analysis.multiple_testing import (
    benjamini_hochberg,
    bonferroni,
    holm,
)


def test_bonferroni_caps_at_one():
    p = np.array([0.01, 0.02, 0.5, 0.9])
    adj = bonferroni(p)
    assert adj[0] == 0.04
    assert adj[1] == 0.08
    assert adj[3] == 1.0


def test_bh_fdr_monotone():
    p = np.array([0.001, 0.01, 0.04, 0.2, 0.7])
    q = benjamini_hochberg(p)
    assert np.all(np.diff(q) >= -1e-12)
    assert np.all(q <= 1.0)
    assert q[0] < q[-1]


def test_holm_less_strict_than_bonferroni():
    p = np.array([0.01, 0.02, 0.04])
    h = holm(p)
    b = bonferroni(p)
    # Holm should never exceed Bonferroni
    assert np.all(h <= b + 1e-12)
    # Smallest p gets the same correction factor as Bonferroni
    assert np.isclose(h[0], b[0])


def test_nan_passthrough():
    p = np.array([0.01, np.nan, 0.2])
    for fn in (bonferroni, holm, benjamini_hochberg):
        adj = fn(p)
        assert np.isnan(adj[1])
        assert np.isfinite(adj[0]) and np.isfinite(adj[2])
