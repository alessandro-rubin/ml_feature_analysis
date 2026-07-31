"""Step-6: lagged relations + mutual-information network."""

from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np
import polars as pl
import pytest

from tessa.analysis import AnalysisContext, LaggedRelations, MutualInfoNetwork
from tessa.analysis.relations import lagged_correlations
from tessa.config import Config


def _lagged_df(n: int = 500, lag: int = 5, seed: int = 0) -> pl.DataFrame:
    """`cause` leads `effect` by `lag` samples; `noise` is independent."""
    rng = np.random.default_rng(seed)
    cause = rng.normal(0, 1, n + lag)
    effect = cause[:-lag] + 0.1 * rng.normal(0, 1, n)
    t0 = datetime(2024, 1, 1)
    return pl.DataFrame(
        {
            "timestamp": [t0 + timedelta(minutes=i) for i in range(n)],
            "cause": cause[lag:],  # shift so effect[t] ~ cause[t - lag]...
            "effect": effect,
            "noise": rng.normal(0, 1, n),
        }
    )


def test_lagged_correlations_finds_the_lag():
    rng = np.random.default_rng(0)
    n, lag = 400, 7
    x = rng.normal(0, 1, n)
    y = np.roll(x, lag) + 0.05 * rng.normal(0, 1, n)
    y[:lag] = rng.normal(0, 1, lag)
    lc = lagged_correlations(x, y, max_lag=15)
    best = lc.iloc[lc["correlation"].abs().idxmax()]
    assert best["lag"] == lag
    assert best["correlation"] > 0.9


def test_lagged_relations_with_reference():
    ctx = AnalysisContext(df=_lagged_df(), cfg=Config())
    out = LaggedRelations(reference="cause", max_lag=10).run(ctx)
    table = out["table"]
    row = table[table["following"] == "effect"].iloc[0]
    assert abs(row["correlation_at_best_lag"]) > 0.9
    assert row["best_lag"] != 0  # the association is lagged, not instant
    noise_row = table[table["following"] == "noise"].iloc[0]
    assert abs(noise_row["correlation_at_best_lag"]) < 0.3
    assert "not causation" in out["note"]


def test_lagged_relations_pair_explosion_guard():
    rng = np.random.default_rng(0)
    df = pl.DataFrame({f"c{i}": rng.normal(0, 1, 50) for i in range(40)})
    ctx = AnalysisContext(df=df, cfg=Config())
    with pytest.raises(ValueError, match="pairs"):
        LaggedRelations(max_lag=3).run(ctx)


def test_mi_network_finds_nonlinear_dependence():
    rng = np.random.default_rng(0)
    n = 400
    x = rng.normal(0, 1, n)
    df = pl.DataFrame(
        {
            "x": x,
            "x_squared": x**2 + 0.05 * rng.normal(0, 1, n),  # nonlinear in x
            "indep": rng.normal(0, 1, n),
        }
    )
    ctx = AnalysisContext(df=df, cfg=Config())
    out = MutualInfoNetwork(edge_threshold=0.1).run(ctx)
    edges = out["edges"]
    top = edges.iloc[0]
    assert {top["feature_a"], top["feature_b"]} == {"x", "x_squared"}
    m = out["matrix"]
    assert m.loc["x", "x_squared"] > 3 * max(m.loc["x", "indep"], 1e-6)


def test_mi_network_caps_features():
    rng = np.random.default_rng(0)
    df = pl.DataFrame({f"c{i}": rng.normal(0, 1, 60) for i in range(12)})
    ctx = AnalysisContext(df=df, cfg=Config())
    out = MutualInfoNetwork(max_features=5).run(ctx)
    assert out["n_features_used"] == 5
    assert out["matrix"].shape == (5, 5)
