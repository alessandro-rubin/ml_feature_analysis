from dataclasses import dataclass, field
from typing import Any

import polars as pl

from tessa import Config
from tessa.analysis import AnalysisContext, prepare_xy, run_analyses


def _toy_df() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "f1": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            "f2": [10.0, 9.0, 8.0, 7.0, 6.0, 5.0],
            "class": ["A", "A", "A", "B", "B", "B"],
            "extra": ["x", "y", "x", "y", "x", "y"],
        }
    )


def test_filter_by_dict():
    ctx = AnalysisContext(
        df=_toy_df(), cfg=Config(), target_col="class",
        label_filter={"class": ["A"]},
    )
    out = ctx.filtered()
    assert out.height == 3


def test_prepare_xy_drops_target_and_strings():
    ctx = AnalysisContext(df=_toy_df(), cfg=Config(), target_col="class")
    prep = prepare_xy(ctx)
    assert prep.feature_cols == ["f1", "f2"]
    assert prep.class_names == ["A", "B"]
    assert len(prep.y) == 6


def test_run_analyses_topo_order():
    log: list[str] = []

    @dataclass
    class A:
        name: str = "a"
        requires: tuple = ()

        def run(self, ctx):
            log.append(self.name)
            return 1

    @dataclass
    class B:
        name: str = "b"
        requires: tuple = ("a",)

        def run(self, ctx):
            log.append(self.name)
            return 2

    ctx = AnalysisContext(df=_toy_df(), cfg=Config(), target_col="class")
    run_analyses([B(), A()], ctx)
    assert log == ["a", "b"]
    assert ctx.results == {"a": 1, "b": 2}
