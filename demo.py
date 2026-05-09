"""
End-to-end demo for ml_analysis.

Generates synthetic ~1 Hz time-series data for three assets, builds a label
table with four event classes (TP / FP / TN / FN) and two replacement-type
strata, then runs the full pipeline:

  data on disk  ->  dataset.build  ->  feature materialisation
               ->  period aggregate  ->  analysis suite

Run:
    python demo.py

No external files required — everything is synthesised in demo_data/.
"""

from __future__ import annotations

import shutil
import textwrap
from datetime import datetime, timedelta
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless

import matplotlib.pyplot as plt
import numpy as np
import polars as pl

from ml_analysis import Config
from ml_analysis.analysis import (
    AnalysisContext,
    ClassifierEvaluation,
    DistributionAnalysis,
    FeatureImportance,
    PairwiseSeparability,
    Stratified,
    run_analyses,
    ClusterAnalysis
)
from ml_analysis.dataset.builder import build
from ml_analysis.features import builtins  # noqa: F401 – registers stock features/aggs
from ml_analysis.features.builtins import (
    make_first_difference,
    make_rolling_mean,
    make_rolling_std,
    make_zscore,
)
from ml_analysis.features.materialize import to_period,to_per_sample,to_windowed

# ── Configuration ─────────────────────────────────────────────────────────────

DATA_ROOT = Path("demo_data")
OUTPUT_DIR = Path("demo_outputs")
RANDOM_SEED = 42
N_EVENTS_PER_CLASS = 20   # events per (class, asset) combination
EVENT_LEN_HOURS = 6       # samples per event at 1-sample/min → 360 rows

cfg = Config(
    data_root=DATA_ROOT,
    output_dir=OUTPUT_DIR,
    random_state=RANDOM_SEED,
    filename_pattern=r"^(?P<asset>[^_]+)_(?P<start>\d{8})_(?P<end>\d{8})\.parquet$",
)


# ── 1. Synthetic data generation ──────────────────────────────────────────────

def _signal(rng: np.random.Generator, n: int, cls: str) -> dict[str, np.ndarray]:
    """Return synthetic signal columns.  TP/FP show a burst; TN/FN are quiet."""
    t = np.arange(n, dtype=float)
    base = rng.normal(0, 1, n)

    if cls in ("TP", "FP"):
        amplitude = rng.uniform(3, 6) if cls == "TP" else rng.uniform(1.5, 3)
        onset = rng.integers(n // 4, 3 * n // 4)
        burst = amplitude * np.exp(-0.5 * ((t - onset) / (n * 0.05)) ** 2)
        temp = base + burst
        vibration = rng.normal(0, 0.5, n) + 0.3 * burst
    else:
        temp = base + rng.normal(0, 0.2, n)
        vibration = rng.normal(0, 0.5, n)

    pressure = rng.normal(10, 1, n) + 0.1 * temp
    return {"temperature": temp, "vibration": vibration, "pressure": pressure}


def generate_synthetic_data(
    assets: list[str],
    classes: list[str],
    replacement_types: list[str],
    n_per_class: int,
    event_len_h: int,
    rng: np.random.Generator,
) -> pl.DataFrame:
    """Write parquet files under DATA_ROOT and return the label table."""
    DATA_ROOT.mkdir(parents=True, exist_ok=True)
    label_rows: list[dict] = []
    base_time = datetime(2024, 1, 1)
    offset_days = 0

    for asset in assets:
        folder = cfg.asset_dir(asset)
        folder.mkdir(parents=True, exist_ok=True)

        for cls in classes:
            repl_type = rng.choice(replacement_types)
            for _ in range(n_per_class):
                start = base_time + timedelta(days=offset_days)
                end = start + timedelta(hours=event_len_h)
                offset_days += 1

                n = event_len_h * 60  # 1-sample/min
                ts = [start + timedelta(minutes=i) for i in range(n)]
                sigs = _signal(rng, n, cls)

                df = pl.DataFrame({"timestamp": ts, **sigs})

                fname = f"{asset}_{start:%Y%m%d}_{end:%Y%m%d}.parquet"
                df.write_parquet(folder / fname)

                label_rows.append(
                    {
                        "asset_id": asset,
                        "start": start,
                        "end": end,
                        "class": cls,
                        "replacement_type": repl_type,
                    }
                )

    return pl.DataFrame(label_rows)


# ── 2. Feature registration ───────────────────────────────────────────────────

def register_features() -> None:
    for signal in ("temperature", "vibration", "pressure"):
        make_rolling_mean(signal, window=10)
        make_rolling_std(signal, window=10)
        make_zscore(signal, window=30)
        make_first_difference(signal)


# ── 3. Pretty-print helpers ───────────────────────────────────────────────────

def _hr(title: str) -> None:
    width = 70
    print(f"\n{'─' * width}")
    print(f"  {title}")
    print(f"{'─' * width}")


def print_importance(result: dict) -> None:
    _hr("Feature Importance")
    tbl = result["table"]
    print(tbl[["rank", "rf_mdi", "perm_mean", "anova_f", "mutual_info", "score_composite"]]
          .head(10)
          .to_string())


def print_classifier(result: dict) -> None:
    _hr("Classifier Evaluation")
    for name, r in result["models"].items():
        print(f"\n  {name}  — accuracy {r['accuracy']:.3f}")
        print(textwrap.indent(
            str(r["confusion_matrix"]), "    "
        ))


def print_pairwise(result: dict) -> None:
    _hr("Pairwise Separability  (top-3 features per pair)")
    for (a, b), df in result["pairs"].items():
        print(f"\n  {a} vs {b}")
        print(textwrap.indent(
            df[["feature", "auc", "cliffs_delta", "ks_p"]].head(3).to_string(index=False),
            "    ",
        ))


def print_distributions(result: dict) -> None:
    _hr("Distribution Analysis  (top-5 features by Kruskal-Wallis)")
    print(result["summary"].head(5)[["feature", "kw_stat", "kw_p"]].to_string(index=False))


def print_clustering(result: dict) -> None:
    _hr(f"Cluster Analysis  (best k = {result['best_k']})")
    metrics = result["metrics"]
    if metrics.empty:
        print("  No clusters with >1 component were found.")
        return
    cols = ["Clusters", "Noise pts", "Silhouette", "ARI", "NMI", "V-measure"]
    print(metrics[cols].round(3).to_string())


def print_stratified(result: dict) -> None:
    _hr("Stratified Importance  (top feature per replacement_type stratum)")
    for stratum, r in result["per_stratum"].items():
        top = r["table"].index[0]
        score = r["table"].loc[top, "score_composite"]
        print(f"  {stratum:12s}  →  {top}  (composite={score:.3f})")


# ── 4. Summary figure ─────────────────────────────────────────────────────────

def save_summary_figure(
    importance_result: dict,
    distributions_result: dict,
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: top-10 composite importance bar chart
    tbl = importance_result["table"].head(10)
    axes[0].barh(tbl.index[::-1], tbl["score_composite"][::-1], color="steelblue")
    axes[0].set_xlabel("Composite importance score")
    axes[0].set_title("Top-10 features by composite importance")

    # Right: top-6 feature Kruskal-Wallis statistic
    summary = distributions_result["summary"].head(6)
    axes[1].bar(range(len(summary)), summary["kw_stat"], color="coral")
    axes[1].set_xticks(range(len(summary)))
    axes[1].set_xticklabels(summary["feature"], rotation=30, ha="right")
    axes[1].set_ylabel("Kruskal-Wallis statistic")
    axes[1].set_title("Top-6 features by class separability (KW)")

    fig.tight_layout()
    path = output_dir / "demo_summary.png"
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  Figure saved → {path}")


# ── 5. Main ───────────────────────────────────────────────────────────────────

def main() -> None:
    print("ml_analysis demo — end-to-end pipeline")

    # --- Regenerate synthetic data ---
    if DATA_ROOT.exists():
        shutil.rmtree(DATA_ROOT)

    rng = np.random.default_rng(RANDOM_SEED)
    assets = ["A01", "A02", "A03"]
    classes = ["TP", "FP", "TN", "FN"]
    replacement_types = ["bearing", "seal"]

    print(f"\n[1/5] Generating synthetic data  ({len(assets)} assets × {len(classes)} classes"
          f" × {N_EVENTS_PER_CLASS} events each) ...")
    labels = generate_synthetic_data(
        assets=assets,
        classes=classes,
        replacement_types=replacement_types,
        n_per_class=N_EVENTS_PER_CLASS,
        event_len_h=EVENT_LEN_HOURS,
        rng=rng,
    )
    print(f"   Label table: {labels.shape[0]} events")

    print("\n[2/5] Registering features ...")
    register_features()

    print("\n[3/5] Building event dataset + materialising period aggregates ...")
    events = build(labels, cfg=cfg)
    period = to_period(
        events,
        cfg=cfg,
        aggregators=["mean", "std", "min", "max", "p05", "p95"],
    )
    print(f"   Period table: {period.shape[0]} rows × {period.shape[1]} columns")

    print("\n[4/5] Running analysis suite ...")
    ctx = AnalysisContext(
        df=period,
        cfg=cfg,
        target_col="class",
        label_filter={"class": ["TP", "FP", "TN", "FN"]},
        stratify_by="replacement_type",
        output_dir=str(OUTPUT_DIR),
    )

    analyses = [
        DistributionAnalysis(),
        PairwiseSeparability(top_n=10),
        FeatureImportance(
            rf_params={"n_estimators": 200, "n_jobs": -1, "random_state": RANDOM_SEED},
            permutation_repeats=5,
        ),
        ClusterAnalysis(),
        ClassifierEvaluation(run_lgb=True, run_xgb=True),
        Stratified(
            inner=FeatureImportance(
                name="importance_strat",
                rf_params={"n_estimators": 100, "n_jobs": -1, "random_state": RANDOM_SEED},
                permutation_repeats=3,
            ),
            by="replacement_type",
        ),
    ]
    results = run_analyses(analyses, ctx)

    print("\n[5/5] Results")
    print_distributions(results["distributions"])
    print_pairwise(results["pairwise"])
    print_importance(results["importance"])
    print_clustering(results["clustering"])
    print_classifier(results["classifier"])
    print_stratified(results["stratified__importance_strat"])
    

    save_summary_figure(
        importance_result=results["importance"],
        distributions_result=results["distributions"],
        output_dir=OUTPUT_DIR,
    )

    print("\nDone.\n")


if __name__ == "__main__":
    main()
