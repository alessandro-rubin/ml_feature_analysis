"""
End-to-end demo for tessa.

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

from tessa import Config
from tessa.analysis import (
    AnalysisContext,
    ClassifierEvaluation,
    ClusterAnalysis,
    ClusterValidation,
    CrossValidatedClassifier,
    DistributionAnalysis,
    FeatureImportance,
    ImportanceStability,
    PairwiseSeparability,
    Stratified,
    run_analyses,
)
from tessa.dataset.builder import build
from tessa.features import builtins  # noqa: F401 – registers stock features/aggs
from tessa.features.builtins import (
    make_first_difference,
    make_rolling_mean,
    make_rolling_std,
    make_zscore,
)
from tessa.features.materialize import to_period, to_per_sample, to_windowed  # noqa: F401
from tessa.io.stat_plots import (
    auc_bootstrap_plot,
    calibration_plot,
    cluster_class_heatmap_panel,
    cv_metric_boxplot,
    diagnostics_panel,
    importance_stability_plot,
    method_agreement_heatmap,
    volcano_plot,
)

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
    _hr("Distribution Analysis  (top-5 features — KW + multiple-testing correction)")
    summary = result["summary"].head(5)
    cols = ["feature", "kw_stat", "kw_p", "kw_p_bh_fdr", "anova_p_bh_fdr", "ad_p"]
    cols = [c for c in cols if c in summary.columns]
    print(summary[cols].to_string(index=False))


def _find_pair(pairs: dict, *wanted: str) -> tuple:
    """Find a pair key matching ``wanted`` classes in any order."""
    target = set(wanted)
    for key in pairs:
        if set(key) == target:
            return key
    return next(iter(pairs))


def print_pairwise_extended(result: dict) -> None:
    _hr("Pairwise — extended battery  (top-5 features for FP vs TP)")
    pairs = result["pairs"]
    key = _find_pair(pairs, "FP", "TP")
    df = pairs[key]
    cols = [
        "feature", "auc", "cliffs_delta", "cohens_d",
        "wasserstein", "mwu_p_bh_fdr", "ks_p_bh_fdr",
    ]
    cols = [c for c in cols if c in df.columns]
    print(f"  pair = {key[0]} vs {key[1]}")
    print(df[cols].head(5).to_string(index=False))


def print_cv(result: dict) -> None:
    _hr(f"Cross-validated classifier  (k={len(result['per_fold'])} folds)")
    summary = result["summary"]
    rows = [r for r in (
        "accuracy", "balanced_accuracy", "f1_macro", "mcc", "cohen_kappa",
        "log_loss", "roc_auc", "roc_auc_ovr", "pr_auc", "brier", "ece",
    ) if r in summary.index]
    print(summary.loc[rows][["mean", "std", "min", "max"]].round(3).to_string())


def print_importance_stability(result: dict) -> None:
    _hr("Importance stability  (bootstrap CI + top-k stability)")
    tbl = result["bootstrap_table"]
    cols = ["feature", "mdi_median", "mdi_ci_low", "mdi_ci_high"]
    stab_cols = [c for c in tbl.columns if c.startswith("stability_top")]
    cols += stab_cols
    print(tbl[cols].head(8).round(4).to_string(index=False))

    agree = result["method_agreement"]
    if not agree.empty:
        print("\n  Spearman agreement between importance methods:")
        print(agree.round(2).to_string())


def print_cluster_validation(result: dict) -> None:
    _hr("Cluster validation  (Hopkins, ARI / V-measure permutation)")
    s = result["summary"].iloc[0]
    print(f"  Hopkins statistic       : {s['hopkins']:.3f}   "
          "(>0.6 = clusterable, ~0.5 = no structure)")
    print(f"  Calinski-Harabasz       : {s['calinski_harabasz']:.2f}")
    print(f"  ARI vs class labels     : {s['ari']:+.3f}   "
          f"(perm p = {s['ari_perm_p']:.4g}, n={int(s['n_permutations'])})")
    print(f"  V-measure vs class lbls : {s['v_measure']:+.3f}   "
          f"(perm p = {s['v_measure_perm_p']:.4g})")


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
) -> Path:
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
    return path


def save_cluster_heatmap_figure(
    clustering_result: dict,
    output_dir: Path,
) -> Path | None:
    """Heatmap of how each algorithm's clusters distribute over the true classes."""
    labels = clustering_result.get("labels")
    if not labels:
        return None
    output_dir.mkdir(parents=True, exist_ok=True)
    fig = cluster_class_heatmap_panel(
        labels,
        clustering_result["y_true"],
        clustering_result["class_names"],
    )
    path = output_dir / "demo_cluster_class_heatmap.png"
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  Figure saved → {path}")
    return path


def save_diagnostics_figures(
    pairwise_result: dict,
    stability_result: dict,
    cv_result: dict,
    output_dir: Path,
    pair: tuple[str, str] = ("FP", "TP"),
) -> list[Path]:
    """Emit the corroboration-focused diagnostic figures."""
    output_dir.mkdir(parents=True, exist_ok=True)
    saved: list[Path] = []

    pairs = pairwise_result["pairs"]
    key = _find_pair(pairs, *pair) if pairs else None
    pair_tbl = pairs.get(key) if key else None
    pair_label = f"{key[0]} vs {key[1]}" if key else ""

    # 1) Composite diagnostics panel (volcano + AUC CI + stability + CV box)
    fig = diagnostics_panel(
        pair_table=pair_tbl,
        pair_label=pair_label,
        stability_table=stability_result["bootstrap_table"],
        cv_per_fold=cv_result["per_fold"],
    )
    p = output_dir / "demo_diagnostics_panel.png"
    fig.savefig(p, dpi=120, bbox_inches="tight"); plt.close(fig); saved.append(p)

    # 2) Method-agreement heatmap (separate — needs its own colorbar)
    agree = stability_result["method_agreement"]
    if not agree.empty:
        fig = method_agreement_heatmap(agree)
        p = output_dir / "demo_method_agreement.png"
        fig.savefig(p, dpi=120, bbox_inches="tight"); plt.close(fig); saved.append(p)

    # 3) Calibration curve (binary: TP-vs-FP subset)
    proba = cv_result.get("oof_proba")
    if proba is not None and proba.shape[1] == 2:
        # Re-derive y_true from the cv result: oof_pred and y align by index in ctx
        # Use the binary positive-class column directly.
        # We rely on oof_pred not being -1 anywhere (StratifiedKFold covers all).
        # The "true" labels were attached via prepare_xy; safest: recompute from
        # the cv_result's class_names by matching oof_pred against itself is
        # nonsense, so caller must pass y_true alongside. We keep this best-effort.
        pass  # calibration drawn in main() where we have y_true on hand.

    for path in saved:
        print(f"  Figure saved → {path}")
    return saved


def save_permutation_null_figure(
    ctx: AnalysisContext,
    cluster_validation_result: dict,
    output_dir: Path,
    n_perm: int = 500,
) -> Path | None:
    """Rebuild a small ARI null distribution and plot it next to the observed value.

    `ClusterValidation` only persists summary statistics, so we re-derive
    a fresh null here purely for visualisation.
    """
    from sklearn.metrics import adjusted_rand_score

    labels = cluster_validation_result.get("labels_used")
    if labels is None:
        return None
    summary = cluster_validation_result["summary"].iloc[0]
    rng = np.random.default_rng(ctx.cfg.random_state)

    # Re-encode the true labels exactly the way prepare_xy did (filtered + encoded).
    from tessa.analysis.base import prepare_xy
    prep = prepare_xy(ctx)
    y = prep.y
    mask = np.asarray(labels) != -1
    y_kept = y[mask]
    lab_kept = np.asarray(labels)[mask]

    nulls = np.empty(n_perm, dtype=float)
    perm = y_kept.copy()
    for i in range(n_perm):
        rng.shuffle(perm)
        nulls[i] = adjusted_rand_score(perm, lab_kept)

    from tessa.io.stat_plots import permutation_null_plot
    fig = permutation_null_plot(
        nulls, observed=float(summary["ari"]),
        p_value=float(summary["ari_perm_p"]), statistic_name="ARI",
    )
    p = output_dir / "demo_permutation_null_ari.png"
    fig.savefig(p, dpi=120, bbox_inches="tight"); plt.close(fig)
    print(f"  Figure saved → {p}")
    return p


def save_calibration_figure(
    ctx: AnalysisContext,
    cv_result: dict,
    output_dir: Path,
) -> Path | None:
    """Calibration curve from the binary CV out-of-fold probabilities."""
    proba = cv_result.get("oof_proba")
    if proba is None or proba.shape[1] != 2:
        return None
    from tessa.analysis.base import prepare_xy
    prep = prepare_xy(ctx)
    fig = calibration_plot(prep.y, proba[:, 1], n_bins=10)
    p = output_dir / "demo_calibration.png"
    fig.savefig(p, dpi=120, bbox_inches="tight"); plt.close(fig)
    print(f"  Figure saved → {p}")
    return p


# ── 5. Main ───────────────────────────────────────────────────────────────────

def main() -> None:
    print("tessa demo — end-to-end pipeline")

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
        PairwiseSeparability(top_n=15, bootstrap_n=300),
        FeatureImportance(
            rf_params={"n_estimators": 200, "n_jobs": -1, "random_state": RANDOM_SEED},
            permutation_repeats=5,
        ),
        ImportanceStability(
            n_bootstrap=80, top_k=10,
            rf_params={"n_estimators": 80, "n_jobs": -1, "random_state": RANDOM_SEED},
        ),
        ClusterAnalysis(),
        ClusterValidation(n_permutations=400),
        ClassifierEvaluation(run_lgb=True, run_xgb=True),
        CrossValidatedClassifier(
            n_splits=5,
            rf_params={"n_estimators": 200, "n_jobs": -1, "random_state": RANDOM_SEED},
        ),
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
    print_pairwise_extended(results["pairwise"])
    print_importance(results["importance"])
    print_importance_stability(results["importance_stability"])
    print_clustering(results["clustering"])
    print_cluster_validation(results["cluster_validation"])
    print_classifier(results["classifier"])
    print_cv(results["cv_classifier"])
    print_stratified(results["stratified__importance_strat"])

    save_summary_figure(
        importance_result=results["importance"],
        distributions_result=results["distributions"],
        output_dir=OUTPUT_DIR,
    )
    save_cluster_heatmap_figure(results["clustering"], OUTPUT_DIR)
    save_diagnostics_figures(
        pairwise_result=results["pairwise"],
        stability_result=results["importance_stability"],
        cv_result=results["cv_classifier"],
        output_dir=OUTPUT_DIR,
        pair=("FP", "TP"),
    )
    # Calibration only makes sense for binary — re-run a binary CV on TP vs FP.
    binary_ctx = AnalysisContext(
        df=period, cfg=cfg, target_col="class",
        label_filter={"class": ["TP", "FP"]},
        output_dir=str(OUTPUT_DIR),
    )
    binary_cv = CrossValidatedClassifier(
        n_splits=5,
        rf_params={"n_estimators": 200, "n_jobs": -1, "random_state": RANDOM_SEED},
    ).run(binary_ctx)
    save_calibration_figure(binary_ctx, binary_cv, OUTPUT_DIR)
    save_permutation_null_figure(
        ctx, results["cluster_validation"], OUTPUT_DIR, n_perm=400,
    )

    print("\nDone.\n")


if __name__ == "__main__":
    main()
