"""
End-to-end demo for tessa.

Generates synthetic 1-sample/min time-series data for three assets, builds a
label table with four event classes (TP / FP / TN / FN) and two
replacement-type strata, then runs the full pipeline:

  data on disk  ->  dataset.build  ->  feature materialisation
               ->  period aggregate  ->  analysis suite
               ->  separability / anomaly / semi-supervised / changepoint
               ->  ResultStore run + static HTML report
               ->  figures: one-page overview + every curated plot

Run:
    python demo.py

No external files required — everything is synthesised in demo_data/.
demo_notebook.ipynb imports this module and reuses `prepare_data` and
`analysis_suite`, so the script and the notebook run the same pipeline.
"""

from __future__ import annotations

import io
import shutil
import sys
import textwrap
from datetime import datetime, timedelta
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polars as pl

from sklearn.metrics import confusion_matrix

from tessa import Config, Run
from tessa.analysis import (
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
from tessa.features.materialize import to_period
from tessa.features.registry import FeatureRegistry
from tessa.io.stat_plots import (
    class_colors,
    confusion_headline,
    confusion_matrix_plot,
    dominant_confusion,
    pair_separability_headline,
    pair_separability_matrix,
    theme,
)
from tessa.results import AnalysisResult
from tessa.results.figures import figures_for_result

# ── Configuration ─────────────────────────────────────────────────────────────

DATA_ROOT = Path("demo_data")
OUTPUT_DIR = Path("demo_outputs")
RANDOM_SEED = 42
ASSETS = ["A01", "A02", "A03"]
CLASSES = ["TP", "FP", "TN", "FN"]
REPLACEMENT_TYPES = ["bearing", "seal"]
N_EVENTS_PER_CLASS = 20  # events per (class, asset) combination
EVENT_LEN_HOURS = 6  # samples per event at 1-sample/min → 360 rows

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
            for _ in range(n_per_class):
                # Drawn per event, so every stratum holds every class.
                repl_type = str(rng.choice(replacement_types))
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


def register_features() -> FeatureRegistry:
    """Return a fresh registry holding the demo's rolling features.

    A private registry rather than the process-wide default, which refuses
    duplicate names, so this can run again (e.g. a re-run notebook cell).
    """
    registry = FeatureRegistry()
    for signal in ("temperature", "vibration", "pressure"):
        make_rolling_mean(signal, window=10, registry=registry)
        make_rolling_std(signal, window=10, registry=registry)
        make_zscore(signal, window=30, registry=registry)
        make_first_difference(signal, registry=registry)
    return registry


# ── 2b. Shared pipeline (also used by demo_notebook.ipynb) ────────────────────


def prepare_data(
    rng: np.random.Generator,
) -> tuple[pl.DataFrame, dict[str, pl.LazyFrame], pl.DataFrame]:
    """Synthesise the data, build the events and aggregate one row per event.

    Returns ``(labels, events, period)``.
    """
    # Only our own asset folders: the AI demo (`tessa-agent --make-demo`) may
    # keep its data under the same root.
    for asset in ASSETS:
        shutil.rmtree(cfg.asset_dir(asset), ignore_errors=True)

    print(
        f"  Synthetic data: {len(ASSETS)} assets × {len(CLASSES)} classes"
        f" × {N_EVENTS_PER_CLASS} events each"
    )
    labels = generate_synthetic_data(
        assets=ASSETS,
        classes=CLASSES,
        replacement_types=REPLACEMENT_TYPES,
        n_per_class=N_EVENTS_PER_CLASS,
        event_len_h=EVENT_LEN_HOURS,
        rng=rng,
    )
    print(f"  Label table : {labels.shape[0]} events")

    events = build(labels, cfg=cfg)
    period = to_period(
        events,
        cfg=cfg,
        aggregators=["mean", "std", "min", "max", "p05", "p95"],
        feature_registry=register_features(),
    )
    print(f"  Period table: {period.shape[0]} rows × {period.shape[1]} columns")
    return labels, events, period


def analysis_suite() -> list:
    """The supervised and corroborating analyses run on the period table."""
    return [
        DistributionAnalysis(),
        # No top_n: the volcano plot needs every feature, not only the winners.
        PairwiseSeparability(bootstrap_n=300),
        FeatureImportance(
            rf_params={"n_estimators": 200, "n_jobs": -1, "random_state": RANDOM_SEED},
            permutation_repeats=5,
        ),
        ImportanceStability(
            n_bootstrap=80,
            top_k=10,
            rf_params={"n_estimators": 80, "n_jobs": -1, "random_state": RANDOM_SEED},
        ),
        ClusterAnalysis(),
        ClusterValidation(n_permutations=400),
        ClassifierEvaluation(run_lgb=True, run_xgb=True),
        CrossValidatedClassifier(
            n_splits=5,  # folds are grouped by asset, so this caps at 3 here
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


# ── 3. Pretty-print helpers ───────────────────────────────────────────────────


def _hr(title: str) -> None:
    width = 70
    print(f"\n{'─' * width}")
    print(f"  {title}")
    print(f"{'─' * width}")


def print_importance(result: dict) -> None:
    _hr("Feature Importance")
    tbl = result["table"]
    print(
        tbl[["rank", "rf_mdi", "perm_mean", "anova_f", "mutual_info", "score_composite"]]
        .head(10)
        .to_string()
    )


def print_classifier(result: dict) -> None:
    _hr("Classifier Evaluation  (confusion matrix: rows = true, columns = predicted)")
    names = result["class_names"]
    for name, r in result["models"].items():
        print(f"\n  {name}  — accuracy {r['accuracy']:.3f}")
        cm = pd.DataFrame(r["confusion_matrix"], index=names, columns=names)
        print(textwrap.indent(cm.to_string(), "    "))


def print_pairwise(result: dict) -> None:
    _hr("Pairwise Separability  (top-3 features per pair)")
    for (a, b), df in result["pairs"].items():
        print(f"\n  {a} vs {b}")
        print(
            textwrap.indent(
                df[["feature", "auc", "cliffs_delta", "ks_p"]].head(3).to_string(index=False),
                "    ",
            )
        )


def print_distributions(result: dict) -> None:
    _hr("Distribution Analysis  (top-5 features — KW + multiple-testing correction)")
    summary = result["summary"].head(5)
    cols = ["feature", "kw_stat", "kw_p", "kw_p_bh_fdr", "anova_p_bh_fdr", "ad_p"]
    cols = [c for c in cols if c in summary.columns]
    print(summary[cols].to_string(index=False))


def find_pair(pairs: dict, *wanted: str) -> tuple:
    """Find a pair key matching ``wanted`` classes in any order."""
    target = set(wanted)
    for key in pairs:
        if set(key) == target:
            return key
    return next(iter(pairs))


def print_pairwise_extended(result: dict) -> None:
    _hr("Pairwise — extended battery  (top-5 features for FP vs TP)")
    pairs = result["pairs"]
    key = find_pair(pairs, "FP", "TP")
    df = pairs[key]
    cols = [
        "feature",
        "auc",
        "cliffs_delta",
        "cohens_d",
        "wasserstein",
        "mwu_p_bh_fdr",
        "ks_p_bh_fdr",
    ]
    cols = [c for c in cols if c in df.columns]
    print(f"  pair = {key[0]} vs {key[1]}")
    print(df[cols].head(5).to_string(index=False))


def print_cv(result: dict) -> None:
    _hr(f"Cross-validated classifier  (k={len(result['per_fold'])} folds)")
    summary = result["summary"]
    rows = [
        r
        for r in (
            "accuracy",
            "balanced_accuracy",
            "f1_macro",
            "mcc",
            "cohen_kappa",
            "log_loss",
            "roc_auc",
            "roc_auc_ovr",
            "pr_auc",
            "brier",
            "ece",
        )
        if r in summary.index
    ]
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
    print(
        f"  Hopkins statistic       : {s['hopkins']:.3f}   "
        "(>0.6 = clusterable, ~0.5 = no structure)"
    )
    print(f"  Calinski-Harabasz       : {s['calinski_harabasz']:.2f}")
    print(
        f"  ARI vs class labels     : {s['ari']:+.3f}   "
        f"(perm p = {s['ari_perm_p']:.4g}, n={int(s['n_permutations'])})"
    )
    print(
        f"  V-measure vs class lbls : {s['v_measure']:+.3f}   "
        f"(perm p = {s['v_measure_perm_p']:.4g})"
    )


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


# ── 4. Figures ────────────────────────────────────────────────────────────────


def example_events_figure(
    events: dict[str, pl.LazyFrame],
    channel: str = "temperature",
    per_class: int = 12,
    smooth_min: int = 15,
    axes: list[plt.Axes] | None = None,
) -> plt.Figure:
    """Overlay a few raw events per class: the signal the analyses separate.

    Each thin line is one event's ``channel`` after a ``smooth_min``-minute
    rolling mean, plotted against hours since the event start. Pass ``axes``
    (one per class) to draw into an existing figure.
    """
    by_class: dict[str, list[pl.DataFrame]] = {cls: [] for cls in CLASSES}
    for lf in events.values():
        if all(len(frames) >= per_class for frames in by_class.values()):
            break
        df = lf.select("timestamp", channel, "class").collect()
        frames = by_class.get(df["class"][0])
        if frames is not None and len(frames) < per_class:
            frames.append(df)

    colors = class_colors(CLASSES)
    with theme():
        if axes is None:
            _, grid = plt.subplots(
                1, len(CLASSES), figsize=(3.3 * len(CLASSES), 3), sharey=True, squeeze=False
            )
            axes = list(grid[0])
        for ax, cls in zip(axes, CLASSES):
            for df in by_class[cls]:
                hours = (df["timestamp"] - df["timestamp"][0]).dt.total_minutes() / 60
                smooth = df[channel].rolling_mean(smooth_min)  # leading nulls: no warm-up spike
                ax.plot(hours, smooth, color=colors[cls], lw=0.8, alpha=0.75)
            ax.set_title(f"{cls}  ({len(by_class[cls])} events)")
            ax.set_xlabel("hours since event start")
            ax.yaxis.grid(True)
        axes[0].set_ylabel(f"{channel}, {smooth_min}-min mean")
    return axes[0].figure


def overview_figure(events: dict[str, pl.LazyFrame], results: dict) -> plt.Figure:
    """The demo's answer on one page.

    Top: raw events per class. Bottom left: which class pairs separate (best
    single-feature AUC among features surviving BH-FDR). Bottom right: where
    the cross-validated classifier's errors go.
    """
    cv = results["cv_classifier"]
    names = list(cv["class_names"])
    cm = confusion_matrix(cv["y_true"], cv["oof_pred"], labels=range(len(names)))
    pair_summary = results["pairwise"]["pair_summary"]

    with theme():
        fig = plt.figure(figsize=(14, 9.5))
        grid = fig.add_gridspec(2, 4, height_ratios=[1, 1.5], hspace=0.45, wspace=0.35)
        top = [fig.add_subplot(grid[0, 0])]
        top += [fig.add_subplot(grid[0, i], sharey=top[0]) for i in range(1, len(CLASSES))]
        example_events_figure(events, axes=top)
        pair_separability_matrix(pair_summary, class_names=names, ax=fig.add_subplot(grid[1, :2]))
        confusion_matrix_plot(
            cm,
            names,
            title="Cross-validated confusion (out-of-fold)\n" + confusion_headline(cm, names),
            ax=fig.add_subplot(grid[1, 2:]),
        )
        headline = pair_separability_headline(pair_summary)
        dom = dominant_confusion(cm, names)
        if dom is not None:
            headline += f" — {dom[0]:.0%} of classifier errors are {dom[1]} ↔ {dom[2]}"
        fig.suptitle(
            f"{headline}\nTP and FP events carry a temperature burst; "
            "TN and FN are generated identically, so nothing should separate them",
            y=0.99,
        )
    return fig


def save_figures(
    run: Run,
    binary: Run,
    events: dict[str, pl.LazyFrame],
    results: dict,
    output_dir: Path,
) -> list[Path]:
    """Write the overview plus every curated figure of both runs as PNGs.

    The per-analysis plots come from the same factory as the HTML report and
    the dashboard, so all three show the same thing. ``figures/`` is rebuilt
    from scratch so renamed or dropped plots don't linger.
    """
    fig_dir = output_dir / "figures"
    shutil.rmtree(fig_dir, ignore_errors=True)
    fig_dir.mkdir(parents=True)

    saved = [output_dir / "overview.png"]
    fig = overview_figure(events, results)
    fig.savefig(saved[0], dpi=120, bbox_inches="tight")
    plt.close(fig)

    for prefix, r in (("", run), ("tp_vs_fp__", binary)):
        # One analysis at a time keeps only a handful of figures open.
        for name, raw in r.ctx.results.items():
            titled = figures_for_result(AnalysisResult.from_raw(name, raw))
            for i, (_, fig) in enumerate(titled):
                path = fig_dir / f"{prefix}{name}_{i}.png"
                fig.savefig(path, dpi=110, bbox_inches="tight")
                plt.close(fig)
                saved.append(path)
    return saved


# ── 4b. New-capabilities tour (v0.1: unsupervised / semi-supervised / report) ──


def run_new_capabilities(
    run: Run,
    period: pl.DataFrame,
    events: dict[str, pl.LazyFrame],
    rng: np.random.Generator,
) -> None:
    """Showcase the v0.1 additions on the same synthetic data:

    separability test, anomaly ensemble, correlation structure, MI network,
    label spreading, PU learning, changepoint + lagged relations on a raw
    event, and ResultStore + static HTML report persistence.

    ``run`` already holds the analysis suite, so the saved run and the
    report cover everything, not just this section.
    """
    _hr("Separability — are the classes distinguishable at all?")
    sep = run.separability(
        n_permutations=200,
        rf_params={"n_estimators": 100, "n_jobs": -1},
    )
    print(sep.frames["summary"].round(4).to_string(index=False))

    _hr("Anomaly detection  (unsupervised; baseline = quiet TN/FN events)")
    ano = run.anomaly(
        baseline_filter={"class": ["TN", "FN"]},
        iforest_params={"n_estimators": 200, "n_jobs": -1},
    )
    scores = ano.frames["scores"].merge(
        period.select("event_id", "class").to_pandas(), on="event_id"
    )
    print("  mean ensemble score by true class (bursty TP/FP should rank high):")
    print(textwrap.indent(scores.groupby("class")["ensemble"].mean().round(3).to_string(), "    "))
    top5 = scores.nlargest(5, "ensemble")
    contributors = ano.objects["top_contributors"]
    print("\n  top-5 anomalous events and their #1 contributing feature:")
    for idx, row in top5.iterrows():
        feat, z = contributors[idx][0]
        print(
            f"    {row['event_id']:38s} class={row['class']:3s} "
            f"score={row['ensemble']:.3f}  ← {feat} (z={z:+.1f})"
        )

    _hr("Correlation structure  (redundant channels)")
    corr = run.correlation_structure()
    print(
        f"  {corr.scalars['n_features']} features → "
        f"{corr.scalars['n_clusters']} correlation clusters"
    )
    dups = corr.frames["duplicates"]
    if len(dups):
        print("  strongest near-duplicates:")
        print(textwrap.indent(dups.head(5).round(3).to_string(index=False), "    "))

    _hr("Mutual-information network  (nonlinear dependences, exploratory)")
    mi = run.mi_network(max_features=15)
    print(textwrap.indent(mi.frames["edges"].head(5).round(3).to_string(index=False), "    "))

    _hr("Label spreading  (15% of labels kept, rest recovered)")
    keep = rng.random(period.height) < 0.15
    sparse = period.with_columns(
        pl.when(pl.Series(keep)).then(pl.col("class")).otherwise(None).alias("class")
    )
    semi = Run(sparse, target_col="class", cfg=cfg)
    ls = semi.label_spreading()
    pred = ls.frames["table"]["predicted_label"].to_numpy()
    truth = period["class"].to_numpy()
    acc = float((pred == truth).mean())
    print(f"  labeled rows used : {ls.scalars['n_labeled']} / {period.height}")
    print(
        f"  recovery accuracy : {acc:.1%} on all rows  "
        "(chance = 25%; TN vs FN are identical by construction, so the "
        "practical ceiling is ~75%)"
    )

    _hr("PU learning  (only TP labeled positive; who else looks like one?)")
    pu_labels = pl.when(pl.col("class") == "TP").then(pl.lit("TP")).otherwise(None)
    pu_run = Run(period.with_columns(pu_labels.alias("class")), target_col="class", cfg=cfg)
    pu = pu_run.pu_learning(
        positive_label="TP",
        n_iterations=20,
        rf_params={"n_estimators": 60, "n_jobs": -1},
    )
    ranked = pu.frames["ranked_unlabeled"].merge(
        period.select("event_id", pl.col("class").alias("true_class")).to_pandas(),
        on="event_id",
    )
    print("  top-5 unlabeled events by PU score (FP bursts should surface):")
    print(
        textwrap.indent(
            ranked[["event_id", "true_class", "pu_score"]].head(5).round(3).to_string(index=False),
            "    ",
        )
    )

    # Raw-signal analyses need a time-indexed series: take one bursty event.
    tp_lf = next(
        lf for lf in events.values() if lf.select(pl.col("class").first()).collect().item() == "TP"
    )
    event_df = tp_lf.collect()

    _hr("Changepoint detection  (CUSUM on one raw TP event)")
    cp = Run(event_df, cfg=cfg).changepoint(channels=["temperature", "vibration"])
    tbl = cp.frames["table"]
    if len(tbl):
        print(
            textwrap.indent(
                tbl[["channel", "position", "direction", "statistic"]]
                .head(5)
                .round(2)
                .to_string(index=False),
                "    ",
            )
        )
    else:
        print("  no regime change detected")

    _hr("Lagged relations  (reference = temperature, exploratory)")
    lr = Run(event_df, cfg=cfg).lagged_relations(
        reference="temperature",
        max_lag=15,
        channels=["temperature", "vibration", "pressure"],
    )
    print(textwrap.indent(lr.frames["table"].round(3).to_string(index=False), "    "))
    print(f"    note: {lr.scalars['note']}")

    _hr("Persistence  (ResultStore run + self-contained HTML report)")
    run_dir = run.save(OUTPUT_DIR / "runs", name="demo_run")
    report = run.report(OUTPUT_DIR / "demo_report.html", title="tessa demo report")
    print(f"  Run saved   → {run_dir}  (manifest + parquet, dashboard-ready)")
    print(f"  Report      → {report}")
    print(
        f"  Dashboard   → streamlit run src/tessa/dashboard/app.py -- --root {OUTPUT_DIR / 'runs'}"
    )


# ── 5. Main ───────────────────────────────────────────────────────────────────


def main() -> None:
    # Script-only setup, kept out of import time so the notebook importing this
    # module keeps its own console and inline plotting backend.
    for stream in (sys.stdout, sys.stderr):
        # Box-drawing / Unicode output on legacy console codepages (e.g. cp1252).
        if isinstance(stream, io.TextIOWrapper):
            stream.reconfigure(encoding="utf-8")
    matplotlib.use("Agg")  # headless

    print("tessa demo — end-to-end pipeline")
    rng = np.random.default_rng(RANDOM_SEED)

    print("\n[1/5] Generating synthetic data + materialising period aggregates ...")
    _, events, period = prepare_data(rng)

    print("\n[2/5] Running analysis suite ...")
    # One Run holds every result, so its save() / report() cover them all.
    run = Run(period, target_col="class", cfg=cfg, label_filter={"class": CLASSES})
    results = run_analyses(analysis_suite(), run.ctx)

    print("\n[3/5] Results")
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

    print("\n[4/5] New capabilities: unsupervised / semi-supervised / persistence")
    run_new_capabilities(run, period, events, rng)

    print("\n[5/5] Figures")
    # Calibration needs a binary problem: cross-validate TP vs FP on its own.
    binary = Run(period, target_col="class", cfg=cfg, label_filter={"class": ["TP", "FP"]})
    binary.cv_classifier(
        n_splits=5,
        rf_params={"n_estimators": 200, "n_jobs": -1, "random_state": RANDOM_SEED},
    )
    saved = save_figures(run, binary, events, results, OUTPUT_DIR)
    print(f"  Overview     → {saved[0]}")
    print(f"  Per-analysis → {OUTPUT_DIR / 'figures'}  ({len(saved) - 1} PNGs)")

    print("\nDone.\n")


if __name__ == "__main__":
    main()
