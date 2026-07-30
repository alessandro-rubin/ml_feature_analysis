# Statistical tests guide

This guide documents the corroborating statistical tests added on top of
the original analyses. The originals already report Kruskal-Wallis, KS,
ARI, NMI, etc.; the additions here cover the standard rigor gaps
(multiple testing, effect sizes, confidence intervals, cross-validation,
stability) so a finding from one analysis can be cross-checked against
several independent angles.

## Contents
- [When to reach for what](#when-to-reach-for-what)
- [Multiple-testing correction](#multiple-testing-correction)
- [Effect sizes and distribution distances](#effect-sizes-and-distribution-distances)
- [DistributionAnalysis — group differences across all classes](#distributionanalysis)
- [PairwiseSeparability — A-vs-B feature ranking](#pairwiseseparability)
- [CrossValidatedClassifier — beyond a single train/test split](#crossvalidatedclassifier)
- [ImportanceStability — is the importance ranking real?](#importancestability)
- [ClusterValidation — is the clustering real and aligned with classes?](#clustervalidation)
- [Recipe: end-to-end corroboration](#recipe-end-to-end-corroboration)

## When to reach for what

| Question | Primary | Corroborate with |
|---|---|---|
| Does feature `f` differ across classes? | `DistributionAnalysis` (KW) | ANOVA F, Anderson-Darling, Levene's, BH-FDR |
| How well does `f` separate A from B? | `PairwiseSeparability` (AUC) | bootstrap CI on AUC; Cohen's d / Hedges' g; Wasserstein |
| Is the per-feature p-value real with N≈90 features? | BH-FDR adjusted p | Bonferroni for strict family-wise control |
| Is the classifier really above chance? | `CrossValidatedClassifier` (acc) | MCC, balanced accuracy, ROC-AUC, Brier, ECE |
| Are the "important" features stable? | `FeatureImportance` (composite) | `ImportanceStability` (bootstrap CI + top-k fraction + method agreement) |
| Do clusters reflect the class labels? | `ClusterAnalysis` (ARI) | `ClusterValidation` (Hopkins + ARI/V-measure permutation p) |

## Multiple-testing correction

`tessa.analysis.multiple_testing` exposes three helpers that take
a 1-D array of raw p-values and return adjusted p-values in the
original order. NaNs pass through.

```python
from tessa.analysis.multiple_testing import (
    bonferroni, holm, benjamini_hochberg,
)
import numpy as np

raw = np.array([0.001, 0.01, 0.04, 0.2, 0.7])
bonferroni(raw)         # strict FWER, multiplies by m
holm(raw)               # step-down FWER, less conservative than Bonferroni
benjamini_hochberg(raw) # FDR-controlled q-values (typical default for ML)
```

Choose:

- **Bonferroni** when any false positive is unacceptable (regulatory,
  publication-grade claims).
- **Holm** as a strictly more powerful drop-in for Bonferroni when you
  still want family-wise error control.
- **Benjamini-Hochberg** for hypothesis-generation / feature
  selection — controls the *fraction* of false discoveries among
  rejections.

Both `DistributionAnalysis` and `PairwiseSeparability` already attach
`*_bonferroni` and `*_bh_fdr` columns to their output tables — you
rarely need to call these helpers directly.

## Effect sizes and distribution distances

`tessa.analysis.effect_sizes` provides:

| Function | Type | Range | Interpretation |
|---|---|---|---|
| `cohens_d(a, b)` | parametric | unbounded | 0.2 small / 0.5 medium / 0.8 large |
| `hedges_g(a, b)` | parametric, small-N corrected | unbounded | same thresholds as `d`; preferred when n < ~30 |
| `cliffs_delta(a, b)` | rank-based | [-1, 1] | 0.147 small / 0.33 medium / 0.474 large |
| `rank_biserial_from_u(u, n1, n2)` | rank-based | [-1, 1] | derived from Mann-Whitney U |
| `js_divergence(a, b)` | distribution-level (bits) | [0, 1] | 0 = identical, 1 = disjoint support |
| `bootstrap_ci(stat_fn, a, b, ...)` | uncertainty | — | returns `(point, lo, hi)` |

Effect sizes always belong **next to** a p-value: a vanishing p-value
attached to a Cliff's delta of 0.01 means "we have lots of data and a
trivially small difference", not "we have a useful discovery".

`bootstrap_ci` works on any scalar two-sample statistic. Example:

```python
from tessa.analysis.effect_sizes import bootstrap_ci, cohens_d
import numpy as np

rng = np.random.default_rng(0)
a, b = rng.normal(0, 1, 80), rng.normal(0.6, 1, 80)
point, lo, hi = bootstrap_ci(cohens_d, a, b, n_resamples=2000, ci=0.95, rng=rng)
print(f"Cohen's d = {point:+.3f}  (95% CI {lo:+.3f}, {hi:+.3f})")
```

## DistributionAnalysis

For *"do feature `f` distributions differ across classes?"* across all
K classes at once.

```python
from tessa.analysis import AnalysisContext, DistributionAnalysis
from tessa import Config

ctx = AnalysisContext(df=period_df, cfg=Config(), target_col="class")
out = DistributionAnalysis().run(ctx)

summary = out["summary"]            # one row per feature
per_class = out["per_feature_class"]  # one row per (feature, class)
```

`summary` columns:

| Column | Meaning |
|---|---|
| `kw_stat`, `kw_p` | Kruskal-Wallis H — rank-based test for k groups |
| `anova_f`, `anova_p` | One-way ANOVA F — parametric counterpart |
| `ad_stat`, `ad_p` | Anderson-Darling k-sample — more sensitive to tail differences |
| `levene_stat`, `levene_p` | Levene's test — equal variances? If not, treat ANOVA with caution |
| `<test>_p_bonferroni`, `<test>_p_bh_fdr` | family-corrected p-values |

Reading the row: if KW, ANOVA, and AD all reject after BH-FDR, the
distributions genuinely differ. If KW rejects but ANOVA does not, the
difference is in rank order / median, not in means — a t-test would
have missed it. If Levene rejects, prefer Welch / non-parametric tests
for that feature.

`per_feature_class` adds **shape** columns to the existing descriptives
so the same row also tells you whether the distributions look
Gaussian-ish or heavy-tailed:

- `skew` — sample skewness
- `kurtosis` — excess kurtosis (0 = Gaussian)
- `mad` — median absolute deviation (robust scale)
- `iqr` — interquartile range

## PairwiseSeparability

For *"for this specific pair (A, B), which features discriminate?"*.

```python
from tessa.analysis import AnalysisContext, PairwiseSeparability

# bootstrap_n=0 (default) is fast; >0 enables AUC and Cliff's delta CIs.
out = PairwiseSeparability(bootstrap_n=500).run(ctx)
table = out["pairs"][("TP", "FP")]
top10 = table.head(10)
```

Per-feature columns:

| Column | What it tells you |
|---|---|
| `auc` (+ CI) | direction-free single-feature ROC-AUC |
| `cliffs_delta` (+ CI) | rank-based effect size in [-1, 1] |
| `rank_biserial` | redundant check on Cliff's delta (derived from MWU) |
| `cohens_d`, `hedges_g` | parametric effect size |
| `wasserstein` | Earth-Mover distance between the two empirical CDFs |
| `js_divergence` | Jensen-Shannon divergence (bits) |
| `ks_stat`, `ks_p` | KS — any CDF difference |
| `mwu_stat`, `mwu_p` | Mann-Whitney U — stochastic dominance |
| `bm_p` | Brunner-Munzel — MWU robust to unequal variance |
| `welch_t`, `welch_p` | Welch's t — parametric, unequal variance |
| `*_bh_fdr`, `*_bonferroni` | multiple-testing-corrected p-values |

The default sort is by AUC then |Cliff's delta|, so the first rows are
the strongest separators. A robust "feature `f` separates A and B"
claim looks like:

- AUC > 0.7, CI does not cross 0.5
- |Cliff's delta| > 0.3, CI does not cross 0
- mwu_p_bh_fdr < 0.05 *and* ks_p_bh_fdr < 0.05
- |Cohen's d| > 0.5

If only KS rejects, the difference is in some moment other than the
location (e.g. variance only) — interpretable but rarely useful for a
binary classifier.

`bootstrap_n` controls bootstrap iterations (per feature, per pair). It
adds non-trivial cost — start at 500, raise to 2000 for final tables.

## CrossValidatedClassifier

A single train/test split (what `ClassifierEvaluation` does) gives you
one accuracy number with no uncertainty. `CrossValidatedClassifier`
runs stratified k-fold and reports a spread.

```python
from tessa.analysis import AnalysisContext, CrossValidatedClassifier

out = CrossValidatedClassifier(n_splits=5).run(ctx)
summary = out["summary"]      # rows = metrics; cols = mean / std / min / max
per_fold = out["per_fold"]    # rows = folds; cols = metrics
```

Metrics reported per fold:

| Metric | When it matters |
|---|---|
| `accuracy` | balanced data, no class-cost asymmetry |
| `balanced_accuracy` | imbalanced classes — the headline metric there |
| `f1_macro`, `f1_weighted` | precision/recall balance, macro for "all classes matter equally" |
| `precision_macro`, `recall_macro` | when one error type costs more |
| `mcc` | single-number summary robust to imbalance, in [-1, 1] |
| `cohen_kappa` | agreement above chance; complements MCC |
| `log_loss` | probabilistic calibration of *all* class probabilities |
| `roc_auc` / `roc_auc_ovr` | threshold-free ranking quality |
| `pr_auc` (binary) | better than ROC-AUC for rare positives |
| `brier` (binary) | mean squared error of predicted probabilities |
| `ece` (binary) | expected calibration error (lower = better calibrated) |

`out["oof_pred"]` and `out["oof_proba"]` give out-of-fold predictions
for every sample — useful for downstream stacking, calibration plots,
or error analysis.

Read the summary table like:
- `accuracy.mean = 0.84, accuracy.std = 0.03` — stable across folds.
- `accuracy.mean = 0.84, accuracy.std = 0.18` — one fold got lucky;
  don't trust the headline.
- `mcc.mean ≈ 0` while `accuracy.mean = 0.9` — classifier is just
  predicting the majority class.

## ImportanceStability

The existing `FeatureImportance` blends 5 importance signals into a
composite score. `ImportanceStability` answers two follow-ups:

1. *Is each top feature's importance reproducible under resampling?*
   (bootstrap RF MDI)
2. *Do the 5 importance methods agree, or is each picking a different
   ranking?* (Spearman rank correlation)

```python
from tessa.analysis import (
    AnalysisContext, FeatureImportance, ImportanceStability, run_analyses,
)

results = run_analyses(
    [FeatureImportance(), ImportanceStability(n_bootstrap=200, top_k=10)],
    ctx,
)
boot = results["importance_stability"]["bootstrap_table"]
agreement = results["importance_stability"]["method_agreement"]
```

`bootstrap_table` columns:

| Column | Meaning |
|---|---|
| `mdi_median` | median RF feature importance across bootstrap resamples |
| `mdi_ci_low`, `mdi_ci_high` | percentile CI |
| `ci_width` | wider = less stable estimate |
| `stability_top<k>` | fraction of resamples where the feature ranked in the top `k` |

A trustworthy "important feature" has high `mdi_median`, a narrow
`ci_width` whose lower bound is well above zero, and a
`stability_top<k>` near 1. A feature with `stability_top10 = 0.4` is
"sometimes important", which is rarely actionable.

`method_agreement` is a square Spearman correlation matrix between
`rf_mdi`, `perm_mean`, `anova_f`, `kw_stat`, `mutual_info`. Strong
off-diagonal entries (> ~0.7) mean the methods agree — the composite
score is well-grounded. Weak entries flag features whose "importance"
is method-dependent: investigate before reporting.

You can also run `ImportanceStability` standalone (without importance
in front) — the bootstrap part still works; only `method_agreement`
will be empty.

## ClusterValidation

`ClusterAnalysis` reports silhouette / Davies-Bouldin / ARI / NMI /
V-measure as scores. `ClusterValidation` adds the missing
"is this real?" answers.

```python
from tessa.analysis import (
    AnalysisContext, ClusterAnalysis, ClusterValidation, run_analyses,
)

results = run_analyses(
    [ClusterAnalysis(), ClusterValidation(n_permutations=1000)],
    ctx,
)
s = results["cluster_validation"]["summary"].iloc[0]
```

Fields in the one-row summary:

| Field | Meaning |
|---|---|
| `hopkins` | cluster-tendency. ≈0.5 = uniform (no structure), ≈1 = highly clustered, ≈0 = regular lattice |
| `calinski_harabasz` | between/within dispersion ratio — third internal validity score |
| `ari` | adjusted Rand index against the true class labels |
| `ari_perm_p` | permutation p — fraction of label shuffles with |ARI| ≥ observed |
| `v_measure`, `v_measure_perm_p` | same for V-measure |

How to read it:

- `hopkins < 0.6` → the data is not really clusterable; downstream
  silhouette / ARI numbers are likely spurious.
- `ari ≈ 0.4` *and* `ari_perm_p < 0.01` → moderate but real alignment.
- `ari ≈ 0.4` *and* `ari_perm_p > 0.1` → looks aligned but isn't
  distinguishable from random chance under permutation.

If `ClusterAnalysis` ran first, `ClusterValidation` reuses its KMeans
labels; otherwise it fits its own KMeans with `k = n_classes`.

## Recipe: end-to-end corroboration

A "is feature `f` really useful to separate TP from FP?" answer that
won't fall over on review uses every layer:

```python
from tessa import Config
from tessa.analysis import (
    AnalysisContext, DistributionAnalysis, PairwiseSeparability,
    FeatureImportance, ImportanceStability,
    CrossValidatedClassifier, ClusterAnalysis, ClusterValidation,
    run_analyses,
)

ctx = AnalysisContext(
    df=period_df,
    cfg=Config(),
    target_col="class",
    label_filter={"class": ["TP", "FP"]},
)

results = run_analyses(
    [
        DistributionAnalysis(),
        PairwiseSeparability(bootstrap_n=500),
        FeatureImportance(),
        ImportanceStability(n_bootstrap=200, top_k=10),
        CrossValidatedClassifier(n_splits=5),
        ClusterAnalysis(),
        ClusterValidation(n_permutations=1000),
    ],
    ctx,
)
```

A feature is "really useful" when *all* of the following agree:

1. `distributions.summary`: `kw_p_bh_fdr < 0.05` *and* `anova_p_bh_fdr < 0.05`.
2. `pairwise.pairs[("TP","FP")]`: AUC CI excludes 0.5 *and*
   |Cohen's d| > 0.5 *and* `mwu_p_bh_fdr < 0.05`.
3. `importance_stability.bootstrap_table`: `mdi_ci_low > 0` *and*
   `stability_top10 > 0.8`; `method_agreement` off-diagonal > 0.7.
4. `cv_classifier.summary`: `mcc.mean` and `balanced_accuracy.mean` both
   meaningfully above 0 / 0.5, with small `std`.
5. `cluster_validation.summary`: `hopkins > 0.6` and
   `ari_perm_p < 0.05` (optional — confirms the structure isn't a
   single-feature artefact).

Disagreement between any two is informative: a high AUC with a wide
bootstrap CI and an unstable importance is *one outlier event* away
from a "negative" finding; report it that way.
