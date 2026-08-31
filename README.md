# Dry Bean Dataset Study

End-to-end reproducible educational study of the UCI Dry Bean dataset, covering source validation, exploratory evidence, deterministic preparation, multiclass model selection, feature-policy sensitivity, sealed final holdout evaluation, model bundling, and trusted independent inference.

## At a glance

| Item | Result |
|---|---:|
| Source | UCI Machine Learning Repository, dataset `602` |
| Source rows | 13,611 |
| Source columns | 17 |
| Predictive features | 16 numerical morphology features |
| Target | `Class` |
| Classes | 7 nominal, unordered bean varieties |
| Problem type | Multiclass classification |
| Selected model | HistGradientBoostingClassifier |
| Selected feature policy | `all_features` |
| Primary metric | Macro F1 |
| Validation macro F1 | 0.937881 |
| Final-test macro F1 | **0.941835** |
| Final-test balanced accuracy | **0.939897** |
| Final-test accuracy | 0.932419 |
| Final-test log loss ↓ | 0.181024 |
| Educational study complete | Yes |
| Operational modeling ready | No |

## Study objective

This repository documents a frozen educational multiclass-classification workflow for dry bean grain varieties. The objective is to determine how effectively numerical morphology measurements derived from images distinguish seven bean classes while preserving a clear separation between exploratory evidence, model selection, final evaluation, and inference.

The study is predictive rather than causal. Morphological associations describe statistical structure in the observed sample; they do not establish causal or agronomic effects.

The study does not claim production readiness, operational validity, temporal validity, external validity, or an implemented deployment/API surface.

## Dataset and source

The dataset is the UCI Machine Learning Repository **Dry Bean** dataset.

| Field | Value |
|---|---|
| UCI dataset ID | `602` |
| Repository URL | <https://archive.ics.uci.edu/dataset/602/dry+bean+dataset> |
| Dataset DOI | `10.24432/C50S4B` |
| Intro paper | *Multiclass classification of dry beans using computer vision and machine learning techniques* |
| Paper DOI | `10.1016/j.compag.2020.105507` |

Each row represents one dry bean grain described by 16 numerical image-derived morphology measurements and one nominal target column, `Class`.

The official class order used by the study is:

```text
SEKER
BARBUNYA
BOMBAY
CALI
DERMASON
HOROZ
SIRA
```

The model operates on numerical measurements already extracted from images. The original image-acquisition, segmentation, and feature-extraction process is outside the scope of this study.

## Analytical question

The study addresses two related questions:

1. **How do the observed morphology profiles differ among the seven dry bean varieties?**
2. **How effectively can the 16 measurements classify previously unseen bean samples?**

The target is nominal and unordered. There is no positive class, no binary threshold, and no ordinal interpretation among varieties.

For prediction, the model estimates a probability distribution across the seven classes and returns the class with the largest score or probability.

## Feature and target contract

| Role | Columns |
|---|---|
| Target | `Class` |
| Numerical predictors | `Area`, `Perimeter`, `MajorAxisLength`, `MinorAxisLength`, `AspectRatio`, `Eccentricity`, `ConvexArea`, `EquivDiameter`, `Extent`, `Solidity`, `Roundness`, `Compactness`, `ShapeFactor1`, `ShapeFactor2`, `ShapeFactor3`, `ShapeFactor4` |
| Identifier | None provided by source |
| Categorical predictors | None |
| Positive class | Not applicable |
| Decision rule | `argmax_class_score_or_probability` |

The target is prohibited from model inputs.

Because the source provides no observation identifier, exact equality between two feature vectors is not treated as proof that the corresponding records represent the same physical bean.

## Data quality and preparation

The prepared dataset preserves the source shape exactly.

| Check | Result |
|---|---:|
| Source rows | 13,611 |
| Prepared rows | 13,611 |
| Source columns | 17 |
| Prepared columns | 17 |
| Row removal | None |
| Generic imputation | None |
| Automatic outlier deletion | None |
| Deterministic materialization rules | None |
| Candidate features retained | 16 of 16 |
| Learned preprocessing in preparation | None |

The static split is stratified with random seed `42`:

| Partition | Rows | Role |
|---|---:|---|
| Train | 9,527 | Family search, cross-validation, and feature-policy analysis |
| Validation | 2,042 | Frozen candidate comparison and selection |
| Final test | 2,042 | One-time sealed holdout evaluation |

The final-test partition remains sealed until the final model family, hyperparameters, feature policy, and imbalance policy have been frozen.

Repeated feature profiles are preserved. Repeated-profile analysis is treated only as sensitivity evidence; it does not establish duplicate identity or leakage.

## Class distribution

| Class | Rows | Share |
|---|---:|---:|
| `SEKER` | 2,027 | 14.89% |
| `BARBUNYA` | 1,322 | 9.71% |
| `BOMBAY` | 522 | 3.84% |
| `CALI` | 1,630 | 11.98% |
| `DERMASON` | 3,546 | 26.05% |
| `HOROZ` | 1,928 | 14.17% |
| `SIRA` | 2,636 | 19.37% |

`DERMASON` is the majority class and `BOMBAY` is the minority class. The majority/minority support ratio is approximately `6.7931`, and normalized class entropy is approximately `0.942737`.

![Dry Bean target class distribution](docs/images/target_class_distribution.png)

Because each class should contribute equally to model selection despite unequal support, **Macro F1** is the primary metric.

## Exploratory evidence

Several numerical morphology measurements show strong univariate association with `Class`.

![Univariate feature-to-target associations](docs/images/feature_target_association_ranking.png)

These associations are descriptive. They indicate that morphology contains predictive signal but do not establish that any individual measurement independently determines bean variety.

The numerical measurements are not independent. Several variables are mathematically related or strongly correlated.

![Numerical feature correlation heatmap](docs/images/numerical_feature_correlation_heatmap.png)

Nine derived dependencies are numerically confirmed by the study:

```text
AspectRatio
Eccentricity
EquivDiameter
Solidity
Roundness
Compactness
ShapeFactor1
ShapeFactor3
ShapeFactor4
```

`ShapeFactor2` remains a distinct provenance case: its audited source formula is not numerically confirmed at the configured tolerance. Its `provenance_status` is therefore `unresolved`, which does **not** mean the feature is invalid.

The PCA projection is exploratory visualization only.

![Exploratory PCA class projection](docs/images/class_pca_projection.png)

It shows meaningful separation alongside overlap among some varieties. It must not be interpreted as a classifier, a causal mechanism, or a replacement for the full 16-dimensional predictive problem.

![Standardized class profiles](docs/images/standardized_class_profiles.png)

## Evaluation protocol

**Macro F1** is the primary model-selection metric. Complementary evidence includes Balanced Accuracy, Macro Recall, Weighted F1, contextual Accuracy, per-class metrics, confusion evidence, and multiclass Log Loss.

Candidate-family search uses training-only five-fold stratified cross-validation:

```text
StratifiedKFold
n_splits = 5
shuffle = True
random_state = 42
```

A candidate must exceed the Dummy validation Macro F1 by more than `0.02`.

A practical validation tie is defined as a Macro-F1 difference of at most `0.002`. Practical ties are resolved using predefined evidence including balanced accuracy, worst-class recall, cross-validation stability, comparable log loss, model simplicity, and stable model identity.

No test evidence participates in model selection.

## Model selection

Model selection occurs in two stages:

1. four model families are searched and ranked using training-only cross-validation;
2. the two strongest families are evaluated under controlled feature policies before validation selection.

Decision Tree and Random Forest do not receive the later validation-policy evaluation because they do not enter the CV-only shortlist.

### Candidate-family comparison

| Model family | Search | Evaluated configurations | CV Macro F1 mean ± std | CV Balanced Accuracy | CV Weighted F1 | CV Log Loss ↓ | Outcome |
|---|---|---:|---:|---:|---:|---:|---|
| HistGradientBoostingClassifier | RandomizedSearchCV | 8 of 64 | **0.937012 ± 0.003825** | **0.935424** | **0.924310** | 0.258114 | **Shortlisted** |
| LogisticRegression | GridSearchCV | 4 of 4 | 0.935031 ± 0.000996 | 0.933826 | 0.922071 | **0.218538** | **Shortlisted** |
| RandomForestClassifier | RandomizedSearchCV | 8 of 24 | 0.933343 ± 0.003635 | 0.931656 | 0.921274 | 0.342768 | Candidate |
| DecisionTreeClassifier | GridSearchCV | 8 of 8 | 0.919904 ± 0.009698 | 0.920557 | 0.908862 | 0.662884 | Candidate |

### Candidate search configuration and hyperparameters

| Model | Search policy | Search space / fixed configuration |
|---|---|---|
| Logistic Regression | GridSearchCV, 4/4 configurations | `C={0.1,1.0}`; `class_weight={None,balanced}`; fixed `solver=lbfgs`, `max_iter=2000`, `random_state=42`; `StandardScaler` fitted inside each fold |
| Decision Tree | GridSearchCV, 8/8 configurations | `max_depth={8,None}`; `min_samples_leaf={1,5}`; `class_weight={None,balanced}`; fixed `random_state=42` |
| Random Forest | RandomizedSearchCV, 8/24 configurations | `n_estimators={120,240}`; `max_depth={None,16}`; `min_samples_leaf={1,3}`; `class_weight={None,balanced,balanced_subsample}`; fixed `random_state=42`, `n_jobs=1` |
| HistGradientBoosting | RandomizedSearchCV, 8/64 configurations | `learning_rate={0.05,0.1}`; `max_iter={150,250}`; `max_leaf_nodes={15,31}`; `min_samples_leaf={20,40}`; `l2_regularization={0.0,1.0}`; `class_weight={None,balanced}`; fixed `random_state=42` |
| Dummy prior | No search | `strategy=prior` |

For non-selected families, the versioned notebook exposes the complete search contract and family-level CV evidence. The exact parameter combination carried forward as a final frozen configuration is persisted only for the selected model; no unobserved best parameter values are invented here.

### Feature-policy sensitivity

The two shortlisted families are evaluated with their frozen family-search parameters under three feature policies: all 16 predictors, all except `ShapeFactor2`, and seven predictors after removing the nine confirmed derived features.

| Frozen candidate | Features | CV Macro F1 mean ± std | Validation Macro F1 | Balanced Accuracy | Worst recall | Log Loss ↓ |
|---|---:|---:|---:|---:|---:|---:|
| HistGradientBoosting — all features | 16 | **0.937012 ± 0.003825** | **0.937881** | **0.939131** | **0.870886** | 0.225172 |
| HistGradientBoosting — without ShapeFactor2 | 15 | 0.936167 ± 0.004014 | 0.936677 | 0.938146 | 0.863291 | 0.229580 |
| Logistic Regression — all features | 16 | 0.935031 ± 0.000996 | 0.928973 | 0.930908 | 0.863291 | **0.222511** |
| Logistic Regression — without ShapeFactor2 | 15 | 0.934860 ± 0.001422 | 0.928917 | 0.930814 | 0.860759 | 0.223445 |
| HistGradientBoosting — without confirmed derived | 7 | 0.918302 ± 0.004924 | 0.919956 | 0.920909 | 0.858586 | 0.278823 |
| Logistic Regression — without confirmed derived | 7 | 0.919941 ± 0.002835 | 0.915406 | 0.914790 | 0.838384 | 0.271961 |

The Dummy validation Macro F1 is `0.059052`.

![Validation model comparison](docs/images/model_validation_comparison.png)

### Selected model and configuration

The selected candidate is `hist_gradient_boosting__all_features`.

| Hyperparameter | Value |
|---|---:|
| `class_weight` | `None` |
| `learning_rate` | 0.05 |
| `max_iter` | 250 |
| `max_leaf_nodes` | 15 |
| `min_samples_leaf` | 40 |
| `l2_regularization` | 0.0 |
| `random_state` | 42 |

All 16 numerical features are retained. Numerical scaling is not used, no resampling is introduced, and the final decision rule is `argmax_class_score_or_probability`.

## Feature-retention evidence

Removing `ShapeFactor2` from HistGradientBoosting changes validation Macro F1 from `0.937881` to `0.936677`, a delta of approximately `-0.001204`.

This is predictive sensitivity evidence only. It does not resolve or invent the source provenance of `ShapeFactor2`.

Removing the nine confirmed derived features reduces validation Macro F1 from `0.937881` to `0.919956`, a delta of approximately `-0.017925`.

The evidence therefore supports retaining all 16 features for the frozen educational model.

![Feature-policy sensitivity](docs/images/feature_policy_sensitivity.png)

## Final holdout evaluation

After all selection decisions are frozen, the selected model is fitted once on **train + validation** and the sealed final-test partition is evaluated once.

| Metric | Validation | Final test | Test − validation |
|---|---:|---:|---:|
| Macro F1 | 0.937881 | **0.941835** | +0.003954 |
| Balanced Accuracy | 0.939131 | **0.939897** | +0.000766 |
| Macro Recall | 0.939131 | **0.939897** | +0.000766 |
| Weighted F1 | 0.926899 | **0.932187** | +0.005288 |
| Accuracy | 0.927032 | **0.932419** | +0.005387 |
| Log Loss ↓ | 0.225172 | **0.181024** | -0.044148 |
| Minimum per-class recall | 0.870886 | **0.868687** | -0.002199 |

The final-test partition contains 2,042 observations and is evaluated exactly once. It is not used for retuning, feature-policy changes, imbalance-policy changes, or model-family changes.

`SIRA` is the class with the lowest final-test recall.

![Final test confusion matrix](docs/images/final_test_confusion_matrix.png)

The largest mutual confusion pairs are:

| Pair | Final-test mutual errors |
|---|---:|
| `DERMASON` ↔ `SIRA` | 58 |
| `BARBUNYA` ↔ `CALI` | 19 |

![Validation vs test per-class recall](docs/images/validation_vs_test_per_class_recall.png)

![Validation vs final-test metrics](docs/images/validation_vs_test_metrics.png)

### Repeated-profile sensitivity

Fifteen final-test rows have a feature profile also observed in train + validation, leaving 2,027 rows in the descriptive sensitivity subset.

| Evaluation | Macro F1 |
|---|---:|
| Official full final test | 0.941835 |
| Excluding repeated-profile rows | 0.941523 |
| Delta | -0.000312 |

The official evaluation is unchanged. This result does not establish duplicate identity or leakage.

## Independent multiclass inference

The final inference input requires the same 16 numerical features used by the selected model. Missing required values are rejected, and `Class` is prohibited as an input.

The output contains `predicted_class`, `class_order`, and `class_probabilities`.

The decision rule is `argmax_class_score_or_probability`. There is no positive class, binary threshold, or operational probability cutoff.

The fitted estimator's internal class order is:

```text
BARBUNYA
BOMBAY
CALI
DERMASON
HOROZ
SEKER
SIRA
```

The official output class order is:

```text
SEKER
BARBUNYA
BOMBAY
CALI
DERMASON
HOROZ
SIRA
```

Probabilities are explicitly realigned from estimator order to official output order before presentation.

## Workflow and notebooks

```text
Raw UCI snapshot
    -> 01 data understanding and exploration
    -> 02 deterministic preparation and stratified split
    -> 03 model-family search, feature-policy sensitivity, and frozen selection
    -> 04 final train+validation fit and sealed one-time test evaluation
    -> 05 independent inference demonstration
```

Each notebook consumes persisted upstream artifacts rather than relying on live variables from a previous notebook.

## Project structure

```text
artifacts/          Artifact documentation and ignored runtime outputs
data/               Data documentation and ignored raw/processed datasets
docs/images/        Curated versionable figures for documentation
notebooks/          Five authoritative study notebooks
scripts/            Reusable validation, analysis, selection, finalization, and inference code
tests/              Unit and contract tests
```

## Environment setup

```bash
python -m pip install -e ".[notebook,test]"
```

Optional Jupyter kernel:

```bash
python -m ipykernel install --user \
  --name dataset-study-dry-bean \
  --display-name "Python (dataset-study-dry-bean)"
```

The final bundle records this runtime:

| Component | Version |
|---|---:|
| Python | 3.13.13 |
| pandas | 3.0.5 |
| scikit-learn | 1.9.0 |
| joblib | 1.5.3 |

## Reproducing the study

Acquire the UCI dataset:

```bash
python -m scripts.download_data uci \
  602 \
  --destination data/raw/dry-bean
```

Execute notebooks in order from 01 to 05. To preserve clean source notebooks, execute copies rather than using `--inplace`:

```bash
jupyter nbconvert --to notebook --execute notebooks/01_data_understanding_and_exploration.ipynb
jupyter nbconvert --to notebook --execute notebooks/02_data_preparation.ipynb
jupyter nbconvert --to notebook --execute notebooks/03_model_selection_and_evaluation.ipynb
jupyter nbconvert --to notebook --execute notebooks/04_final_model_and_bundle.ipynb
jupyter nbconvert --to notebook --execute notebooks/05_inference_demo.ipynb
```

Generated `*.nbconvert.ipynb` files are ignored by Git.

## Tests

```bash
PYTHONPATH=. python -m pytest -q
```

## Reproducibility and integrity

The project preserves deterministic seeds, source and split fingerprints, persisted handoff contracts, artifact hashes, model serialization checks, and fresh-process inference validation.

Raw data, processed data, model binaries, and runtime artifacts remain outside the normal versioned workflow. Curated documentation figures are versioned under `docs/images/`.

## Limitations

- The evaluation is a stratified random-snapshot benchmark rather than a temporal, prospective, or external validation.
- External representativeness is not established.
- The model receives numerical features derived from images; the upstream image-acquisition and feature-extraction pipeline is not evaluated.
- Operational availability and stability of the 16 measurements are unconfirmed.
- `ShapeFactor2` source provenance remains unresolved at the study's numerical formula tolerance.
- Exact equality between feature profiles cannot prove duplicate physical grains because the source contains no observation identifier.
- Class probabilities are not established as universally calibrated measures of biological certainty.
- Morphological overlap remains between several classes.
- Distribution shift across equipment, environments, cultivars, populations, or acquisition protocols is not evaluated.
- No production monitoring or drift policy is established.
- No industrial or commercial error-cost analysis is included.
- Exploratory associations must not be interpreted as causal or agronomic effects.
- Operational modeling readiness and operational validity remain unconfirmed.

## Responsible interpretation

The evidence supports a clear predictive conclusion:

> Numerical measurements of bean size and shape contain strong information for distinguishing seven dry bean varieties in this dataset. A HistGradientBoostingClassifier using all 16 morphology features achieves a final-test Macro F1 of approximately **0.9418** and Balanced Accuracy of approximately **0.9399** under the frozen stratified random-snapshot evaluation protocol.

Four trainable model families are compared under training-only cross-validation. HistGradientBoostingClassifier and LogisticRegression form the CV shortlist, after which controlled feature-policy analysis supports retaining all 16 features and selecting HistGradientBoostingClassifier.

The final result demonstrates effective classification within the observed dataset and evaluation protocol. It does not establish causal morphology-to-variety relationships or guarantee equivalent performance under different measurement pipelines, populations, or operating conditions.

Any use beyond this analytical setting would require independent validation of the complete measurement pipeline, representative external data, probability behavior, distribution stability, monitoring requirements, and application-specific error costs.

## Source and citation

Use the UCI dataset and paper metadata above when citing the source.

- Dataset DOI: `10.24432/C50S4B`
- Intro paper: *Multiclass classification of dry beans using computer vision and machine learning techniques*
- Paper DOI: `10.1016/j.compag.2020.105507`
