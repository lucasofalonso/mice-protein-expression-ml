# From Measurements to Mice

Predicting the experimental group of unseen mice from cortical protein expression.

## Table of contents

- [The experiment behind the dataset](#the-experiment-behind-the-dataset)
- [Data](#data)
- [Validation unit: the animal](#validation-unit-the-animal)
- [Model selection and results](#model-selection-and-results)
- [Reproduce](#reproduce)
  - [Train](#train)
  - [Predict (CLI)](#predict-cli)
  - [Web app (Streamlit)](#web-app-streamlit)
  - [Docker](#docker)
- [Project structure](#project-structure)

## The experiment behind the dataset

- The experiment is a learning test on **72 mice**, using one of two protocols:
  1. **Context-shock (CS):** the mouse is placed in a new chamber, explores it for 3 minutes and then receives a shock. The mouse learns to associate the context with the shock.
  2. **Shock-context (SC):** the mouse receives the shock immediately on entering. This measures the effect of the shock without associating it with the context.
- The mice are either **controls** or **Ts65Dn**, a mouse model of Down syndrome.
- Each mouse is injected with either **saline** or the drug **memantine**.
- Earlier studies indicated that control mice learn under the CS protocol, that Ts65Dn mice struggle to learn under it, and that memantine can restore that ability.
- The mice are euthanized 60 minutes after training. Then **77 proteins and protein modifications** are measured in the nuclear fraction of the cortex, with an antibody-based technique. Each protein is measured 15 times per mouse.
- The eight classes in the dataset come from three **experimental factors**. These are not features of the ML model:
  - `Genotype`: control or Ts65Dn?
  - `Behavior`: stimulated to learn (CS) or not (SC)?
  - `Treatment`: memantine or saline?

| Class | Genotype | Behavior | Treatment | Mice |
|---|---|---|---|---|
| c-CS-m | control | CS | memantine | 10 |
| c-CS-s | control | CS | saline | 9 |
| c-SC-m | control | SC | memantine | 10 |
| c-SC-s | control | SC | saline | 9 |
| t-CS-m | Ts65Dn | CS | memantine | 9 |
| t-CS-s | Ts65Dn | CS | saline | 7 |
| t-SC-m | Ts65Dn | SC | memantine | 9 |
| t-SC-s | Ts65Dn | SC | saline | 9 |

> **Central question:** do the profiles of the 77 proteins distinguish the experimental groups? Which proteins contribute most to the differences found?

The delivered model answers the first part for mice it has never seen. It uses only the 77 proteins as inputs. The experimental factors are never inputs, because together they define the class. The second part is explored descriptively in `notebooks/01_data_audit.ipynb` (PCA loadings and per-class box plots). That analysis is exploratory and does not establish which proteins drive the predictions.

## Data

[Mice Protein Expression](https://archive.ics.uci.edu/dataset/342/mice+protein+expression), from the UCI Machine Learning Repository (dataset 342, DOI [10.24432/C50S3Z](https://doi.org/10.24432/C50S3Z)), introduced by Higuera, Gardiner & Cios (2015), *PLoS ONE* 10(6): e0129126. `train.py` downloads it with `ucimlrepo`.

There are 1080 rows (72 mice × 15 measurements) and 77 protein columns. `MouseID` has the form `<animal>_<measurement>`.

## Validation unit: the animal

The 15 measurements of a mouse are much closer to each other than to those of other mice (`notebooks/01_data_audit.ipynb`). They behave like replicates, not like 15 independent mice. A random split of the rows would put measurements of the same mouse in both train and test, so the score would reward recognizing mice the model has already seen.

So every split in this project is made over the 72 animals:

- **Train/test:** 80/20 split of the animal IDs, stratified by class (`random_state=42`). full_train has **57 animals (855 measurements)** and test has **15 animals (225 measurements)**. All measurements of an animal stay on the same side.
- **Model selection:** 5-fold cross-validation over the 57 full_train animals, with 11–12 validation animals per fold.
- **Metric:** one prediction per animal. Each measurement is predicted, and the class probabilities of the animal's measurements are averaged.

## Model selection and results

`notebooks/02_classification.ipynb` compares a dummy baseline, logistic regression and random forest. Each is tried on two representations of the data: the original measurements and the per-animal mean of each protein (72 × 77). The grid search covers each model's hyperparameters and the imputation strategy (median or mean). The model was chosen by the **cross-validated F1 macro on full_train**. The test animals were used neither to select the model nor its hyperparameters.

Cross-validation on full_train (mean ± standard deviation over 5 folds, one prediction per animal):

| Data | Model | F1 macro | Accuracy |
|---|---|---|---|
| measurements | **Logistic regression (median imputation, C=0.1)** | **0.569 ± 0.072** | **0.648 ± 0.055** |
| measurements | Random forest | 0.501 ± 0.067 | 0.559 ± 0.069 |
| per-animal means | Logistic regression | 0.503 ± 0.107 | 0.541 ± 0.110 |
| per-animal means | Random forest | 0.458 ± 0.152 | 0.476 ± 0.153 |
| either | Dummy (most frequent class) | 0.030 ± 0.008 | 0.139 ± 0.040 |

The selected model is a pipeline of median imputation, standardization and logistic regression (`C=0.1`), trained on the full_train measurements.

**Test (15 animals, evaluated once):** accuracy **0.867** (13 of 15 animals), F1 macro **0.792**.

The test score is above the cross-validation score, but it is a noisy single estimate: each test animal is worth about 7 percentage points, and t-CS-s has only one test animal. The cross-validation is the more stable reference. The CS/SC distinction is always correct. The errors are confusions between classes with the same behavior.

## Reproduce

Requires [uv](https://docs.astral.sh/uv/).

```bash
uv sync
```

### Train

```bash
uv run python train.py
```

This downloads the data, repeats the animal-level split, trains the selected pipeline on the 57 full_train animals and prints the held-out test result. It saves:

- `models/model.joblib`: the full pipeline (imputer, scaler, logistic regression).
- `models/protein_columns.json`: the 77 protein columns, in training order.
- `examples/animal.csv`: the 225 measurements of the 15 test animals, without labels, as an example input.
- `examples/animal_labeled.csv`: the same rows plus the optional `class` column, to compare predictions with the actual class.

### Predict (CLI)

```bash
uv run python predict.py --input examples/animal.csv
```

The input CSV needs an `animal_id` column and the 77 protein columns, with 15 measurements (rows) per animal. Labels (`class`, `Genotype`, `Treatment`, `Behavior`) are not required, and extra columns are ignored by the model. Missing columns and non-numeric values are rejected. Missing protein values are imputed by the pipeline. The CLI warns if an animal does not have 15 measurements.

The output has one row per animal, with the predicted class and the averaged probability of each class. For the 15 test animals in `examples/animal.csv`:

```text
animal_id  n_measurements predicted_class  prob_c-CS-m  prob_c-CS-s  prob_c-SC-m  prob_c-SC-s  prob_t-CS-m  prob_t-CS-s  prob_t-SC-m  prob_t-SC-s
      321              15          c-CS-m        0.801        0.070        0.000        0.085        0.006        0.022        0.014        0.002
     3507              15          c-CS-m        0.901        0.011        0.006        0.014        0.056        0.003        0.003        0.006
     3412              15          c-SC-m        0.000        0.000        0.979        0.002        0.000        0.000        0.017        0.002
     3420              15          c-SC-m        0.000        0.003        0.689        0.006        0.001        0.000        0.295        0.006
     3484              15          t-CS-m        0.000        0.000        0.000        0.000        0.958        0.041        0.000        0.000
   50810D              15          c-CS-s        0.233        0.727        0.001        0.001        0.021        0.005        0.000        0.013
     3424              15          c-SC-s        0.036        0.002        0.026        0.655        0.004        0.007        0.262        0.008
     3481              15          c-SC-s        0.007        0.001        0.042        0.872        0.002        0.000        0.073        0.003
     3504              15          t-CS-m        0.092        0.002        0.001        0.000        0.864        0.039        0.000        0.002
     3522              15          t-CS-m        0.088        0.200        0.001        0.001        0.535        0.089        0.000        0.087
     3501              15          t-SC-m        0.001        0.000        0.001        0.027        0.000        0.000        0.948        0.023
     3534              15          t-SC-m        0.001        0.000        0.113        0.006        0.004        0.000        0.871        0.005
     3476              15          t-CS-m        0.012        0.325        0.002        0.000        0.421        0.056        0.001        0.182
     3425              15          t-SC-s        0.038        0.017        0.023        0.020        0.008        0.027        0.003        0.863
     3491              15          t-SC-s        0.025        0.032        0.061        0.008        0.037        0.003        0.059        0.776
```

If the CSV has a `class` column, the output adds `actual_class` and `correct` and ends with a summary. The label is never passed to the model:

```bash
uv run python predict.py --input examples/animal_labeled.csv
```

```text
animal_id  n_measurements predicted_class actual_class  correct  prob_c-CS-m  ...
      321              15          c-CS-m       c-CS-m     True        0.801  ...
     3484              15          t-CS-m       c-CS-s    False        0.000  ...
     3476              15          t-CS-m       t-CS-s    False        0.012  ...
      ...

Correct: 13 of 15 labeled animals (accuracy 0.867)
```

The two errors are confusions within the CS block: 3484 (c-CS-s) and 3476 (t-CS-s), both predicted as t-CS-m.

Add `--output predictions.csv` to save the result.

### Web app (Streamlit)

```bash
uv run streamlit run app.py
```

Upload a CSV in the same format, or tick the box to use the labeled example (`examples/animal_labeled.csv`). The app calls the same inference code as the CLI (`src/mice_protein_expression_ml/inference.py`). When the CSV has a `class` column, the app shows:

- a summary: labeled animals, correct predictions and accuracy;
- the accuracy per actual class;
- the predictions table, with each animal marked ✓ correct or ✗ wrong.

### Docker

The image copies the trained model from `models/`, so run `train.py` before building.

```bash
docker build -t mice-protein-expression-ml .
docker run --rm -p 8501:8501 mice-protein-expression-ml
```

Then open http://localhost:8501.

## Project structure

```text
notebooks/
  01_data_audit.ipynb         # audit, within/between-animal variability, PCA, K-means
  02_classification.ipynb     # animal-level split, grid search, test evaluation
src/mice_protein_expression_ml/
  data.py                     # data download and animal-level train/test split
  inference.py                # input validation and per-animal prediction (CLI and app)
models/                       # trained pipeline and protein column order
examples/                     # test animals without / with labels (written by train.py)
train.py                      # trains and saves the selected model and the example input
predict.py                    # command-line predictions
app.py                        # Streamlit app
Dockerfile
pyproject.toml, uv.lock
```
