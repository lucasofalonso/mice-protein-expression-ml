"""Train the final model on the full_train animals and save it to models/.

Also writes example inputs for predict.py and app.py with the measurements of the test
animals: examples/animal.csv (no labels) and examples/animal_labeled.csv (with class).

Model chosen in notebooks/02_classification.ipynb by the cross-validated F1 macro
(5 animal-level folds on full_train): median imputation + standardization +
logistic regression (C=0.1) on the original measurements, one prediction per
animal by averaging the probabilities of its 15 measurements.
"""

import json
from pathlib import Path

import joblib
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from mice_protein_expression_ml.data import load_measurements, protein_columns, split_animals
from mice_protein_expression_ml.inference import MODEL_FILE, MODELS_DIR, PROTEINS_FILE, predict_animals

EXAMPLES_DIR = Path(__file__).parent / "examples"


def build_pipeline() -> Pipeline:
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(C=0.1, max_iter=5000)),
    ])


def main() -> None:
    df = load_measurements()
    protein_cols = protein_columns(df)
    train_animals, test_animals = split_animals(df)

    full_train = df[df["animal_id"].isin(train_animals)]
    assert set(full_train["animal_id"]).isdisjoint(test_animals)

    model = build_pipeline().fit(full_train[protein_cols], full_train["class"])

    MODELS_DIR.mkdir(exist_ok=True)
    joblib.dump(model, MODELS_DIR / MODEL_FILE)
    (MODELS_DIR / PROTEINS_FILE).write_text(json.dumps(protein_cols, indent=1) + "\n")

    print(f"Trained on {len(train_animals)} animals ({len(full_train)} measurements)")
    print(f"Saved {MODELS_DIR / MODEL_FILE} and {MODELS_DIR / PROTEINS_FILE}")

    test = df[df["animal_id"].isin(test_animals)]

    # Example inputs from the test animals only: without labels, and with the optional class
    # column to compare predictions with the actual class (Genotype, Treatment, Behavior left out)
    EXAMPLES_DIR.mkdir(exist_ok=True)
    test[["animal_id", *protein_cols]].to_csv(EXAMPLES_DIR / "animal.csv", index=False)
    test[["animal_id", *protein_cols, "class"]].to_csv(EXAMPLES_DIR / "animal_labeled.csv", index=False)
    print(f"Saved {EXAMPLES_DIR}/animal.csv and animal_labeled.csv ({len(test_animals)} test animals, {len(test)} measurements)")

    # Held-out report only: the model and its hyperparameters were not chosen with these animals.
    predictions = predict_animals(model, protein_cols, test).set_index("animal_id")
    y_true = test.groupby("animal_id")["class"].first().loc[predictions.index]
    y_pred = predictions["predicted_class"]
    print(
        f"Test ({len(test_animals)} animals, one prediction per animal): "
        f"accuracy={accuracy_score(y_true, y_pred):.3f}, "
        f"F1 macro={f1_score(y_true, y_pred, average='macro'):.3f}"
    )


if __name__ == "__main__":
    main()
