"""Animal-level inference shared by predict.py and app.py."""

import json
from pathlib import Path

import joblib
import pandas as pd

MODELS_DIR = Path(__file__).resolve().parents[2] / "models"
MODEL_FILE = "model.joblib"
PROTEINS_FILE = "protein_columns.json"
EXPECTED_MEASUREMENTS = 15
LABEL_COL = "class"


def load_model(models_dir: Path = MODELS_DIR):
    """Return the fitted pipeline and the ordered list of the 77 protein columns."""
    models_dir = Path(models_dir)
    model = joblib.load(models_dir / MODEL_FILE)
    protein_cols = json.loads((models_dir / PROTEINS_FILE).read_text())
    return model, protein_cols


def validate_input(df: pd.DataFrame, protein_cols: list[str]) -> pd.DataFrame:
    """Check required columns and return animal_id plus the proteins in training order.

    Extra columns (class, Genotype, Treatment, Behavior, MouseID...) are ignored.
    """
    missing = [c for c in ["animal_id", *protein_cols] if c not in df.columns]
    if missing:
        raise ValueError(f"Missing {len(missing)} required column(s): {', '.join(missing)}")
    if df.empty:
        raise ValueError("The input has no rows.")
    if df["animal_id"].isna().any():
        raise ValueError("animal_id has empty values.")

    proteins = df[protein_cols].apply(pd.to_numeric, errors="coerce")
    non_numeric = proteins.isna() & df[protein_cols].notna()
    if non_numeric.any().any():
        bad = non_numeric.any()[lambda s: s].index.tolist()
        raise ValueError(f"Non-numeric values in: {', '.join(bad)}")

    return pd.concat([df["animal_id"].astype(str), proteins], axis=1)


def predict_animals(model, protein_cols: list[str], df: pd.DataFrame) -> pd.DataFrame:
    """Predict every measurement, average the probabilities per animal and return one row per animal.

    Missing protein values are imputed by the pipeline (median from training).
    If the input has an optional `class` column, the result also gets `actual_class`
    and `correct`; the label is never passed to the model.
    """
    data = validate_input(df, protein_cols)
    proba = pd.DataFrame(
        model.predict_proba(data[protein_cols]), index=data.index, columns=model.classes_
    )
    by_animal = proba.groupby(data["animal_id"], sort=False).mean()

    result = pd.DataFrame({
        "n_measurements": data.groupby("animal_id", sort=False).size(),
        "predicted_class": by_animal.idxmax(axis=1),
    })
    if LABEL_COL in df.columns:
        result["actual_class"] = actual_class_per_animal(df).reindex(result.index)
        result["correct"] = (result["predicted_class"] == result["actual_class"]).where(
            result["actual_class"].notna()
        )
    result = result.join(by_animal.add_prefix("prob_"))
    return result.rename_axis("animal_id").reset_index()


def actual_class_per_animal(df: pd.DataFrame) -> pd.Series:
    """The optional label of each animal; an animal cannot have more than one class."""
    labels = df[[LABEL_COL]].assign(animal_id=df["animal_id"].astype(str)).dropna()
    n_classes = labels.groupby("animal_id")[LABEL_COL].nunique()
    if (n_classes > 1).any():
        raise ValueError(f"Animals with more than one class: {', '.join(n_classes[n_classes > 1].index)}")
    return labels.groupby("animal_id")[LABEL_COL].first()


def evaluation_summary(predictions: pd.DataFrame) -> dict | None:
    """Correct predictions over the animals with a class label; None if there are no labels."""
    if "actual_class" not in predictions.columns:
        return None
    labeled = predictions.dropna(subset=["actual_class"])
    if labeled.empty:
        return None
    correct = labeled["correct"].astype(bool)
    per_class = (
        labeled.assign(correct=correct)
        .groupby("actual_class")["correct"]
        .agg(animals="size", correct="sum")
        .assign(accuracy=lambda t: t["correct"] / t["animals"])
        .rename_axis("class")
        .reset_index()
    )
    return {
        "animals": len(labeled),
        "correct": int(correct.sum()),
        "accuracy": correct.mean(),
        "per_class": per_class,
    }


def animals_with_unexpected_count(predictions: pd.DataFrame) -> list[str]:
    """Animals whose number of measurements differs from the 15 the model was validated with."""
    return predictions.loc[
        predictions["n_measurements"] != EXPECTED_MEASUREMENTS, "animal_id"
    ].tolist()
