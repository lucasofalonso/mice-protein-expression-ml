"""Animal-level inference shared by predict.py and app.py."""

import json
from pathlib import Path

import joblib
import pandas as pd

MODELS_DIR = Path(__file__).resolve().parents[2] / "models"
MODEL_FILE = "model.joblib"
PROTEINS_FILE = "protein_columns.json"
EXPECTED_MEASUREMENTS = 15


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
    """
    data = validate_input(df, protein_cols)
    proba = pd.DataFrame(
        model.predict_proba(data[protein_cols]), index=data.index, columns=model.classes_
    )
    by_animal = proba.groupby(data["animal_id"], sort=False).mean()

    result = pd.DataFrame({
        "n_measurements": data.groupby("animal_id", sort=False).size(),
        "predicted_class": by_animal.idxmax(axis=1),
    }).join(by_animal.add_prefix("prob_"))
    return result.rename_axis("animal_id").reset_index()


def animals_with_unexpected_count(predictions: pd.DataFrame) -> list[str]:
    """Animals whose number of measurements differs from the 15 the model was validated with."""
    return predictions.loc[
        predictions["n_measurements"] != EXPECTED_MEASUREMENTS, "animal_id"
    ].tolist()
