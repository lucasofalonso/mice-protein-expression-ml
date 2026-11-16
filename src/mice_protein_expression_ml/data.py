"""Data ingestion and animal-level train/test split, as in notebooks/02_classification.ipynb."""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

UCI_DATASET_ID = 342
META_COLS = ["Genotype", "Treatment", "Behavior", "class"]
RANDOM_STATE = 42
TEST_SIZE = 0.2


def load_measurements() -> pd.DataFrame:
    """Download the dataset (1080 measurements) and split MouseID into animal_id and measurement."""
    from ucimlrepo import fetch_ucirepo

    df_full = fetch_ucirepo(id=UCI_DATASET_ID).data.original

    mouse_parts = df_full["MouseID"].str.split("_", expand=True)
    return df_full.assign(
        animal_id=mouse_parts[0],
        measurement=pd.to_numeric(mouse_parts[1], errors="raise"),
    )


def protein_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c.endswith("_N")]


def split_animals(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """80/20 split over animal IDs, stratified by class; all measurements of an animal stay together."""
    y_animal = df.groupby("animal_id")["class"].first()
    # NumPy array: train_test_split cannot index the Arrow-backed string Index pandas uses with pyarrow
    train_animals, test_animals = train_test_split(
        y_animal.index.to_numpy(),
        test_size=TEST_SIZE,
        stratify=y_animal,
        random_state=RANDOM_STATE,
    )
    return train_animals, test_animals
