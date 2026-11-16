"""Minimal Streamlit app: upload a CSV of protein measurements and get one prediction per animal."""

from pathlib import Path

import pandas as pd
import streamlit as st

from mice_protein_expression_ml.inference import (
    EXPECTED_MEASUREMENTS,
    animals_with_unexpected_count,
    load_model,
    predict_animals,
)

EXAMPLE_CSV = Path(__file__).parent / "examples" / "animal.csv"


@st.cache_resource
def cached_model():
    return load_model()


st.title("Mice Protein Expression: class per animal")
st.write(
    "Upload a CSV with `animal_id` and the 77 protein columns "
    f"({EXPECTED_MEASUREMENTS} measurements per animal). "
    "The model predicts each measurement and averages the probabilities per animal."
)

uploaded = st.file_uploader("CSV file", type="csv")
use_example = st.checkbox(f"Use the example file ({EXAMPLE_CSV.name})")

source = uploaded if uploaded is not None else (EXAMPLE_CSV if use_example else None)
if source is not None:
    model, protein_cols = cached_model()
    try:
        predictions = predict_animals(model, protein_cols, pd.read_csv(source))
    except ValueError as error:
        st.error(f"Invalid input: {error}")
    else:
        unexpected = animals_with_unexpected_count(predictions)
        if unexpected:
            st.warning(
                f"Expected {EXPECTED_MEASUREMENTS} measurements per animal; "
                f"different count for: {', '.join(unexpected)}"
            )
        st.dataframe(predictions, hide_index=True)
