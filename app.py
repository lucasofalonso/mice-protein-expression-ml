"""Minimal Streamlit app: upload a CSV of protein measurements and get one prediction per animal."""

from pathlib import Path

import pandas as pd
import streamlit as st

from mice_protein_expression_ml.inference import (
    EXPECTED_MEASUREMENTS,
    animals_with_unexpected_count,
    evaluation_summary,
    load_model,
    predict_animals,
)

EXAMPLE_CSV = Path(__file__).parent / "examples" / "animal_labeled.csv"


@st.cache_resource
def cached_model():
    return load_model()


st.title("Mice Protein Expression: class per animal")
st.write(
    "Upload a CSV with `animal_id` and the 77 protein columns "
    f"({EXPECTED_MEASUREMENTS} measurements per animal). "
    "The model predicts each measurement and averages the probabilities per animal. "
    "An optional `class` column is used only to compare the predictions with the actual class."
)

uploaded = st.file_uploader("CSV file", type="csv")
use_example = st.checkbox("Use the example file (15 test animals, with labels)")

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

        summary = evaluation_summary(predictions)
        if summary:
            st.subheader("Summary")
            animals_col, correct_col, accuracy_col = st.columns(3)
            animals_col.metric("Labeled animals", summary["animals"])
            correct_col.metric("Correct", f"{summary['correct']} of {summary['animals']}")
            accuracy_col.metric("Accuracy", f"{summary['accuracy']:.1%}")

            st.dataframe(
                summary["per_class"],
                hide_index=True,
                column_config={
                    "class": "Actual class",
                    "animals": "Animals",
                    "correct": "Correct",
                    "accuracy": st.column_config.ProgressColumn(
                        "Accuracy", min_value=0, max_value=1, format="percent"
                    ),
                },
            )
            # Icon + label, so the result does not rely on color alone
            predictions["correct"] = predictions["correct"].map({True: "✓ correct", False: "✗ wrong"})

        st.subheader("Predictions per animal")
        st.dataframe(
            predictions,
            hide_index=True,
            column_config={
                col: st.column_config.NumberColumn(format="%.3f")
                for col in predictions.columns if col.startswith("prob_")
            },
        )
