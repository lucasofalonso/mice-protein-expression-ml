"""Predict the experimental class of each animal from a CSV of protein measurements.

Usage:
    uv run python predict.py --input examples/animal.csv [--output predictions.csv]

The CSV needs an animal_id column and the 77 protein columns (15 measurements per
animal). Labels (class, Genotype, Treatment, Behavior) are not required; if a class
column is present, the output also compares the predictions with it.
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

from mice_protein_expression_ml.inference import (
    EXPECTED_MEASUREMENTS,
    MODELS_DIR,
    animals_with_unexpected_count,
    evaluation_summary,
    load_model,
    predict_animals,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--input", required=True, type=Path, help="CSV with animal_id and the 77 proteins")
    parser.add_argument("--output", type=Path, help="optional CSV to save the predictions")
    parser.add_argument("--models-dir", type=Path, default=MODELS_DIR)
    args = parser.parse_args()

    model, protein_cols = load_model(args.models_dir)
    try:
        predictions = predict_animals(model, protein_cols, pd.read_csv(args.input))
    except ValueError as error:
        sys.exit(f"Invalid input: {error}")

    unexpected = animals_with_unexpected_count(predictions)
    if unexpected:
        print(
            f"Warning: expected {EXPECTED_MEASUREMENTS} measurements per animal; "
            f"different count for: {', '.join(unexpected)}",
            file=sys.stderr,
        )

    print(predictions.to_string(index=False, float_format=lambda value: f"{value:.3f}"))

    summary = evaluation_summary(predictions)
    if summary:
        print(
            f"\nCorrect: {summary['correct']} of {summary['animals']} labeled animals "
            f"(accuracy {summary['accuracy']:.3f})"
        )

    if args.output:
        predictions.to_csv(args.output, index=False)
        print(f"\nSaved {args.output}")


if __name__ == "__main__":
    main()
