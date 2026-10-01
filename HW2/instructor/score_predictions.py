"""Score a student's CSV against the private instructor test labels."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


PRIVATE = Path(__file__).resolve().parents[1] / "instructor_private"


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("predictions", type=Path, help="CSV with id,predicted_label")
    args = parser.parse_args()

    truth_rows = read_rows(PRIVATE / "manifest.csv")
    pred_rows = read_rows(args.predictions)
    truth = {row["id"]: row["label"] for row in truth_rows}
    if not pred_rows or set(pred_rows[0]) != {"id", "predicted_label"}:
        raise ValueError("Prediction CSV must have exactly id,predicted_label columns")
    pred = {row["id"]: row["predicted_label"] for row in pred_rows}
    if len(pred) != len(pred_rows):
        raise ValueError("Duplicate prediction IDs")
    if set(pred) != set(truth):
        raise ValueError(f"ID mismatch: missing={len(set(truth)-set(pred))}, extra={len(set(pred)-set(truth))}")
    labels = sorted(set(truth.values()))
    if any(value not in labels for value in pred.values()):
        raise ValueError("Invalid class code")

    accuracy = sum(pred[i] == truth[i] for i in truth) / len(truth)
    f1_scores = []
    for label in labels:
        tp = sum(truth[i] == label and pred[i] == label for i in truth)
        fp = sum(truth[i] != label and pred[i] == label for i in truth)
        fn = sum(truth[i] == label and pred[i] != label for i in truth)
        f1_scores.append(2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0)
    macro_f1 = sum(f1_scores) / len(f1_scores)
    print(f"n={len(truth)} accuracy={accuracy:.4f} macro_F1={macro_f1:.4f}")
    print("Confusion matrix (rows=true, columns=predicted):")
    print("      " + " ".join(f"{label:>3}" for label in labels))
    for true_label in labels:
        counts = [sum(truth[i] == true_label and pred[i] == predicted_label for i in truth)
                  for predicted_label in labels]
        print(f"{true_label:>3}   " + " ".join(f"{count:>3}" for count in counts))


if __name__ == "__main__":
    main()
