"""Run one clean training/prediction command and score the hidden set.

Install a student's requirements in the selected Python environment first.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path


HW2 = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("submission_dir", type=Path, help="Unpacked submission ZIP root")
    parser.add_argument("results_dir", type=Path, help="Directory for logs and predictions")
    parser.add_argument("--python", default=sys.executable, help="Python with student's requirements installed")
    parser.add_argument("--timeout-seconds", type=int, default=900)
    args = parser.parse_args()

    submission = args.submission_dir.resolve()
    results = args.results_dir.resolve()
    for required in ("submission.py", "requirements.txt", "report.pdf"):
        if not (submission / required).is_file():
            raise FileNotFoundError(submission / required)
    results.mkdir(parents=True, exist_ok=True)
    predictions = results / "predictions.csv"
    predictions.unlink(missing_ok=True)
    with tempfile.TemporaryDirectory(prefix="hw2-grade-") as staged_dir:
        staged = Path(staged_dir)
        # Give the student process only unlabeled hidden inputs in its paths.
        shutil.copytree(HW2 / "student_release" / "dataset", staged / "dataset")
        shutil.copytree(HW2 / "instructor_private" / "test_audio", staged / "hidden_audio")
        shutil.copyfile(HW2 / "instructor_private" / "input.csv", staged / "hidden_input.csv")
        command = [
            args.python, str(submission / "submission.py"),
            "--data-root", str(staged / "dataset"),
            "--predict-manifest", str(staged / "hidden_input.csv"),
            "--predict-audio-dir", str(staged / "hidden_audio"),
            "--output", str(predictions),
            "--seed", "745",
        ]
        start = time.monotonic()
        try:
            completed = subprocess.run(
                command, cwd=submission, text=True, capture_output=True,
                timeout=args.timeout_seconds, check=False,
            )
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError(f"Training exceeded {args.timeout_seconds} seconds") from exc
    seconds = time.monotonic() - start
    (results / "stdout.txt").write_text(completed.stdout)
    (results / "stderr.txt").write_text(completed.stderr)
    if completed.returncode != 0:
        raise RuntimeError(f"Submission exited {completed.returncode}; see {results / 'stderr.txt'}")
    print(f"Training and prediction finished in {seconds:.1f} s")
    score = subprocess.run(
        [args.python, str(HW2 / "instructor" / "score_predictions.py"), str(predictions)],
        text=True, capture_output=True, check=False,
    )
    (results / "score.txt").write_text(score.stdout + score.stderr)
    if score.returncode != 0:
        raise RuntimeError(f"Invalid predictions; see {results / 'score.txt'}")
    print(score.stdout, end="")


if __name__ == "__main__":
    main()
