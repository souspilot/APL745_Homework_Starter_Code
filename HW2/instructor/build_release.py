"""Build the HW2 student audio package."""

from __future__ import annotations

import csv
import random
import secrets
import shutil
from collections import Counter
from itertools import combinations
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np
from scipy.io import wavfile
from scipy.signal import resample_poly


HW2 = Path(__file__).resolve().parents[1]
SOURCE = HW2 / "original_dataset"
RELEASE = HW2 / "student_release"
PRIVATE = HW2 / "instructor_private"

CATEGORIES = (
    "chirping_birds",
    "vacuum_cleaner",
    "door_wood_knock",
    "clapping",
    "airplane",
    "mouse_click",
    "pouring_water",
    "keyboard_typing",
    "footsteps",
    "washing_machine",
)


def split_by_source(rows: list[dict[str, str]]) -> dict[str, str]:
    """Split the fifth source fold into balanced, source-disjoint tests."""
    assignment = {}
    for row in rows:
        if row["fold"] != "5":
            assignment[row["filename"]] = "train" if row["fold"] in ("1", "2", "3") else "validation"
    for category in CATEGORIES:
        fifth = [r for r in rows if r["fold"] == "5" and r["category"] == category]
        groups = {}
        for row in fifth:
            groups.setdefault(row["src_file"], []).append(row["filename"])
        names = sorted(groups)
        random.Random(f"HW2-source-split-{category}").shuffle(names)
        selected = next(
            (set(choice) for n in range(1, len(names) + 1)
             for choice in combinations(names, n)
             if sum(len(groups[name]) for name in choice) == 4),
            None,
        )
        if selected is None:
            raise ValueError(f"Cannot create balanced source-disjoint test split for {category}")
        for row in fifth:
            assignment[row["filename"]] = "test" if row["src_file"] in selected else "instructor_test"
    return assignment


def main() -> None:
    if not SOURCE.is_dir():
        raise FileNotFoundError(SOURCE)
    with (SOURCE / "meta" / "data.csv").open(newline="") as f:
        rows = [row for row in csv.DictReader(f) if row["category"] in CATEGORIES]
    assert len(rows) == 400
    assert all(Counter(row["category"] for row in rows if row["fold"] == str(fold))
               == Counter({category: 8 for category in CATEGORIES})
               for fold in range(1, 6))
    split_for = split_by_source(rows)
    source_by_filename = {row["filename"]: row for row in rows}
    source_splits = {}
    for filename, split in split_for.items():
        row = source_by_filename[filename]
        source_splits.setdefault((row["category"], row["src_file"]), set()).add(split)
    if any(len(splits) != 1 for splits in source_splits.values()):
        raise ValueError("A source recording crosses train/validation/test boundaries")

    # Reuse an existing private key so rebuilding does not change published IDs.
    PRIVATE.mkdir(exist_ok=True)
    key_path = PRIVATE / "source_key.csv"
    if key_path.exists():
        with key_path.open(newline="") as f:
            key = list(csv.DictReader(f))
        if {r["source_filename"] for r in key} != {r["filename"] for r in rows}:
            raise ValueError("Existing instructor key does not match source selection")
    else:
        category_codes = [f"C{i:02d}" for i in range(10)]
        rng = secrets.SystemRandom()
        rng.shuffle(category_codes)
        label_for = dict(zip(CATEGORIES, category_codes))
        rng.shuffle(rows)
        key = []
        for row in rows:
            key.append({
                "id": "clip_" + secrets.token_hex(10) + ".wav",
                "fold": row["fold"],
                "label": label_for[row["category"]],
                "source_filename": row["filename"],
                "source_category": row["category"],
            })
    for entry in key:
        entry["split"] = split_for[entry["source_filename"]]
    with key_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(key[0]))
        writer.writeheader()
        writer.writerows(key)

    audio_dir = RELEASE / "dataset" / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)
    private_audio_dir = PRIVATE / "test_audio"
    private_audio_dir.mkdir(parents=True, exist_ok=True)
    manifest = []
    private_manifest = []
    for entry in key:
        is_private = entry["split"] == "instructor_test"
        destination = (private_audio_dir if is_private else audio_dir) / entry["id"]
        if not destination.exists():
            rate, x = wavfile.read(SOURCE / "audio" / entry["source_filename"])
            if x.ndim == 2:
                x = x.astype(np.float64).mean(axis=1)
            if rate != 16000:
                # Recordings are 44.1 kHz; this also handles other rates.
                from math import gcd
                factor = gcd(rate, 16000)
                x = resample_poly(x, 16000 // factor, rate // factor)
            x = np.clip(np.rint(x), -32768, 32767).astype(np.int16)
            wavfile.write(destination, 16000, x)
        (private_manifest if is_private else manifest).append(
            {k: entry[k] for k in ("id", "split", "label")}
        )

    public_ids = {row["id"] for row in manifest}
    private_ids = {row["id"] for row in private_manifest}
    for path in audio_dir.glob("clip_*.wav"):
        if path.name not in public_ids:
            path.unlink()
    for path in private_audio_dir.glob("clip_*.wav"):
        if path.name not in private_ids:
            path.unlink()

    with (RELEASE / "dataset" / "manifest.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=("id", "split", "label"))
        writer.writeheader()
        writer.writerows(sorted(manifest, key=lambda r: r["id"]))
    with (PRIVATE / "manifest.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=("id", "split", "label"))
        writer.writeheader()
        writer.writerows(sorted(private_manifest, key=lambda r: r["id"]))
    with (PRIVATE / "input.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=("id", "split"))
        writer.writeheader()
        writer.writerows({k: row[k] for k in ("id", "split")}
                         for row in sorted(private_manifest, key=lambda r: r["id"]))
    shutil.copyfile(SOURCE / "LICENSE", RELEASE / "ATTRIBUTION.txt")
    archive = HW2 / "HW2_student_release.zip"
    with ZipFile(archive, "w", compression=ZIP_DEFLATED, compresslevel=2) as z:
        for path in sorted(RELEASE.rglob("*")):
            if path.is_file() and not any(part.startswith(".") for part in path.relative_to(RELEASE).parts):
                z.write(path, Path("HW2_student") / path.relative_to(RELEASE))
    expected = {"train": 240, "validation": 80, "test": 40}
    assert Counter(row["split"] for row in manifest) == expected
    assert len(private_manifest) == 40
    print(f"Built {len(manifest)} student clips in {archive} and {len(private_manifest)} private clips; instructor key: {key_path}")


if __name__ == "__main__":
    main()
