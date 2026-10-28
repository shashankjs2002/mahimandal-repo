"""Check package files, frozen split and saved benchmark averages. No GPU needed."""
import csv
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parent
MANIFEST_HASH = "6bf97403ee40ccf0cbd38b2e6e4932e01973e5d00e211e0acfb1e0b810efc5e7"


def sha256(path):
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def check_files():
    count = 0
    for line in (ROOT / "checksums.sha256").read_text(encoding="utf-8").splitlines():
        expected, name = line.split("  ", 1)
        path = (ROOT / name).resolve()
        path.relative_to(ROOT)
        if not path.is_file() or sha256(path) != expected:
            raise ValueError("Missing or changed file: " + name)
        count += 1
    print(f"File checksums: {count} passed")


def check_split():
    path = ROOT / "metadata/paper_release_manifest.csv"
    if sha256(path) != MANIFEST_HASH:
        raise ValueError("Frozen release manifest changed")
    with path.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    counts = {s: sum(r["split"] == s for r in rows) for s in ("train", "val", "test")}
    if counts != {"train": 3424, "val": 645, "test": 731}:
        raise ValueError("Unexpected split: " + str(counts))
    if len({r["pair_id"] for r in rows}) != len(rows):
        raise ValueError("Duplicate pair IDs")
    print("Frozen split:", counts)


def check_scores():
    checks, folders = 0, 0
    for path in sorted((ROOT / "results/benchmark").glob("reflectance16/*/*/seed_*/evaluation/*/*/metrics.json")):
        saved = json.loads(path.read_text(encoding="utf-8"))
        with path.with_name("per_image.csv").open(encoding="utf-8", newline="") as stream:
            rows = list(csv.DictReader(stream))
        if len(rows) != saved["count"] or len({r["pair_id"] for r in rows}) != len(rows):
            raise ValueError("Wrong evaluation IDs/count: " + str(path))
        for metric, expected in saved.items():
            if isinstance(expected, bool) or not isinstance(expected, (int, float)) or metric not in rows[0]:
                continue
            values = [float(r[metric]) for r in rows if r.get(metric, "") not in ("", "nan", "NaN")]
            values = [v for v in values if math.isfinite(v)]
            if not values:
                continue
            actual = math.fsum(values) / len(values)
            if not math.isclose(actual, expected, rel_tol=1e-6, abs_tol=1e-7):
                raise ValueError(f"Saved mean differs: {path}, {metric}: {actual} vs {expected}")
            checks += 1
        folders += 1
    if not folders:
        raise ValueError("No benchmark evaluations found")
    print(f"Saved benchmark means: {checks} passed in {folders} evaluation folders")
    print("These checks reaggregate saved scores; they do not run model inference.")


if __name__ == "__main__":
    check_files()
    check_split()
    check_scores()
