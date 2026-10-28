"""Audit source-product overlap in the frozen Mahimandal release manifest."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path


EXPECTED_SHA256 = "6bf97403ee40ccf0cbd38b2e6e4932e01973e5d00e211e0acfb1e0b810efc5e7"
EXPECTED_COUNTS = {"train": 3424, "val": 645, "test": 731}
FIELDS = ("landsat_id", "sentinel_id", "spatial_block_id")
COMPARISONS = (("train", "test"), ("train", "val"), ("val", "test"))


def audit(manifest: Path) -> dict:
    digest = hashlib.sha256(manifest.read_bytes()).hexdigest()
    if digest != EXPECTED_SHA256:
        raise ValueError(f"Not the frozen reflectance16 release manifest: {digest}")
    with manifest.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    by_split = {split: [row for row in rows if row["split"] == split]
                for split in EXPECTED_COUNTS}
    counts = Counter(row["split"] for row in rows)
    if dict(counts) != EXPECTED_COUNTS:
        raise ValueError(f"Unexpected split counts: {dict(counts)}")
    if len({row["pair_id"] for row in rows}) != len(rows):
        raise ValueError("Duplicate pair IDs in release manifest")
    if any(row["release_included"].casefold() != "true" for row in rows):
        raise ValueError("Manifest contains a non-release row")
    if any(not row[field] for row in rows for field in FIELDS):
        raise ValueError("Missing source ID or spatial block ID")

    sets = {split: {field: {row[field] for row in items} for field in FIELDS}
            for split, items in by_split.items()}
    source_pairs = {split: {(row["landsat_id"], row["sentinel_id"])
                            for row in items}
                    for split, items in by_split.items()}
    report = {
        "release_manifest_sha256": digest,
        "counts": dict(counts),
        "unique_landsat_ids": len({row["landsat_id"] for row in rows}),
        "unique_sentinel_ids": len({row["sentinel_id"] for row in rows}),
        "unique_spatial_blocks": len({row["spatial_block_id"] for row in rows}),
        "per_split": {split: {
            "patches": len(items),
            "landsat_ids": len(sets[split]["landsat_id"]),
            "sentinel_ids": len(sets[split]["sentinel_id"]),
            "spatial_blocks": len(sets[split]["spatial_block_id"]),
        } for split, items in by_split.items()},
        "comparisons": {},
    }
    for first, second in COMPARISONS:
        common = {field: sorted(sets[first][field] & sets[second][field])
                  for field in FIELDS}
        paired = sorted(source_pairs[first] & source_pairs[second])
        report["comparisons"][f"{first}_{second}"] = {
            "shared_landsat_ids": common["landsat_id"],
            "shared_sentinel_ids": common["sentinel_id"],
            "shared_spatial_block_ids": common["spatial_block_id"],
            "shared_source_pairs": [
                {"landsat_id": landsat, "sentinel_id": sentinel}
                for landsat, sentinel in paired
            ],
            "shared_landsat_count": len(common["landsat_id"]),
            "shared_sentinel_count": len(common["sentinel_id"]),
            "shared_spatial_block_count": len(common["spatial_block_id"]),
            "shared_source_pair_count": len(paired),
            "second_split_patches_with_first_landsat": sum(
                row["landsat_id"] in sets[first]["landsat_id"]
                for row in by_split[second]
            ),
            "second_split_patches_with_first_sentinel": sum(
                row["sentinel_id"] in sets[first]["sentinel_id"]
                for row in by_split[second]
            ),
            "second_split_patches_with_first_source_pair": sum(
                (row["landsat_id"], row["sentinel_id"]) in source_pairs[first]
                for row in by_split[second]
            ),
        }
    if any(report["comparisons"][key]["shared_spatial_block_count"]
           for key in report["comparisons"]):
        raise ValueError("Spatial blocks overlap across splits")
    return report


def write_tex_table(report: dict, path: Path) -> None:
    lines = [
        r"\begin{tabular}{lrrrrr}",
        r"\toprule",
        r"Splits & Blocks & Landsat & Sentinel-2 & Source pairs & Second-split patches \\",
        r"\midrule",
    ]
    for label in ("train_test", "train_val", "val_test"):
        row = report["comparisons"][label]
        cells = [
            label.replace("_", "--"),
            str(row["shared_spatial_block_count"]),
            str(row["shared_landsat_count"]),
            str(row["shared_sentinel_count"]),
            str(row["shared_source_pair_count"]),
            str(row["second_split_patches_with_first_source_pair"]),
        ]
        lines.append(" & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tex-output", type=Path)
    args = parser.parse_args()
    result = audit(args.manifest)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    if args.tex_output:
        write_tex_table(result, args.tex_output)
    for label, comparison in result["comparisons"].items():
        print(label, "shared Landsat", comparison["shared_landsat_count"],
              "Sentinel", comparison["shared_sentinel_count"],
              "source pairs", comparison["shared_source_pair_count"],
              "blocks", comparison["shared_spatial_block_count"])
