"""Audit the frozen release manifest without training or reading model outputs.

The distance calculation reproduces the collector's 4,800 m square footprint
approximation in its spherical cylindrical equal-area coordinate system. It is
not a survey of true geodetic footprint boundaries or a registration RMSE.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


EXPECTED_SHA256 = "6bf97403ee40ccf0cbd38b2e6e4932e01973e5d00e211e0acfb1e0b810efc5e7"
EXPECTED_COUNTS = {"train": 3424, "val": 645, "test": 731}
SPLITS = ("train", "val", "test")
FOOTPRINT_HALF_SIDE_M = 2400.0


def nearest_footprints(first: list[dict], second: list[dict]) -> dict:
    """Find the closest modeled square footprints across two splits."""
    a = np.asarray([(float(r["equal_area_x_m"]), float(r["equal_area_y_m"]))
                    for r in first], dtype=np.float64)
    b = np.asarray([(float(r["equal_area_x_m"]), float(r["equal_area_y_m"]))
                    for r in second], dtype=np.float64)
    best = (math.inf, None, None)
    for start in range(0, len(a), 128):
        dx = np.maximum(np.abs(a[start:start + 128, None, 0] - b[None, :, 0])
                        - 2 * FOOTPRINT_HALF_SIDE_M, 0.0)
        dy = np.maximum(np.abs(a[start:start + 128, None, 1] - b[None, :, 1])
                        - 2 * FOOTPRINT_HALF_SIDE_M, 0.0)
        distances = np.hypot(dx, dy)
        local = np.unravel_index(np.argmin(distances), distances.shape)
        value = float(distances[local])
        if value < best[0]:
            best = (value, start + int(local[0]), int(local[1]))
    distance, i, j = best
    return {
        "minimum_modeled_edge_distance_m": round(distance, 3),
        "first_pair_id": first[i]["pair_id"],
        "second_pair_id": second[j]["pair_id"],
        "first_block_id": first[i]["spatial_block_id"],
        "second_block_id": second[j]["spatial_block_id"],
    }


def audit(manifest: Path, expected_sha256: str = EXPECTED_SHA256) -> dict:
    digest = hashlib.sha256(manifest.read_bytes()).hexdigest()
    if digest != expected_sha256:
        raise ValueError(f"Manifest hash differs from frozen release: {digest}")
    with manifest.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    counts = Counter(row["split"] for row in rows)
    if dict(counts) != EXPECTED_COUNTS:
        raise ValueError(f"Unexpected split counts: {dict(counts)}")
    if any(row["release_included"].casefold() != "true" for row in rows):
        raise ValueError("Manifest includes a rejected pair")
    if len({row["pair_id"] for row in rows}) != len(rows):
        raise ValueError("Duplicate pair IDs")

    by_split = {split: [row for row in rows if row["split"] == split]
                for split in SPLITS}
    blocks = {split: {row["spatial_block_id"] for row in items}
              for split, items in by_split.items()}
    if any(blocks[a] & blocks[b] for a, b in
           (("train", "val"), ("train", "test"), ("val", "test"))):
        raise ValueError("Spatial block ID appears in multiple splits")

    landscapes = sorted({row["landscape_class"] for row in rows})
    class_rows = []
    for landscape in landscapes:
        line = {"landscape_class": landscape}
        for split in SPLITS:
            selected = [row for row in by_split[split]
                        if row["landscape_class"] == landscape]
            line[f"{split}_pairs"] = len(selected)
            line[f"{split}_blocks"] = len({r["spatial_block_id"] for r in selected})
        class_rows.append(line)

    cloud_exceptions = []
    cloud_boundary = []
    for row in rows:
        clear_fraction = float(row["clear_frac"])
        if clear_fraction <= 0.99:
            item = {
                "pair_id": row["pair_id"], "split": row["split"],
                "manifest_clear_frac": clear_fraction,
                "manifest_obstructed_pct": round(100 * (1 - clear_fraction), 4),
                "quality_flag": row["quality_flag"],
            }
            (cloud_exceptions if clear_fraction < 0.99 else cloud_boundary).append(item)
    months = Counter(row["date_L"][:7] for row in rows)
    years = Counter(row["date_L"][:4] for row in rows)
    distances = {f"{a}_{b}": nearest_footprints(by_split[a], by_split[b])
                 for a, b in (("train", "val"), ("train", "test"), ("val", "test"))}
    return {
        "manifest_sha256": digest,
        "counts": dict(counts),
        "spatial_blocks": {split: len(blocks[split]) for split in SPLITS},
        "landscape_split": class_rows,
        "acquisition_years": dict(sorted(years.items())),
        "acquisition_months": dict(sorted(months.items())),
        "cloud_threshold": {
            "manifest_precision_note": "clear_frac is rounded in the manifest; exact acceptance requires raw masks",
            "pairs_below_0_99_clear": len(cloud_exceptions),
            "pairs_at_rounded_0_99_clear_boundary": len(cloud_boundary),
            "exceptions": cloud_exceptions,
            "boundary_pairs_requiring_raw_mask_check": cloud_boundary,
        },
        "cross_split_distances": {
            "method": "collector's 4800 m axis-aligned square approximation in spherical cylindrical equal-area metres",
            "limitations": "not native-CRS footprint geometry, a geodesic buffer, or surveyed registration accuracy",
            "pairs": distances,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    result = audit(args.manifest)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    table = args.output.with_name(args.output.stem + "_landscape_split.csv")
    with table.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=result["landscape_split"][0].keys())
        writer.writeheader()
        writer.writerows(result["landscape_split"])
    print(json.dumps({"report": str(args.output), "landscape_table": str(table),
                      "cloud_exceptions": result["cloud_threshold"]["pairs_below_0_99_clear"],
                      "cloud_boundary_pairs": result["cloud_threshold"]["pairs_at_rounded_0_99_clear_boundary"],
                      "cross_split_distances": result["cross_split_distances"]["pairs"]}, indent=2))


if __name__ == "__main__":
    main()
