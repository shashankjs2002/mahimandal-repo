"""Inventory stored zeros without treating zero reflectance as invalid data.

Read-only full-release screening. Candidate IDs require original-source masks
to distinguish valid clipped-dark pixels from fill. No metric masks are changed.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import rasterio

from audit_sentinel_product_metadata import EXPECTED_MANIFEST


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--release-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if hashlib.sha256(args.manifest.read_bytes()).hexdigest() != EXPECTED_MANIFEST:
        raise ValueError("Not the frozen release")
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError("Choose a new output")
    args.output.mkdir(parents=True, exist_ok=True)
    with args.manifest.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != 4800 or len({row["pair_id"] for row in rows}) != 4800:
        raise ValueError("Unexpected release cohort")
    records = []
    for index, row in enumerate(rows):
        for sensor, key, size in (("landsat", "lr_path", 160), ("sentinel", "hr_path", 480)):
            path = (args.release_root / row[key]).resolve()
            path.relative_to(args.release_root.resolve())
            with rasterio.open(path) as src:
                values = src.read()
                if values.shape != (4, size, size) or values.dtype != np.dtype("int16"):
                    raise ValueError(f"Unexpected release raster: {row['pair_id']}")
                zeros = values == 0
                record = {"pair_id": row["pair_id"], "split": row["split"],
                          "landscape_class": row["landscape_class"], "sensor": sensor,
                          "pixels": size * size, "zero_any_band_pixels": int(zeros.any(0).sum()),
                          "zero_all_bands_pixels": int(zeros.all(0).sum()),
                          "masked_pixels_any_band": int((src.read_masks() == 0).any(0).sum()),
                          "outside_physical_storage_values": int(((values < 0) | (values > 10000)).sum())}
                for band, name in enumerate(("blue", "green", "red", "nir")):
                    record[f"zero_{name}_pixels"] = int(zeros[band].sum())
                records.append(record)
        if (index + 1) % 500 == 0:
            print(f"Scanned {index + 1}/4800 pairs", flush=True)
    for name, selected in (("per_raster.csv", records),
                           ("source_followup_candidates.csv", [r for r in records if r["zero_any_band_pixels"]])):
        with (args.output / name).open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(records[0]))
            writer.writeheader()
            writer.writerows(selected)
    summary = {"manifest_sha256": EXPECTED_MANIFEST,
               "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
               "pairs": len(rows), "rasters": len(records),
               "candidate_pairs": len({r["pair_id"] for r in records if r["zero_any_band_pixels"]}),
               "by_sensor": {}, "invalidity_inferred_from_zero": False,
               "dataset_or_metric_masks_changed": False,
               "note": "Stored zeros are ambiguous: valid clipped reflectance and source fill can coincide. Candidates are not failed pairs."}
    for sensor in ("landsat", "sentinel"):
        selected = [r for r in records if r["sensor"] == sensor]
        summary["by_sensor"][sensor] = {"rasters_with_zero": sum(r["zero_any_band_pixels"] > 0 for r in selected),
            "zero_any_band_pixels": sum(r["zero_any_band_pixels"] for r in selected),
            "total_spatial_pixels": sum(r["pixels"] for r in selected),
            "masked_pixels": sum(r["masked_pixels_any_band"] for r in selected),
            "out_of_range_values": sum(r["outside_physical_storage_values"] for r in selected)}
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
