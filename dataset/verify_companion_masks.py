"""Check existing identified-fill masks against their inventory and release grids."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    paper = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release", type=Path, default=paper / "data/reflectance16")
    parser.add_argument("--masks", type=Path, default=paper / "results/audits/identified_fill_masks_v1")
    parser.add_argument("--output", type=Path, default=paper / "results/cpu_support/verification")
    args = parser.parse_args()
    manifest_path = args.release / "paper_release_manifest.csv"
    manifest = pd.read_csv(manifest_path).set_index("pair_id")
    inventory_path = args.masks / "summary.json"
    inventory = json.loads(inventory_path.read_text())
    if not inventory["complete"] or digest(manifest_path) != inventory["manifest_sha256"]:
        raise ValueError("Incomplete masks or release mismatch")
    files = inventory["files"]
    if len(files) != inventory["mask_files"] or len({(r["pair_id"], r["sensor"]) for r in files}) != len(files):
        raise ValueError("Duplicate or missing inventory rows")
    records, masks_by_pair = [], {}
    for record in files:
        pair_id, sensor = record["pair_id"], record["sensor"]
        row = manifest.loc[pair_id]
        path = (args.masks / record["path"]).resolve()
        path.relative_to(args.masks.resolve())
        if digest(path) != record["sha256"] or record["split"] != row.split:
            raise ValueError("Mask inventory mismatch: " + str(path))
        source_path = args.release / row["lr_path" if sensor == "landsat" else "hr_path"]
        with rasterio.open(path) as mask, rasterio.open(source_path) as source:
            expected = 160 if sensor == "landsat" else 480
            if (mask.count, mask.width, mask.height) != (1, expected, expected):
                raise ValueError("Mask dimensions differ")
            if mask.crs != source.crs or mask.transform != source.transform or mask.bounds != source.bounds:
                raise ValueError("Mask grid differs from imagery")
            values = mask.read(1)
        if not np.isin(values, [0, 1]).all() or int(values.sum()) != record["pixels_identified"]:
            raise ValueError("Mask values differ from inventory")
        hr = np.repeat(np.repeat(values, 3, axis=0), 3, axis=1) if sensor == "landsat" else values
        masks_by_pair.setdefault(pair_id, np.zeros((480, 480), dtype=bool))
        masks_by_pair[pair_id] |= hr.astype(bool)
        records.append(dict(pair_id=pair_id, split=row.split, landscape_class=row.landscape_class,
                            sensor=sensor, identified_native_pixels=int(values.sum()),
                            source_grid_matches=True, inventory_hash_matches=True))
    if len(masks_by_pair) != inventory["distinct_pairs"]:
        raise ValueError("Affected pair count differs")
    frame = pd.DataFrame(records)
    pair_rows = []
    for pair_id, mask in masks_by_pair.items():
        row = manifest.loc[pair_id]
        pair_rows.append(dict(pair_id=pair_id, split=row.split, landscape_class=row.landscape_class,
                              identified_hr_union=int(mask.sum()),
                              identified_hr_union_after_border3=int(mask[3:-3, 3:-3].sum())))
    pairs = pd.DataFrame(pair_rows)
    args.output.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output / "mask_file_checks.csv", index=False)
    pairs.to_csv(args.output / "identified_fill_pair_support.csv", index=False)
    pairs.groupby(["split", "landscape_class"]).agg(
        affected_pairs=("pair_id", "size"), identified_hr_union=("identified_hr_union", "sum"),
        identified_hr_union_after_border3=("identified_hr_union_after_border3", "sum")
    ).to_csv(args.output / "identified_fill_by_class.csv")
    summary = dict(manifest_sha256=digest(manifest_path), mask_inventory_sha256=digest(inventory_path),
                   script_sha256=digest(__file__), verified_files=len(frame), affected_pairs=len(pairs),
                   native_pixel_counts=frame.groupby("sensor").identified_native_pixels.sum().to_dict(),
                   affected_pairs_by_split=pairs.split.value_counts().to_dict(),
                   all_inventory_hashes_and_source_grids_match=True,
                   semantics="1 identifies source fill recovered by the prior audit. 0 means no fill identified. These are not exhaustive QA/SCL or saturation masks.",
                   hr_union_rule="Nearest-neighbour 3x expansion of the Landsat mask, union with the Sentinel mask where present; count locations once per pair.",
                   scope="Only inventory-listed identified-fill masks are verified; absent files do not establish universally valid pixels.",
                   original_images_masks_and_scores_modified=False)
    (args.output / "mask_verification.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({k: summary[k] for k in ("verified_files", "affected_pairs", "native_pixel_counts", "affected_pairs_by_split")}, indent=2))


if __name__ == "__main__":
    main()
