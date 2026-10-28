"""Read-only lineage recovery from the user's older Mahimandal preparation tree.

Recovers only recorded evidence. Does not infer historical asset IDs from a
current catalogue search or execute the original notebook.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

import pandas as pd
import rasterio


EXPECTED = "6bf97403ee40ccf0cbd38b2e6e4932e01973e5d00e211e0acfb1e0b810efc5e7"
LINEAGE_TERMS = ("dem_item", "dem_asset", "dem_tile", "dem_version", "esri_item", "esri_raster",
                 "esri_object", "landcover_tile", "landcover_item", "raster_ids", "mosaic_rule")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def classify_record(row) -> str:
    value = lambda key: float(row[key])
    elevation, slope = value("elevation_p90_m"), value("slope_p90_deg")
    if value("esri2025_snow_ice_pct") >= 1 or elevation >= 3000 or (elevation >= 2500 and slope >= 15):
        return "alpine"
    if elevation >= 1500 or slope >= 15:
        return "mountain"
    for column, threshold, category in (
        ("built", 12, "urban"), ("bare", 35, "desert_bare"),
        ("trees", 40, "forest"), ("crops", 40, "cropland"),
    ):
        if value(f"esri2025_{column}_pct") >= threshold:
            return category
    if value("esri2025_water_pct") + value("esri2025_flooded_vegetation_pct") >= 25:
        return "water_wetland"
    if value("esri2025_rangeland_pct") >= 40:
        return "shrub_grass"
    return "mixed"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--legacy", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if sha(args.manifest) != EXPECTED:
        raise ValueError("Not the frozen release manifest")
    if args.output.resolve().is_relative_to(args.legacy.resolve()):
        raise ValueError("Never write in the older input tree")
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError("Choose a new audit output directory")
    args.output.mkdir(parents=True, exist_ok=True)
    frozen = pd.read_csv(args.manifest, keep_default_na=False)
    legacy = pd.read_csv(args.legacy / "manifest.csv", keep_default_na=False)
    if not legacy.pair_id.is_unique:
        raise ValueError("Older manifest pair IDs are not unique")
    joined = frozen.set_index("pair_id").join(legacy.set_index("pair_id"), rsuffix="_legacy", how="left")
    missing = sorted(set(frozen.pair_id) - set(legacy.pair_id))
    extras = sorted(set(legacy.pair_id) - set(frozen.pair_id))
    columns = [c for c in legacy if c.startswith("esri2025_") or c in (
        "landsat_id", "sentinel_id", "landscape_class", "landcover_year", "landcover_source",
        "landcover_service", "dem_source", "worldcover_source", "elevation_p90_m", "slope_p90_deg")]
    mismatches = []
    for column in columns:
        different = joined[column].astype(str) != joined[column + "_legacy"].astype(str)
        for pair_id in joined.index[different]:
            mismatches.append({"pair_id": pair_id, "field": column,
                               "frozen": str(joined.loc[pair_id, column]),
                               "legacy": str(joined.loc[pair_id, column + "_legacy"])})
    pd.DataFrame(mismatches, columns=["pair_id", "field", "frozen", "legacy"]).to_csv(args.output / "metadata_differences.csv", index=False)
    reproduced = frozen[["pair_id", "landscape_class", "split", "spatial_block_id"]].copy()
    reproduced["recomputed_from_saved_metadata"] = [classify_record(row) for row in frozen.to_dict("records")]
    reproduced["matches_recorded_class"] = reproduced.landscape_class == reproduced.recomputed_from_saved_metadata
    reproduced.to_csv(args.output / "classification_reproduction.csv", index=False)
    inventory, lineage_records = [], []
    tag_counts = Counter()
    rasters = 0
    for path in sorted(args.legacy.rglob("*")):
        if not path.is_file() or path.is_symlink():
            continue
        relative = path.relative_to(args.legacy).as_posix()
        if path.suffix.lower() in (".tif", ".tiff"):
            rasters += 1
            with rasterio.open(path) as source:
                tags = source.tags()
                # Domain and band tags are checked too, not only the default tag map.
                domains = {"default": tags}
                for domain in source.tag_namespaces():
                    domains[domain] = source.tags(ns=domain)
                for band in range(1, source.count + 1):
                    domains[f"band_{band}"] = source.tags(band)
                for domain, values in domains.items():
                    for key, value in values.items():
                        tag_counts[key] += 1
                        if any(term in key.casefold() for term in LINEAGE_TERMS):
                            lineage_records.append({"file": relative, "domain": domain, "key": key, "value": value})
            if rasters % 1000 == 0:
                print(f"Older TIFF headers scanned: {rasters}", flush=True)
        elif path.suffix.lower() in (".json", ".csv", ".txt", ".md", ".log", ".jsonl", ".xml"):
            raw = path.read_bytes()
            text = raw.decode("utf-8-sig", errors="replace")
            found = sorted({term for term in LINEAGE_TERMS if term in text.casefold()})
            inventory.append({"file": relative, "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest(),
                              "lineage_terms_found": "|".join(found)})
    pd.DataFrame(inventory).to_csv(args.output / "older_metadata_inventory.csv", index=False)
    (args.output / "raster_tag_lineage_hits.json").write_text(json.dumps(lineage_records, indent=2) + "\n", encoding="utf-8")
    older_protocol = json.loads((args.legacy / "protocol.json").read_text(encoding="utf-8"))
    summary = {"frozen_manifest_sha256": EXPECTED, "legacy_manifest_sha256": sha(args.legacy / "manifest.csv"),
        "script_sha256": sha(Path(__file__)), "legacy_pairs": len(legacy), "frozen_pairs": len(frozen),
        "missing_frozen_pair_ids_in_legacy": missing, "legacy_only_pair_ids": extras,
        "compared_metadata_fields": columns, "metadata_difference_count": len(mismatches),
        "classification_agreements": int(reproduced.matches_recorded_class.sum()),
        "classification_disagreements": int((~reproduced.matches_recorded_class).sum()),
        "older_tiff_headers_scanned": rasters, "raster_tag_key_counts": dict(tag_counts),
        "explicit_auxiliary_lineage_tag_hits": len(lineage_records),
        "older_protocol_split_fractions": older_protocol.get("split_fractions"),
        "split_note": "Older protocol targets are historical configuration, not a replacement for frozen realized assignments.",
        "lineage_status": "partial; inspect retained hits and metadata files; absent receipts are not reconstructed as fact",
        "classification_scope": "Reproduction from rounded saved percentages and terrain descriptors, not independent semantic ground truth or source-raster reclassification"}
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    rows = [f"{sha(path)}  {path.relative_to(args.output).as_posix()}" for path in sorted(args.output.rglob("*")) if path.is_file()]
    (args.output / "checksums.sha256").write_text("\n".join(rows) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
