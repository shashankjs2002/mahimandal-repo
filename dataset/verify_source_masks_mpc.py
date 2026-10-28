"""Check rounded cloud-boundary pairs against original Planetary Computer assets.

Run in the dataset-preparation Kaggle environment (requires internet, rasterio,
planetary-computer, and pystac-client). Does not change the frozen release.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import planetary_computer as pc
import rasterio
from affine import Affine
from pystac_client import Client
from rasterio.enums import Resampling
from rasterio.vrt import WarpedVRT
from rasterio.windows import Window

from audit_release_quality import EXPECTED_SHA256


STAC_URL = "https://planetarycomputer.microsoft.com/api/stac/v1"
LR_SIZE = 160
HR_SIZE = 480
S2_CLEAR_SCL = (4, 5, 6, 11)


def source_item(catalog: Client, collection: str, identifier: str):
    matches = list(catalog.search(collections=[collection], ids=[identifier]).items())
    if len(matches) != 1 or matches[0].id != identifier:
        raise RuntimeError(f"Expected exactly one {collection} item {identifier}; found {len(matches)}")
    return matches[0]


def asset_metadata(item, key: str) -> dict:
    asset = item.assets[key]
    with rasterio.open(pc.sign(asset.href)) as source:
        return {
            "asset": key,
            "source_dtype": source.dtypes[0],
            "source_nodata": source.nodata,
            "source_scale": source.scales[0],
            "source_offset": source.offsets[0],
            "stac_raster_bands": asset.extra_fields.get("raster:bands", []),
        }


def check_pair(catalog: Client, row: dict, release_root: Path) -> dict:
    landsat = source_item(catalog, "landsat-c2-l2", row["landsat_id"])
    sentinel = source_item(catalog, "sentinel-2-l2a", row["sentinel_id"])
    crs = row["crs"]
    x_ul, y_ul = float(row["ul_x"]), float(row["ul_y"])
    hr_transform = Affine(10, 0, x_ul, 0, -10, y_ul)

    with rasterio.open(pc.sign(landsat.assets["qa_pixel"].href)) as source:
        if str(source.crs) != crs:
            raise RuntimeError(f"Landsat CRS mismatch for {row['pair_id']}")
        raw_col, raw_line = ~source.transform * (x_ul, y_ul)
        col, line = round(raw_col), round(raw_line)
        if abs(col - raw_col) > 0.01 or abs(line - raw_line) > 0.01:
            raise RuntimeError("LR patch is not aligned to QA raster grid")
        qa = source.read(1, window=Window(col, line, LR_SIZE, LR_SIZE))
    if qa.shape != (LR_SIZE, LR_SIZE):
        raise RuntimeError(f"QA patch has wrong shape: {qa.shape}")
    landsat_clear = (qa & 31) == 0

    with rasterio.open(pc.sign(landsat.assets["red"].href)) as source:
        raw_landsat_red = source.read(1, window=Window(col, line, LR_SIZE, LR_SIZE))

    with rasterio.open(pc.sign(sentinel.assets["SCL"].href)) as source:
        with WarpedVRT(source, crs=crs, transform=hr_transform,
                       width=HR_SIZE, height=HR_SIZE,
                       resampling=Resampling.nearest, src_nodata=0, nodata=0) as vrt:
            scl = vrt.read(1)
    sentinel_clear = np.isin(scl, S2_CLEAR_SCL)
    with rasterio.open(pc.sign(sentinel.assets["B04"].href)) as source:
        with WarpedVRT(source, crs=crs, transform=hr_transform,
                       width=HR_SIZE, height=HR_SIZE,
                       resampling=Resampling.bilinear, src_nodata=0, nodata=0) as vrt:
            raw_sentinel_red = vrt.read(1)

    lr_path = release_root / row["lr_path"]
    hr_path = release_root / row["hr_path"]
    with rasterio.open(lr_path) as source:
        released_landsat_red = source.read(3)
    with rasterio.open(hr_path) as source:
        released_sentinel_red = source.read(3)
    baseline = str(sentinel.properties.get("s2:processing_baseline", "missing"))
    offset = 1000 if float(baseline if baseline != "missing" else "04.00") >= 4.0 else 0
    expected_landsat_red = np.clip(
        raw_landsat_red.astype("float32") * 0.275 - 2000.0, 0, 10000
    ).round().astype("int16")
    expected_sentinel_red = np.clip(
        raw_sentinel_red.astype("float32") - offset, 0, 10000
    ).astype("int16")
    joint = (np.repeat(np.repeat(landsat_clear, 3, axis=0), 3, axis=1)
             & sentinel_clear)
    clear = float(joint.mean())
    return {
        "pair_id": row["pair_id"],
        "manifest_clear_frac_rounded": float(row["clear_frac"]),
        "raw_joint_clear_pixels": int(joint.sum()),
        "raw_joint_total_pixels": int(joint.size),
        "raw_joint_clear_frac": clear,
        "raw_joint_obstructed_pct": 100 * (1 - clear),
        "passes_collector_threshold": clear >= 0.99,
        "landsat_clear_frac": float(landsat_clear.mean()),
        "sentinel_scl_clear_frac": float(sentinel_clear.mean()),
        "sentinel_processing_baseline": baseline,
        "notebook_sentinel_offset_dn": offset,
        "landsat_red_source_zero_pixels": int(np.count_nonzero(raw_landsat_red == 0)),
        "sentinel_red_source_zero_pixels": int(np.count_nonzero(raw_sentinel_red == 0)),
        "landsat_red_conversion_mismatch_pixels": int(np.count_nonzero(
            expected_landsat_red != released_landsat_red)),
        "sentinel_red_conversion_mismatch_pixels": int(np.count_nonzero(
            expected_sentinel_red != released_sentinel_red)),
        "landsat_red_metadata": asset_metadata(landsat, "red"),
        "sentinel_red_metadata": asset_metadata(sentinel, "B04"),
        "radiometry_note": "Pixel equality proves faithful reproduction of notebook conversion, not that the Sentinel BOA offset was physically appropriate; inspect STAC scale/offset and processing documentation.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--release-root", required=True, type=Path,
                        help="Folder containing train/LR and train/HR GeoTIFFs")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    digest = hashlib.sha256(args.manifest.read_bytes()).hexdigest()
    if digest != EXPECTED_SHA256:
        raise ValueError(f"Not the frozen reflectance16 release manifest: {digest}")
    with args.manifest.open(newline="", encoding="utf-8") as stream:
        boundary = [row for row in csv.DictReader(stream)
                    if float(row["clear_frac"]) == 0.99]
    if len(boundary) != 3:
        raise ValueError(f"Expected 3 frozen-release boundary pairs, found {len(boundary)}")
    catalog = Client.open(STAC_URL)
    results = [check_pair(catalog, row, args.release_root) for row in boundary]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print(json.dumps([{
        "pair_id": result["pair_id"],
        "raw_joint_clear_frac": result["raw_joint_clear_frac"],
        "passes_collector_threshold": result["passes_collector_threshold"],
    } for result in results], indent=2))
    print("Saved", args.output)


if __name__ == "__main__":
    main()
