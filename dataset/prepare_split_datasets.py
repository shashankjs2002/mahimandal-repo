"""Export locked Mahimandal pairs into split-first GeoTIFF releases.

The exporter never redraws train/validation/test assignments. It copies the
assignments from ``paper_release_manifest.csv`` into two storage variants:

* ``reflectance16``: original int16 reflectance x 10000 GeoTIFFs;
* ``normalized_float32``: the same samples stored as float32 reflectance [0, 1].
"""
from __future__ import annotations

import argparse
import csv
import json
import os
from collections import Counter
from pathlib import Path
import shutil

import numpy as np
import rasterio

from data import digest, find_release_manifest, read_rows, resolve


VARIANTS = ("reflectance16", "normalized_float32")
SPLITS = ("train", "val", "test")
RELEASE_SPLIT_OBJECTIVE = {"train": 0.78, "val": 0.11, "test": 0.11}
SOURCE_DECLARED_SPLIT_OBJECTIVE = {"train": 0.80, "val": 0.10, "test": 0.10}
DATA_LICENSE_NOTICE = """# MahimandalSR data license and source attribution

The original MahimandalSR curation, manifest, documentation, and arrangement are
released under the Creative Commons Attribution 4.0 International license
(CC BY 4.0): https://creativecommons.org/licenses/by/4.0/

The license does not replace terms attached to source observations or auxiliary
products. Landsat imagery is provided by the U.S. Geological Survey and is
public domain; acknowledge USGS as the source. Sentinel-2 and Copernicus data
remain subject to the Copernicus Sentinel Data Terms and Conditions. Esri/Impact
Observatory land-cover, geoBoundaries, ESA WorldCover, and Copernicus DEM data
retain their respective source terms. See dataset_card.json and the manuscript
for citations. No additional rights over source imagery are asserted here.
"""


def _write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _validate_source(path: Path) -> None:
    with rasterio.open(path) as source:
        if source.count != 4 or set(source.dtypes) != {"int16"}:
            raise ValueError(f"Expected four-band int16 reflectance TIFF: {path}")
        names = tuple(str(value or "").casefold() for value in source.descriptions)
        if names != ("blue", "green", "red", "nir"):
            raise ValueError(f"Unexpected source band order in {path}: {source.descriptions}")


def _copy_reflectance16(source: Path, destination: Path, overwrite: bool) -> None:
    if destination.is_file() and not overwrite:
        if destination.stat().st_size != source.stat().st_size:
            raise ValueError(f"Existing native copy is incomplete: {destination}")
        _validate_source(destination)
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(".tmp.tif")
    shutil.copy2(source, temporary)
    os.replace(temporary, destination)


def _write_normalized(source_path: Path, destination: Path, overwrite: bool) -> None:
    if destination.is_file() and not overwrite:
        with rasterio.open(source_path) as source, rasterio.open(destination) as existing:
            if (existing.count == 4 and set(existing.dtypes) == {"float32"}
                    and existing.shape == source.shape and existing.crs == source.crs
                    and existing.transform == source.transform
                    and existing.descriptions == source.descriptions):
                return
        raise ValueError(f"Existing normalized output has the wrong contract: {destination}")

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(".tmp.tif")
    with rasterio.open(source_path) as source:
        if set(source.dtypes) != {"int16"}:
            raise ValueError(f"Normalization requires int16 reflectance x10000: {source_path}")
        values = np.clip(source.read().astype(np.float32) / 10000.0, 0.0, 1.0)
        profile = source.profile.copy()
        profile.update(dtype="float32", compress="deflate", predictor=3, nodata=None)
        tags = source.tags()
        band_tags = [source.tags(index) for index in range(1, source.count + 1)]
        descriptions = source.descriptions
        colorinterp = source.colorinterp
        mask = source.dataset_mask()
    with rasterio.Env(GDAL_TIFF_INTERNAL_MASK=True):
        with rasterio.open(temporary, "w", **profile) as output:
            output.write(values)
            output.write_mask(mask)
            output.update_tags(**tags, storage_variant="normalized_float32",
                               value_domain="reflectance_[0,1]")
            output.descriptions = descriptions
            output.colorinterp = colorinterp
            for index, values_for_band in enumerate(band_tags, 1):
                output.update_tags(index, **values_for_band)
    os.replace(temporary, destination)


def _write_manifest(path: Path, rows: list[dict]) -> None:
    fields = list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _assert_release_closure(root: Path, rows: list[dict]) -> dict:
    expected = {
        (root / row[field]).resolve()
        for row in rows
        for field in ("lr_path", "hr_path")
    }
    actual = {
        path.resolve()
        for split in SPLITS
        for role in ("LR", "HR")
        for path in (root / split / role).glob("*.tif")
    }
    missing = sorted(str(path.relative_to(root.resolve())) for path in expected - actual)
    orphaned = sorted(str(path.relative_to(root.resolve())) for path in actual - expected)
    if missing or orphaned:
        raise ValueError(
            "Release is not manifest-closed: "
            f"missing={missing[:5]}, orphaned={orphaned[:5]}"
        )
    return {
        "expected_tiffs": len(expected),
        "actual_tiffs": len(actual),
        "missing_tiffs": 0,
        "orphan_tiffs": 0,
    }


def _dataset_card(source_root: Path, variant: str, rows: list[dict]) -> dict:
    counts = Counter(row["split"] for row in rows)
    realized = {split: counts[split] / len(rows) for split in SPLITS}
    return {
        "format": "mahimandal-split-geotiff-v1",
        "variant": variant,
        "pairs": len(rows),
        "split_counts": {split: counts[split] for split in SPLITS},
        "release_split_objective": RELEASE_SPLIT_OBJECTIVE,
        "realized_split_fractions": realized,
        "split_source": "authoritative paper_release_manifest.csv; no reshuffling",
        "split_resolution": (
            "The collection notebook originally recorded an 80/10/10 planning field. "
            "The report-stage allocator actually used 78/11/11 with indivisible 100-km "
            "blocks. Release v1 freezes those assignments and their realized fractions."
        ),
        "layout": "{train,val,test}/{LR,HR}/{pair_id}.tif",
        "geometry": {"LR": [4, 160, 160], "HR": [4, 480, 480], "scale": 3},
        "band_order_on_disk": ["blue", "green", "red", "nir"],
        "stored_dtype": "int16" if variant == "reflectance16" else "float32",
        "stored_value_domain": ("reflectance_x10000_[0,10000]"
                                if variant == "reflectance16" else "reflectance_[0,1]"),
        "model_value_domain": "float32 reflectance [0,1]",
        "model_decode": ("clip(pixel/10000,0,1)" if variant == "reflectance16"
                         else "identity followed by clip(pixel,0,1)"),
        "source_root": str(source_root.resolve()),
        "source_manifest_sha256": digest(find_release_manifest(source_root)),
        "license": "CC-BY-4.0 for original curation and metadata; source-data terms retained",
        "scientific_note": (
            "The two releases encode the same reflectance samples. They are storage variants, "
            "not independent data splits or distinct radiometric experiments."
        ),
    }


def export_release(source_root: str | Path, output_root: str | Path, *,
                   variants: tuple[str, ...] = VARIANTS, overwrite: bool = False,
                   checksums: bool = True, progress_every: int = 100) -> dict[str, dict]:
    source_root = Path(source_root)
    output_root = Path(output_root)
    unknown = set(variants).difference(VARIANTS)
    if unknown:
        raise ValueError(f"Unknown variants: {sorted(unknown)}")
    rows = read_rows(source_root)
    counts = Counter(row["split"] for row in rows)
    if set(counts) != set(SPLITS) or min(counts.values()) < 1:
        raise ValueError(f"Source release must contain train/val/test: {dict(counts)}")
    source_protocol = source_root / "protocol.json"
    summaries: dict[str, dict] = {}

    for variant in variants:
        variant_root = output_root / variant
        variant_root.mkdir(parents=True, exist_ok=True)
        exported_rows: list[dict] = []
        for index, source_row in enumerate(rows, 1):
            row = dict(source_row)
            split = row["split"]
            if split not in SPLITS:
                raise ValueError(f"Unexpected split {split!r} for {row['pair_id']}")
            for role, field, fallback in (
                    ("LR", "lr_path", f"landsat/{row['pair_id']}.tif"),
                    ("HR", "hr_path", f"sentinel/{row['pair_id']}.tif")):
                source_path = resolve(source_root, row.get(field) or fallback)
                _validate_source(source_path)
                relative = Path(split) / role / f"{row['pair_id']}.tif"
                destination = variant_root / relative
                if variant == "reflectance16":
                    _copy_reflectance16(source_path, destination, overwrite)
                else:
                    _write_normalized(source_path, destination, overwrite)
                row[field] = relative.as_posix()
            row["storage_variant"] = variant
            exported_rows.append(row)
            if progress_every and (index % progress_every == 0 or index == len(rows)):
                print(f"[{variant}] {index}/{len(rows)} pairs", flush=True)

        _write_manifest(variant_root / "paper_release_manifest.csv", exported_rows)
        closure = _assert_release_closure(variant_root, exported_rows)
        card = _dataset_card(source_root, variant, exported_rows)
        card["manifest_closure"] = closure
        _write_json(variant_root / "dataset_card.json", card)
        protocol = json.loads(source_protocol.read_text(encoding="utf-8")) if source_protocol.is_file() else {}
        protocol.update(
            storage_variant=variant,
            split_layout="split/LR_or_HR/pair_id.tif",
            release_manifest="paper_release_manifest.csv",
            source_declared_split_objective=SOURCE_DECLARED_SPLIT_OBJECTIVE,
            release_split_objective=RELEASE_SPLIT_OBJECTIVE,
            realized_split_counts=card["split_counts"],
            realized_split_fractions=card["realized_split_fractions"],
            split_configuration_resolution=card["split_resolution"],
            manifest_closure=closure,
        )
        _write_json(variant_root / "protocol.json", protocol)
        (variant_root / "DATA_LICENSE.md").write_text(
            DATA_LICENSE_NOTICE, encoding="utf-8"
        )
        if checksums:
            inventory = sorted(
                path for path in variant_root.rglob("*")
                if path.is_file() and path.name != "checksums.sha256"
            )
            with (variant_root / "checksums.sha256").open(
                    "w", encoding="utf-8", newline="\n") as stream:
                for path in inventory:
                    stream.write(f"{digest(path)}  {path.relative_to(variant_root).as_posix()}\n")
        summaries[variant] = card
    return summaries


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True,
                        help="Release containing landsat/, sentinel/ and the paper manifest")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--variant", action="append", choices=VARIANTS,
                        help="Export one variant; repeat for both. Default: both")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--skip-checksums", action="store_true")
    parser.add_argument("--progress-every", type=int, default=100)
    args = parser.parse_args()
    summaries = export_release(args.source, args.output,
                               variants=tuple(args.variant or VARIANTS),
                               overwrite=args.overwrite,
                               checksums=not args.skip_checksums,
                               progress_every=args.progress_every)
    print(json.dumps(summaries, indent=2))


if __name__ == "__main__":
    main()
