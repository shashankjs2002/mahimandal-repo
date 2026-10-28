"""Read-only radiometry gate: reader semantics, notebook order, original windows.

No imagery is modified. Synthetic reader tests are not original-source evidence.
Remote mode checks two deterministic pairs with distinct Sentinel products.
Unsigned STAC/XML metadata and numeric examples are retained, never SAS tokens.
"""
from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import rasterio
from affine import Affine
from rasterio.enums import Resampling
from rasterio.io import MemoryFile
from rasterio.vrt import WarpedVRT
from rasterio.windows import Window

from audit_sentinel_product_metadata import (
    EXPECTED_MANIFEST, STAC, assess, fetch, signed_metadata_url,
)


BANDS = (("blue", "B02"), ("green", "B03"), ("red", "B04"), ("nir08", "B08"))


def sha(data):
    return hashlib.sha256(data).hexdigest()


def decode(raw, sensor, offset=1000):
    raw = np.asarray(raw, dtype=np.float32)
    value = raw * 0.275 - 2000 if sensor == "landsat" else raw - offset
    clipped = np.clip(value, 0, 10000)
    return np.rint(clipped).astype(np.int16) if sensor == "landsat" else clipped.astype(np.int16)


def reader_probe():
    """A file with non-default scale/offset must still return unscaled DN."""
    raw = np.array([[0, 500, 1000, 2000, 12000]], dtype=np.uint16)
    with MemoryFile() as memory:
        with memory.open(driver="GTiff", width=5, height=1, count=1, dtype="uint16",
                         transform=Affine(10, 0, 300000, 0, -10, 2500000),
                         crs="EPSG:32644", nodata=0) as dst:
            dst.write(raw, 1)
            dst.scales = (0.0001,)
            dst.offsets = (-0.1,)
        with memory.open() as src:
            plain = src.read(1)
            with WarpedVRT(src, crs=src.crs, transform=src.transform,
                           width=5, height=1, resampling=Resampling.bilinear,
                           src_nodata=0, nodata=0) as vrt:
                warped = vrt.read(1)
                valid = vrt.read_masks(1) > 0
            passed = np.array_equal(plain, raw) and np.array_equal(warped, raw)
            if not passed or valid[0, 0] or not valid[0, 1:].all():
                raise ValueError("Reader did not preserve expected raw DN/mask semantics")
            return {"passed": True, "synthetic_not_source_evidence": True,
                    "rasterio_version": rasterio.__version__,
                    "gdal_version": rasterio.__gdal_version__,
                    "scale": src.scales[0], "offset": src.offsets[0],
                    "raw_dn": raw.tolist(), "plain_read": plain.tolist(),
                    "identity_vrt_read": warped.tolist(), "valid_before_clipping": valid.tolist(),
                    "single_offset_stored": decode(raw, "sentinel").tolist(),
                    "double_offset_stored": decode(raw, "sentinel", 2000).tolist(),
                    "note": "DN 500 is valid but clips to zero; DN 0 is fill. Zero reflectance cannot recover validity."}


def notebook_order(path):
    cells = json.loads(path.read_text(encoding="utf-8"))["cells"]
    definitions = {}
    for index, cell in enumerate(cells):
        source = "".join(cell.get("source", []))
        if cell.get("cell_type") != "code" or "def build_patch(" not in source:
            continue
        tree = ast.parse(source)
        for node in tree.body:
            if isinstance(node, ast.FunctionDef) and node.name in ("build_patch", "read_l", "read_s2"):
                definitions[node.name] = {"cell_index_zero_based": index,
                                          "source": ast.get_source_segment(source, node)}
    if "build_patch" not in definitions:
        raise ValueError("Notebook build_patch not found")
    source = definitions["build_patch"]["source"]
    marker = "if (l_dn == 0).any(0).mean() > 0.001"
    if marker not in source or source.index(marker) > source.index("lr = np.clip"):
        raise ValueError("Expected raw-DN rejection before clipping not established")
    return {"notebook_sha256": sha(path.read_bytes()), "definitions": definitions,
            "raw_zero_test_before_clipping": True,
            "zero_fraction_tolerance": 0.001,
            "separate_validity_mask_exported_by_collector": False,
            "note": "Static source inspection; zero-DN heuristic is not an exhaustive vendor-mask audit."}


def source_item(collection, identifier):
    from urllib.parse import quote
    content = fetch(f"{STAC}/collections/{collection}/items/{quote(identifier, safe='')}")
    item = json.loads(content)
    if item["id"] != identifier:
        raise ValueError("Source identity mismatch")
    return item, content


def read_window(asset, crs, transform, size, warped):
    with rasterio.open(signed_metadata_url(asset["href"])) as src:
        metadata = {"dtype": src.dtypes[0], "nodata": src.nodata,
                    "scale": src.scales[0], "offset": src.offsets[0],
                    "raster_bands": asset.get("raster:bands", [])}
        if warped:
            with WarpedVRT(src, crs=crs, transform=transform, width=size, height=size,
                           resampling=Resampling.bilinear, src_nodata=0, nodata=0) as vrt:
                values, mask = vrt.read(1), vrt.read_masks(1) > 0
        else:
            if src.crs != rasterio.crs.CRS.from_string(crs):
                raise ValueError("Native Landsat CRS mismatch")
            col, row = ~src.transform * (transform.c, transform.f)
            if abs(col - round(col)) > .01 or abs(row - round(row)) > .01:
                raise ValueError("Unaligned Landsat window")
            window = Window(round(col), round(row), size, size)
            values, mask = src.read(1, window=window), src.read_masks(1, window=window) > 0
        if values.shape != (size, size) or values.dtype.kind not in "ui":
            raise ValueError("Expected original integer DN window")
        return values, mask, metadata


def example_pixels(raw, valid, released, sensor):
    eligible = np.argwhere(valid & (raw > (2000 if sensor == "sentinel" else 0)))
    if not len(eligible):
        eligible = np.argwhere(valid)
    samples = []
    for y, x in eligible[:2]:
        dn = int(raw[y, x])
        physical = dn * .0000275 - .2 if sensor == "landsat" else (dn - 1000) / 10000
        samples.append({"row": int(y), "col": int(x), "raw_dn": dn,
                        "valid_before_clipping": True, "physical_before_clipping": physical,
                        "expected_stored": int(decode([dn], sensor)[0]),
                        "released_stored": int(released[y, x])})
    return samples


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--notebook", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--release-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--remote", action="store_true", help="Read original COG windows; no full scene downloads")
    args = parser.parse_args()
    if sha(args.manifest.read_bytes()) != EXPECTED_MANIFEST:
        raise ValueError("Not the frozen release manifest")
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError("Use a new output directory; preserve prior evidence")
    args.output.mkdir(parents=True, exist_ok=True)
    summary = {"script_sha256": sha(Path(__file__).read_bytes()),
               "manifest_sha256": EXPECTED_MANIFEST, "new_training_or_inference": False,
               "reader_probe": reader_probe(), "notebook_inspection": notebook_order(args.notebook),
               "source_status": "not_run", "source_results": [],
               "remaining": "No full-release vendor-mask audit or persistent per-pixel validity masks established."}
    try:
        if args.remote:
            with args.manifest.open(newline="", encoding="utf-8") as stream:
                rows = sorted(csv.DictReader(stream), key=lambda row: sha(row["pair_id"].encode()))
            selected, seen = [], set()
            for row in rows:
                if row["sentinel_id"] not in seen:
                    selected.append(row)
                    seen.add(row["sentinel_id"])
                if len(selected) == 2:
                    break
            if len(selected) != 2:
                raise ValueError("Two distinct Sentinel examples required")
            summary["selected_pair_ids"] = [row["pair_id"] for row in selected]
            summary["source_status"] = "in_progress"
            for row in selected:
                print("Source audit:", row["pair_id"], flush=True)
                pair_dir = args.output / sha(row["pair_id"].encode())[:16]
                pair_dir.mkdir()
                for sensor, collection, id_key, path_key, size in (
                    ("landsat", "landsat-c2-l2", "landsat_id", "lr_path", 160),
                    ("sentinel", "sentinel-2-l2a", "sentinel_id", "hr_path", 480),
                ):
                    item, content = source_item(collection, row[id_key])
                    (pair_dir / f"{sensor}_item.json").write_bytes(content)
                    if sensor == "sentinel":
                        xml_asset = item["assets"].get("product-metadata")
                        if not xml_asset:
                            raise ValueError("Sentinel product XML asset missing")
                        xml = fetch(signed_metadata_url(xml_asset["href"]))
                        (pair_dir / "product.xml").write_bytes(xml)
                        assessment = assess(item, xml, row[id_key])
                        if assessment.get("status") != "metadata_supports_conversion":
                            raise ValueError(f"Product coefficient assessment failed: {assessment}")
                    release_path = (args.release_root / row[path_key]).resolve()
                    release_path.relative_to(args.release_root.resolve())
                    with rasterio.open(release_path) as src:
                        released, crs, transform = src.read(), str(src.crs), src.transform
                    for index, keys in enumerate(BANDS):
                        asset_key = keys[0 if sensor == "landsat" else 1]
                        raw, mask, metadata = read_window(item["assets"][asset_key], crs, transform,
                                                          size, sensor == "sentinel")
                        expected = decode(raw, sensor)
                        valid = mask & (raw != 0)
                        mismatch = int(np.count_nonzero(expected != released[index]))
                        result = {"pair_id": row["pair_id"], "sensor": sensor, "asset": asset_key,
                                  "source_metadata": metadata, "raw_array_sha256": sha(raw.tobytes()),
                                  "release_file_sha256": sha(release_path.read_bytes()),
                                  "mismatch_pixels": mismatch, "raw_zero_pixels": int((raw == 0).sum()),
                                  "invalid_before_clipping_pixels": int((~valid).sum()),
                                  "valid_clipped_zero_pixels": int((valid & (expected == 0)).sum()),
                                  "examples": example_pixels(raw, valid, released[index], sensor)}
                        if sensor == "sentinel":
                            result["double_offset_mismatch_pixels"] = int(np.count_nonzero(
                                decode(raw, sensor, 2000) != released[index]))
                            if result["double_offset_mismatch_pixels"] == 0:
                                raise ValueError("Selected source cannot discriminate single/double offset")
                        summary["source_results"].append(result)
                        if mismatch:
                            raise ValueError("Source/release mismatch: stop gate and investigate; do not alter release")
            summary["source_status"] = "two_pair_four_band_reproduction_passed"
    except Exception as exc:
        summary["source_status"] = "blocked_or_failed"
        # Signed URLs can occur in transport exceptions; retain type only.
        summary["failure_type"] = type(exc).__name__
        raise RuntimeError("Source audit blocked or failed; inspect summary failure_type and completed numeric checks. Signed transport details withheld.") from None
    finally:
        (args.output / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        files = sorted(path for path in args.output.rglob("*") if path.is_file())
        (args.output / "checksums.sha256").write_text("".join(
            f"{sha(path.read_bytes())}  {path.relative_to(args.output).as_posix()}\n" for path in files), encoding="utf-8")
    print("Saved", args.output / "summary.json")


if __name__ == "__main__":
    with rasterio.Env(GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR", GDAL_HTTP_TIMEOUT="30",
                      GDAL_HTTP_MAX_RETRY="1", GDAL_HTTP_RETRY_DELAY="1"):
        main()
