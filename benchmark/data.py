"""Locked Mahimandal 160->480 RGB/RGB+NIR dataset reader."""
from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np
import rasterio
from rasterio.enums import ColorInterp
import torch
from torch.utils.data import Dataset


SCALE = 3
LR_SIZE = 160
HR_SIZE = 480
BAND_ORDER = ("red", "green", "blue", "nir")


def digest(path: str | Path) -> str:
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def resolve(root: str | Path, path: str | Path) -> Path:
    root = Path(root).resolve()
    candidate = Path(path)
    candidate = candidate if candidate.is_absolute() else root / candidate
    candidate = candidate.resolve()
    try:
        candidate.relative_to(root)
    except ValueError:
        raise ValueError(f"Path escapes dataset root: {path}")
    return candidate


def find_release_manifest(root: str | Path) -> Path:
    root = Path(root)
    candidates = (
        root / "paper_reports" / "paper_release_manifest.csv",
        root / "paper_release_manifest.csv",
    )
    found = [path for path in candidates if path.is_file()]
    if len(found) != 1:
        raise FileNotFoundError(f"Expected one paper_release_manifest.csv under {root}; found {found}")
    return found[0]


def read_rows(root: str | Path) -> list[dict]:
    root = Path(root)
    with find_release_manifest(root).open(newline="", encoding="utf-8-sig") as stream:
        rows = list(csv.DictReader(stream))
    required = {"pair_id", "split", "landscape_class", "spatial_block_id"}
    missing = required.difference(rows[0] if rows else {})
    if missing:
        raise ValueError(f"Release manifest lacks required columns: {sorted(missing)}")
    for row in rows:
        pair_id = row["pair_id"]
        row["lr_path"] = row.get("lr_path") or f"landsat/{pair_id}.tif"
        row["hr_path"] = row.get("hr_path") or f"sentinel/{pair_id}.tif"
    return rows


def _band_indices(dataset: rasterio.io.DatasetReader, modality: str) -> list[int]:
    names = [str(name or "").strip().casefold() for name in dataset.descriptions]
    aliases = {"red": {"red", "b4", "sr_b4"}, "green": {"green", "b3", "sr_b3"},
               "blue": {"blue", "b2", "sr_b2"}, "nir": {"nir", "b8", "sr_b5", "b5"}}
    requested = BAND_ORDER[:3] if modality == "rgb" else BAND_ORDER
    indices = []
    for band in requested:
        matches = [index + 1 for index, value in enumerate(names) if value in aliases[band]]
        if len(matches) != 1:
            raise ValueError(f"Cannot resolve one {band} band from descriptions {dataset.descriptions}")
        indices.append(matches[0])
    return indices


def _reflectance(array: np.ndarray, source_dtype: str, radiometry: str) -> np.ndarray:
    array = array.astype(np.float32, copy=False)
    is_integer = np.issubdtype(np.dtype(source_dtype), np.integer)
    if radiometry == "normalized_float32" and is_integer:
        raise ValueError("normalized_float32 requires floating-point TIFF storage")
    if is_integer:
        array = array / 10000.0
    elif np.nanmax(array) > 1.5:
        raise ValueError("Floating TIFF exceeds reflectance range; scaling is ambiguous")
    array = np.clip(array, 0.0, 1.0)
    if radiometry == "quantized8":
        array = np.rint(array * 255.0) / 255.0
    elif radiometry not in {"reflectance16", "normalized_float32"}:
        raise ValueError(f"Unknown radiometry protocol: {radiometry}")
    return array


class Pairs(Dataset):
    def __init__(self, root: str | Path, split: str, modality: str = "rgb",
                 radiometry: str = "reflectance16", augment: bool = False,
                 lr_crop: int | None = None):
        if modality not in {"rgb", "rgbnir"}:
            raise ValueError("modality must be rgb or rgbnir")
        if lr_crop is not None and not 1 <= lr_crop <= LR_SIZE:
            raise ValueError(f"lr_crop must be in [1,{LR_SIZE}]")
        self.root = Path(root)
        self.modality = modality
        self.radiometry = radiometry
        self.augment = augment
        self.lr_crop = lr_crop
        self.rows = sorted((row for row in read_rows(root) if row["split"] == split),
                           key=lambda row: row["pair_id"])
        if not self.rows:
            raise ValueError(f"Empty split {split!r}")

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int):
        row = self.rows[index]
        with rasterio.open(resolve(self.root, row["lr_path"])) as source:
            lr = _reflectance(source.read(_band_indices(source, self.modality)), source.dtypes[0], self.radiometry)
            lr_mask = np.all(source.read_masks() > 0, axis=0)
        with rasterio.open(resolve(self.root, row["hr_path"])) as source:
            hr = _reflectance(source.read(_band_indices(source, self.modality)), source.dtypes[0], self.radiometry)
            hr_mask = np.all(source.read_masks() > 0, axis=0)
        channels = 3 if self.modality == "rgb" else 4
        if lr.shape != (channels, LR_SIZE, LR_SIZE) or hr.shape != (channels, HR_SIZE, HR_SIZE):
            raise ValueError(f"Unexpected geometry for {row['pair_id']}: {lr.shape}, {hr.shape}")
        # TIFF validity on the target grid; the published all-valid masks retain the full frame.
        joint_mask = hr_mask & np.repeat(np.repeat(lr_mask, SCALE, axis=0), SCALE, axis=1)
        if not joint_mask.any() or not np.isfinite(lr).all() or not np.isfinite(hr).all():
            raise ValueError(f"No finite jointly valid pixels for {row['pair_id']}")
        if self.lr_crop and self.lr_crop < LR_SIZE:
            top = int(torch.randint(0, LR_SIZE - self.lr_crop + 1, ()).item())
            left = int(torch.randint(0, LR_SIZE - self.lr_crop + 1, ()).item())
            bottom, right = top + self.lr_crop, left + self.lr_crop
            lr = lr[:, top:bottom, left:right]
            hr = hr[:, top * SCALE:bottom * SCALE, left * SCALE:right * SCALE]
            joint_mask = joint_mask[top * SCALE:bottom * SCALE, left * SCALE:right * SCALE]
        arrays = [lr, hr, joint_mask[None].astype(np.float32)]
        if self.augment:
            rotations = int(torch.randint(4, ()).item())
            arrays = [np.rot90(array, rotations, axes=(-2, -1)) for array in arrays]
            if torch.rand(()) < 0.5:
                arrays = [array[..., ::-1] for array in arrays]
        return (*[torch.from_numpy(array.copy()) for array in arrays], index)


def audit(root: str | Path) -> dict:
    root = Path(root)
    rows = read_rows(root)
    counts = Counter(row["split"] for row in rows)
    if set(counts) != {"train", "val", "test"} or min(counts.values()) < 1:
        raise ValueError(f"Release requires nonempty train/val/test: {dict(counts)}")
    if len({row["pair_id"] for row in rows}) != len(rows):
        raise ValueError("Release manifest contains duplicate pair IDs")
    blocks: dict[str, set[str]] = {}
    classes: dict[str, Counter] = {split: Counter() for split in counts}
    regions: dict[str, Counter] = {split: Counter() for split in counts}
    source_dtypes: set[str] = set()
    for row in rows:
        blocks.setdefault(row["spatial_block_id"], set()).add(row["split"])
        classes[row["split"]][row.get("landscape_class", "unknown")] += 1
        regions[row["split"]][row.get("geographic_stratum", "unknown")] += 1
        for field, shape, resolution in (("lr_path", (LR_SIZE, LR_SIZE), 30.0),
                                         ("hr_path", (HR_SIZE, HR_SIZE), 10.0)):
            path = resolve(root, row[field])
            if not path.is_file():
                raise FileNotFoundError(path)
            with rasterio.open(path) as source:
                if (source.height, source.width) != shape or source.count != 4:
                    raise ValueError(f"Unexpected TIFF contract: {path}")
                if not np.allclose(source.res, (resolution, resolution), atol=1e-6):
                    raise ValueError(f"Unexpected resolution: {path}: {source.res}")
                source_dtypes.update(source.dtypes)
                _band_indices(source, "rgbnir")
    leaked = {block: sorted(splits) for block, splits in blocks.items() if len(splits) > 1}
    if leaked:
        raise ValueError(f"Spatial blocks cross release splits: {list(leaked.items())[:5]}")
    protocol_path = root / "protocol.json"
    protocol = json.loads(protocol_path.read_text(encoding="utf-8")) if protocol_path.is_file() else {}
    return {
        "pairs": len(rows), "counts": dict(counts),
        "realized_fractions": {key: value / len(rows) for key, value in counts.items()},
        "intended_split_fractions": protocol.get(
            "release_split_objective",
            protocol.get("split_fractions", {"train": 0.78, "val": 0.11, "test": 0.11}),
        ),
        "split_note": (
            "The release allocator objective is 78/11/11; complete 100-km "
            "blocks produce the locked realized counts."
        ),
        "geometry": {"lr": [4, LR_SIZE, LR_SIZE], "hr": [4, HR_SIZE, HR_SIZE], "scale": SCALE},
        "source_band_order": ["blue", "green", "red", "nir"],
        "model_band_order": list(BAND_ORDER),
        "source_dtypes": sorted(source_dtypes),
        "reflectance_decode": ("clip(source/10000,0,1)" if source_dtypes == {"int16"}
                               else "floating reflectance clipped to [0,1]"),
        "classes_by_split": {key: dict(value) for key, value in classes.items()},
        "geographic_strata_by_split": {key: dict(value) for key, value in regions.items()},
        "release_manifest": str(find_release_manifest(root)),
        "release_manifest_sha256": digest(find_release_manifest(root)),
        "protocol_sha256": digest(protocol_path) if protocol_path.is_file() else None,
        "spatial_blocks": len(blocks), "cross_split_spatial_blocks": 0,
    }


def save_prediction(path: str | Path, image: np.ndarray, mask: np.ndarray,
                    reference: str | Path, modality: str, tags: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    names = BAND_ORDER[:3] if modality == "rgb" else BAND_ORDER
    with rasterio.open(reference) as source:
        profile = source.profile.copy()
        metadata = source.tags()
    profile.update(count=len(names), dtype="float32", compress="deflate", predictor=3, nodata=None)
    temporary = path.with_suffix(".tmp.tif")
    with rasterio.Env(GDAL_TIFF_INTERNAL_MASK=True):
        with rasterio.open(temporary, "w", **profile) as output:
            output.write(image.astype(np.float32))
            output.write_mask(mask.squeeze().astype(np.uint8) * 255)
            metadata.update(sensor="model_prediction", band_order=",".join(names), **tags)
            output.update_tags(**{key: str(value) for key, value in metadata.items()})
            if modality == "rgb":
                output.colorinterp = (ColorInterp.red, ColorInterp.green, ColorInterp.blue)
            for index, name in enumerate(names, 1):
                output.set_band_description(index, name)
    temporary.replace(path)
