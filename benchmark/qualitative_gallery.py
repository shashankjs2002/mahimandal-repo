"""Deterministic, complete-model qualitative gallery from saved test predictions.

This reads existing evaluation GeoTIFFs. It never runs inference or changes
checkpoints, predictions, the dataset, or the locked evaluation protocol.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import zipfile
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import rasterio

from data import Pairs, digest, find_release_manifest


def evaluation_dirs(root, radiometry, modality, seed, split, border_crop):
    base = root / radiometry / modality
    found = {}
    for metrics in sorted(base.glob(f"*/seed_{seed}/evaluation/{split}/*/metrics.json")):
        directory = metrics.parent
        protocol_path = directory / "protocol.json"
        if not protocol_path.is_file():
            continue
        protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
        if (protocol.get("border_crop") != border_crop or
                protocol.get("radiometry") != radiometry or
                protocol.get("modality") != modality or
                protocol.get("split") != split):
            continue
        model = directory.relative_to(root).parts[2]
        if model in found:
            raise ValueError(f"Ambiguous {model} evaluation: {found[model]} and {directory}")
        found[model] = (directory, protocol)
    if "bicubic" not in found:
        raise FileNotFoundError(f"No matching bicubic evaluation in {base}")
    return dict(sorted(found.items(), key=lambda item: (item[0] != "bicubic", item[0].casefold())))


def _complete_prediction_paths(pair_id, evaluations):
    paths = {}
    for modality, models in evaluations.items():
        paths[modality] = {}
        for name, (directory, _) in models.items():
            path = directory / "predictions" / f"{pair_id}.tif"
            if not path.is_file():
                return None
            paths[modality][name] = path
    return paths


def select_cases(rows, evaluations, group_by, per_group, seed):
    eligible = []
    for index, row in enumerate(rows):
        paths = _complete_prediction_paths(row["pair_id"], evaluations)
        if paths is not None:
            eligible.append((index, row, paths))
    if not eligible:
        raise FileNotFoundError("No test pair has predictions from every evaluated model")
    groups = defaultdict(list)
    expected = defaultdict(int)
    for row in rows:
        for field in group_by:
            expected[(field, row.get(field, "unknown") or "unknown")] += 1
    for item in eligible:
        _, row, _ = item
        for field in group_by:
            groups[(field, row.get(field, "unknown") or "unknown")].append(item)
    selections = []
    for (field, label), total in sorted(expected.items()):
        candidates = groups.get((field, label), [])
        if len(candidates) < per_group:
            raise ValueError(
                f"{field}={label}: need {per_group} pairs with ALL model predictions; "
                f"found {len(candidates)} of {total}. The compact bundle cannot "
                "make three distinct examples per group; use full DGX output."
            )
        shuffled = sorted(candidates, key=lambda item: hashlib.sha256(
            f"{seed}:{item[1]['pair_id']}".encode("utf-8")
        ).digest())
        chosen = []
        for _ in range(per_group):
            used_blocks = {item[1].get("spatial_block_id") for item in chosen}
            used_states = {item[1].get("state_or_ut") for item in chosen}
            remaining = [item for item in shuffled if item not in chosen]
            best = max(remaining, key=lambda item: (
                item[1].get("spatial_block_id") not in used_blocks,
                item[1].get("state_or_ut") not in used_states,
            ))
            chosen.append(best)
        for rank, (index, row, paths) in enumerate(chosen, 1):
            selections.append((field, label, rank, index, row, paths))
    return selections, len(eligible), expected


def _rgb(array, display_max):
    return np.clip(array[:3].transpose(1, 2, 0) / display_max, 0.0, 1.0)


def render_case(dataset, index, predictions, output_path, title, display_max,
                error_max, dpi):
    lr, hr, valid_mask, _ = dataset[index]
    target = hr.numpy()
    valid = valid_mask.numpy()[0] > 0
    lr_up = np.repeat(np.repeat(lr.numpy(), 3, axis=1), 3, axis=2)
    panels = [("Landsat LR\n(nearest display)", lr_up),
              ("Sentinel-2\ntarget", target)]
    for name, path in predictions.items():
        with rasterio.open(path) as source:
            image = source.read()
        if image.shape != target.shape or not np.isfinite(image).all():
            raise ValueError(f"Invalid prediction shape/values: {path}: {image.shape}")
        panels.append((name.replace("_", "\n", 1), image))
    columns = len(panels)
    figure, axes = plt.subplots(3, columns, figsize=(2.55 * columns, 7.7), squeeze=False)
    height, width = target.shape[-2:]
    zoom = (slice(height // 3, 2 * height // 3),
            slice(width // 3, 2 * width // 3))
    for column, (name, image) in enumerate(panels):
        axes[0, column].imshow(_rgb(image, display_max), interpolation="nearest")
        axes[0, column].set_title(name, fontsize=8)
        axes[1, column].imshow(_rgb(image[:, zoom[0], zoom[1]], display_max),
                               interpolation="nearest")
        if column == 1:
            error = np.zeros((height, width), dtype=np.float32)
        else:
            error = np.mean(np.abs(image[:3] - target[:3]), axis=0)
        error = np.where(valid, error, np.nan)
        axes[2, column].imshow(error, cmap="magma", vmin=0, vmax=error_max,
                               interpolation="nearest")
        for row_index in range(3):
            axes[row_index, column].set_xticks([])
            axes[row_index, column].set_yticks([])
    for row_index, text in enumerate(("Full", "Center zoom", "RGB |error|")):
        axes[row_index, 0].set_ylabel(text, fontsize=9)
    figure.suptitle(title + f"\nShared RGB display [0,{display_max:g}]; "
                    f"shared error scale [0,{error_max:g}]. Neither affects metrics.",
                    fontsize=10)
    figure.tight_layout(rect=(0, 0, 1, 0.94))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=dpi, facecolor="white")
    plt.close(figure)


def _write_contact_sheet(items, destination, label, display_max, dpi):
    figure, axes = plt.subplots(2, len(items), figsize=(3.7 * len(items), 7.8),
                               squeeze=False)
    for column, (rank, image, row) in enumerate(items):
        lr, hr, _, _ = image
        axes[0, column].imshow(_rgb(hr.numpy(), display_max))
        axes[1, column].imshow(_rgb(np.repeat(np.repeat(lr.numpy(), 3, 1), 3, 2),
                                    display_max), interpolation="nearest")
        axes[0, column].set_title(f"Candidate {rank}: {row.get('state_or_ut', '')}\n"
                                  f"{row['pair_id']}", fontsize=8)
        for r in (0, 1):
            axes[r, column].axis("off")
    axes[0, 0].set_ylabel("Sentinel-2 HR")
    axes[1, 0].set_ylabel("Landsat LR")
    figure.suptitle(label + " | source pairs only; model panels in individual PNGs")
    figure.tight_layout(rect=(0, 0, 1, 0.96))
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, dpi=dpi, facecolor="white")
    plt.close(figure)


def build_gallery(output, dataset_root, destination, archive, group_by,
                  per_group=3, seed=42, split="test", radiometry="reflectance16",
                  modalities=("rgb", "rgbnir"), border_crop=3, display_max=0.3,
                  error_max=0.10, dpi=110, verify_manifest=True):
    output, dataset_root = Path(output).resolve(), Path(dataset_root).resolve()
    destination, archive = Path(destination).resolve(), Path(archive).resolve()
    if destination == output or output in destination.parents:
        raise ValueError("Gallery destination must be outside OUTPUT to avoid recursive bundling")
    if archive == output or output in archive.parents:
        raise ValueError("Gallery archive must be outside OUTPUT")
    if not output.is_dir() or not dataset_root.is_dir():
        raise FileNotFoundError("OUTPUT and DATASET_ROOT must both exist")
    manifest_sha = digest(find_release_manifest(dataset_root))
    evaluations = {modality: evaluation_dirs(output, radiometry, modality, seed,
                                              split, border_crop)
                   for modality in modalities}
    mismatches = []
    for modality, models in evaluations.items():
        for name, (_, protocol) in models.items():
            recorded = protocol.get("dataset", {}).get("release_manifest_sha256")
            if recorded != manifest_sha:
                mismatches.append((modality, name, recorded))
    if mismatches and verify_manifest:
        raise ValueError(f"Dataset manifest SHA mismatch; refusing mixed data: {mismatches[:3]}. "
                         f"Actual SHA256={manifest_sha}")
    datasets = {modality: Pairs(dataset_root, split, modality, radiometry)
                for modality in modalities}
    rows = datasets[modalities[0]].rows
    for dataset in datasets.values():
        if [row["pair_id"] for row in dataset.rows] != [row["pair_id"] for row in rows]:
            raise ValueError("Modality datasets do not have the same pair order")
    selections, eligible_count, expected = select_cases(
        rows, evaluations, group_by, per_group, seed)
    destination.mkdir(parents=True, exist_ok=True)
    inventory = []
    sheets = defaultdict(list)
    for field, label, rank, index, row, paths in selections:
        safe_label = "".join(c if c.isalnum() or c in "_-" else "_" for c in label)
        basename = f"{rank:02d}_{row['pair_id']}"
        title = (f"{split}/{index} | {row['pair_id']} | {row.get('landscape_class', '')} | "
                 f"{row.get('geographic_stratum', '')} | {row.get('state_or_ut', '')}")
        for modality in modalities:
            path = destination / field / safe_label / modality / f"{basename}.png"
            render_case(datasets[modality], index, paths[modality], path, title,
                        display_max, error_max, dpi)
            inventory.append({
                "group_by": field, "group": label, "candidate": rank,
                "modality": modality, "pair_id": row["pair_id"], "test_index": index,
                "landscape_class": row.get("landscape_class", ""),
                "geographic_stratum": row.get("geographic_stratum", ""),
                "state_or_ut": row.get("state_or_ut", ""),
                "spatial_block_id": row.get("spatial_block_id", ""),
                "center_lon": row.get("center_lon", ""),
                "center_lat": row.get("center_lat", ""),
                "model_count": len(paths[modality]),
                "models": ";".join(paths[modality]),
                "figure": path.relative_to(destination).as_posix(),
            })
        sheets[(field, label)].append((rank, index, row))
    for (field, label), items in sheets.items():
        safe_label = "".join(c if c.isalnum() or c in "_-" else "_" for c in label)
        source_items = [(rank, datasets[modalities[0]][index], row)
                        for rank, index, row in items]
        _write_contact_sheet(source_items, destination / field / safe_label / "contact_sheet.png",
                             f"{field}: {label}", display_max, dpi)
    with (destination / "selection.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(inventory[0]))
        writer.writeheader()
        writer.writerows(inventory)
    coverage = [{"group_by": field, "group": label, "test_pairs": count,
                 "complete_prediction_pairs": sum(
                     1 for item in rows if (item.get(field, "unknown") or "unknown") == label
                     and _complete_prediction_paths(item["pair_id"], evaluations) is not None)}
                for (field, label), count in sorted(expected.items())]
    with (destination / "coverage.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(coverage[0]))
        writer.writeheader()
        writer.writerows(coverage)
    metadata = {
        "dataset_manifest_sha256": manifest_sha,
        "manifest_verified": not mismatches,
        "split": split, "radiometry": radiometry, "seed": seed,
        "selection_seed": seed,
        "selection_method": "SHA256(seed:pair_id) order; prefer distinct spatial blocks and states",
        "group_by": list(group_by), "per_group": per_group,
        "eligible_pairs_with_all_model_predictions": eligible_count,
        "models": {modality: list(models) for modality, models in evaluations.items()},
        "border_crop_for_evaluation": border_crop,
        "display_max": display_max, "error_max": error_max,
        "note": "Figures use model predictions saved by locked evaluation; source display scaling does not affect metrics.",
    }
    (destination / "gallery_protocol.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    (destination / "README.md").write_text(
        "# Mahimandal qualitative candidates\n\n"
        "Open `contact_sheet.png` in each group, then inspect the three detailed PNGs "
        "under `rgb/` and `rgbnir/`. Each detailed panel includes every model with a "
        "matching saved prediction. The first row is the full scene, second a fixed "
        "center zoom, third RGB absolute error against Sentinel-2. All columns share "
        "the same display and error scales. Invalid pixels are transparent in error maps.\n\n"
        "Selection uses a fixed seed and favors distinct spatial blocks and states; it "
        "does not rank learned models by test results. Candidate choices for the paper "
        "must be documented. `selection.csv` records exact pair IDs and figure paths. "
        "No TIFF predictions or model checkpoints are copied into this gallery.\n",
        encoding="utf-8")
    archive.parent.mkdir(parents=True, exist_ok=True)
    if archive.exists():
        raise FileExistsError(f"Archive already exists; choose a new path: {archive}")
    with zipfile.ZipFile(archive, "x", zipfile.ZIP_DEFLATED, compresslevel=3) as bundle:
        hashes = []
        for path in sorted(destination.rglob("*")):
            if path.is_file():
                relative = path.relative_to(destination).as_posix()
                bundle.write(path, relative)
                hashes.append(f"{digest(path)}  {relative}")
        bundle.writestr("checksums.sha256", "\n".join(hashes) + "\n")
    return archive, metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--group-by", nargs="+", default=["landscape_class", "geographic_stratum"],
                        choices=["landscape_class", "geographic_stratum", "state_or_ut"])
    parser.add_argument("--per-group", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--split", choices=["val", "test"], default="test")
    parser.add_argument("--modalities", nargs="+", default=["rgb", "rgbnir"],
                        choices=["rgb", "rgbnir"])
    parser.add_argument("--dpi", type=int, default=110)
    args = parser.parse_args()
    if args.per_group < 1 or args.dpi < 50:
        parser.error("per-group must be positive and dpi must be at least 50")
    archive, metadata = build_gallery(
        args.output, args.dataset, args.destination, args.archive,
        args.group_by, per_group=args.per_group, seed=args.seed,
        split=args.split, modalities=args.modalities, dpi=args.dpi)
    print(json.dumps(metadata, indent=2, ensure_ascii=False))
    print("Gallery archive:", archive)


if __name__ == "__main__":
    main()
