"""CPU-only, held-out feature-correspondence diagnostic for frozen Mahimandal.

No image corrections are written and no model is executed. Automatically
matched landmarks are not independent surveyed ground control. Failures remain
in the denominators, and all held-out matches are scored without residual gating.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import rasterio


EXPECTED = "6bf97403ee40ccf0cbd38b2e6e4932e01973e5d00e211e0acfb1e0b810efc5e7"
PROTOCOL = {
    "seed": 42, "sample_count": 100,
    "sampling": "11 per landscape (7 train, 2 val, 2 test), plus one training cropland; prefer new blocks and source pairs",
    "feature_grid_m": 30, "hr_reduction": "non-overlapping 3x3 area mean",
    "detection": "SIFT on separately 1st/99th-percentile-stretched visible luminance; detection only",
    "sift": {"nfeatures": 2000, "contrastThreshold": 0.02, "edgeThreshold": 10, "sigma": 1.6},
    "matching": "mutual two-nearest-neighbour L2 ratio test at 0.75; remove repeated half-pixel source/target locations; no displacement gate",
    "split": "4x4 source-image cells; hash-shuffle occupied cells, hold out ceil(one third) of cells before model fitting",
    "minimum_training_matches": 8, "minimum_heldout_matches": 6,
    "transform": "estimateAffinePartial2D, train matches only, RANSAC threshold 1.5 LR pixels, 5000 iterations, confidence 0.999",
    "heldout_scoring": "all held-out descriptor matches, including outliers; no residual-based selection",
    "scope": "feature-match relative consistency, not manually validated landmarks, absolute geodetic accuracy or GCP RMSE",
}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rank(value: str) -> str:
    return hashlib.sha256(("42/" + value).encode()).hexdigest()


def choose_sample(manifest: pd.DataFrame) -> pd.DataFrame:
    classes = sorted(manifest.landscape_class.unique())
    if len(classes) != 9:
        raise ValueError("Expected nine release classes")
    chosen, used_blocks, used_sources = [], set(), set()
    for category in classes:
        for split, quota in (("train", 8 if category == "cropland" else 7), ("val", 2), ("test", 2)):
            candidates = manifest.loc[(manifest.landscape_class == category) & (manifest.split == split)].to_dict("records")
            for _ in range(quota):
                if not candidates:
                    raise ValueError(f"Cannot fill predefined quota: {category}/{split}")
                candidates.sort(key=lambda row: (
                    row["spatial_block_id"] in used_blocks,
                    (row["landsat_id"], row["sentinel_id"]) in used_sources,
                    rank(row["pair_id"])))
                row = candidates.pop(0)
                chosen.append(row)
                used_blocks.add(row["spatial_block_id"])
                used_sources.add((row["landsat_id"], row["sentinel_id"]))
    result = pd.DataFrame(chosen)
    if len(result) != 100 or not result.pair_id.is_unique:
        raise ValueError("Predefined sample is not 100 unique pairs")
    return result


def root_path(root: Path, relative: str) -> Path:
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError("Manifest path escapes release root")
    return path


def stretch(image: np.ndarray) -> np.ndarray:
    if not np.isfinite(image).all():
        raise ValueError("Nonfinite input image")
    low, high = np.percentile(image, (1, 99))
    if high - low < 1e-6:
        raise ValueError("Insufficient intensity range")
    return np.round(255 * np.clip((image - low) / (high - low), 0, 1)).astype("uint8")


def cell_holdout(points: np.ndarray, pair_id: str) -> np.ndarray:
    cells = np.minimum((points // 40).astype(int), 3)
    labels = [f"{x}_{y}" for x, y in cells]
    occupied = sorted(set(labels), key=lambda cell: rank(pair_id + "/" + cell))
    heldout = set(occupied[:int(np.ceil(len(occupied) / 3))])
    return np.array([label in heldout for label in labels])


def mutual_matches(first, second) -> list:
    matcher = cv2.BFMatcher(cv2.NORM_L2)

    def accepted(a, b):
        return {pair[0].queryIdx: pair[0] for pair in matcher.knnMatch(a, b, k=2)
                if len(pair) == 2 and pair[0].distance < 0.75 * pair[1].distance}

    forward, reverse = accepted(first, second), accepted(second, first)
    return sorted([match for index, match in forward.items()
                   if match.trainIdx in reverse and reverse[match.trainIdx].trainIdx == index],
                  key=lambda match: (match.distance, match.queryIdx, match.trainIdx))


def unique_locations(matches, first, second) -> list:
    used_first, used_second, output = set(), set(), []
    for match in matches:
        a = tuple(np.round(np.array(first[match.queryIdx].pt) * 2).astype(int))
        b = tuple(np.round(np.array(second[match.trainIdx].pt) * 2).astype(int))
        if a not in used_first and b not in used_second:
            output.append(match)
            used_first.add(a)
            used_second.add(b)
    return output


def errors(source: np.ndarray, target: np.ndarray, matrix=None) -> np.ndarray:
    transformed = source if matrix is None else source @ matrix[:, :2].T + matrix[:, 2]
    return np.linalg.norm(transformed - target, axis=1)


def panel(lr, hr, source, target, holdout, path: Path, title: str) -> None:
    figure, axes = plt.subplots(1, 3, figsize=(13, 4), gridspec_kw={"width_ratios": [1, 1, 2]})
    axes[0].imshow(np.clip(lr[[2, 1, 0]].transpose(1, 2, 0) / .3, 0, 1))
    axes[0].set_title("Landsat RGB, 160x160")
    axes[1].imshow(np.clip(hr[[2, 1, 0]].transpose(1, 2, 0) / .3, 0, 1))
    axes[1].set_title("Sentinel RGB, area-reduced to 160")
    left = stretch(.114 * lr[0] + .587 * lr[1] + .299 * lr[2])
    right = stretch(.114 * hr[0] + .587 * hr[1] + .299 * hr[2])
    axes[2].imshow(np.concatenate([left, right], axis=1), cmap="gray", vmin=0, vmax=255)
    for a, b, is_test in zip(source, target, holdout):
        color = "tab:orange" if is_test else "tab:cyan"
        axes[2].plot([a[0], b[0] + 160], [a[1], b[1]], color=color, linewidth=.6, alpha=.8)
        axes[2].scatter([a[0], b[0] + 160], [a[1], b[1]], color=color, s=5)
    axes[2].set_title("All descriptor matches: cyan fit / orange held out")
    for axis in axes:
        axis.set_axis_off()
    figure.suptitle(title, fontsize=8)
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)


def audit_pair(row: dict, root: Path, output: Path) -> tuple[dict, list]:
    identity = {key: row[key] for key in ("pair_id", "split", "landscape_class", "spatial_block_id", "landsat_id", "sentinel_id")}
    key = rank(row["pair_id"])[:16]
    result = {**identity, "status": "unresolved", "review_panel": f"panels/{key}.png"}
    points = []
    try:
        lr_path, hr_path = root_path(root, row["lr_path"]), root_path(root, row["hr_path"])
        with rasterio.open(lr_path) as a, rasterio.open(hr_path) as b:
            if a.shape != (160, 160) or b.shape != (480, 480) or a.count != 4 or b.count != 4:
                raise ValueError("Unexpected full-frame dimensions or channels")
            if a.crs != b.crs or not np.allclose(a.bounds, b.bounds, atol=.01, rtol=0):
                raise ValueError("LR/HR georeferencing mismatch")
            if not np.allclose(a.res, (30, 30), atol=1e-6):
                raise ValueError("Expected 30 m LR grid")
            lr, hr = a.read().astype("float32") / 10000, b.read().astype("float32") / 10000
            transform = a.transform
        result.update(lr_sha256=sha(lr_path), hr_sha256=sha(hr_path))
        reduced = hr.reshape(4, 160, 3, 160, 3).mean(axis=(2, 4))
        first = stretch(.114 * lr[0] + .587 * lr[1] + .299 * lr[2])
        second = stretch(.114 * reduced[0] + .587 * reduced[1] + .299 * reduced[2])
        detector = cv2.SIFT_create(**PROTOCOL["sift"])
        ka, da = detector.detectAndCompute(first, None)
        kb, db = detector.detectAndCompute(second, None)
        if da is None or db is None or len(da) < 2 or len(db) < 2:
            raise ValueError("Insufficient descriptors")
        matches = unique_locations(mutual_matches(da, db), ka, kb)
        source = np.array([ka[m.queryIdx].pt for m in matches], dtype="float64").reshape(-1, 2)
        target = np.array([kb[m.trainIdx].pt for m in matches], dtype="float64").reshape(-1, 2)
        holdout = cell_holdout(source, row["pair_id"])
        result.update(descriptor_matches=len(matches), fit_matches=int((~holdout).sum()), heldout_matches=int(holdout.sum()))
        panel(lr, reduced, source, target, holdout, output / result["review_panel"],
              row["pair_id"] + " | " + row["landscape_class"])
        matrix = None
        if (~holdout).sum() >= 8 and holdout.sum() >= 6:
            cv2.setRNGSeed(int(key[:7], 16))
            matrix, inliers = cv2.estimateAffinePartial2D(source[~holdout], target[~holdout],
                method=cv2.RANSAC, ransacReprojThreshold=1.5, maxIters=5000, confidence=.999, refineIters=10)
            if matrix is None or not np.isfinite(matrix).all():
                raise ValueError("Training-only robust transform could not be estimated")
            result.update(fit_inliers=int(inliers.sum()), fit_inlier_fraction=float(inliers.mean()),
                          transform_json=json.dumps(matrix.tolist()))
            if inliers.sum() >= 6 and inliers.mean() >= .5:
                before, after = errors(source[holdout], target[holdout]), errors(source[holdout], target[holdout], matrix)
                for name, values in (("identity", before), ("fitted", after)):
                    result.update({f"heldout_{name}_rmse_lr_px": float(np.sqrt(np.mean(values ** 2))),
                                   f"heldout_{name}_median_lr_px": float(np.median(values)),
                                   f"heldout_{name}_p95_lr_px": float(np.percentile(values, 95))})
                result.update(status="automated_fit_evaluable",
                    center_model_displacement_lr_px=float(errors(np.array([[79.5, 79.5]]), np.array([[79.5, 79.5]]), matrix)[0]),
                    scale=float(np.hypot(matrix[0, 0], matrix[1, 0])),
                    rotation_degrees=float(np.degrees(np.arctan2(matrix[1, 0], matrix[0, 0]))))
            else:
                result.update(status="insufficient_training_consensus")
        else:
            result.update(status="insufficient_spatially_heldout_matches")
        residual = errors(source, target, matrix) if matrix is not None else np.full(len(source), np.nan)
        for index, (a, b, test, distance) in enumerate(zip(source, target, holdout, residual)):
            ax, ay = transform * (a[0] + .5, a[1] + .5)
            bx, by = transform * (b[0] + .5, b[1] + .5)
            points.append({**identity, "match_index": index, "partition": "heldout" if test else "fit",
                "lr_x": a[0], "lr_y": a[1], "hr_area_reduced_x": b[0], "hr_area_reduced_y": b[1],
                "hr_native_x": 3 * b[0] + 1, "hr_native_y": 3 * b[1] + 1,
                "lr_world_x": ax, "lr_world_y": ay, "hr_world_x": bx, "hr_world_y": by,
                "identity_distance_lr_px": float(np.linalg.norm(a - b)),
                "fitted_residual_lr_px": float(distance), "human_match_valid": "", "reviewer": "", "remarks": ""})
    except Exception as exc:
        result.update(error_type=type(exc).__name__, error=str(exc))
    return result, points


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    manifest_path = args.dataset / "paper_release_manifest.csv"
    if sha(manifest_path) != EXPECTED:
        raise ValueError("Not the frozen release manifest")
    if args.output.resolve().is_relative_to(args.dataset.resolve()):
        raise ValueError("Never write inside the frozen dataset")
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError("Choose a new output directory; do not overwrite diagnostic evidence")
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "panels").mkdir()
    cv2.setNumThreads(1)
    cv2.ocl.setUseOpenCL(False)
    manifest = pd.read_csv(manifest_path, keep_default_na=False)
    sample = choose_sample(manifest)
    sample.to_csv(args.output / "sample_manifest.csv", index=False)
    protocol = {**PROTOCOL, "manifest_sha256": EXPECTED, "script_sha256": sha(Path(__file__)),
                "versions": {"opencv": cv2.__version__, "numpy": np.__version__, "rasterio": rasterio.__version__}}
    (args.output / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n", encoding="utf-8")
    results, matches = [], []
    for row in sample.to_dict("records"):
        result, points = audit_pair(row, args.dataset, args.output)
        results.append(result)
        matches.extend(points)
        if len(results) % 10 == 0:
            print(f"Pairs: {len(results)}/100; {dict(Counter(item['status'] for item in results))}", flush=True)
    frame = pd.DataFrame(results)
    frame.to_csv(args.output / "per_pair.csv", index=False)
    pd.DataFrame(matches).to_csv(args.output / "correspondences_for_manual_review.csv", index=False)
    checklist = frame[["pair_id", "landscape_class", "split", "status", "review_panel"]].copy()
    checklist["human_reviewed"], checklist["reviewer"], checklist["decision"], checklist["remarks"] = False, "", "", ""
    checklist.to_csv(args.output / "manual_review_checklist.csv", index=False)
    evaluable = frame.loc[frame.status == "automated_fit_evaluable"]
    summaries = []
    for category, group in frame.groupby("landscape_class"):
        good = group.loc[group.status == "automated_fit_evaluable"]
        summaries.append({"landscape_class": category, "attempted": len(group), "evaluable": len(good),
            "heldout_identity_rmse_lr_px_mean": float(good.heldout_identity_rmse_lr_px.mean()) if len(good) else None,
            "heldout_fitted_rmse_lr_px_mean": float(good.heldout_fitted_rmse_lr_px.mean()) if len(good) else None})
    pd.DataFrame(summaries).to_csv(args.output / "by_landscape.csv", index=False)
    columns = [c for c in evaluable if c.startswith("heldout_") and c.endswith("_lr_px")]
    means = {c: float(evaluable[c].mean()) for c in columns}
    summary = {"attempted_pairs": 100, "status_counts": dict(Counter(frame.status)),
        "evaluable_pairs": len(evaluable), "human_reviewed_pairs": 0,
        "sample_blocks": int(sample.spatial_block_id.nunique()),
        "sample_source_pairs": len(set(zip(sample.landsat_id, sample.sentinel_id))),
        "means_over_evaluable_pairs": means,
        "conversion": {"lr_px_to_sentinel_px": 3, "lr_px_to_m": 30},
        "interpretation": PROTOCOL["scope"],
        "failure_note": "Unevaluable cases are retained; evaluable-only errors must not be generalized to all 100 pairs or the release.",
        "manual_review": "CSV and panels provided; no automatic match is asserted to be a human-validated ground control point."}
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    inventory = [f"{sha(path)}  {path.relative_to(args.output).as_posix()}" for path in sorted(args.output.rglob("*")) if path.is_file()]
    (args.output / "checksums.sha256").write_text("\n".join(inventory) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
