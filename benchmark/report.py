"""Paper tables, paired inference, qualitative panels and evidence bundles."""
from __future__ import annotations

import json
from pathlib import Path
import zipfile
from datetime import datetime, timezone

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio

from data import Pairs, digest


IDENTITY = ["radiometry", "modality", "model", "seed"]
HIGHER = {"psnr", "ssim", "rgb_psnr", "rgb_ssim", "y_psnr", "y_ssim",
          "edge_f1", "gradient_correlation", "scc", "uiqi", "ndvi_correlation"}
LOWER = {"mae", "rmse", "ergas", "sam_degrees", "lpips", "ndvi_mae", "ndvi_rmse"}


def _identity(path: Path, root: Path) -> dict:
    relative = path.relative_to(root).parts
    seed_component = relative[3]
    if not seed_component.startswith("seed_"):
        raise ValueError(f"Invalid run seed directory: {seed_component}")
    return {"radiometry": relative[0], "modality": relative[1], "model": relative[2],
            "seed": int(seed_component[len("seed_"):]), "split": relative[5],
            "evaluation_id": relative[6]}


def _bootstrap(values: np.ndarray, clusters: np.ndarray | None = None,
               repeats: int = 2000, seed: int = 42) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    estimates = []
    if clusters is None:
        for _ in range(repeats):
            estimates.append(values[rng.integers(0, len(values), len(values))].mean())
    else:
        groups = np.unique(clusters)
        index = {group: np.flatnonzero(clusters == group) for group in groups}
        for _ in range(repeats):
            selected = rng.choice(groups, size=len(groups), replace=True)
            estimates.append(np.concatenate([values[index[group]] for group in selected]).mean())
    return tuple(np.percentile(estimates, (2.5, 97.5)).tolist())


def collect(root: str | Path, split: str = "test") -> tuple[pd.DataFrame, pd.DataFrame]:
    root = Path(root)
    summaries, records = [], []
    for metrics_path in root.glob(f"*/*/*/seed_*/evaluation/{split}/*/metrics.json"):
        identity = _identity(metrics_path, root)
        summaries.append({**identity, **json.loads(metrics_path.read_text())})
        per_image = metrics_path.with_name("per_image.csv")
        if per_image.is_file():
            frame = pd.read_csv(per_image)
            for key, value in identity.items():
                frame[key] = value
            frame["evaluation_root"] = str(metrics_path.parent)
            records.append(frame)
    if not summaries:
        raise FileNotFoundError(f"No {split} metrics below {root}")
    return pd.DataFrame(summaries), pd.concat(records, ignore_index=True)


def paired_tables(per_image: pd.DataFrame) -> pd.DataFrame:
    rows = []
    metrics = sorted((HIGHER | LOWER).intersection(per_image.columns))
    for key, frame in per_image.groupby(["radiometry", "modality", "seed"], dropna=False):
        baseline = frame[frame.model == "bicubic"].set_index("pair_id")
        if baseline.empty:
            continue
        for model, candidate in frame[frame.model != "bicubic"].groupby("model"):
            candidate = candidate.set_index("pair_id")
            common = baseline.index.intersection(candidate.index)
            if not len(common):
                continue
            clusters = candidate.loc[common, "spatial_block_id"].astype(str).to_numpy()
            for metric in metrics:
                direction = 1 if metric in HIGHER else -1
                improvement = direction * (candidate.loc[common, metric].to_numpy(dtype=float) -
                                            baseline.loc[common, metric].to_numpy(dtype=float))
                patch_low, patch_high = _bootstrap(improvement)
                cluster_low, cluster_high = _bootstrap(improvement, clusters)
                rows.append({"radiometry": key[0], "modality": key[1], "seed": key[2],
                             "model": model, "metric": metric,
                             "direction": "higher" if direction == 1 else "lower",
                             "mean_improvement_vs_bicubic": improvement.mean(),
                             "patch_ci95_low": patch_low, "patch_ci95_high": patch_high,
                             "spatial_block_ci95_low": cluster_low,
                             "spatial_block_ci95_high": cluster_high,
                             "fraction_better": float((improvement > 0).mean()),
                             "count": len(improvement), "spatial_blocks": len(np.unique(clusters))})
    return pd.DataFrame(rows)


def grouped_tables(per_image: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    metrics = [name for name in ("psnr", "ssim", "mae", "sam_degrees", "lpips",
                                  "edge_f1", "ndvi_rmse", "ndvi_correlation")
               if name in per_image]
    keys = ["radiometry", "modality", "model", "seed"]
    by_class = per_image.groupby(keys + ["landscape_class"], dropna=False)[metrics].agg(["mean", "std", "count"])
    by_class.columns = ["_".join(column) for column in by_class.columns]
    by_region = per_image.groupby(keys + ["geographic_stratum"], dropna=False)[metrics].agg(["mean", "std", "count"])
    by_region.columns = ["_".join(column) for column in by_region.columns]
    return by_class.reset_index(), by_region.reset_index()


def ablation_tables(summary: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rgb_rows, quantization_rows = [], []
    for (model, seed), frame in summary.groupby(["model", "seed"]):
        reference = frame[frame.radiometry == "reflectance16"]
        rgb = reference[reference.modality == "rgb"]
        rgbnir = reference[reference.modality == "rgbnir"]
        if len(rgb) == 1 and len(rgbnir) == 1:
            rgb_rows.append({"model": model, "seed": seed,
                             "rgb_only_rgb_psnr": float(rgb.iloc[0].rgb_psnr),
                             "rgbnir_rgb_psnr": float(rgbnir.iloc[0].rgb_psnr),
                             "rgb_psnr_delta_with_nir": float(rgbnir.iloc[0].rgb_psnr - rgb.iloc[0].rgb_psnr),
                             "rgb_only_rgb_ssim": float(rgb.iloc[0].rgb_ssim),
                             "rgbnir_rgb_ssim": float(rgbnir.iloc[0].rgb_ssim),
                             "rgb_ssim_delta_with_nir": float(rgbnir.iloc[0].rgb_ssim - rgb.iloc[0].rgb_ssim),
                             "ndvi_rmse": float(rgbnir.iloc[0].get("ndvi_rmse", np.nan))})
        for modality, subset in frame.groupby("modality"):
            full = subset[subset.radiometry == "reflectance16"]
            quantized = subset[subset.radiometry == "quantized8"]
            if len(full) == 1 and len(quantized) == 1:
                quantization_rows.append({"model": model, "seed": seed, "modality": modality,
                                          "reflectance16_psnr": float(full.iloc[0].psnr),
                                          "quantized8_psnr": float(quantized.iloc[0].psnr),
                                          "quantized_minus_reflectance_psnr": float(quantized.iloc[0].psnr - full.iloc[0].psnr),
                                          "reflectance16_sam": float(full.iloc[0].sam_degrees),
                                          "quantized8_sam": float(quantized.iloc[0].sam_degrees)})
    return pd.DataFrame(rgb_rows), pd.DataFrame(quantization_rows)


def _plot_summary(summary: pd.DataFrame, destination: Path) -> None:
    subset = summary[(summary.radiometry == "reflectance16") & (summary.modality == "rgb")]
    if subset.empty:
        return
    subset = subset.sort_values("psnr", ascending=False)
    figure, axes = plt.subplots(2, 2, figsize=(15, 10))
    for axis, metric, title in zip(axes.flat, ("psnr", "ssim", "lpips", "inference_mean_ms"),
                                   ("PSNR (dB)", "SSIM", "LPIPS", "Inference latency (ms)")):
        if metric not in subset:
            axis.axis("off")
            continue
        axis.barh(subset.model, subset[metric], color="#315b7d")
        axis.invert_yaxis()
        axis.set_title(title)
        axis.grid(axis="x", alpha=0.25)
    figure.suptitle("Mahimandal 160→480 test benchmark: reflectance16 RGB")
    figure.tight_layout()
    figure.savefig(destination, dpi=180)
    plt.close(figure)


def summarize(root: str | Path, split: str = "test") -> pd.DataFrame:
    root = Path(root)
    summary, per_image = collect(root, split)
    report = root / "reports" / split
    report.mkdir(parents=True, exist_ok=True)
    summary.sort_values(["radiometry", "modality", "psnr"], ascending=[True, True, False]).to_csv(
        report / "all_runs.csv", index=False)
    per_image.to_csv(report / "all_per_image.csv", index=False)
    paired_tables(per_image).to_csv(report / "paired_vs_bicubic.csv", index=False)
    by_class, by_region = grouped_tables(per_image)
    by_class.to_csv(report / "by_landscape_class.csv", index=False)
    by_region.to_csv(report / "by_geographic_stratum.csv", index=False)
    nir, quantization = ablation_tables(summary)
    nir.to_csv(report / "rgb_vs_rgbnir.csv", index=False)
    quantization.to_csv(report / "reflectance16_vs_quantized8.csv", index=False)
    complexity_columns = [column for column in ("model", "modality", "parameters", "flops_160",
                                                 "inference_mean_ms", "inference_p95_ms",
                                                 "images_per_second", "peak_allocated_mb")
                          if column in summary]
    summary[complexity_columns].drop_duplicates().to_csv(report / "complexity.csv", index=False)
    for (radiometry, modality), frame in summary.groupby(["radiometry", "modality"]):
        ordered = frame.sort_values("psnr", ascending=False)
        ordered.to_csv(
            report / f"main_{radiometry}_{modality}.csv", index=False)
        paper_columns = [column for column in
                         ("model", "count", "psnr", "ssim", "mae", "rmse", "edge_f1",
                          "ergas", "sam_degrees", "uiqi", "scc", "lpips", "ndvi_rmse",
                          "ndvi_correlation", "parameters", "flops_160",
                          "inference_mean_ms", "peak_allocated_mb") if column in ordered]
        ordered[paper_columns].to_latex(report / f"main_{radiometry}_{modality}.tex",
                                        index=False, float_format="%.4f")
    _plot_summary(summary, report / "main_metrics.png")
    inventory = {"split": split, "runs": len(summary), "per_image_rows": len(per_image),
                 "models": sorted(summary.model.unique().tolist()),
                 "modalities": sorted(summary.modality.unique().tolist()),
                 "radiometry": sorted(summary.radiometry.unique().tolist()),
                 "warning": "Do not select models or hyperparameters using test results."}
    (report / "report_inventory.json").write_text(json.dumps(inventory, indent=2), encoding="utf-8")
    return summary


def _rgb(array: np.ndarray, display_max: float = 0.3) -> np.ndarray:
    return np.clip(array[:3].transpose(1, 2, 0) / display_max, 0, 1)


def visualize(root: str | Path, dataset_root: str | Path, index: int = 0,
              radiometry: str = "reflectance16", modality: str = "rgb",
              seed: int = 42, split: str = "test", models: list[str] | None = None,
              display_max: float = 0.3) -> Path:
    root, dataset_root = Path(root), Path(dataset_root)
    dataset = Pairs(dataset_root, split, modality, radiometry)
    lr, hr, _, _ = dataset[index]
    record = dataset.rows[index]
    predictions = []
    for metrics_path in root.glob(f"{radiometry}/{modality}/*/seed_{seed}/evaluation/{split}/*/metrics.json"):
        model = metrics_path.relative_to(root).parts[2]
        if models is not None and model not in models:
            continue
        path = metrics_path.parent / "predictions" / f"{record['pair_id']}.tif"
        if path.is_file():
            with rasterio.open(path) as source:
                predictions.append((model, source.read()))
    lr_up = np.repeat(np.repeat(lr.numpy(), 3, axis=1), 3, axis=2)
    panels = [("Landsat LR (nearest display)", lr_up),
              ("Sentinel-2 target", hr.numpy())] + sorted(predictions)
    target = hr.numpy()
    columns = len(panels)
    figure, axes = plt.subplots(3, columns, figsize=(3.25 * columns, 9.5), squeeze=False)
    height, width = target.shape[-2:]
    zoom = (slice(height // 3, 2 * height // 3), slice(width // 3, 2 * width // 3))
    for column, (title, image) in enumerate(panels):
        axes[0, column].imshow(_rgb(image, display_max), interpolation="nearest")
        axes[0, column].set_title(title, fontsize=9)
        axes[1, column].imshow(_rgb(image[:, zoom[0], zoom[1]], display_max), interpolation="nearest")
        error = np.mean(np.abs(image[:3] - target[:3]), axis=0)
        axes[2, column].imshow(error, cmap="magma", vmin=0, vmax=0.10, interpolation="nearest")
        for row in range(3):
            axes[row, column].axis("off")
    axes[0, 0].set_ylabel("Full frame")
    axes[1, 0].set_ylabel("Center zoom")
    axes[2, 0].set_ylabel("RGB absolute error")
    figure.suptitle(f"{split}/{index} | {record['pair_id']} | {record.get('landscape_class', '')}\n"
                    f"Shared display stretch [0,{display_max}]; never used for metrics")
    figure.tight_layout()
    destination = root / "reports" / split / "qualitative"
    destination.mkdir(parents=True, exist_ok=True)
    path = destination / f"{radiometry}_{modality}_{index}.png"
    figure.savefig(path, dpi=180)
    plt.close(figure)
    return path


def bundle(root: str | Path, include_predictions: bool = False) -> Path:
    root = Path(root)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    suffix = "full_predictions" if include_predictions else "paper_results"
    destination = root.parent / f"{root.name}_{suffix}_{stamp}.zip"
    allowed = {".json", ".csv", ".tex", ".txt", ".png", ".pdf"}
    if include_predictions:
        allowed.add(".tif")
    with zipfile.ZipFile(destination, "x", zipfile.ZIP_DEFLATED, compresslevel=3) as archive:
        inventory = []
        for path in root.rglob("*"):
            if not path.is_file() or path.suffix not in allowed:
                continue
            if not include_predictions and "predictions" in path.parts:
                continue
            relative = path.relative_to(root).as_posix()
            archive.write(path, relative)
            inventory.append(f"{digest(path)}  {relative}")
        archive.writestr("checksums.sha256", "\n".join(inventory) + "\n")
    return destination


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--split", choices=("val", "test"), default="test")
    parser.add_argument("--bundle", action="store_true")
    args = parser.parse_args()
    print(summarize(args.root, args.split).to_string(index=False))
    if args.bundle:
        print(bundle(args.root))
