"""Train and evaluate pinned x3 SR architectures on the Mahimandal 160 benchmark."""
from __future__ import annotations

import argparse
import csv
import importlib.metadata
import json
import os
from pathlib import Path
import random
import sys
import time
import traceback

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

from adapters import REGISTRY, fetch, load, load_discriminator, predict
from data import LR_SIZE, Pairs, audit, digest, find_release_manifest, resolve, save_prediction
from metrics import mae, metric_bundle, psnr


def write_json(path, value) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def write_csv(path, rows) -> None:
    if not rows:
        return
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def save_checkpoint(path, state) -> None:
    temporary = Path(path).with_suffix(".tmp.pt")
    torch.save(state, temporary)
    temporary.replace(path)


def canonical(value):
    return json.loads(json.dumps(value, allow_nan=False))


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def environment(device: torch.device) -> dict:
    packages = {}
    for name in ("numpy", "rasterio", "pandas", "matplotlib", "scipy", "lpips", "fvcore"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    return {"python": sys.version, "torch": torch.__version__, "cuda": torch.version.cuda,
            "device": str(device),
            "hardware": torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU",
            "precision": "float32", "packages": packages}


def code_inventory() -> dict:
    root = Path(__file__).resolve().parent
    return {name: digest(root / name) for name in
            ("run.py", "adapters.py", "data.py", "metrics.py", "registry.json")}


def run_root(args) -> Path:
    return args.output / args.radiometry / args.modality / args.model / f"seed_{args.seed}"


def signature(args, spec: dict) -> dict:
    release = find_release_manifest(args.dataset)
    value = {
        "model": spec, "code_sha256": code_inventory(),
        "release_manifest_sha256": digest(release), "protocol_sha256": digest(args.dataset / "protocol.json"),
        "task": "actual Landsat 30 m 160x160 to Sentinel-2 10 m 480x480 paired SR",
        "scale": 3, "modality": args.modality, "radiometry": args.radiometry,
        "radiometry_definition": {
            "reflectance16": "clip(int16/10000,0,1)",
            "normalized_float32": "clip(float32,0,1)",
            "quantized8": "round(clip(int16/10000,0,1)*255)/255",
        }[args.radiometry],
        "training": {"seed": args.seed, "batch_size": args.batch_size,
                     "learning_rate": args.lr, "train_lr_crop": args.train_lr_crop,
                     "loss": ("masked L1" if args.model != "RealESRGAN_SRVGG_X3"
                              else "masked L1 + RGB LPIPS + RGB logistic GAN after warmup"),
                     "optimizer": "Adam", "augmentation": "aligned D4",
                     "sampling": "one random aligned crop per release training pair per epoch",
                     "model_selection": "highest full-frame validation PSNR"},
    }
    return canonical(value)


def strict_load(model: nn.Module, checkpoint: Path):
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    weights = state.get("model", state.get("state_dict", state)) if isinstance(state, dict) else state
    if isinstance(weights, nn.Module):
        weights = weights.state_dict()
    nn.Module.load_state_dict(model, weights, strict=True)
    return state


def lpips_network(device: torch.device):
    import lpips
    network = lpips.LPIPS(net="alex").to(device).eval()
    for parameter in network.parameters():
        parameter.requires_grad_(False)
    return network


@torch.no_grad()
def validation(model, spec, dataset, device) -> tuple[float, float]:
    model.eval()
    scores = []
    for lr, hr, mask, _ in DataLoader(dataset, batch_size=1, num_workers=0):
        sr = predict(model, spec, lr.to(device)).float().clamp(0, 1).cpu()
        scores.append((float(psnr(sr, hr, mask)), float(mae(sr, hr, mask))))
    return tuple(np.mean(scores, axis=0).tolist())


def trailing_nonimprovements(history: list[dict]) -> int:
    best, stale = -float("inf"), 0
    for row in history:
        value = row["val_psnr"]
        if value > best:
            best, stale = value, 0
        else:
            stale += 1
    return stale


def train(args) -> None:
    if args.model == "bicubic":
        raise ValueError("Bicubic has no training stage")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device(args.device)
    model, spec = load(args.model, args.sources, args.modality)
    root = run_root(args)
    root.mkdir(parents=True, exist_ok=True)
    receipt = signature(args, spec)
    protocol_path = root / "protocol.json"
    if protocol_path.exists() and json.loads(protocol_path.read_text()) != receipt:
        raise ValueError("Existing run uses different model/data/training settings; choose a new output root")
    audit_path = args.output / "dataset_audit.json"
    if not audit_path.exists():
        write_json(audit_path, audit(args.dataset))
    previous_audit = json.loads(audit_path.read_text())
    if previous_audit["release_manifest_sha256"] != receipt["release_manifest_sha256"]:
        raise ValueError("Dataset release manifest differs from the locked audit")
    write_json(protocol_path, receipt)
    (root / "architecture.txt").write_text(str(model), encoding="utf-8")
    write_json(root / "environment.json", environment(device))

    model = model.to(device)
    optimizer = torch.optim.Adam((parameter for parameter in model.parameters() if parameter.requires_grad),
                                 lr=args.lr, betas=(0.9, 0.999))
    discriminator = discriminator_optimizer = perceptual = None
    if args.model == "RealESRGAN_SRVGG_X3":
        discriminator = load_discriminator(args.sources).to(device)
        discriminator_optimizer = torch.optim.Adam(discriminator.parameters(), lr=args.lr,
                                                   betas=(0.9, 0.999))
        perceptual = lpips_network(device)
    start_epoch, best_psnr, history = 0, -float("inf"), []
    last_path = root / "last.pt"
    if last_path.exists():
        state = strict_load(model, last_path)
        if state["protocol"] != receipt:
            raise ValueError("Checkpoint lineage differs from the requested run")
        optimizer.load_state_dict(state["optimizer"])
        start_epoch, best_psnr, history = state["epoch"], state["best_psnr"], state["history"]
        if discriminator is not None:
            discriminator.load_state_dict(state["discriminator"], strict=True)
            discriminator_optimizer.load_state_dict(state["discriminator_optimizer"])
        torch.set_rng_state(state["torch_rng"])
        np.random.set_state(state["numpy_rng"])
        random.setstate(state["python_rng"])
        if device.type == "cuda" and state.get("cuda_rng") is not None:
            torch.cuda.set_rng_state_all(state["cuda_rng"])
    stale = trailing_nonimprovements(history)
    if stale >= args.patience:
        print(f"Already early-stopped after {stale} non-improving validation epochs", flush=True)
        return
    training = Pairs(args.dataset, "train", args.modality, args.radiometry,
                     augment=True, lr_crop=args.train_lr_crop)
    validation_set = Pairs(args.dataset, "val", args.modality, args.radiometry)
    loader_generator = torch.Generator().manual_seed(args.seed)
    loader = DataLoader(training, batch_size=args.batch_size, shuffle=True,
                        num_workers=args.workers, generator=loader_generator,
                        pin_memory=device.type == "cuda")
    gan_start = max(1, args.epochs // 3)
    for epoch in range(start_epoch + 1, args.epochs + 1):
        model.train()
        losses, pixel_losses, perceptual_losses, adversarial_losses = [], [], [], []
        epoch_start = time.perf_counter()
        for lr, hr, mask, _ in loader:
            lr, hr, mask = lr.to(device), hr.to(device), mask.to(device)
            sr = predict(model, spec, lr)
            pixel_loss = mae(sr, hr, mask)
            perceptual_loss = adversarial_loss = sr.new_zeros(())
            use_gan = discriminator is not None and epoch > gan_start
            if use_gan:
                discriminator.train()
                discriminator_optimizer.zero_grad(set_to_none=True)
                real_logits = discriminator(hr[:, :3])
                fake_logits = discriminator(sr[:, :3].detach())
                discriminator_loss = F.softplus(-real_logits).mean() + F.softplus(fake_logits).mean()
                discriminator_loss.backward()
                discriminator_optimizer.step()
                for parameter in discriminator.parameters():
                    parameter.requires_grad_(False)
                perceptual_loss = perceptual(sr[:, :3] * 2 - 1, hr[:, :3] * 2 - 1).mean()
                adversarial_loss = F.softplus(-discriminator(sr[:, :3])).mean()
                for parameter in discriminator.parameters():
                    parameter.requires_grad_(True)
            loss = pixel_loss + 0.1 * perceptual_loss + 0.005 * adversarial_loss
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.detach()))
            pixel_losses.append(float(pixel_loss.detach()))
            perceptual_losses.append(float(perceptual_loss.detach()))
            adversarial_losses.append(float(adversarial_loss.detach()))
        val_psnr, val_mae = validation(model, spec, validation_set, device)
        improved = val_psnr > best_psnr
        best_psnr = max(best_psnr, val_psnr)
        row = {"epoch": epoch, "loss": float(np.mean(losses)),
               "pixel_loss": float(np.mean(pixel_losses)),
               "perceptual_loss": float(np.mean(perceptual_losses)),
               "adversarial_loss": float(np.mean(adversarial_losses)),
               "val_psnr": val_psnr, "val_mae": val_mae,
               "seconds": time.perf_counter() - epoch_start, "improved": improved}
        history.append(row)
        state = {"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                 "discriminator": discriminator.state_dict() if discriminator is not None else None,
                 "discriminator_optimizer": (discriminator_optimizer.state_dict()
                                             if discriminator_optimizer is not None else None),
                 "epoch": epoch, "best_psnr": best_psnr, "history": history,
                 "protocol": receipt, "torch_rng": torch.get_rng_state(),
                 "cuda_rng": torch.cuda.get_rng_state_all() if device.type == "cuda" else None,
                 "numpy_rng": np.random.get_state(), "python_rng": random.getstate()}
        save_checkpoint(last_path, state)
        if improved:
            save_checkpoint(root / "best.pt", state)
        write_csv(root / "training.csv", history)
        stale = trailing_nonimprovements(history)
        print(f"[{args.model}/{args.modality}/{args.radiometry}] epoch {epoch}/{args.epochs} "
              f"loss={row['loss']:.6f} val_psnr={val_psnr:.4f} stale={stale}/{args.patience}", flush=True)
        if stale >= args.patience:
            break
    write_json(root / "training_status.json",
               {"last_completed_epoch": history[-1]["epoch"], "requested_epoch_ceiling": args.epochs,
                "early_stopped": trailing_nonimprovements(history) >= args.patience,
                "patience": args.patience, "best_validation_psnr": best_psnr,
                "best_checkpoint_sha256": digest(root / "best.pt")})


def _flops(model, sample: torch.Tensor) -> tuple[float | None, str]:
    if model is None:
        return None, "not_applicable"
    try:
        from fvcore.nn import FlopCountAnalysis
        return float(FlopCountAnalysis(model, (sample,)).total()), "fvcore_operator_count"
    except Exception as error:
        return None, f"unavailable: {type(error).__name__}: {error}"


@torch.no_grad()
def evaluate(args) -> None:
    device = torch.device(args.device)
    model, spec = load(args.model, args.sources, args.modality)
    root = run_root(args)
    checkpoint = args.checkpoint or root / "best.pt"
    checkpoint_hash = None
    if model is not None:
        if not checkpoint.is_file():
            raise FileNotFoundError(f"Train first or supply --checkpoint: {checkpoint}")
        state = strict_load(model, checkpoint)
        checkpoint_hash = digest(checkpoint)
        expected = signature(args, spec)
        if state.get("protocol") != expected:
            raise ValueError("Checkpoint does not match the requested dataset/modality/radiometry protocol")
        model = model.to(device).eval()
    dataset = Pairs(args.dataset, args.split, args.modality, args.radiometry)
    audit_info = audit(args.dataset)
    evaluation_id = checkpoint_hash[:12] if checkpoint_hash else "bicubic"
    destination = root / "evaluation" / args.split / evaluation_id
    receipt = canonical({"model": spec, "checkpoint_sha256": checkpoint_hash,
                         "code_sha256": code_inventory(), "dataset": audit_info,
                         "split": args.split, "modality": args.modality,
                         "radiometry": args.radiometry, "border_crop": args.border_crop,
                         "test_time_augmentation": False, "inference": "whole 160x160 frame, float32"})
    destination.mkdir(parents=True, exist_ok=True)
    protocol_path = destination / "protocol.json"
    if protocol_path.exists() and json.loads(protocol_path.read_text()) != receipt:
        raise ValueError("Existing evaluation protocol differs")
    write_json(protocol_path, receipt)
    write_json(destination / "environment.json", environment(device))
    if (destination / "metrics.json").exists() and (destination / "per_image.csv").exists():
        print("Completed evaluation reused:", destination, flush=True)
        return
    perceptual = lpips_network(device)
    sample = dataset[0][0][None].to(device)
    for _ in range(3):
        predict(model, spec, sample)
    synchronize(device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    flop_count, flop_status = _flops(model, sample)
    rows, inference_times = [], []
    started = time.perf_counter()
    for index, (lr, hr, mask, _) in enumerate(DataLoader(dataset, batch_size=1, num_workers=0)):
        record = dataset.rows[index]
        lr_device = lr.to(device)
        synchronize(device)
        tick = time.perf_counter()
        sr_device = predict(model, spec, lr_device)
        synchronize(device)
        milliseconds = (time.perf_counter() - tick) * 1000
        sr_device = sr_device.float().clamp(0, 1)
        if not torch.isfinite(sr_device).all():
            raise FloatingPointError(record["pair_id"])
        values = metric_bundle(sr_device, hr.to(device), mask.to(device),
                               border=args.border_crop, lpips_model=perceptual)
        rows.append({"pair_id": record["pair_id"], "model": args.model,
                     "modality": args.modality, "radiometry": args.radiometry,
                     "landscape_class": record.get("landscape_class", ""),
                     "geographic_stratum": record.get("geographic_stratum", ""),
                     "state_or_ut": record.get("state_or_ut", ""),
                     "priority_region": record.get("priority_region", ""),
                     "spatial_block_id": record.get("spatial_block_id", ""),
                     "day_gap": record.get("day_gap", ""),
                     "inference_ms": milliseconds, **values})
        inference_times.append(milliseconds)
        sr = sr_device.cpu()
        save_prediction(destination / "predictions" / f"{record['pair_id']}.tif",
                        sr[0].numpy(), mask[0].numpy() > 0,
                        resolve(args.dataset, record["hr_path"]), args.modality,
                        {"model": args.model, "checkpoint_sha256": checkpoint_hash or "none",
                         "radiometry": args.radiometry, "border_crop_for_metrics": args.border_crop})
        if (index + 1) % 25 == 0 or index + 1 == len(dataset):
            print(f"[{args.model}/{args.split}] {index + 1}/{len(dataset)} "
                  f"PSNR={np.mean([row['psnr'] for row in rows]):.4f}", flush=True)
    metric_names = [key for key in rows[0] if key not in {
        "pair_id", "model", "modality", "radiometry", "landscape_class",
        "geographic_stratum", "state_or_ut", "priority_region", "spatial_block_id", "day_gap"
    }]
    summary = {key: float(np.mean([row[key] for row in rows])) for key in metric_names}
    timing = np.asarray(inference_times)
    summary.update({"model": args.model, "modality": args.modality,
                    "radiometry": args.radiometry, "split": args.split, "count": len(rows),
                    "seed": args.seed, "parameters": spec.get("parameters", 0),
                    "flops_160": flop_count, "flops_status": flop_status,
                    "seconds_total": time.perf_counter() - started,
                    "inference_mean_ms": float(timing.mean()),
                    "inference_median_ms": float(np.median(timing)),
                    "inference_p95_ms": float(np.percentile(timing, 95)),
                    "images_per_second": 1000 / float(timing.mean()),
                    "peak_allocated_mb": (torch.cuda.max_memory_allocated(device) / 2**20
                                          if device.type == "cuda" else None),
                    "metric_domain": "clipped reflectance [0,1]",
                    "border_crop": args.border_crop,
                    "lpips_domain": "RGB AlexNet features; invalid pixels replaced with target"})
    write_csv(destination / "per_image.csv", rows)
    write_json(destination / "per_image.json", rows)
    write_json(destination / "metrics.json", summary)
    print(json.dumps(summary, indent=2), flush=True)


def smoke(args) -> None:
    model, spec = load(args.model, args.sources, args.modality)
    device = torch.device(args.device)
    if model is not None:
        model = model.to(device).train(args.backward)
    channels = 3 if args.modality == "rgb" else 4
    value = torch.rand(1, channels, LR_SIZE, LR_SIZE, device=device)
    with torch.set_grad_enabled(args.backward):
        output = predict(model, spec, value)
        if args.backward and model is not None:
            output.square().mean().backward()
    if tuple(output.shape) != (1, channels, 480, 480) or not torch.isfinite(output).all():
        raise RuntimeError(f"Invalid preflight output: {tuple(output.shape)}")
    report = {"status": "passed", "shape": list(output.shape), "backward": args.backward, **spec}
    write_json(args.output / "preflight" / f"{args.model}_{args.modality}.json", report)
    print(json.dumps(report, indent=2), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("fetch", "audit", "smoke", "train", "evaluate"))
    parser.add_argument("--sources", type=Path, default=Path("/kaggle/working/mahimandal_sources"))
    parser.add_argument("--model", choices=[*REGISTRY, "bicubic"], default="EDSR")
    parser.add_argument("--dataset", type=Path)
    parser.add_argument("--output", type=Path, default=Path("/kaggle/working/mahimandal160_benchmark"))
    parser.add_argument("--modality", choices=("rgb", "rgbnir"), default="rgb")
    parser.add_argument("--radiometry", choices=("reflectance16", "normalized_float32", "quantized8"),
                        default="reflectance16")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--train-lr-crop", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--split", choices=("val", "test"), default="val")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--border-crop", type=int, default=3)
    parser.add_argument("--backward", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(min(4, os.cpu_count() or 1))
    if min(args.epochs, args.patience, args.batch_size, args.train_lr_crop) < 1:
        parser.error("epochs, patience, batch size and crop must be positive")
    if args.border_crop < 0 or args.border_crop >= 240:
        parser.error("invalid border crop")
    if args.action == "fetch":
        fetch(args.sources)
        return
    if args.action == "smoke":
        smoke(args)
        return
    if args.dataset is None:
        parser.error("--dataset is required")
    if args.action == "audit":
        result = audit(args.dataset)
        write_json(args.output / "dataset_audit.json", result)
        print(json.dumps(result, indent=2))
        return
    try:
        (train if args.action == "train" else evaluate)(args)
    except Exception:
        failure = run_root(args) / f"{args.action}_failure.json"
        write_json(failure, {"status": "failed", "model": args.model,
                             "modality": args.modality, "radiometry": args.radiometry,
                             "error": traceback.format_exc()})
        raise


if __name__ == "__main__":
    main()
