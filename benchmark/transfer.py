"""Full-frame OLI2MSI transfer study for the pinned benchmark models.

The source checkpoints stay immutable. OLI2MSI validation is derived only from
its official training directory; the official 100-pair test directory is locked.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import time
from pathlib import Path

import numpy as np
import rasterio
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

from adapters import load as load_benchmark, load_discriminator, predict as predict_benchmark
from data import Pairs as MahimandalPairs, digest
from metrics import mae, metric_bundle, psnr
PROFILES = {"rgb": (3, 3), "rgbnir": (4, 4)}


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(dict.fromkeys(k for r in rows for k in r)))
        writer.writeheader()
        writer.writerows(rows)


def oli_layout(root: Path) -> dict[str, Path]:
    aliases = {"trainlr": "train_lr", "trainhr": "train_hr",
               "testlr": "test_lr", "testhr": "test_hr"}
    found = {}
    for child in root.iterdir():
        if child.is_dir():
            name = "".join(c for c in child.name.casefold() if c.isalnum())
            if name in aliases:
                found[aliases[name]] = child
    if set(found) != set(aliases.values()):
        raise ValueError(f"Expected train_lr/train_hr/test_lr/test_hr under {root}")
    return found


def oli_inventory(root: Path, seed: int = 42, val_percent: int = 10) -> dict:
    layout = oli_layout(root)
    inventory = {}
    manifest = hashlib.sha256()
    for split in ("train", "test"):
        sides = {}
        for side in ("lr", "hr"):
            paths = [p for p in layout[f"{split}_{side}"].iterdir()
                     if p.is_file() and p.suffix.casefold() in {".tif", ".tiff"}]
            files = {p.stem.casefold(): p for p in paths}
            if len(files) != len(paths):
                raise ValueError(f"Duplicate pair identifiers in {layout[split + '_' + side]}")
            sides[side] = files
        if not sides["lr"] or sides["lr"].keys() != sides["hr"].keys():
            raise ValueError(f"Unpaired or empty OLI2MSI {split} files")
        inventory[split] = []
        for name in sorted(sides["lr"]):
            lr, hr = sides["lr"][name], sides["hr"][name]
            manifest.update(f"{split}|{name}|{lr.stat().st_size}|{hr.stat().st_size}\n".encode())
            inventory[split].append((name, lr, hr))
    if {row[0] for row in inventory["train"]} & {row[0] for row in inventory["test"]}:
        raise ValueError("OLI2MSI train and test share pair identifiers")
    with rasterio.open(inventory["train"][0][1]) as lr, rasterio.open(inventory["train"][0][2]) as hr:
        channels = min(lr.count, hr.count)
        descriptions = (lr.descriptions, hr.descriptions)
        dtypes = (lr.dtypes[0], hr.dtypes[0])
    nir_available = channels >= 4 and all(
        name and "nir" in name.casefold() for name in (descriptions[0][3], descriptions[1][3])
    )
    return {"rows": inventory, "identity": manifest.hexdigest(), "channels": channels,
            "nir_available": nir_available, "descriptions": descriptions, "dtypes": dtypes,
            "counts": {"official_train": len(inventory["train"]),
                       "official_test": len(inventory["test"])},
            "seed": seed, "validation_percent": val_percent}


def oli_split(inventory: dict, split: str) -> list[tuple[str, Path, Path]]:
    if split == "test":
        return inventory["rows"]["test"]
    if split not in {"train", "val"}:
        raise ValueError(split)
    result = []
    for item in inventory["rows"]["train"]:
        value = hashlib.sha256(f"{inventory['seed']}:{item[0]}".encode()).digest()
        assigned = "val" if int.from_bytes(value[:4], "big") % 100 < inventory["validation_percent"] else "train"
        if assigned == split:
            result.append(item)
    return result


def decode(image: np.ndarray, dtype: str, radiometry: str) -> np.ndarray:
    if np.dtype(dtype) == np.dtype("uint8"):
        image = image.astype(np.float32) / 255.0
    elif np.issubdtype(np.dtype(dtype), np.integer):
        image = image.astype(np.float32) / 10000.0
    else:
        image = image.astype(np.float32)
        if np.nanmax(image) > 1.5:
            raise ValueError("Floating OLI2MSI reflectance exceeds 1.5; inspect source scaling")
    image = np.clip(image, 0, 1)
    if radiometry == "quantized8":
        image = np.rint(image * 255) / 255
    elif radiometry != "reflectance16":
        raise ValueError(radiometry)
    return image


class OliPairs(Dataset):
    def __init__(self, inventory: dict, split: str, modality: str,
                 radiometry: str, augment: bool = False):
        self.rows = oli_split(inventory, split)
        self.modality, self.radiometry, self.augment = modality, radiometry, augment
        if not self.rows:
            raise ValueError(f"Empty OLI2MSI {split} split")
        if modality != "rgb" and not inventory["nir_available"]:
            raise ValueError("OLI2MSI lacks verified paired NIR; RGB+NIR experiments cannot run")

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int):
        name, lr_path, hr_path = self.rows[index]
        bands = [1, 2, 3] if self.modality == "rgb" else [1, 2, 3, 4]
        arrays = []
        for path, size in ((lr_path, 160), (hr_path, 480)):
            with rasterio.open(path) as src:
                if src.width != size or src.height != size:
                    raise ValueError(f"Expected {size}x{size}: {path}")
                if src.count < len(bands):
                    raise ValueError(f"Missing bands in {path}")
                raw = src.read(bands)
                image = decode(raw, src.dtypes[0], self.radiometry)
                mask = np.all(src.read_masks(bands) > 0, axis=0) & np.isfinite(raw).all(axis=0)
                arrays.append((image, mask))
        (lr, lr_mask), (hr, hr_mask) = arrays
        mask = hr_mask & np.repeat(np.repeat(lr_mask, 3, 0), 3, 1)
        if not mask.any():
            raise ValueError(f"No jointly valid pixels: {name}")
        payload = [lr, hr, mask[None].astype(np.float32)]
        if self.augment:
            rotation = int(torch.randint(4, ()).item())
            payload = [np.rot90(x, rotation, axes=(-2, -1)) for x in payload]
            if torch.rand(()) < 0.5:
                payload = [x[..., ::-1] for x in payload]
        return (*[torch.from_numpy(x.copy()) for x in payload], name)


class StudyPairs(Dataset):
    def __init__(self, domain: str, root: Path, split: str, modality: str,
                 radiometry: str, inventory: dict | None = None, augment: bool = False):
        self.modality = modality
        channels = "rgb" if modality == "rgb" else "rgbnir"
        if domain == "oli2msi":
            self.source = OliPairs(inventory or oli_inventory(root), split, channels, radiometry, augment)
        elif domain == "mahimandal":
            self.source = MahimandalPairs(root, split, channels, radiometry, augment, 160)
        else:
            raise ValueError(domain)

    def __len__(self) -> int:
        return len(self.source)

    def __getitem__(self, index: int):
        lr, hr, mask, row = self.source[index]
        if self.modality == "rgbnir_rgb":
            hr = hr[:3]
        pair_id = row if isinstance(row, str) else self.source.rows[index]["pair_id"]
        return lr, hr, mask, pair_id


def load_model(kind: str, name: str, sources: Path, modality: str):
    if kind == "benchmark":
        model, spec = load_benchmark(name, sources, modality)
        return model, spec
    raise ValueError(kind)


def predict(kind: str, model, spec: dict, lr: torch.Tensor):
    if model is None:
        return F.interpolate(lr, scale_factor=3, mode="bicubic", align_corners=False)
    if kind == "benchmark":
        return predict_benchmark(model, spec, lr)
    raise ValueError(kind)


def benchmark_checkpoint_name(saved_protocol: dict) -> str | None:
    """Read both OLI-trained and original benchmark checkpoint identities."""
    names = {saved_protocol.get("model_name")}
    for key in ("model_spec", "model"):
        value = saved_protocol.get(key)
        if isinstance(value, dict):
            names.add(value.get("name"))
    names.discard(None)
    if len(names) > 1:
        raise ValueError(f"Conflicting checkpoint benchmark identities: {names}")
    return next(iter(names), None)


def protocol(args, identity: str, spec: dict) -> dict:
    value = {"source_dataset": "oli2msi", "dataset_identity": identity,
            "model_kind": args.kind, "model_name": args.model, "model_spec": spec,
            "modality": args.modality, "radiometry": args.radiometry,
            "seed": args.seed, "batch_size": args.batch_size, "lr": args.lr,
            "optimizer": "Adam", "loss": (
                "masked L1 + RGB LPIPS + logistic GAN after warmup"
                if args.kind == "benchmark" and args.model == "RealESRGAN_SRVGG_X3"
                else "masked L1"), "augmentation": "D4",
            "train_geometry": [160, 480], "selection": "full-frame OLI2MSI validation PSNR"}
    if args.kind == "benchmark" and args.model == "RealESRGAN_SRVGG_X3":
        value["gan_warmup_epochs"] = max(1, args.epochs // 3)
    return json.loads(json.dumps(value, allow_nan=False))


def training_root(args) -> Path:
    return args.output / "trained_oli2msi" / args.kind / args.radiometry / args.modality / args.model / f"seed_{args.seed}"


def train(args, inventory: dict) -> Path:
    root = training_root(args)
    root.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    model, spec = load_model(args.kind, args.model, args.sources, args.modality)
    if model is None:
        raise ValueError("Bicubic is an evaluation baseline")
    signature = protocol(args, inventory["identity"], spec)
    lock = root / "protocol.json"
    if lock.is_file() and json.loads(lock.read_text()) != signature:
        raise ValueError(f"Existing training protocol differs: {lock}")
    write_json(lock, signature)
    device = torch.device(args.device)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    use_gan = args.kind == "benchmark" and args.model == "RealESRGAN_SRVGG_X3"
    discriminator = discriminator_optimizer = perceptual = None
    if use_gan:
        import lpips
        discriminator = load_discriminator(args.sources).to(device)
        discriminator_optimizer = torch.optim.Adam(discriminator.parameters(), lr=args.lr)
        perceptual = lpips.LPIPS(net="alex").to(device).eval()
        for parameter in perceptual.parameters():
            parameter.requires_grad_(False)
    training = StudyPairs("oli2msi", args.oli, "train", args.modality, args.radiometry, inventory, True)
    validation = StudyPairs("oli2msi", args.oli, "val", args.modality, args.radiometry, inventory)
    last, best = root / "last.pt", root / "best.pt"
    epoch0, best_psnr, stale = 0, -float("inf"), 0
    if last.is_file():
        state = torch.load(last, map_location="cpu", weights_only=False)
        if state["protocol"] != signature:
            raise ValueError("Checkpoint protocol differs from current request")
        model.load_state_dict(state["model"], strict=True)
        optimizer.load_state_dict(state["optimizer"])
        if use_gan:
            discriminator.load_state_dict(state["discriminator"], strict=True)
            discriminator_optimizer.load_state_dict(state["discriminator_optimizer"])
        epoch0, best_psnr, stale = state["epoch"], state["best_psnr"], state["stale"]
    for epoch in range(epoch0 + 1, args.epochs + 1):
        if stale >= args.patience:
            break
        torch.manual_seed(args.seed + epoch)
        loader = DataLoader(training, batch_size=args.batch_size, shuffle=True,
                            num_workers=args.workers, pin_memory=device.type == "cuda")
        model.train()
        losses = []
        for lr, hr, mask, _ in loader:
            lr, hr, mask = lr.to(device), hr.to(device), mask.to(device)
            output = predict(args.kind, model, spec, lr)
            loss = mae(output, hr, mask)
            if use_gan and epoch > max(1, args.epochs // 3):
                discriminator_optimizer.zero_grad(set_to_none=True)
                real_logits = discriminator(hr[:, :3])
                fake_logits = discriminator(output[:, :3].detach())
                d_loss = F.softplus(-real_logits).mean() + F.softplus(fake_logits).mean()
                d_loss.backward()
                discriminator_optimizer.step()
                for parameter in discriminator.parameters():
                    parameter.requires_grad_(False)
                loss = (loss + 0.1 * perceptual(output[:, :3] * 2 - 1,
                                                 hr[:, :3] * 2 - 1).mean()
                        + 0.005 * F.softplus(-discriminator(output[:, :3])).mean())
                for parameter in discriminator.parameters():
                    parameter.requires_grad_(True)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
            optimizer.step()
            losses.append(float(loss.detach()))
        model.eval()
        scores = []
        with torch.no_grad():
            for lr, hr, mask, _ in DataLoader(validation, batch_size=1):
                output = predict(args.kind, model, spec, lr.to(device)).float().clamp(0, 1).cpu()
                scores.append(float(psnr(output, hr, mask)))
        score = float(np.mean(scores))
        improved = score > best_psnr
        stale = 0 if improved else stale + 1
        best_psnr = max(best_psnr, score)
        state = {"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                 "epoch": epoch, "best_psnr": best_psnr, "stale": stale,
                 "protocol": signature}
        if use_gan:
            state["discriminator"] = discriminator.state_dict()
            state["discriminator_optimizer"] = discriminator_optimizer.state_dict()
        temporary = root / "checkpoint.tmp"
        torch.save(state, temporary)
        temporary.replace(last)
        if improved:
            torch.save(state, temporary)
            temporary.replace(best)
        print(f"{args.kind}/{args.model}/{args.modality}: epoch {epoch}/{args.epochs} "
              f"train_l1={np.mean(losses):.6f} val_psnr={score:.4f} stale={stale}/{args.patience}", flush=True)
    if not best.is_file():
        raise FileNotFoundError(f"No best checkpoint: {best}")
    return best


@torch.no_grad()
def evaluate(args, inventory: dict, checkpoint: Path, domain: str) -> dict:
    model, spec = load_model(args.kind, args.model, args.sources, args.modality)
    if model is not None:
        state = torch.load(checkpoint, map_location="cpu", weights_only=False)
        if state["protocol"].get("modality") != args.modality:
            raise ValueError("Checkpoint modality does not match request")
        if state["protocol"].get("radiometry") != args.radiometry:
            raise ValueError("Checkpoint radiometry does not match request")
        if args.kind == "benchmark" and benchmark_checkpoint_name(state["protocol"]) != args.model:
            raise ValueError("Checkpoint benchmark architecture does not match request")
        model.load_state_dict(state["model"], strict=True)
    checkpoint_hash = digest(checkpoint) if model is not None else "bicubic"
    target_root = args.oli if domain == "oli2msi" else args.mahimandal
    target_identity = inventory["identity"] if domain == "oli2msi" else digest(target_root / "paper_release_manifest.csv")
    destination = args.output / "evaluation" / args.trained_on / domain / args.kind / args.radiometry / args.modality / args.model / checkpoint_hash[:12]
    receipt = {"trained_on": args.trained_on, "tested_on": domain,
               "source_checkpoint_sha256": checkpoint_hash, "target_identity": target_identity,
               "modality": args.modality, "radiometry": args.radiometry,
               "geometry": [160, 480], "metric_domain": "reflectance [0,1]",
               "border_crop": args.border_crop}
    lock = destination / "protocol.json"
    if lock.is_file() and json.loads(lock.read_text()) != receipt:
        raise ValueError(f"Existing evaluation differs: {lock}")
    write_json(lock, receipt)
    result = destination / "summary.json"
    if result.is_file() and (destination / "per_image.csv").is_file():
        print("Reused:", result, flush=True)
        return json.loads(result.read_text())
    device = torch.device(args.device)
    if model is not None:
        model.to(device).eval()
    dataset = StudyPairs(domain, target_root, "test", args.modality,
                         args.radiometry, inventory if domain == "oli2msi" else None)
    rows = []
    pictured_classes = set()
    for index, (lr, hr, mask, pair_id) in enumerate(DataLoader(dataset, batch_size=1)):
        if device.type == "cuda":
            torch.cuda.synchronize()
        started = time.perf_counter()
        output = predict(args.kind, model, spec, lr.to(device)).float().clamp(0, 1)
        if device.type == "cuda":
            torch.cuda.synchronize()
        elapsed = (time.perf_counter() - started) * 1000
        if tuple(output.shape) != tuple(hr.shape) or not torch.isfinite(output).all():
            raise ValueError(f"Invalid prediction for {pair_id[0]}")
        measures = metric_bundle(output, hr.to(device), mask.to(device), border=args.border_crop)
        metadata = dataset.source.rows[index] if domain == "mahimandal" else {}
        scene = metadata.get("landscape_class", "unlabeled")
        rows.append({"pair_id": str(pair_id[0]),
                     "landscape_class": scene,
                     "geographic_stratum": metadata.get("geographic_stratum", "unlabeled"),
                     "spatial_block_id": metadata.get("spatial_block_id", ""),
                     "day_gap": metadata.get("day_gap", ""),
                     "inference_ms": elapsed, **measures})
        first_of_class = domain == "mahimandal" and scene not in pictured_classes
        if index < 5 or first_of_class:
            import matplotlib.pyplot as plt
            name = (f"class_{''.join(c if c.isalnum() else '_' for c in scene)}.png"
                    if first_of_class else f"test_{index:04d}.png")
            visual = destination / "figures" / name
            visual.parent.mkdir(parents=True, exist_ok=True)
            lr_rgb = F.interpolate(lr[:, :3], scale_factor=3, mode="nearest")[0].numpy()
            target_rgb = hr[0, :3].numpy()
            prediction_rgb = output[0, :3].cpu().numpy()
            panels = (lr_rgb, target_rgb, prediction_rgb,
                      np.abs(prediction_rgb - target_rgb))
            figure, axes = plt.subplots(1, 4, figsize=(16, 4))
            for axis, image, title in zip(axes, panels,
                                          ("Landsat LR", "Sentinel HR", args.model, "RGB absolute error")):
                shown = image.transpose(1, 2, 0)
                axis.imshow(np.clip(shown / 0.3, 0, 1) if title != "RGB absolute error"
                            else np.clip(shown / 0.1, 0, 1))
                axis.set_title(title)
                axis.axis("off")
            figure.suptitle(f"{domain} test/{index}: {pair_id[0]} | {scene} | display stretch only")
            figure.tight_layout()
            figure.savefig(visual, dpi=140)
            plt.close(figure)
            pictured_classes.add(scene)
    write_csv(destination / "per_image.csv", rows)
    summary = {"model": args.model, "kind": args.kind, "modality": args.modality,
               "radiometry": args.radiometry, "trained_on": args.trained_on,
               "tested_on": domain, "count": len(rows), "parameters": spec["parameters"],
               "checkpoint_sha256": checkpoint_hash,
               **{key: float(np.mean([r[key] for r in rows])) for key in rows[0]
                  if isinstance(rows[0][key], (int, float))}}
    write_json(result, summary)
    print(json.dumps(summary, indent=2), flush=True)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("audit", "train", "evaluate"))
    parser.add_argument("--oli", type=Path, required=True)
    parser.add_argument("--mahimandal", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sources", type=Path)
    parser.set_defaults(kind="benchmark")
    parser.add_argument("--model", default="bicubic")
    parser.add_argument("--modality", choices=tuple(PROFILES), default="rgb")
    parser.add_argument("--radiometry", choices=("reflectance16", "quantized8"), default="reflectance16")
    parser.add_argument("--trained-on", choices=("mahimandal", "oli2msi"), default="mahimandal")
    parser.add_argument("--tested-on", choices=("mahimandal", "oli2msi"), default="oli2msi")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--border-crop", type=int, default=3)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    inventory = oli_inventory(args.oli, args.seed)
    if args.action == "audit":
        result = {key: value for key, value in inventory.items() if key != "rows"}
        result["splits"] = {name: len(oli_split(inventory, name)) for name in ("train", "val", "test")}
        result["geometry"] = [160, 480]
        write_json(args.output / "oli_audit.json", result)
        print(json.dumps(result, indent=2), flush=True)
        return
    if args.action == "train":
        print(train(args, inventory))
        return
    if args.tested_on == "mahimandal" and args.mahimandal is None:
        parser.error("--mahimandal is required for that test domain")
    checkpoint = args.checkpoint or training_root(args) / "best.pt"
    evaluate(args, inventory, checkpoint, args.tested_on)


if __name__ == "__main__":
    main()
