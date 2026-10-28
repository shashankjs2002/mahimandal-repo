"""Pinned upstream x3 model adapters with explicit modality capability checks."""
from __future__ import annotations

import contextlib
import importlib.util
import io
import json
from pathlib import Path
import subprocess
import sys
import types
from types import SimpleNamespace

import torch
from torch import nn
import yaml


REGISTRY = json.loads(Path(__file__).with_name("registry.json").read_text(encoding="utf-8"))


class _Registry:
    def register(self, obj=None, **_):
        def decorator(value):
            return value
        return decorator(obj) if obj is not None else decorator


def _install_basicsr_stubs() -> None:
    """Provide only the two BasicSR interfaces used by pinned architecture files."""
    if "basicsr.utils.registry" in sys.modules:
        return
    basicsr = types.ModuleType("basicsr")
    basicsr.__path__ = []
    utils = types.ModuleType("basicsr.utils")
    utils.__path__ = []
    registry = types.ModuleType("basicsr.utils.registry")
    registry.ARCH_REGISTRY = _Registry()
    archs = types.ModuleType("basicsr.archs")
    archs.__path__ = []
    arch_util = types.ModuleType("basicsr.archs.arch_util")
    arch_util.to_2tuple = lambda value: value if isinstance(value, tuple) else (value, value)
    arch_util.trunc_normal_ = torch.nn.init.trunc_normal_
    sys.modules.update({"basicsr": basicsr, "basicsr.utils": utils,
                        "basicsr.utils.registry": registry, "basicsr.archs": archs,
                        "basicsr.archs.arch_util": arch_util})


def git(root: str | Path, *arguments: str) -> str:
    return subprocess.check_output(["git", "-C", str(root), *arguments], text=True).strip()


def verify_source(root: str | Path, spec: dict) -> None:
    root = Path(root)
    if git(root, "rev-parse", "HEAD") != spec["revision"]:
        raise ValueError(f"Wrong source revision: {root}; use a separate sources directory")
    if git(root, "status", "--porcelain", "--untracked-files=no"):
        raise ValueError(f"Tracked upstream files were modified: {root}")


def fetch(sources: str | Path) -> None:
    sources = Path(sources)
    sources.mkdir(parents=True, exist_ok=True)
    unique = {spec["source_dir"]: spec for spec in REGISTRY.values()}
    for spec in unique.values():
        root = sources / spec["source_dir"]
        if not root.exists():
            subprocess.run(["git", "clone", "--filter=blob:none", "--no-checkout",
                            f"https://github.com/{spec['repo']}.git", str(root)], check=True)
            subprocess.run(["git", "-C", str(root), "sparse-checkout", "set", "--no-cone",
                            "*.py", "*.yaml", "*.yml", "*.md", "*LICENSE*"], check=True)
            subprocess.run(["git", "-C", str(root), "checkout", spec["revision"]], check=True)
        verify_source(root, spec)
        print(spec["source_dir"], spec["revision"], flush=True)


def _load_file(name: str, path: Path):
    module_spec = importlib.util.spec_from_file_location(name, path)
    if module_spec is None or module_spec.loader is None:
        raise ImportError(f"Cannot import {name} from {path}")
    module = importlib.util.module_from_spec(module_spec)
    sys.modules[name] = module
    module_spec.loader.exec_module(module)
    return module


def _clear_upstream_modules() -> None:
    for key in list(sys.modules):
        if key.split(".")[0] in {"model", "models", "ops", "blocks", "components"}:
            del sys.modules[key]


def _classic_model(name: str, module, channels: int):
    args = dict(scale=[3], n_colors=channels, n_feats=64, rgb_range=255,
                res_scale=1, n_resgroups=10, n_resblocks=20, reduction=16,
                n_basic_modules=10, G0=64, RDNkSize=3, RDNconfig="B",
                no_upsampling=False)
    if name == "EDSR":
        args.update(n_feats=256, n_resblocks=32, res_scale=0.1)
    if name == "SAN":
        args.update(n_resgroups=20, n_resblocks=10)
    return module.make_model(SimpleNamespace(**args)), args


def load(name: str, sources: str | Path, modality: str):
    if modality not in {"rgb", "rgbnir"}:
        raise ValueError(modality)
    channels = 3 if modality == "rgb" else 4
    if name == "bicubic":
        return None, {"name": "bicubic", "scale": 3, "input_channels": channels,
                      "output_channels": channels, "parameters": 0,
                      "provenance": "PyTorch bicubic interpolation; align_corners=False",
                      "suite": "baseline", "range_multiplier": 1, "range_offset": 0}
    if name not in REGISTRY:
        raise KeyError(name)
    spec = dict(REGISTRY[name])
    if modality not in spec["modalities"]:
        raise ValueError(f"{name} does not have a verified {modality} construction; supported={spec['modalities']}")
    root = (Path(sources) / spec["source_dir"]).resolve()
    verify_source(root, spec)
    entry = root / spec["entry"]
    _clear_upstream_modules()
    if name in {"DCM", "HSENet", "HAUNet", "RCAN", "SAN", "RDN", "EDSR"}:
        import_root = entry.parent.parent
    else:
        import_root = root
    sys.path.insert(0, str(import_root))
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            if name == "HAT":
                _install_basicsr_stubs()
            if name == "RealESRGAN_SRVGG_X3":
                _install_basicsr_stubs()
            module = _load_file("upstream_" + name, entry)
            if name == "CAFRN":
                kwargs = dict(in_channels=channels, out_channels=channels, num_features=64,
                              num_steps=7, reduction=16, upscale_factor=3,
                              act_type="prelu", norm_type=None)
                model = module.CAFRN(**kwargs)
            elif name == "FENet":
                kwargs = dict(upscale_factor=3)
                model = module.FENet(**kwargs)
            elif name == "MHAN":
                kwargs = dict(in_channels=channels, num_features=64, scale=3)
                model = module.Net(**kwargs)
            elif name == "OmniSR":
                config = yaml.safe_load((root / "train_yamls/train_OmniSR_X3_DIV2K.yaml").read_text())
                kwargs = dict(num_in_ch=channels, num_out_ch=channels,
                              num_feat=config["feature_num"], **config["module_params"])
                model = module.OmniSR(**kwargs)
            elif name == "SwinIR":
                kwargs = dict(img_size=64, patch_size=1, in_chans=channels, embed_dim=180,
                              depths=(6, 6, 6, 6, 6, 6), num_heads=(6, 6, 6, 6, 6, 6),
                              window_size=8, mlp_ratio=2, upscale=3, img_range=1.0,
                              upsampler="pixelshuffle", resi_connection="1conv",
                              use_checkpoint=False)
                model = module.SwinIR(**kwargs)
            elif name == "HAT":
                kwargs = dict(img_size=64, in_chans=channels, embed_dim=144,
                              depths=(6, 6, 6, 6, 6, 6), num_heads=(6, 6, 6, 6, 6, 6),
                              window_size=16, compress_ratio=24, squeeze_factor=24,
                              conv_scale=0.01, overlap_ratio=0.5, mlp_ratio=2,
                              upscale=3, img_range=1.0, upsampler="pixelshuffle",
                              resi_connection="1conv", use_checkpoint=False)
                model = module.HAT(**kwargs)
            elif name == "RealESRGAN_SRVGG_X3":
                kwargs = dict(num_in_ch=channels, num_out_ch=channels, num_feat=64,
                              num_conv=32, upscale=3, act_type="prelu")
                model = module.SRVGGNetCompact(**kwargs)
            else:
                model, kwargs = _classic_model(name, module, channels)
    finally:
        if str(import_root) in sys.path:
            sys.path.remove(str(import_root))
    spec.update(name=name, constructor=kwargs, input_channels=channels,
                output_channels=channels, scale=3,
                parameters=sum(parameter.numel() for parameter in model.parameters()),
                trainable_parameters=sum(parameter.numel() for parameter in model.parameters()
                                         if parameter.requires_grad))
    return model, spec


def load_discriminator(sources: str | Path) -> nn.Module:
    spec = REGISTRY["RealESRGAN_SRVGG_X3"]
    root = (Path(sources) / spec["source_dir"]).resolve()
    verify_source(root, spec)
    _install_basicsr_stubs()
    module = _load_file("upstream_realesrgan_discriminator",
                        root / "realesrgan/archs/discriminator_arch.py")
    # The adversarial signal is deliberately RGB-only in RGB+NIR experiments.
    return module.UNetDiscriminatorSN(num_in_ch=3, num_feat=64, skip_connection=True)


def predict(model: nn.Module | None, spec: dict, lr: torch.Tensor) -> torch.Tensor:
    if model is None:
        result = torch.nn.functional.interpolate(lr, scale_factor=3, mode="bicubic",
                                                 align_corners=False)
    else:
        multiplier = float(spec["range_multiplier"])
        offset = float(spec["range_offset"])
        result = model(lr * multiplier + offset)
        if not isinstance(result, torch.Tensor):
            raise TypeError("Unexpected upstream output; auxiliary outputs are not silently selected")
        result = (result - offset) / multiplier
    expected = (lr.shape[0], spec["output_channels"], lr.shape[-2] * 3, lr.shape[-1] * 3)
    if tuple(result.shape) != expected:
        raise ValueError(f"Model returned {tuple(result.shape)}, expected {expected}")
    return result
