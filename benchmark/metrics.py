"""Common full-reference metrics for reflectance-domain cross-sensor SR."""
from __future__ import annotations

import math

import torch
from torch.nn import functional as F


def _weighted_mean(value: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Average over included channel-pixels using shared spatial TIFF validity."""
    if mask.shape[1] == 1 and value.shape[1] != 1:
        mask = mask.expand(-1, value.shape[1], -1, -1)
    return (value * mask).sum() / mask.sum().clamp_min(1)


def mae(prediction: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return _weighted_mean((prediction - target).abs(), mask)


def mse(prediction: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return _weighted_mean((prediction - target).square(), mask)


def psnr(prediction: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return -10.0 * torch.log10(mse(prediction, target, mask).clamp_min(1e-12))


def _gaussian_kernel(channels: int, device, dtype) -> torch.Tensor:
    axis = torch.arange(11, device=device, dtype=dtype) - 5
    gaussian = torch.exp(-(axis.square()) / (2 * 1.5**2))
    gaussian = gaussian / gaussian.sum()
    kernel = torch.outer(gaussian, gaussian)[None, None]
    return kernel.expand(channels, 1, 11, 11).contiguous()


def ssim(prediction: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    channels = prediction.shape[1]
    kernel = _gaussian_kernel(channels, prediction.device, prediction.dtype)
    mu_x = F.conv2d(prediction, kernel, padding=5, groups=channels)
    mu_y = F.conv2d(target, kernel, padding=5, groups=channels)
    sigma_x = F.conv2d(prediction.square(), kernel, padding=5, groups=channels) - mu_x.square()
    sigma_y = F.conv2d(target.square(), kernel, padding=5, groups=channels) - mu_y.square()
    sigma_xy = F.conv2d(prediction * target, kernel, padding=5, groups=channels) - mu_x * mu_y
    c1, c2 = 0.01**2, 0.03**2
    value = ((2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)) / (
        (mu_x.square() + mu_y.square() + c1) * (sigma_x + sigma_y + c2)
    ).clamp_min(1e-12)
    return _weighted_mean(value, mask)


def _correlation(first: torch.Tensor, second: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    if mask.shape[1] == 1 and first.shape[1] != 1:
        mask = mask.expand(-1, first.shape[1], -1, -1)
    weight = mask.sum().clamp_min(1)
    first_mean = (first * mask).sum() / weight
    second_mean = (second * mask).sum() / weight
    first_centered = (first - first_mean) * mask
    second_centered = (second - second_mean) * mask
    return (first_centered * second_centered).sum() / torch.sqrt(
        first_centered.square().sum() * second_centered.square().sum()
    ).clamp_min(1e-12)


def _sobel(image: torch.Tensor) -> torch.Tensor:
    gray = image[:, :3].mean(dim=1, keepdim=True)
    x_kernel = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
                            device=image.device, dtype=image.dtype)[None, None] / 8
    y_kernel = x_kernel.transpose(-1, -2)
    dx = F.conv2d(gray, x_kernel, padding=1)
    dy = F.conv2d(gray, y_kernel, padding=1)
    return torch.sqrt(dx.square() + dy.square() + 1e-12)


def edge_f1(prediction: torch.Tensor, target: torch.Tensor, mask: torch.Tensor,
            threshold: float = 0.04) -> torch.Tensor:
    predicted = (_sobel(prediction) >= threshold).float() * mask
    reference = (_sobel(target) >= threshold).float() * mask
    predicted_dilated = F.max_pool2d(predicted, 3, stride=1, padding=1)
    reference_dilated = F.max_pool2d(reference, 3, stride=1, padding=1)
    precision = (predicted * reference_dilated).sum() / predicted.sum().clamp_min(1)
    recall = (reference * predicted_dilated).sum() / reference.sum().clamp_min(1)
    return 2 * precision * recall / (precision + recall).clamp_min(1e-12)


def _laplacian(image: torch.Tensor) -> torch.Tensor:
    channels = image.shape[1]
    kernel = torch.tensor([[0, 1, 0], [1, -4, 1], [0, 1, 0]],
                          device=image.device, dtype=image.dtype)[None, None]
    return F.conv2d(image, kernel.expand(channels, 1, 3, 3), padding=1, groups=channels)


def _crop(values: tuple[torch.Tensor, ...], border: int) -> tuple[torch.Tensor, ...]:
    if border == 0:
        return values
    return tuple(value[..., border:-border, border:-border] for value in values)


def _y(image: torch.Tensor) -> torch.Tensor:
    weights = image.new_tensor((0.299, 0.587, 0.114)).view(1, 3, 1, 1)
    return (image[:, :3] * weights).sum(dim=1, keepdim=True)


def _ndvi(image: torch.Tensor) -> torch.Tensor:
    red, nir = image[:, 0:1], image[:, 3:4]
    return (nir - red) / (nir + red).clamp_min(1e-6)


def metric_bundle(prediction: torch.Tensor, target: torch.Tensor, mask: torch.Tensor,
                  border: int = 3, lpips_model=None) -> dict[str, float]:
    prediction, target, mask = prediction.float(), target.float(), mask.float()
    prediction, target, mask = _crop((prediction, target, mask), border)
    channels = prediction.shape[1]
    values = {
        "mae": float(mae(prediction, target, mask)),
        "rmse": float(torch.sqrt(mse(prediction, target, mask))),
        "psnr": float(psnr(prediction, target, mask)),
        "ssim": float(ssim(prediction, target, mask)),
        "edge_f1": float(edge_f1(prediction, target, mask)),
        "edge_psnr": float(psnr(_sobel(prediction), _sobel(target), mask)),
        "gradient_correlation": float(_correlation(_sobel(prediction), _sobel(target), mask)),
        "scc": float(_correlation(_laplacian(prediction), _laplacian(target), mask)),
    }
    rgb_prediction, rgb_target = prediction[:, :3], target[:, :3]
    values.update(
        rgb_psnr=float(psnr(rgb_prediction, rgb_target, mask)),
        rgb_ssim=float(ssim(rgb_prediction, rgb_target, mask)),
        y_psnr=float(psnr(_y(rgb_prediction), _y(rgb_target), mask)),
        y_ssim=float(ssim(_y(rgb_prediction), _y(rgb_target), mask)),
    )
    names = ("red", "green", "blue", "nir")[:channels]
    band_mse, band_uiqi, band_means = [], [], []
    for index, name in enumerate(names):
        predicted_band, target_band = prediction[:, index:index + 1], target[:, index:index + 1]
        values[f"{name}_psnr"] = float(psnr(predicted_band, target_band, mask))
        error = mse(predicted_band, target_band, mask)
        band_mse.append(error)
        band_means.append(_weighted_mean(target_band, mask).abs().clamp_min(1e-6))
        predicted_mean = _weighted_mean(predicted_band, mask)
        target_mean = _weighted_mean(target_band, mask)
        predicted_variance = _weighted_mean((predicted_band - predicted_mean).square(), mask)
        target_variance = _weighted_mean((target_band - target_mean).square(), mask)
        covariance = _weighted_mean((predicted_band - predicted_mean) * (target_band - target_mean), mask)
        band_uiqi.append((4 * covariance * predicted_mean * target_mean) /
                         ((predicted_variance + target_variance) *
                          (predicted_mean.square() + target_mean.square())).clamp_min(1e-12))
    values["ergas"] = float((100 / 3) * torch.sqrt(torch.stack([
        error / mean.square() for error, mean in zip(band_mse, band_means)
    ]).mean()))
    values["uiqi"] = float(torch.stack(band_uiqi).mean())
    dot = (prediction * target).sum(dim=1)
    denominator = torch.linalg.vector_norm(prediction, dim=1) * torch.linalg.vector_norm(target, dim=1)
    angle = torch.acos((dot / denominator.clamp_min(1e-8)).clamp(-1 + 1e-7, 1 - 1e-7))
    values["sam_degrees"] = float(_weighted_mean(angle[:, None], mask) * 180 / math.pi)
    if channels == 4:
        predicted_ndvi, target_ndvi = _ndvi(prediction), _ndvi(target)
        values.update(ndvi_mae=float(mae(predicted_ndvi, target_ndvi, mask)),
                      ndvi_rmse=float(torch.sqrt(mse(predicted_ndvi, target_ndvi, mask))),
                      ndvi_correlation=float(_correlation(predicted_ndvi, target_ndvi, mask)))
    if lpips_model is not None:
        # Invalid pixels are replaced by the target before the RGB feature comparison.
        lpips_mask = mask.expand(-1, 3, -1, -1)
        safe_prediction = rgb_prediction * lpips_mask + rgb_target * (1 - lpips_mask)
        values["lpips"] = float(lpips_model(safe_prediction * 2 - 1, rgb_target * 2 - 1).mean())
    return values
