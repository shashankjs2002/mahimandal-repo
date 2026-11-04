"""Summarize archived test scores without changing benchmark inputs or results."""
import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def block_summary(differences, draws):
    blocks = differences.groupby("spatial_block_id", sort=True).delta.agg(["sum", "count"])
    sums, counts = blocks["sum"].to_numpy(), blocks["count"].to_numpy()
    sampled = sums[draws].sum(axis=1) / counts[draws].sum(axis=1)
    omitted = (sums.sum() - sums) / (counts.sum() - counts)
    return blocks, {
        "patch_weighted": float(sums.sum() / counts.sum()),
        "equal_block": float((sums / counts).mean()),
        "ci_low": float(np.quantile(sampled, .025)),
        "ci_high": float(np.quantile(sampled, .975)),
        "leave_one_out_min": float(omitted.min()),
        "leave_one_out_max": float(omitted.max()),
    }, omitted


def main():
    paper = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scores", type=Path, default=paper / "results/benchmark_three_seed/all_seed_per_image.csv")
    parser.add_argument("--means", type=Path, default=paper / "results/benchmark_three_seed/per_seed.csv")
    parser.add_argument("--manifest", type=Path, default=paper / "metadata/paper_release_manifest.csv")
    parser.add_argument("--output", type=Path, default=paper / "results/cpu_support/statistical_analysis")
    args = parser.parse_args()
    manifest = pd.read_csv(args.manifest)
    scores, means = pd.read_csv(args.scores), pd.read_csv(args.means)
    if digest(args.manifest) != "6bf97403ee40ccf0cbd38b2e6e4932e01973e5d00e211e0acfb1e0b810efc5e7":
        raise ValueError("Expected the published reflectance16 manifest")
    if not manifest.pair_id.is_unique or manifest.split.value_counts().to_dict() != {"train": 3424, "test": 731, "val": 645}:
        raise ValueError("Release membership differs")
    if manifest.spatial_block_id.nunique() != 274 or manifest.groupby("spatial_block_id").split.nunique().max() != 1:
        raise ValueError("Release blocks differ")
    test = manifest.loc[manifest.split.eq("test")].set_index("pair_id")
    if len(test) != 731 or test.spatial_block_id.nunique() != 13:
        raise ValueError("Test geography differs")
    identity = ["modality", "model", "seed"]
    if scores.duplicated(identity + ["pair_id"]).any() or means.duplicated(identity).any():
        raise ValueError("Duplicate score identities")
    keys = set(map(tuple, scores[identity].drop_duplicates().to_numpy()))
    if keys != set(map(tuple, means[identity].to_numpy())) or len(keys) != 36:
        raise ValueError("Incomplete archived fits")
    checks = []
    for key, group in scores.groupby(identity):
        if len(group) != 731 or set(group.pair_id) != set(test.index):
            raise ValueError("Test cohort mismatch: " + str(key))
        if not np.array_equal(group.spatial_block_id.to_numpy(), test.loc[group.pair_id, "spatial_block_id"].to_numpy()):
            raise ValueError("Block assignment mismatch: " + str(key))
        if set(group.radiometry) != {"reflectance16"}:
            raise ValueError("Unexpected radiometry")
        row = means.set_index(identity).loc[key]
        for metric in ("psnr", "ssim", "mae", "rmse", "edge_f1", "ergas", "uiqi", "sam_degrees", "lpips"):
            values = group[metric].to_numpy()
            calculated = float(values.mean())
            if not np.isfinite(values).all() or not np.isclose(calculated, row[metric], rtol=0, atol=1e-6):
                raise ValueError("Archived mean mismatch: " + str((key, metric)))
            checks.append(dict(zip(identity, key), metric=metric, archived_mean=float(row[metric]), recalculated_mean=calculated))
    for _, group in scores.groupby(["modality", "model"]):
        if set(group.seed) != {42, 43, 44}:
            raise ValueError("Missing fitting seed")
    comparisons = [("rgb", "RDN", "bicubic"), ("rgb", "RDN", "EDSR"),
                   ("rgbnir", "RDN", "bicubic"), ("rgbnir", "RDN", "CAFRN")]
    draws = np.random.default_rng(42).integers(0, 13, size=(2000, 13))
    results, by_seed, by_block, leave_out, paired = [], [], [], [], []
    for modality, model, reference in comparisons:
        data = scores.loc[scores.modality.eq(modality) & scores.model.isin([model, reference])]
        pivot = data.pivot(index=["pair_id", "spatial_block_id", "seed"], columns="model", values="psnr")
        if len(pivot) != 731 * 3 or pivot.isna().any().any():
            raise ValueError("Incomplete pairing")
        delta = (pivot[model] - pivot[reference]).rename("delta")
        label = dict(modality=modality, model=model, reference=reference)
        for seed, value in delta.groupby(level="seed").mean().items():
            by_seed.append(dict(label, seed=int(seed), mean_delta=float(value)))
        differences = delta.groupby(level=["pair_id", "spatial_block_id"]).mean().reset_index()
        blocks, result, omitted = block_summary(differences, draws)
        results.append(dict(label, pairs=731, blocks=13, **result))
        for i, (block, row) in enumerate(blocks.iterrows()):
            by_block.append(dict(label, spatial_block_id=block, pairs=int(row["count"]), mean_delta=float(row["sum"] / row["count"])))
            leave_out.append(dict(label, omitted_block=block, remaining_pairs=731-int(row["count"]), mean_delta=float(omitted[i])))
        paired.append(differences.assign(**label))
    out = args.output
    if out.resolve() in {args.scores.parent.resolve(), args.manifest.parent.resolve()}:
        raise ValueError("Use a separate supplementary output folder")
    out.mkdir(parents=True, exist_ok=True)
    summary = pd.DataFrame(results)
    summary.to_csv(out / "comparisons.csv", index=False)
    pd.DataFrame(by_seed).to_csv(out / "per_seed_differences.csv", index=False)
    pd.DataFrame(by_block).to_csv(out / "per_block_differences.csv", index=False)
    pd.DataFrame(leave_out).to_csv(out / "leave_one_block_out.csv", index=False)
    pd.concat(paired).to_csv(out / "paired_image_differences.csv", index=False)
    pd.DataFrame(checks).to_csv(out / "archived_mean_checks.csv", index=False)
    receipt = dict(inputs={p.name: digest(p) for p in (args.manifest, args.scores, args.means)},
                   script_sha256=digest(__file__), pairs=4800, split_counts=manifest.split.value_counts().to_dict(),
                   spatial_blocks=274, test_blocks=13, learned_fits=30, deterministic_bicubic_rows=6,
                   checked_run_metrics=len(checks), bootstrap_seed=42, bootstrap_draws=2000,
                   method="Average image-paired PSNR differences over fitting seeds, then resample whole blocks with multiplicity and pool their images.",
                   scope="Descriptive intervals conditional on three fits; blocks may share source products. Equal-block and patch-weighted means target different summaries.",
                   dataset_or_benchmark_files_written=False)
    (out / "verification.json").write_text(json.dumps(receipt, indent=2) + "\n")
    lines = [r"\begin{tabular}{@{}llrrrr@{}}", r"\toprule",
             r"Output & Comparison & Patch mean & Equal-block & 95\% block interval & LOO range \\", r"\midrule"]
    for row in summary.itertuples():
        lines.append(f"{'RGB' if row.modality == 'rgb' else 'Four-band'} & RDN--{row.reference.capitalize() if row.reference == 'bicubic' else row.reference} & {row.patch_weighted:+.3f} & {row.equal_block:+.3f} & [{row.ci_low:+.3f}, {row.ci_high:+.3f}] & [{row.leave_one_out_min:+.3f}, {row.leave_one_out_max:+.3f}]" + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (out / "paired_comparisons.tex").write_text("\n".join(lines) + "\n")
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10})
    fig, ax = plt.subplots(figsize=(9, 3.1))
    for i, row in enumerate(summary.itertuples()):
        ax.plot([row.ci_low, row.ci_high], [i, i], color="#245a81", lw=2)
        ax.scatter(row.patch_weighted, i, color="#245a81", s=45, label="Patch-weighted mean and 95% block interval" if i == 0 else None)
        ax.scatter(row.equal_block, i, marker="D", color="#c57131", s=36, label="Equal-block mean" if i == 0 else None)
    ax.axvline(0, color="#777777", ls="--", lw=1)
    ax.set(yticks=range(4), yticklabels=[("RGB" if r.modality == "rgb" else "Four-band") + ": RDN - " + r.reference for r in summary.itertuples()], xlabel="Paired PSNR difference (dB)")
    ax.invert_yaxis()
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="x", alpha=.18)
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "paired_block_comparisons.pdf", bbox_inches="tight")
    plt.close(fig)
    print(summary.to_string(index=False))
    print("Verified", len(checks), "archived run/metric means; original files read only.")


if __name__ == "__main__":
    main()
