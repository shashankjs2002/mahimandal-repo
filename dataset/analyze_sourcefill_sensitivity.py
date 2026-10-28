"""Reaggregate archived fixed-baseline scores after source-fill pair exclusion.

Not pixel-remasked metrics or retraining. Conservative common four-band cohort
exclusion measures sensitivity to identified source-invalid locations only.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from audit_sentinel_product_metadata import EXPECTED_MANIFEST


def paired_interval(frame, repeats=2000, seed=42):
    grouped = frame.groupby('spatial_block_id').delta.agg(['sum', 'count'])
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(grouped), size=(repeats, len(grouped)))
    values = grouped['sum'].to_numpy()[draws].sum(1) / grouped['count'].to_numpy()[draws].sum(1)
    return np.quantile(values, [.025, .975]).tolist()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--source-audit', type=Path, required=True)
    parser.add_argument('--benchmark', type=Path, required=True)
    parser.add_argument('--overlap-excluded', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if hashlib.sha256(args.manifest.read_bytes()).hexdigest() != EXPECTED_MANIFEST:
        raise ValueError('Not the frozen release')
    source = json.loads((args.source_audit / 'summary.json').read_text())
    if not source['complete'] or source['conversion_mismatch_pixels']:
        raise ValueError('Original-source adjudication must complete without conversion disagreement')
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError('Choose a new output')
    args.output.mkdir(parents=True, exist_ok=True)
    records = [json.loads(line) for line in (args.source_audit / 'completed.jsonl').read_text().splitlines()]
    invalid_ids = {row['pair_id'] for row in records if row['invalid_any_checked_band_pixels']}
    manifest = pd.read_csv(args.manifest)
    affected = manifest.loc[manifest.pair_id.isin(invalid_ids)].copy()
    affected.to_csv(args.output / 'source_fill_affected_pairs.csv', index=False)
    test = manifest.loc[manifest.split == 'test'].copy()
    all_ids = set(test.pair_id)
    overlap = set(pd.read_csv(args.overlap_excluded).pair_id)
    cohorts = {'full731': all_ids, 'sourcefill_excluded': all_ids - invalid_ids,
               'sourcefill_and_overlap_excluded': all_ids - invalid_ids - overlap}
    rows, cohort_rows = [], []
    metrics = ('psnr', 'ssim', 'mae', 'rmse', 'edge_f1', 'ergas', 'uiqi', 'sam_degrees', 'lpips', 'ndvi_mae', 'ndvi_rmse')
    for name, ids in cohorts.items():
        selected = test.loc[test.pair_id.isin(ids)]
        selected.to_csv(args.output / (name + '_manifest.csv'), index=False)
        cohort_rows.append({'cohort': name, 'pairs': len(ids), 'blocks': selected.spatial_block_id.nunique(),
                            'classes': selected.landscape_class.nunique()})
    for modality in ('rgb', 'rgbnir'):
        root = args.benchmark / 'reflectance16' / modality
        paths = list(root.glob('*/seed_42/evaluation/test/*/per_image.csv'))
        tables = {}
        for path in paths:
            model = path.relative_to(root).parts[0]
            if model in tables:
                raise ValueError('Ambiguous archived evaluation')
            frame = pd.read_csv(path)
            if frame.pair_id.duplicated().any() or set(frame.pair_id) != all_ids:
                raise ValueError('Archived cohort mismatch')
            tables[model] = frame.set_index('pair_id')
        for model, frame in tables.items():
            for cohort, ids in cohorts.items():
                selected = frame.loc[sorted(ids)]
                delta = selected.psnr - tables['bicubic'].loc[selected.index].psnr
                paired = pd.DataFrame({'pair_id': selected.index, 'delta': delta.to_numpy()}).merge(
                    test[['pair_id', 'spatial_block_id']], on='pair_id', validate='one_to_one')
                low, high = paired_interval(paired)
                row = {'modality': modality, 'model': model, 'cohort': cohort, 'count': len(ids),
                       'delta_psnr_vs_bicubic': float(delta.mean()), 'ci_low': low, 'ci_high': high}
                row.update({metric: float(selected[metric].mean()) for metric in metrics if metric in selected
                            and selected[metric].notna().any()})
                rows.append(row)
    results = pd.DataFrame(rows)
    results.to_csv(args.output / 'metrics.csv', index=False)
    pd.DataFrame(cohort_rows).to_csv(args.output / 'cohorts.csv', index=False)
    counts = test.assign(source_fill_affected=test.pair_id.isin(invalid_ids),
                         sourcefill_overlap_retained=test.pair_id.isin(cohorts['sourcefill_and_overlap_excluded']))
    counts.groupby('landscape_class').agg(full=('pair_id', 'size'),
        affected=('source_fill_affected', 'sum'), combined_retained=('sourcefill_overlap_retained', 'sum')).to_csv(
            args.output / 'class_counts.csv')
    tex = [r'\begin{tabular}{llrrr}', r'\toprule',
           r'Output & Model & PSNR full & PSNR fill-excluded & $\Delta$ vs bicubic \\', r'\midrule']
    for modality in ('rgb', 'rgbnir'):
        for model in sorted(results.model.unique()):
            group = results.loc[(results.modality == modality) & (results.model == model)]
            if group.empty:
                continue
            full = group.loc[group.cohort == 'full731'].iloc[0]
            kept = group.loc[group.cohort == 'sourcefill_excluded'].iloc[0]
            label = model.replace('_', r'\_')
            tex.append(f"{modality.upper()} & {label} & {full.psnr:.3f} & {kept.psnr:.3f} & {kept.delta_psnr_vs_bicubic:+.3f} " + r'\\')
    tex.extend([r'\bottomrule', r'\end{tabular}'])
    (args.output / 'benchmark_psnr.tex').write_text('\n'.join(tex) + '\n', encoding='utf-8')
    summary = {'manifest_sha256': EXPECTED_MANIFEST, 'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
               'source_checkpoint_sha256': hashlib.sha256((args.source_audit / 'completed.jsonl').read_bytes()).hexdigest(),
               'affected_pairs_by_split': affected.split.value_counts().to_dict(), 'cohorts': cohort_rows,
               'new_training_or_inference': False,
               'interpretation': 'Entire-pair exclusion using archived scores; not pixel-masked reevaluation or corrected training. Existing full731 results unchanged.'}
    (args.output / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(summary, indent=2))
    print(results.loc[(results.model.isin(['RDN', 'bicubic']))].to_string(index=False))


if __name__ == '__main__':
    main()
