"""Score independently annotated relative landmarks; never fabricate GCP accuracy.

CSV coordinates use pixel-edge origin (top-left pixel center is 0.5,0.5).
HR coordinates are in the 480 grid; LR in the 160 grid. Original images stay
unchanged. Annotator independence is an attestation, not verified by this script.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from audit_sentinel_product_metadata import EXPECTED_MANIFEST


FIELDS = ['pair_id', 'feature_id', 'lr_x', 'lr_y', 'hr_x', 'hr_y',
          'annotator', 'independently_reviewed', 'used_for_registration_fit']


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--template', action='store_true')
    parser.add_argument('--annotations', type=Path)
    parser.add_argument('--manifest', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--min-pairs', type=int, default=50)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('Do not overwrite prior annotation evidence')
    args.output.mkdir(parents=True)
    if args.template:
        pd.DataFrame(columns=FIELDS).to_csv(args.output / 'independent_landmarks.csv', index=False)
        (args.output / 'README.txt').write_text(
            'Annotate at least 50 pairs covering the nine strata; at least five distinct stable features per pair.\n'
            'Do not copy automatic SIFT matches as independent validation.\n'
            'Coordinates have pixel-edge origin; top-left pixel center is (0.5,0.5).\n'
            'independently_reviewed=yes; used_for_registration_fit=no only if true.\n'
            'This measures relative feature displacement, not surveyed geodetic accuracy.\n', encoding='utf-8')
        return
    if args.annotations is None or args.manifest is None:
        parser.error('Provide annotations and frozen manifest')
    if hashlib.sha256(args.manifest.read_bytes()).hexdigest() != EXPECTED_MANIFEST:
        raise ValueError('Not the frozen release')
    frame = pd.read_csv(args.annotations, keep_default_na=False)
    if not set(FIELDS).issubset(frame) or frame.empty:
        raise ValueError('Unfilled/incomplete annotation form')
    if frame.duplicated(['pair_id', 'feature_id']).any() or (frame.annotator.astype(str).str.strip() == '').any():
        raise ValueError('Duplicate landmarks or missing annotator')
    if not frame.independently_reviewed.astype(str).str.casefold().eq('yes').all() or not frame.used_for_registration_fit.astype(str).str.casefold().eq('no').all():
        raise ValueError('Independent held-out landmark attestations required')
    manifest = pd.read_csv(args.manifest)
    if not set(frame.pair_id).issubset(set(manifest.pair_id)):
        raise ValueError('Foreign pair ID')
    for key, limit in (('lr_x', 160), ('lr_y', 160), ('hr_x', 480), ('hr_y', 480)):
        frame[key] = pd.to_numeric(frame[key], errors='raise')
        if not (np.isfinite(frame[key]) & (frame[key] >= 0) & (frame[key] < limit)).all():
            raise ValueError('Invalid landmark coordinates')
    if frame.groupby('pair_id').size().min() < 5 or frame.pair_id.nunique() < args.min_pairs:
        raise ValueError('Need five landmarks per pair and required independent pair count')
    if frame.duplicated(['pair_id', 'lr_x', 'lr_y']).any() or frame.duplicated(['pair_id', 'hr_x', 'hr_y']).any():
        raise ValueError('Repeated coordinates are not independent landmarks')
    cells = (frame.lr_x // 40).astype(int) + 4 * (frame.lr_y // 40).astype(int)
    if frame.assign(spatial_cell=cells).groupby('pair_id').spatial_cell.nunique().min() < 3:
        raise ValueError('Landmarks must cover at least three 40x40 LR spatial cells per pair')
    frame['dx_lr'] = frame.hr_x / 3 - frame.lr_x
    frame['dy_lr'] = frame.hr_y / 3 - frame.lr_y
    frame['distance_lr_pixels'] = np.hypot(frame.dx_lr, frame.dy_lr)
    frame['relative_distance_m'] = frame.distance_lr_pixels * 30
    frame.to_csv(args.output / 'per_landmark.csv', index=False)
    paired = frame.groupby('pair_id').agg(landmarks=('feature_id', 'size'),
        rms_lr_pixels=('distance_lr_pixels', lambda x: np.sqrt(np.mean(x ** 2))),
        median_lr_pixels=('distance_lr_pixels', 'median')).reset_index()
    paired = paired.merge(manifest[['pair_id', 'landscape_class', 'spatial_block_id']], on='pair_id', validate='one_to_one')
    paired.to_csv(args.output / 'per_pair.csv', index=False)
    paired.groupby('landscape_class').agg(pairs=('pair_id', 'size'),
        mean_pair_rms_lr_pixels=('rms_lr_pixels', 'mean')).to_csv(args.output / 'by_class.csv')
    summary = {'annotated_pairs': len(paired), 'annotated_classes': paired.landscape_class.nunique(),
               'all_nine_strata_represented': paired.landscape_class.nunique() == 9,
               'landmarks': len(frame), 'mean_pair_rms_lr_pixels': float(paired.rms_lr_pixels.mean()),
               'annotations_sha256': hashlib.sha256(args.annotations.read_bytes()).hexdigest(),
               'not_surveyed_geodetic_accuracy': True, 'independent_annotation_attestations_verified_externally': False}
    (args.output / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
