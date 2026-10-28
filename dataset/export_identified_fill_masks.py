"""Export separate georeferenced identified-fill masks, never edit release TIFFs.

One means a detected invalid/fill location, zero means no fill identified by
this audit. These are not cloud, saturation or exhaustive quality masks.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import rasterio

from audit_radiometry_reader import read_window
from audit_sentinel_product_metadata import EXPECTED_MANIFEST
from audit_zero_source_followup import cached_item


def export(record, rows, release_root, audit, output):
    row = rows[record['pair_id']]
    sensor = record['sensor']
    key = 'lr_path' if sensor == 'landsat' else 'hr_path'
    path = (release_root / row[key]).resolve()
    path.relative_to(release_root.resolve())
    collection = 'landsat-c2-l2' if sensor == 'landsat' else 'sentinel-2-l2a'
    item, _ = cached_item(collection, record['source_id'], audit)
    with rasterio.open(path) as src:
        crs, transform, size = src.crs, src.transform, src.width
    invalid = np.zeros((size, size), dtype=bool)
    with rasterio.Env(GDAL_DISABLE_READDIR_ON_OPEN='EMPTY_DIR', GDAL_HTTP_TIMEOUT='30', GDAL_HTTP_MAX_RETRY='1'):
        for band in record['bands']:
            if not band['source_invalid_pixels']:
                continue
            raw, mask, _ = read_window(item['assets'][band['asset']], str(crs), transform, size, sensor == 'sentinel')
            if hashlib.sha256(raw.tobytes()).hexdigest() != band['raw_sha256']:
                raise ValueError('Source window changed from audited evidence')
            invalid |= (~mask) | (raw == 0)
    if int(invalid.sum()) != record['invalid_any_checked_band_pixels']:
        raise ValueError('Fill-mask count does not reproduce source audit')
    destination = output / 'masks' / row['split'] / sensor / (row['pair_id'] + '.tif')
    destination.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(destination, 'w', driver='GTiff', width=size, height=size, count=1,
                       dtype='uint8', crs=crs, transform=transform, compress='deflate') as dst:
        dst.write(invalid.astype(np.uint8), 1)
        dst.set_band_description(1, 'identified_source_fill_1_yes_0_not_identified')
        dst.update_tags(pair_id=row['pair_id'], role='audit_only_not_applied_to_original_scores',
                        exclusions='not_cloud_or_saturation_mask')
    return {'pair_id': row['pair_id'], 'sensor': sensor, 'split': row['split'],
            'pixels_identified': int(invalid.sum()), 'path': destination.relative_to(output).as_posix(),
            'sha256': hashlib.sha256(destination.read_bytes()).hexdigest()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('manifest', 'release-root', 'source-audit', 'output'):
        parser.add_argument('--' + key, type=Path, required=True)
    parser.add_argument('--workers', type=int, default=6)
    args = parser.parse_args()
    if hashlib.sha256(args.manifest.read_bytes()).hexdigest() != EXPECTED_MANIFEST:
        raise ValueError('Wrong release')
    status = json.loads((args.source_audit / 'summary.json').read_text())
    if not status['complete'] or status['conversion_mismatch_pixels']:
        raise ValueError('Source audit not complete')
    if args.output.exists():
        raise FileExistsError('Never overwrite a mask export')
    args.output.mkdir(parents=True)
    with args.manifest.open(newline='', encoding='utf-8') as stream:
        rows = {r['pair_id']: r for r in csv.DictReader(stream)}
    records = [json.loads(line) for line in (args.source_audit / 'completed.jsonl').read_text().splitlines()]
    records = [r for r in records if r['invalid_any_checked_band_pixels']]
    exported = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        jobs = [pool.submit(export, r, rows, args.release_root, args.source_audit, args.output) for r in records]
        for future in as_completed(jobs):
            exported.append(future.result())
            if len(exported) % 50 == 0:
                print('Exported', len(exported), '/', len(records), flush=True)
    exported.sort(key=lambda r: (r['pair_id'], r['sensor']))
    summary = {'complete': len(exported) == len(records), 'mask_files': len(exported),
               'distinct_pairs': len({r['pair_id'] for r in exported}), 'manifest_sha256': EXPECTED_MANIFEST,
               'source_checkpoint_sha256': hashlib.sha256((args.source_audit / 'completed.jsonl').read_bytes()).hexdigest(),
               'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
               'original_rasters_and_metric_masks_modified': False,
               'semantics': '1 identified fill; 0 no fill identified; not a claim of all-source validity',
               'files': exported}
    (args.output / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n', encoding='utf-8')
    print('Completed:', len(exported), 'separate georeferenced masks')


if __name__ == '__main__':
    main()
