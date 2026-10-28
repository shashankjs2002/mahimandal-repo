"""Resume-safe original DN/mask adjudication of stored-zero candidate rasters.

Reads only original band windows with zeros in the frozen release. Source fill
and valid clipped-dark pixels are counted separately, never changing imagery or
scoring masks. This does not assess clouds, saturation or geodetic accuracy.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
import hashlib
import json
from pathlib import Path
import shutil
from threading import Lock
import xml.etree.ElementTree as ET

import numpy as np
import rasterio

from audit_radiometry_reader import BANDS, decode, read_window, source_item
from audit_sentinel_product_metadata import EXPECTED_MANIFEST


CACHE_LOCK = Lock()


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def adjudicate(raw, valid_mask, released, sensor):
    expected = decode(raw, sensor)
    valid = valid_mask & (raw != 0)
    zeros = released == 0
    return {"release_zero_pixels": int(zeros.sum()),
            "source_invalid_pixels": int((~valid).sum()),
            "zero_from_source_invalid_pixels": int((zeros & ~valid).sum()),
            "zero_from_valid_clipping_pixels": int((zeros & valid & (expected == 0)).sum()),
            "unexplained_zero_pixels": int((zeros & valid & (expected != 0)).sum()),
            "conversion_mismatch_pixels": int((expected != released).sum())}


def cached_item(collection, identifier, output):
    key = hashlib.sha256((collection + '/' + identifier).encode()).hexdigest()[:24]
    path = output / 'source_metadata' / (key + '.json')
    # Metadata only; serialize first fetch to avoid duplicate writers.
    with CACHE_LOCK:
        if path.is_file():
            item = json.loads(path.read_text(encoding='utf-8'))
        else:
            item, content = source_item(collection, identifier)
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary = path.with_suffix('.tmp')
            temporary.write_bytes(content)
            temporary.replace(path)
    if item['id'] != identifier:
        raise ValueError('Cached source identity mismatch')
    return item, digest(path)


def inspect(candidate, manifest, release_root, output, xml_cache=None):
    row = manifest[candidate['pair_id']]
    sensor = candidate['sensor']
    collection, id_key, path_key, size = (
        ('landsat-c2-l2', 'landsat_id', 'lr_path', 160) if sensor == 'landsat'
        else ('sentinel-2-l2a', 'sentinel_id', 'hr_path', 480))
    item, metadata_hash = cached_item(collection, row[id_key], output)
    if sensor == 'sentinel' and float(item['properties']['s2:processing_baseline']) < 4:
        raise ValueError('Unexpected pre-offset baseline')
    xml_digest = None
    if sensor == 'sentinel':
        if xml_cache is None:
            raise ValueError('Original product XML cache required')
        xml_path = xml_cache / hashlib.sha256(row[id_key].encode()).hexdigest()[:24] / 'product.xml'
        xml = ET.fromstring(xml_path.read_bytes())
        specials = {}
        for node in xml.iter():
            children = {child.tag.rsplit('}', 1)[-1]: child.text for child in node}
            if 'SPECIAL_VALUE_TEXT' in children and 'SPECIAL_VALUE_INDEX' in children:
                specials[children['SPECIAL_VALUE_TEXT']] = int(children['SPECIAL_VALUE_INDEX'])
        if specials.get('NODATA') != 0:
            raise ValueError('Original XML does not confirm DN zero as nodata')
        xml_digest = digest(xml_path)
    path = (release_root / row[path_key]).resolve()
    path.relative_to(release_root.resolve())
    with rasterio.open(path) as src:
        values, crs, transform = src.read(), str(src.crs), src.transform
    if values.shape != (4, size, size):
        raise ValueError('Wrong release geometry')
    bands = []
    union_invalid = np.zeros((size, size), dtype=bool)
    union_zero = (values == 0).any(0)
    with rasterio.Env(GDAL_DISABLE_READDIR_ON_OPEN='EMPTY_DIR', GDAL_HTTP_TIMEOUT='30',
                      GDAL_HTTP_MAX_RETRY='1', GDAL_HTTP_RETRY_DELAY='1'):
        for index, names in enumerate(BANDS):
            count = int((values[index] == 0).sum())
            if count != int(candidate[f'zero_{("blue", "green", "red", "nir")[index]}_pixels']):
                raise ValueError('Candidate inventory changed')
            if not count:
                continue
            asset = names[0 if sensor == 'landsat' else 1]
            raw, mask, header = read_window(item['assets'][asset], crs, transform, size, sensor == 'sentinel')
            if header['nodata'] != 0 and not (sensor == 'sentinel' and header['nodata'] is None and xml_digest):
                raise ValueError('Nodata convention differs: broaden audit before closure')
            invalid = (~mask) | (raw == 0)
            union_invalid |= invalid
            bands.append({'asset': asset, **adjudicate(raw, mask, values[index], sensor),
                          'source_header': header,
                          'raw_sha256': hashlib.sha256(raw.tobytes()).hexdigest(),
                          'valid_mask_sha256': hashlib.sha256(mask.tobytes()).hexdigest()})
    return {'pair_id': row['pair_id'], 'sensor': sensor, 'split': row['split'],
            'landscape_class': row['landscape_class'], 'source_id': row[id_key],
            'source_metadata_sha256': metadata_hash, 'release_sha256': digest(path),
            'sentinel_nodata_xml_sha256': xml_digest,
            'release_zero_any_band_pixels': int(union_zero.sum()),
            'invalid_any_checked_band_pixels': int(union_invalid.sum()),
            'conversion_mismatch_pixels': sum(b['conversion_mismatch_pixels'] for b in bands),
            'bands': bands}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', required=True, type=Path)
    parser.add_argument('--release-root', required=True, type=Path)
    parser.add_argument('--candidates', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--workers', type=int, default=6)
    parser.add_argument('--xml-cache', type=Path, required=True)
    parser.add_argument('--import-prior', type=Path)
    args = parser.parse_args()
    if not 1 <= args.workers <= 12:
        raise ValueError('Use 1-12 I/O workers')
    if digest(args.manifest) != EXPECTED_MANIFEST:
        raise ValueError('Not the frozen release')
    with args.manifest.open(newline='', encoding='utf-8') as stream:
        manifest = {r['pair_id']: r for r in csv.DictReader(stream)}
    with args.candidates.open(newline='', encoding='utf-8') as stream:
        candidates = list(csv.DictReader(stream))
    expected = {(r['pair_id'], r['sensor']) for r in candidates}
    if len(expected) != len(candidates) or any(r['pair_id'] not in manifest or
            r['sensor'] not in ('landsat', 'sentinel') for r in candidates):
        raise ValueError('Duplicate/foreign candidate identities')
    args.output.mkdir(parents=True, exist_ok=True)
    protocol = {'script_sha256': digest(Path(__file__)), 'manifest_sha256': EXPECTED_MANIFEST,
                'candidate_sha256': digest(args.candidates), 'candidate_rasters': len(candidates),
                'imported_checkpoint_sha256': digest(args.import_prior / 'completed.jsonl') if args.import_prior else None,
                'scope': 'Original nodata/read-mask and raw-zero checks for bands containing stored zeros; not cloud/saturation masks.'}
    protocol_path = args.output / 'protocol.json'
    if protocol_path.exists() and json.loads(protocol_path.read_text()) != protocol:
        raise ValueError('Resume protocol changed; choose a new output')
    protocol_path.write_text(json.dumps(protocol, indent=2) + '\n', encoding='utf-8')
    checkpoint = args.output / 'completed.jsonl'
    if args.import_prior and not checkpoint.exists():
        prior_protocol = json.loads((args.import_prior / 'protocol.json').read_text())
        if any(prior_protocol[k] != protocol[k] for k in ('manifest_sha256', 'candidate_sha256', 'candidate_rasters')):
            raise ValueError('Prior evidence belongs to another cohort')
        shutil.copyfile(args.import_prior / 'completed.jsonl', checkpoint)
        shutil.copytree(args.import_prior / 'source_metadata', args.output / 'source_metadata', dirs_exist_ok=True)
    completed = {}
    if checkpoint.exists():
        for line in checkpoint.read_text(encoding='utf-8').splitlines():
            result = json.loads(line)
            key = result['pair_id'], result['sensor']
            if key not in expected or key in completed:
                raise ValueError('Foreign/duplicate resume record')
            completed[key] = result
    failures = []
    remaining = [r for r in candidates if (r['pair_id'], r['sensor']) not in completed]
    with checkpoint.open('a', encoding='utf-8') as stream, ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(inspect, r, manifest, args.release_root, args.output, args.xml_cache): r for r in remaining}
        for index, future in enumerate(as_completed(futures)):
            row = futures[future]
            try:
                result = future.result()
                completed[result['pair_id'], result['sensor']] = result
                stream.write(json.dumps(result, allow_nan=False) + '\n')
                stream.flush()
            except Exception as exc:
                failures.append({'pair_id': row['pair_id'], 'sensor': row['sensor'],
                                 'failure_type': type(exc).__name__})
            if (index + 1) % 25 == 0:
                print(f'Checked {len(completed)}/{len(expected)} rasters; failures this attempt {len(failures)}', flush=True)
    results = sorted(completed.values(), key=lambda r: (r['pair_id'], r['sensor']))
    summary = {**protocol, 'completed_rasters': len(results), 'complete': len(results) == len(expected),
               'failed_this_attempt': failures,
               'pairs_with_source_invalid_pixels': len({r['pair_id'] for r in results if r['invalid_any_checked_band_pixels']}),
               'conversion_mismatch_pixels': sum(r['conversion_mismatch_pixels'] for r in results),
               'by_sensor': {}, 'release_or_scoring_masks_changed': False,
               'unchecked_scope': 'Bands/rasters without stored zeros; saturation, QA/SCL flags and original invalid contributor weights during interpolation.'}
    for sensor in ('landsat', 'sentinel'):
        selected = [r for r in results if r['sensor'] == sensor]
        summary['by_sensor'][sensor] = {'checked_rasters': len(selected),
            'rasters_with_source_invalid_pixels': sum(r['invalid_any_checked_band_pixels'] > 0 for r in selected),
            'invalid_any_checked_band_pixels': sum(r['invalid_any_checked_band_pixels'] for r in selected),
            'zero_from_valid_clipping_band_values': sum(b['zero_from_valid_clipping_pixels'] for r in selected for b in r['bands']),
            'zero_from_source_invalid_band_values': sum(b['zero_from_source_invalid_pixels'] for r in selected for b in r['bands'])}
    (args.output / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n', encoding='utf-8')
    (args.output / 'failures.json').write_text(json.dumps(failures, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({k:v for k,v in summary.items() if k != 'failed_this_attempt'}, indent=2))
    if failures or summary['conversion_mismatch_pixels']:
        raise SystemExit('Incomplete or mismatching source evidence; inspect and resume, do not declare closure')


if __name__ == '__main__':
    main()
