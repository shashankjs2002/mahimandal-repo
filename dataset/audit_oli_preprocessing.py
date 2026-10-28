"""Audit all 100 local OLI2MSI test pairs and pin published preparation code.

No model execution or radiometric harmonization. TIFF interpretation tags and
publisher code support RGB/TOA provenance but do not authenticate repackaged
pixels against a checksum-signed publisher archive.
"""
import argparse
import ast
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import rasterio

from audit_sentinel_product_metadata import fetch


def digest(data):
    return hashlib.sha256(data).hexdigest()


def published_contract(extraction, utilities):
    tree = ast.parse(extraction)
    bands = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Tuple):
            if any(isinstance(t, ast.Subscript) and isinstance(t.slice, ast.Constant)
                   and t.slice.value == 'bands' for t in node.targets):
                bands.append(ast.literal_eval(node.value))
    if (4, 3, 2) not in bands:
        raise ValueError('Publisher RGB band selection not established')
    utils = ast.parse(utilities)
    extractor_calls = []
    for cls in utils.body:
        if isinstance(cls, ast.ClassDef) and cls.name == 'Extractor':
            for node in ast.walk(cls):
                if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == 'get_TOA':
                    extractor_calls.append(ast.get_source_segment(utilities, node))
    if len(extractor_calls) < 2:
        raise ValueError('Both publisher TOA reader calls not established')
    return {'publisher_band_selection': [4, 3, 2], 'extractor_calls': extractor_calls,
            'publisher_radiometry': 'top_of_atmosphere_reflectance',
            'not_claimed': 'Same atmospheric correction or surface-reflectance harmonization as Mahimandal'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', required=True, type=Path)
    parser.add_argument('--transfer-scores', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--publisher-cache', type=Path)
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError('Choose new output')
    args.output.mkdir(parents=True, exist_ok=True)
    commit_bytes = ((args.publisher_cache / 'publisher_commit.json').read_bytes() if args.publisher_cache
                    else fetch('https://api.github.com/repos/wjwjww/OLI2MSI/commits/main'))
    commit = json.loads(commit_bytes)['sha']
    (args.output / 'publisher_commit.json').write_bytes(commit_bytes)
    sources = {}
    for name in ('extract_subimgs_from_granules.py', 'utils.py'):
        content = ((args.publisher_cache / ('publisher_' + name)).read_bytes() if args.publisher_cache
                   else fetch(f'https://raw.githubusercontent.com/wjwjww/OLI2MSI/{commit}/{name}'))
        (args.output / ('publisher_' + name)).write_bytes(content)
        sources[name] = content.decode('utf-8-sig')
    contract = published_contract(sources['extract_subimgs_from_granules.py'], sources['utils.py'])
    paths = {}
    for side in ('lr', 'hr'):
        files = [p for p in (args.dataset / ('test_' + side)).iterdir() if p.suffix.casefold() in ('.tif', '.tiff')]
        paths[side] = {p.stem.casefold(): p for p in files}
        if len(files) != len(paths[side]):
            raise ValueError('Duplicate OLI IDs')
    if len(paths['lr']) != 100 or paths['lr'].keys() != paths['hr'].keys():
        raise ValueError('Expected paired 100-image OLI test release')
    with args.transfer_scores.open(newline='', encoding='utf-8') as stream:
        transfer_ids = {r['pair_id'] for r in csv.DictReader(stream) if r['kind'] == 'benchmark'
                        and r['radiometry'] == 'reflectance16' and r['tested_on'] == 'oli2msi'}
    if transfer_ids != paths['lr'].keys():
        raise ValueError('Local test cohort differs from archived transfer cohort')
    results, geometry = [], []
    for pair_id in sorted(paths['lr']):
        pair_headers = []
        for side, size in (('lr', 160), ('hr', 480)):
            path = paths[side][pair_id]
            with rasterio.open(path) as src:
                values = src.read()
                if values.shape != (3, size, size) or values.dtype != np.dtype('float32'):
                    raise ValueError('Unexpected OLI geometry/storage')
                colors = [color.name for color in src.colorinterp]
                if colors != ['red', 'green', 'blue']:
                    raise ValueError('Non-RGB interpretation tags; investigate before closure')
                finite = np.isfinite(values)
                if not finite.all():
                    raise ValueError('Nonfinite OLI test values')
                header = {'pair_id': pair_id, 'side': side, 'file_sha256': digest(path.read_bytes()),
                          'colorinterp': colors, 'descriptions': list(src.descriptions),
                          'crs': str(src.crs), 'bounds': list(src.bounds),
                          'scales': list(src.scales), 'offsets': list(src.offsets),
                          'minimum': float(values.min()), 'maximum': float(values.max()),
                          'values_clipped_below_zero': int((values < 0).sum()),
                          'values_clipped_above_one': int((values > 1).sum()),
                          'masked_pixels_any_band': int((src.read_masks() == 0).any(0).sum())}
                pair_headers.append(header)
                results.append(header)
        same_crs = pair_headers[0]['crs'] == pair_headers[1]['crs']
        difference = np.asarray(pair_headers[0]['bounds']) - np.asarray(pair_headers[1]['bounds'])
        geometry.append({'pair_id': pair_id, 'same_crs': same_crs,
                         'equal_bounds_at_0_01_crs_units': bool(same_crs and (np.abs(difference) <= .01).all()),
                         'lr_minus_hr_bounds_crs_units': difference.tolist() if same_crs else None,
                         'note': 'Header-coordinate difference, not an image-registration residual; no rewarping performed.'})
    summary = {'script_sha256': digest(Path(__file__).read_bytes()), 'publisher_commit': commit,
               'publisher_url': f'https://github.com/wjwjww/OLI2MSI/tree/{commit}',
               'publisher_contract': contract, 'test_pairs': 100, 'test_files': len(results),
               'all_files_rgb_interpretation': True, 'cohort_matches_archived_transfer': True,
               'below_zero_values': sum(r['values_clipped_below_zero'] for r in results),
               'above_one_values': sum(r['values_clipped_above_one'] for r in results),
               'pairs_with_different_crs': sum(not r['same_crs'] for r in geometry),
               'pairs_with_unequal_header_bounds': sum(not r['equal_bounds_at_0_01_crs_units'] for r in geometry),
               'source_hashes': {name: digest((args.output / ('publisher_' + name)).read_bytes()) for name in sources},
               'transfer_scores_sha256': digest(args.transfer_scores.read_bytes()),
               'no_harmonization_no_model_execution': True,
               'limitation': 'Publisher code and local RGB tags support interpretation, not independent atmospheric harmonization or authenticated pixel lineage of the repackaged archive.'}
    for name, payload in (('per_file.json', results), ('pair_geometry.json', geometry), ('summary.json', summary)):
        (args.output / name).write_text(json.dumps(payload, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
