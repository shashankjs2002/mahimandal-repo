"""Measure strict holdout feasibility without changing the frozen split.

Connect source IDs, existing block IDs and positive-area footprint overlaps.
Components are indivisible if all three exclusions are required simultaneously.
No new split, buffer claim, training or test evaluation is produced.
"""
import argparse
from collections import defaultdict
from functools import lru_cache
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from rasterio.warp import transform
from shapely.geometry import Polygon
from shapely.strtree import STRtree

from audit_sentinel_product_metadata import EXPECTED_MANIFEST


class Components:
    def __init__(self, size):
        self.parent = list(range(size))

    def find(self, node):
        while self.parent[node] != node:
            self.parent[node] = self.parent[self.parent[node]]
            node = self.parent[node]
        return node

    def join(self, left, right):
        a, b = self.find(left), self.find(right)
        if a != b:
            self.parent[max(a, b)] = min(a, b)


@lru_cache(maxsize=None)
def transformer(crs):
    return lambda x, y: transform(crs, 'EPSG:6933', x, y)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if hashlib.sha256(args.manifest.read_bytes()).hexdigest() != EXPECTED_MANIFEST:
        raise ValueError('Not the frozen release')
    if args.output.exists():
        raise FileExistsError('Choose new output')
    args.output.mkdir(parents=True)
    frame = pd.read_csv(args.manifest)
    components = Components(len(frame))
    for name in ('landsat_id', 'sentinel_id', 'spatial_block_id'):
        for indices in frame.groupby(name).indices.values():
            for index in indices[1:]:
                components.join(int(indices[0]), int(index))
    polygons = []
    for row in frame.itertuples():
        x, y = float(row.ul_x), float(row.ul_y)
        corners = np.array([(x, y), (x + 4800, y), (x + 4800, y - 4800), (x, y - 4800)])
        xx, yy = transformer(row.crs)(corners[:, 0], corners[:, 1])
        polygons.append(Polygon(zip(xx, yy)))
    tree = STRtree(polygons)
    overlap_edges = 0
    for left, polygon in enumerate(polygons):
        for right in tree.query(polygon, predicate='intersects'):
            right = int(right)
            if right > left and polygon.intersection(polygons[right]).area > 1:
                components.join(left, right)
                overlap_edges += 1
    groups = defaultdict(list)
    for index in range(len(frame)):
        groups[components.find(index)].append(index)
    labels = {root: f'component_{rank:03d}' for rank, root in enumerate(sorted(groups))}
    frame['strict_component'] = [labels[components.find(index)] for index in range(len(frame))]
    membership = frame[['pair_id', 'split', 'landscape_class', 'landsat_id', 'sentinel_id', 'spatial_block_id', 'strict_component']]
    membership.to_csv(args.output / 'component_membership.csv', index=False)
    sizes = membership.groupby('strict_component').size().sort_values(ascending=False)
    sizes.rename('pairs').to_csv(args.output / 'component_sizes.csv')
    support = pd.crosstab(membership.landscape_class, membership.strict_component)
    support.to_csv(args.output / 'class_component_counts.csv')
    class_support = (support > 0).sum(1)
    impossible = class_support[class_support < 3].to_dict()
    summary = {'manifest_sha256': EXPECTED_MANIFEST,
               'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
               'pairs': len(frame), 'indivisible_components': len(groups),
               'largest_component_pairs': int(sizes.iloc[0]),
               'positive_area_overlap_edges': overlap_edges,
               'class_distinct_components': class_support.to_dict(),
               'all_classes_in_three_splits_necessary_condition': not impossible and len(groups) >= 3,
               'classes_failing_necessary_condition': impossible,
               'split_assignments_modified': False,
               'note': 'Necessary condition only. No allocation, source-disjoint benchmark or nonzero spatial buffer is established. Footprints use manifest UL/CRS and nominal 4800-m square geometry.'}
    (args.output / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
