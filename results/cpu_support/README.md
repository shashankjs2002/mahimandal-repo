# Supporting CPU results

These reports supplement the published 731-pair test benchmark. The original
images, 3,424/645/731 partition, 274 blocks, checkpoints and primary scores are
preserved.

- `statistical_analysis/`: paired RDN-bicubic and RDN-EDSR RGB differences,
  RDN-bicubic and RDN-CAFRN four-band differences, equal-block means,
  conditional geographic intervals and leave-one-block-out ranges.
- `verification/`: inventory, grid and binary-value checks for 410 existing
  identified-fill masks covering 404 pairs; target-grid union counts and
  landscape breakdowns accompany the native-grid counts.
- `../benchmark_three_seed/`: archived test scores, means and training receipts
  for the three-seed analysis (30 learned fits plus six deterministic bicubic rows).
- `../audits/identified_fill_masks_v1/`: the separate masks and original inventory.

From the repository root:

```bash
python dataset/analyze_paired_blocks.py
python dataset/verify_companion_masks.py --release data/reflectance16
```

The statistical analysis checks 324 archived run/metric means, averages paired
differences across seeds 42/43/44 per image, and resamples 13 whole test blocks
2,000 times with bootstrap seed 42. Intervals describe geographic sensitivity
conditional on the fitted models; shared source products remain a dependence.
Equal-block weighting is a complementary summary, while primary benchmark
tables retain equal image weighting.

A fill-mask value of 1 identifies fill recovered by the original audit; 0 means
no fill was identified. These are not exhaustive cloud/shadow or source-validity
masks, and they are not applied to the archived scores. All 410 mask hashes and
release grids reproduced, with 148 Landsat and 4,540 Sentinel native-grid
locations and 80 affected test pairs.

The registration audit remains automated: its inspection forms require human
review before human-validated correspondence statistics can be reported.
The two-pair all-band source reconstruction uses Landsat 8. Shared published
Landsat 8/9 scaling supports the conversion rule, but additional Landsat-9
all-band source reconstruction has not been added. No new training or inference
is represented by these reports.
