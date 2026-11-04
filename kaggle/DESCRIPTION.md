# MahimandalSR: Landsat-Sentinel-2 RGB-NIR pairs across Bharat

MahimandalSR contains 4,800 georeferenced Landsat 8/9 and Sentinel-2
surface-reflectance image pairs for 3x super-resolution. Each pair covers the
same nominal 4.8 km square: a 160 x 160 Landsat input at 30 m and a 480 x 480
Sentinel-2 reference at 10 m.

All pairs share a UTC acquisition date. The mean acquisition gap is 8.895
minutes and the maximum is 19.8 minutes. Mean joint patch-level cloud/shadow
obstruction is 0.123%, with all pairs passing the stated below-1% screening rule.

## Files and reflectance

The release contains 9,600 GeoTIFFs organized under train, val and test, with
LR and HR folders. Four bands are stored as blue, green, red and near-infrared.
Signed 16-bit values encode surface reflectance multiplied by 10,000. Decode by
dividing by 10,000 and clipping to [0, 1]. Reorder bands to RGB or RGB-NIR for
model inputs. Display stretches and gamma are for visualization; metrics use
linear reflectance.

## Published split

| Split | Pairs | Share | 100-km blocks |
| --- | ---: | ---: | ---: |
| Train | 3,424 | 71.33% | 247 |
| Validation | 645 | 13.44% | 14 |
| Test | 731 | 15.23% | 13 |

The 274 whole blocks are assigned to one split each, balancing landscape and
regional representation with a minimum of 11% of pairs in each holdout.
All nine landscape strata occur in all three splits. Some source products recur
across partitions, and 22 test footprints intersect another partition. Audits
and overlap-excluded sensitivities document these properties.

The scientific release is identified by `paper_release_manifest.csv`, SHA-256:
`6bf97403ee40ccf0cbd38b2e6e4932e01973e5d00e211e0acfb1e0b810efc5e7`.

## Landscape and quality metadata

Esri 2025 annual land-cover percentages and Copernicus DEM GLO-30 descriptors
define nine sampling strata: cropland, urban, mountain, shrub/grass, forest,
alpine, mixed, water/wetland and desert/bare. The collection includes 828
western-Himalayan priority-phase pairs. Class and seasonal coverage are uneven;
these labels describe patch-level sampling conditions.

## Training and evaluation support

The benchmark uses full 160 x 160 LR inputs and 480 x 480 references for
training, validation and testing. The L1 loss mask intersects the TIFF raster
masks, expanding the LR mask 3x and sharing it across output channels. The
released masks are all-valid, giving ordinary full-frame L1. QA/SCL masks
screened pairs during collection. Separate identified-fill masks document
recovered source fill and are not applied to training or the published scores.
Test metrics use a three-pixel HR border exclusion and linear reflectance.

The paper includes 30 learned baseline fits across seeds 42/43/44, classwise
metrics, source and alignment audits, paired model comparisons, equal-block and
leave-one-block-out summaries, and separate seed-42 OLI2MSI transfer experiments.
Supporting CPU analyses preserve the released dataset and primary benchmark.

## Code, paper and evidence

- Code: https://github.com/shashankjs2002/mahimandal-repo
- Paper with appendices: https://github.com/shashankjs2002/mahimandal-repo/blob/main/paper/MahimandalSR.pdf
- Three-seed scores: `results/benchmark_three_seed/` in the code repository.
- CPU reports: `results/cpu_support/` in the code repository.
- Companion identified-fill masks: `results/audits/identified_fill_masks_v1/`.
- Checkpoint inventory: `weights/checkpoints.csv`. Model states and the
  full-evidence bundle will accompany the confidential journal-review submission.

Original curation and metadata are licensed under CC BY 4.0. Landsat,
Sentinel-2, Copernicus and auxiliary products retain their applicable source terms;
the repository records source attributions and model licences.
