# MahimandalSR paper

`MahimandalSR.pdf` is the current dataset paper with appendices included.
It describes the reflectance16 release, three-seed fixed-baseline benchmark,
OLI2MSI transfer, paired geographic summaries and data-quality audits.

The manuscript's source-tree paths map to this review repository as follows:

| Manuscript path | Repository path |
| --- | --- |
| `scripts/analyze_paired_blocks.py` | `dataset/analyze_paired_blocks.py` |
| `scripts/verify_companion_masks.py` | `dataset/verify_companion_masks.py` |
| `generated/multiseed_review/` score CSVs | `results/benchmark_three_seed/` |
| `revision_cpu_only/statistical_analysis/` | `results/cpu_support/statistical_analysis/` |
| `revision_cpu_only/verification/` | `results/cpu_support/verification/` |

No NAS architecture results are included in this dataset paper.
