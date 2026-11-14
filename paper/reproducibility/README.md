# MahimandalSR reproducibility evidence

These checks use the frozen reflectance16 manifest and saved benchmark
artifacts. They leave the release, split, scores, and weights unchanged.

| File | Evidence |
| --- | --- |
| [release_configuration.json](release_configuration.json) | Frozen manifest digest and counts; notebook and protocol-metadata split replay; recorded versus notebook-default collection caps. |
| [model_configurations.json](model_configurations.json) | Recorded constructor settings, upstream commits and entry paths, channels, and parameter counts for all ten learned model/modality configurations. |
| [verified_fits.csv](verified_fits.csv) | All 30 fit receipts and selected-checkpoint digests, checked against the locally archived run artifacts. |
| [rdn_rgb_seed43_training.csv](rdn_rgb_seed43_training.csv) | Five archived epochs used for the manuscript's RGB RDN learning curves. |
| [rdn_rgb_seed43_learning_curves.png](rdn_rgb_seed43_learning_curves.png) | Training L1 and validation PSNR plotted from the archived epoch log. |

## Which settings govern reproduction?

The published manifest is authoritative for train, validation, and test
membership. Starting with its retained pairs, the preparation notebook's
seed-42 allocation with 78/11/11 targets reproduces all 274 block assignments
(3,424 / 645 / 731 pairs). Substituting the historical 80/10/10 targets
recorded in `metadata/protocol.json`
changes two blocks containing 84 pairs. The JSON records both settings
without modifying the historical metadata or the release manifest.

The collector defaults and metadata disagree on some class caps. For example,
the metadata lists 700 for mountain and urban, while the notebook defaults
and the released counts for those classes are 800. A replay from the retained
manifest establishes the split, but cannot establish every historical
collection cap or resume setting.

The archived training protocols agree across seeds for each constructor and
recorded parameter count. Their upstream revisions and source modules match
the benchmark registry. This check uses receipts and checkpoint files rather
than instantiating, running, or retraining the models.

## Access check and reviewer handoff

The [manuscript's pinned reproduction repository](https://github.com/shashankjs2002/mahimandal-repo/tree/4f2d5ef1adef89517f071d97b4b14b73549de0f1)
and the [pinned notebook commit](https://github.com/shashankjs2002/mahimandal-repo/tree/4b79913ff6605f477b843b91224f86dd3e5854b6)
resolved through GitHub's API without authentication. At the full repository
pin, the manifest, registry, three-seed score directory, and checkpoint index
were visible. The [Kaggle listing](https://www.kaggle.com/datasets/twilight2002/mahimandal-sr-reflectance16/)
and its metadata endpoint resolved without authentication and identify the
release as public version 1. The manifest from the local frozen release
matches the published SHA-256 checksum. The pinned Git copy has the same CSV
records with LF line endings; use the frozen release copy for an exact byte
check. An anonymous download of the full imagery archive was not tested.

The selected weights and full run histories are present in the local seed-42
benchmark export and seed-43/44 compact export. The public checkpoint index
identifies seed-42 weights; the fit CSV above identifies the selected weights
for all 30 fits. A journal deposit or reviewer link for the private bundles
is still needed. Test that link in a signed-out session before submission.

## What the verification covers

The pinned analysis replay checked 324 archived run-level metric means,
all four paired PSNR comparisons and all 52 leave-one-block-out results.
A separate CPU run recomputed bicubic PSNR from all 731 released test pairs
in both output profiles: 35.3775 dB (RGB) and 33.8920 dB (RGB+NIR). All
410 companion masks passed the release-grid check. The locally archived
selected checkpoint files matched all 30 SHA-256 values in
`verified_fits.csv`. Matching their hashes verifies file identity, not
learned-model predictions. These checks did not rerun training, learned-model
inference or catalogue collection.

The [FeNet authors' repository](https://github.com/wangzheyuan-666/FeNet)
identifies the implementation as Wang et al.'s 2022 Feature Enhancement
Network (DOI: 10.1109/TGRS.2022.3168787). The benchmark receipt records the
pinned revision, source module and threefold constructor; the audit did not
re-execute that model.
