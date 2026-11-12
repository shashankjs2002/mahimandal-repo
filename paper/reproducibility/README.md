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
seed-42 allocation reproduces all 274 block assignments (3,424 / 645 / 731
pairs). Substituting the fractions recorded in `metadata/protocol.json`
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

The [pinned reproduction repository](https://github.com/shashankjs2002/mahimandal-repo/tree/18bb9f9a78f0f1de0c04241ee8c6918d7b3b0f8b)
and the [pinned notebook commit](https://github.com/shashankjs2002/mahimandal-repo/tree/4b79913ff6605f477b843b91224f86dd3e5854b6)
resolved through GitHub's API without authentication. At the full repository
pin, the manifest, registry, three-seed score directory, and checkpoint index
were visible. The [Kaggle listing](https://www.kaggle.com/datasets/twilight2002/mahimandal-sr-reflectance16/)
and its metadata endpoint resolved without authentication and identify the
release as public version 1. The local Kaggle upload manifest matches the
repository manifest byte for byte. An anonymous download of the full imagery
archive was not tested.

The selected weights and full run histories are present in the local seed-42
benchmark export and seed-43/44 compact export. The public checkpoint index
identifies seed-42 weights; the fit CSV above identifies the selected weights
for all 30 fits. A journal deposit or reviewer link for the private bundles
is still needed. Test that link in a signed-out session before submission.
