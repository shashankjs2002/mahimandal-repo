# MahimandalSR

Code and saved results for the MahimandalSR dataset paper: paired Landsat
30 m and Sentinel-2 10 m images from Bharat, with RGB and near-infrared bands.

The frozen release has 4,800 pairs: 3,424 train, 645 validation and 731 test.
Inputs are 160 x 160; targets are 480 x 480. Model tensors use RGB or RGBNIR;
the TIFF band descriptions determine the conversion from stored BGRNIR order.
Integer reflectance is divided by 10,000 and clipped to [0, 1].

## Contents

```
notebooks/       Data collection, benchmark and dataset checks
benchmark/       Model adapters, training, evaluation and metrics
dataset/         Release export and dataset-quality checks
metadata/        Frozen manifest, protocol, dataset card and image checksums
results/         Saved benchmark, transfer and dataset-audit results
paper/           Manuscript and supplement PDFs
weights/         Index of the separately supplied checkpoints
verify.py        File integrity and saved-score checks
```

This repository concerns the dataset paper only. It does not contain the NAS
model paper, architecture search, internal review correspondence or notebook
generation tools. TIFF imagery, model weights and third-party model source
repositories are distributed separately; they are not embedded in this ZIP.

## Check the reported results

Python 3.10+ is recommended. No GPU or additional Python packages are needed:

```bash
python verify.py
```

This checks the distributed files against `checksums.sha256`, verifies the frozen
split, and recomputes benchmark means from the saved per-image CSVs. It does not
run inference or claim independent reproduction of the predictions.

## Rerun the benchmark

Use a project environment. Install PyTorch/torchvision appropriate for your
machine, then install `requirements.txt`. Do not change a shared machine's
global environment. The historical runs used an A100 and PyTorch 2.1.0a0;
`environment.json` in each saved run records the actual environment.

Put the released imagery in `data/reflectance16/`, including its original
`paper_release_manifest.csv` and `protocol.json`. Both the original
`landsat/`, `sentinel/` layout and the release's split-first LR/HR paths are
handled by the manifest reader. Do not redraw the split.

Open `notebooks/02_benchmark.ipynb`, or run from this folder:

```bash
python benchmark/run.py fetch --sources models/upstream
python benchmark/run.py smoke --model RDN --modality rgb --sources models/upstream --output runs --device cpu --backward
python benchmark/run.py train --model RDN --dataset data/reflectance16 --sources models/upstream --output runs --radiometry reflectance16 --modality rgb --epochs 50 --patience 3 --batch-size 1 --train-lr-crop 160 --lr 0.0001 --seed 42 --device cuda
python benchmark/run.py evaluate --model RDN --dataset data/reflectance16 --sources models/upstream --output runs --radiometry reflectance16 --modality rgb --train-lr-crop 160 --seed 42 --split test --border-crop 3 --device cuda
```

Use only a GPU assigned to you. `fetch` downloads the pinned upstream repositories;
the other steps do not fetch data. All paper runs use the explicit
`--train-lr-crop 160` flag. The unchanged historical harness also supports smaller
crops; its legacy CLI default is not the setting used in the paper.

The notebook separates training, validation and test execution. Choose the best
validation checkpoint before enabling test. Use a new `runs/` directory for new
experiments, not `results/`. Historical checkpoint evaluation requires matching
the archived protocol and source hashes; do not remove those checks to force a
checkpoint to load. `weights/checkpoints.csv` identifies the archived files.

RGB results cover EDSR, RCAN, RDN, CAFRN, DCM, FENet and the paired x3 SRVGG
adaptation, plus bicubic. Four-band results cover registry-supported RDN, CAFRN
and SRVGG, plus bicubic. The registry retains the original candidate implementations;
being listed there does not mean a model completed the paper benchmark.
SRVGG did not reach its adversarial stage in the supplied runs. DCM is the HSENet
comparison implementation; the FENet acronym match to Land2Sent is unverified.

## Data preparation and checks

`notebooks/01_prepare_dataset.ipynb` contains the collection workflow using
Microsoft Planetary Computer. It requires internet access and the geospatial
dependencies in `requirements.txt`. It can collect new pairs, but changed remote
catalogues or rerunning with different date windows can produce a different
release. Use the supplied frozen manifest and imagery for paper reproduction.
The original auxiliary-raster collection receipts are incomplete; current queries
do not recreate that historical provenance.

`notebooks/03_check_dataset.ipynb` runs local manifest and source-overlap checks.
The scripts in `dataset/` additionally cover source masks, radiometry, registration
diagnostics and source-fill sensitivity. Run an individual script with `--help`
for its required inputs. Network audits can be slow; they are not run by `verify.py`.

The stored split is block-held-out, not acquisition-disjoint or spatially buffered.
Some source products and boundary footprints overlap across splits. Sensitivity
analyses in `results/audits/` supplement the original 731-pair evaluation; they do
not replace its split or metrics. Automatic registration proxies are not surveyed
ground-control accuracy. Separate fill masks were not applied to the original fits.

`benchmark/transfer.py` contains the fixed-baseline OLI2MSI transfer evaluator;
its CLI is `python benchmark/transfer.py --help`. OLI2MSI has RGB float32 inputs
and publisher TOA provenance, unlike Mahimandal surface reflectance. The supplied
transfer results are descriptive, not evidence of radiometric harmonization.

## Source and data terms

`metadata/file_sources.csv` records the source and SHA-256 of copied files.
Benchmark training/metric code is unchanged; the transfer copy removes the
unrelated NAS branches. Preparation notebook outputs and its optional web app
were removed, not its collection logic. Saved scores and historical protocols
are kept as recorded. Paths inside historical receipts describe the original
run; the setup instructions above use relative paths.

See `LICENSES.md` before redistribution. This repository does not grant new
rights over satellite imagery, auxiliary products, upstream code or checkpoints.

