# MahimandalSR

Dataset preparation and benchmark code for Landsat-to-Sentinel-2 super-resolution.
The published dataset has 4,800 pairs: 3,424 train, 645 validation and 731 test.
Inputs are 160 x 160 and targets are 480 x 480, using RGB or RGB+NIR.

Dataset: https://www.kaggle.com/datasets/twilight2002/mahimandal-sr-reflectance16/

[Paper with appendices](paper/MahimandalSR.pdf). The release is identified by
`metadata/paper_release_manifest.csv` (4,800 IDs with their original splits).
Its SHA-256 is
`6bf97403ee40ccf0cbd38b2e6e4932e01973e5d00e211e0acfb1e0b810efc5e7`.

## Setup

Use Python 3.10+ in a project environment. Install PyTorch and torchvision for
your CPU/CUDA version, then:

```bash
pip install -r requirements.txt
```

Put the released dataset in `data/reflectance16/`, including its original
`paper_release_manifest.csv` and `protocol.json`. Images and weights are supplied
separately. `metadata/` contains the published manifest; do not change its split.

## Run

```bash
python benchmark/run.py fetch --sources models/upstream
python benchmark/run.py train --model RDN --dataset data/reflectance16 --sources models/upstream --output runs --modality rgb --epochs 50 --patience 3 --batch-size 1 --train-lr-crop 160 --seed 42 --device cuda
python benchmark/run.py evaluate --model RDN --dataset data/reflectance16 --sources models/upstream --output runs --modality rgb --train-lr-crop 160 --split val --seed 42 --device cuda
python benchmark/run.py evaluate --model RDN --dataset data/reflectance16 --sources models/upstream --output runs --modality rgb --train-lr-crop 160 --split test --seed 42 --device cuda
```

Use `--device cpu` without a GPU. On shared machines, use only your assigned GPU.
Change `--model` to select another implementation; `benchmark/registry.json`
lists supported modalities and pinned upstream revisions. Use `--modality rgbnir`
for supported four-band models. Bicubic needs evaluation only, not training.

Paper training uses full 160 x 160 inputs, Adam at 1e-4 and validation-PSNR selection.
Test uses full160 inputs and a three-pixel HR border crop. Evaluation always
recomputes scores, so replacing `best.pt` cannot silently reuse old metrics.
Resume preserves optimizer and RNG states. Use a new output folder if settings
change; choose checkpoints on validation before opening test results.

## Files

- `notebooks/01_prepare_dataset.ipynb`: Planetary Computer collection workflow.
- `notebooks/02_benchmark.ipynb`: the same benchmark commands in notebook form.
- `dataset/`: optional dataset-quality analyses used in the paper.
- `benchmark/transfer.py`: fixed-baseline OLI2MSI transfer; run with `--help`.
- `results/`: original reported scores, settings, histories and audit records.
- `paper/`: manuscript and supplement; `weights/`: checkpoint inventory.

## Training masks and scoring

The spatial loss mask intersects the HR TIFF raster mask with the LR raster
mask expanded 3x onto the target grid. Validity must hold in all stored bands;
the mask is shared across output channels and normalizes the L1 average.
The released masks are all-valid, so this gives ordinary full-frame L1.
QA/SCL masks screen candidate pairs during collection; companion identified-fill
masks document recovered source fill separately. Neither enters training loss
or the published scoring mask. Test metrics exclude a three-pixel HR border.

## Supporting CPU analysis

```bash
python dataset/analyze_paired_blocks.py
python dataset/verify_companion_masks.py --release data/reflectance16
```

The first command uses the saved three-seed scores in
`results/benchmark_three_seed/`. It checks cohort membership and archived metric
means, then calculates paired model differences, equal-block means and
leave-one-block-out summaries. The second checks the existing companion masks
against their inventory and imagery grids. See `results/cpu_support/README.md`.
Neither command runs model inference or replaces the primary benchmark.

## Data and checkpoint access

The Kaggle download supplies reflectance16 images. Archived scores and audit
reports are under `results/`; the checkpoint inventory is
`weights/checkpoints.csv`. Validation-selected model states and the full-evidence
bundle will be supplied as confidential journal-review material, with access
instructions accompanying the manuscript.

Band descriptions determine RGB/RGBNIR ordering. Integer reflectance is divided
by 10,000 and clipped to [0, 1]; display stretches never enter metric calculation.
The reader checks shape, bands and validity. Training/evaluation use the
published manifest and recorded model/data protocol.
Model settings and strict weight loading still guard checkpoint compatibility.

Archived file identities remain in historical result records, and the ID-based
OLI2MSI split algorithm is unchanged.
Fresh collection may differ from the frozen release as remote catalogues change;
use the published imagery and split for reproduction. New training can also vary
with hardware/software; the archived environments document the reported runs.

Source products and some boundary footprints recur across the published splits;
this is not a source-disjoint or buffered benchmark. SRVGG stopped before its
adversarial phase; DCM uses the HSENet comparison code and FeNet is the
feature-enhancement network identified in the paper. See `LICENSES.md`
for third-party terms.
