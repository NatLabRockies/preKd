# PreKd

PreKd is a graph neural network that predicts the partition coefficient
(`log Kp`) of a compound in a biphasic solvent system directly from molecular
structure — the solute's SMILES plus the SMILES and mole fractions of the
solvents making up the system.

Partition coefficients are the central design parameter for liquid–liquid
extraction and counter-current chromatography, and measuring one for every
compound/solvent-system combination of interest is slow and material-intensive.
PreKd is intended to narrow that search so experiments can be spent on the most
promising candidates.

The model runs message passing over the solute and over each solvent
independently, and weights each solvent's contribution by its mole fraction, so
a solvent system is represented by its composition rather than by an arbitrary
system label. This means predictions can be made for solvent systems and
compositions that were never seen during training.

This repository contains the TensorFlow/Keras implementation built on
[`nfp`](https://github.com/NREL/nfp).

## Installation

PreKd requires the TF 2.20 / Keras 3 branch of `nfp`, which is **not** the
version published on PyPI — the PyPI release predates Keras 3 and will fail when
the model is built. Both `environment.yml` and `requirements.txt` already pin the
correct commit, so install through one of them rather than `pip install nfp`.

```bash
conda env create -f environment.yml
conda activate preKd
pip install -e .
```

A GPU is recommended for training but is not required; prediction runs fine on
CPU.

## Input data format

PreKd reads a CSV (or Excel) table with one measurement per row:

| column | contents |
| --- | --- |
| `solute_smiles` | SMILES of the solute |
| `solvent_smiles` | `;`-delimited SMILES of the solvent-system components |
| `solvent_mol_fractions` | `;`-delimited mole fractions, in the same order, summing to 1 |
| `log_kp` | the target value |

All four column names are configurable with `--solute_col`, `--solvents_col`,
`--solvent_fracs_col` and `--pred_cols`. Any other columns are carried through
and reappear in the prediction output. For single-solvent data (solubility, for
example) pass `--solvent_fracs_col None`.

`example_data/example_measurements.csv` shows the format. Note that its values
are **synthetic** — they exist to exercise the pipeline, not to train a usable
model.

## Quickstart

```bash
# 1. Validate the table and build the training-data dump with its CV folds
python prepare_data.py \
    --input example_data/example_measurements.csv \
    --out data/train_dump.pk \
    --kfolds 4

# 2. Train one cross-validation fold (repeat, or see "Training all folds" below)
python train_solute_solvent.py \
    --mm_dump data/train_dump.pk \
    --save_folder models/run1 \
    --kfolds 0 \
    --epochs 100

# 3. Predict with the ensemble of trained folds
python predict.py \
    --models models/run1 \
    --input queries.csv \
    --out predictions.csv \
    --dov
```

### 1. Preparing data

`prepare_data.py` turns a table of measurements into the pickled `MultiModel`
dump the training script expects. It drops rows RDKit cannot parse, rows whose
fraction list does not line up with the solvent list, and rows whose fractions do
not sum to 1, reporting how many went for each reason. The cleaned frame is
written alongside the dump as `<out>.cleaned.csv` so it is always clear which
rows were used.

Fold assignment is seeded (`--seed`) and saved to `<out>.kfolds.json`, so a split
can be reproduced or reused. By default folds are a shuffled split over rows. Use
`--group_col solute_smiles` to keep every measurement of the same compound in a
single fold, which is the honest setting if you want to know how the model does
on compounds it has never seen — expect noticeably lower scores than a random
split, because a random split lets other measurements of the same compound sit in
the training set.

### 2. Training

Each run of `train_solute_solvent.py` trains one fold and writes it to
`--save_folder/model_<k>/`. The main architecture and optimization arguments are
`--n_messages`, `--af`/`--bf`/`--mf` (atom, bond and molecule feature widths;
atom and bond widths must match), `--epochs`, `--batch_size`, `--learning_rate`,
`--dropout` and `--decay`. Pass `--seed` for a reproducible run.

`train_solute_solvent_TL.py` is the transfer-learning variant: it starts from a
model trained on a related dataset (solubility, for instance) via `--base_model`
and fine-tunes on the target data.

### Training all folds

`prekd.utils.write_slurm_submit` writes a SLURM script that runs several folds
concurrently on one GPU node. Cluster-specific settings are arguments, and can
also come from the environment so nothing site-specific has to be edited into the
source:

```bash
export PREKD_SLURM_ACCOUNT=my_allocation
export PREKD_SLURM_GRES=gpu:h100:1
export PREKD_ENV_SETUP="module load cuda; conda activate preKd"
```

```python
from pathlib import Path
from prekd.parameters import Parameters
from prekd.utils import write_slurm_submit

params = Parameters(prediction_columns=["log_kp"], epochs=100)
write_slurm_submit(
    out_dir=Path("models/run1"),
    mm_data_file=Path("data/train_dump.pk"),
    params=params,
    job_name="prekd_cv",
    n_runs_per_node=5,
    submit=True,
)
```

Without a scheduler, just loop over `--kfolds 0 … k-1`.

### 3. Predicting

`predict.py` loads every fold in the model folder and predicts with all of them.
The reported `log_kp_pred` is the ensemble mean and `log_kp_pred_std` is the
spread across folds, which is the model's confidence signal: when the folds
disagree, the prediction deserves less trust.

The query table needs the same solute and solvent columns the model was trained
on — the names are read back off the saved preprocessor, so a mismatch names the
missing columns rather than failing deeper in.

### Domain of validity

Passing `--dov` also reports `fps_notin_train`: the number of Morgan
substructures in the query that never appear anywhere in the training data, and
an `outside_domain` flag for queries above the heuristic cutoff of 6. A GNN can
only interpolate over chemistry it has seen; a large compound built from
unfamiliar fragments will still get a prediction, and that prediction should be
treated as extrapolation. Checking this alongside the ensemble spread is the
intended way to decide whether a given prediction is worth acting on.

`prekd.domain_of_validity.DoV` also offers nearest-neighbor analysis, to see
which training compounds a query most resembles.

## Repository layout

```
.
├── prepare_data.py             # table -> validated training-data dump + CV folds
├── train_solute_solvent.py     # train one cross-validation fold
├── train_solute_solvent_TL.py  # transfer learning from a pretrained base model
├── predict.py                  # ensemble prediction + domain of validity
├── example_data/               # synthetic example table (format illustration)
└── prekd/
    ├── model_handler.py        # SingleModel / MultiModel: training, saving, prediction
    ├── parameters.py           # model and training parameters
    ├── preprocessor.py         # solute + weighted-solvent graph preprocessing
    ├── domain_of_validity.py   # unseen-substructure counts, nearest neighbors
    ├── utils.py                # SLURM submission helper, small IO helpers
    └── models/                 # network definitions, losses, callbacks
```

## Citation

<!-- TODO: fill in once the accompanying paper is published. -->

## License

BSD 3-Clause. See [LICENSE](LICENSE) and [NOTICE](NOTICE).
