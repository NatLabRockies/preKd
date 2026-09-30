"""Turn a CSV/Excel table of partition measurements into a PreKd training-data dump.

``train_solute_solvent.py`` expects a pickled ``MultiModel`` ("mm dump") that already
contains the cleaned data frame and the cross-validation fold assignment. This script
builds that dump from a plain table, so the full workflow is::

    python prepare_data.py --input measurements.csv --out data/train_dump.pk
    python train_solute_solvent.py --mm_dump data/train_dump.pk --save_folder models/run1 --kfolds 0
    python predict.py --models models/run1 --input queries.csv --out predictions.csv

Expected input columns (all names are configurable):

==========================  ==========================================================
column                      contents
==========================  ==========================================================
``solute_smiles``           SMILES of the solute, one per row
``solvent_smiles``          SMILES of the solvent-system components, ``;``-delimited
``solvent_mol_fractions``   mole fractions matching ``solvent_smiles``, ``;``-delimited
``log_kp``                  the target value (``--pred_cols``)
==========================  ==========================================================

Rows whose SMILES RDKit cannot parse, whose fraction list does not line up with the
solvent list, or whose targets are all missing are dropped and reported. The cleaned
frame is written next to the dump as ``<out>.cleaned.csv`` so it is clear which rows
were used.
"""

import json
from argparse import ArgumentParser, RawDescriptionHelpFormatter
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from sklearn import model_selection

from prekd.model_handler import MultiModel

RDLogger.DisableLog("rdApp.*")


def _split_list(value, sep=";"):
    """Split a delimited cell into a list of stripped strings."""
    if isinstance(value, (list, tuple)):
        return [str(v).strip() for v in value]
    if pd.isna(value):
        return []
    return [v.strip() for v in str(value).split(sep) if v.strip()]


def validate_rows(df, solute_col, solvents_col, fracs_col, sep=";", frac_tol=0.02):
    """Drop rows PreKd cannot featurize and report why.

    Returns the kept frame and a dict of per-reason drop counts.
    """
    reasons = {}
    keep = pd.Series(True, index=df.index)

    def _reject(mask, reason):
        n = int((keep & mask).sum())
        if n:
            reasons[reason] = reasons.get(reason, 0) + n
        return keep & ~mask

    # Solute SMILES must parse
    bad_solute = df[solute_col].apply(
        lambda s: pd.isna(s) or Chem.MolFromSmiles(str(s)) is None
    )
    keep = _reject(bad_solute, "unparseable solute SMILES")

    # Every solvent SMILES must parse
    def _bad_solvents(value):
        smiles = _split_list(value, sep)
        if not smiles:
            return True
        return any(Chem.MolFromSmiles(s) is None for s in smiles)

    keep = _reject(df[solvents_col].apply(_bad_solvents), "unparseable solvent SMILES")

    if fracs_col is not None:
        # Fractions must parse, match the solvent count, and sum to ~1
        def _frac_problem(row):
            smiles = _split_list(row[solvents_col], sep)
            raw = _split_list(row[fracs_col], sep)
            try:
                fracs = [float(f) for f in raw]
            except ValueError:
                return "unparseable solvent fractions"
            if len(fracs) != len(smiles):
                return "solvent/fraction count mismatch"
            if not fracs or abs(sum(fracs) - 1.0) > frac_tol:
                return "solvent fractions do not sum to 1"
            return None

        problems = df.apply(_frac_problem, axis=1)
        for reason in problems.dropna().unique():
            keep = _reject(problems == reason, reason)

    return df[keep].reset_index(drop=True), reasons


def build_folds(df, kfolds, seed, group_col=None):
    """Assign rows to cross-validation folds, keyed by ``data_id``.

    Without ``group_col`` this is a plain shuffled K-fold over rows. With it, every
    row sharing a value (e.g. the same solute) lands in the same fold, which gives a
    leave-group-out style evaluation and is the honest setting when the same compound
    is measured many times.

    Unlike ``MultiModel.split_data``'s internal split, the fold assignment here is
    seeded and therefore reproducible.
    """
    data_ids = df["data_id"].to_numpy()
    if group_col is None:
        splitter = model_selection.KFold(n_splits=kfolds, shuffle=True, random_state=seed)
        splits = splitter.split(data_ids)
        return {
            i: {
                "train": [int(data_ids[j]) for j in train_idx],
                "validate": [int(data_ids[j]) for j in val_idx],
            }
            for i, (train_idx, val_idx) in enumerate(splits)
        }

    # Grouped: shuffle the unique groups, then deal them into folds
    groups = df[group_col].astype(str).to_numpy()
    unique = np.unique(groups)
    rng = np.random.default_rng(seed)
    rng.shuffle(unique)
    assignment = {g: i % kfolds for i, g in enumerate(unique)}
    fold_of_row = np.array([assignment[g] for g in groups])
    return {
        i: {
            "train": [int(d) for d in data_ids[fold_of_row != i]],
            "validate": [int(d) for d in data_ids[fold_of_row == i]],
        }
        for i in range(kfolds)
    }


def main(args):
    in_path = Path(args.input)
    if in_path.suffix == ".csv":
        df = pd.read_csv(in_path)
    elif in_path.suffix in (".xlsx", ".xls"):
        df = pd.read_excel(in_path)
    else:
        raise SystemExit(f"Input must be .csv or .xlsx, got '{in_path.suffix}'")
    print(f"Read {len(df)} rows from {in_path}")

    pred_cols = [c.strip() for c in args.pred_cols.split(",")]
    fracs_col = None if args.solvent_fracs_col in (None, "None") else args.solvent_fracs_col

    required = [args.solute_col, args.solvents_col] + pred_cols
    if fracs_col is not None:
        required.append(fracs_col)
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise SystemExit(
            f"Missing required column(s): {missing}\nColumns present: {list(df.columns)}"
        )

    # Rows with no target at all cannot train anything
    n_before = len(df)
    df = df.dropna(subset=pred_cols, how="all").reset_index(drop=True)
    if len(df) < n_before:
        print(f"Dropped {n_before - len(df)} rows with no value in {pred_cols}")

    df, reasons = validate_rows(df, args.solute_col, args.solvents_col, fracs_col)
    for reason, count in reasons.items():
        print(f"Dropped {count} rows: {reason}")
    if df.empty:
        raise SystemExit("No usable rows remain after validation.")
    print(f"{len(df)} rows kept ({df[args.solute_col].nunique()} unique solutes)")

    # split_data() matches folds on data_id, so it must line up with the kept frame
    if "data_id" in df.columns and not args.reassign_data_id:
        print("Using the existing data_id column")
    else:
        df["data_id"] = np.arange(len(df))

    if args.group_col and args.group_col not in df.columns:
        raise SystemExit(f"--group_col '{args.group_col}' is not a column in the input")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cleaned_path = out_path.with_suffix(".cleaned.csv")
    df.to_csv(cleaned_path, index=False)
    print(f"Wrote the cleaned frame to {cleaned_path}")

    mm = MultiModel()
    mm.load_dataset(cleaned_path, pred_cols)

    if args.solvent_features:
        feat = pd.read_csv(args.solvent_features, index_col=0)
        mm.df_solvent_features = feat
        mm.solvent_feature_cols = (
            [c.strip() for c in args.solvent_feature_cols.split(",")]
            if args.solvent_feature_cols
            else feat.columns.tolist()
        )
        print(f"Using {len(mm.solvent_feature_cols)} solvent feature column(s)")

    folds = build_folds(df, args.kfolds, args.seed, args.group_col)
    for i, fold in folds.items():
        print(f"  fold {i}: {len(fold['train'])} train / {len(fold['validate'])} validate")

    fold_path = out_path.with_suffix(".kfolds.json")
    mm.split_data(kfolds=args.kfolds, load_kfold=folds)
    with open(fold_path, "w") as f:
        json.dump({str(k): v for k, v in folds.items()}, f)
    print(f"Wrote the fold assignment to {fold_path}")

    mm.dump_training_data(out_path)
    print(f"Wrote the training-data dump to {out_path}")
    print(
        f"\nNext:\n  python train_solute_solvent.py --mm_dump {out_path} "
        f"--save_folder models/run1 --kfolds 0 --pred_cols {','.join(pred_cols)}"
    )


if __name__ == "__main__":
    parser = ArgumentParser(
        description=__doc__, formatter_class=RawDescriptionHelpFormatter
    )
    parser.add_argument("--input", required=True, help="Input .csv or .xlsx of measurements")
    parser.add_argument("--out", default="data/train_dump.pk", help="Where to write the dump")
    parser.add_argument("--pred_cols", default="log_kp", help="Comma-separated target columns")
    parser.add_argument("--solute_col", default="solute_smiles")
    parser.add_argument("--solvents_col", default="solvent_smiles")
    parser.add_argument(
        "--solvent_fracs_col",
        default="solvent_mol_fractions",
        help="Use 'None' for single-solvent data with no fractions column",
    )
    parser.add_argument("--kfolds", type=int, default=10, help="Number of CV folds")
    parser.add_argument("--seed", type=int, default=42, help="Seed for the fold assignment")
    parser.add_argument(
        "--group_col",
        default=None,
        help="Keep rows sharing this column's value in the same fold (e.g. solute_smiles)",
    )
    parser.add_argument("--solvent_features", default=None, help="Optional solvent-feature CSV")
    parser.add_argument("--solvent_feature_cols", default=None, help="Comma-separated subset")
    parser.add_argument(
        "--reassign_data_id",
        action="store_true",
        help="Overwrite an existing data_id column instead of reusing it",
    )
    main(parser.parse_args())
