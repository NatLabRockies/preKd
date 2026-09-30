"""Predict log Kp for new solute / solvent-system pairs with a trained PreKd ensemble.

PreKd is trained as a k-fold cross-validation ensemble: ``train_solute_solvent.py``
writes one model per fold into ``--save_folder``. This script loads all of them,
predicts with each, and reports the ensemble mean together with the spread across
folds, which is the model's confidence signal -- a large standard deviation means
the folds disagree and the prediction should be treated with caution.

    python predict.py --models models/run1 --input queries.csv --out predictions.csv

The input table needs the same solute/solvent columns the model was trained on; the
names are read back off the model's preprocessor, so a mismatch is reported by name.
With ``--dov`` each query is additionally scored for how far it sits outside the
training data, by counting the Morgan substructures that never occur in training.
Predictions for compounds well beyond that range are extrapolation, not
interpolation, and the paper's heuristic cutoff is 6 unseen substructures.
"""

from argparse import ArgumentParser, RawDescriptionHelpFormatter
from pathlib import Path

import pandas as pd

from prekd.model_handler import MultiModel

DOV_CUTOFF = 6


def load_queries(path, preprocessor):
    """Read the query table and check it carries the columns the model expects."""
    path = Path(path)
    if path.suffix == ".csv":
        df = pd.read_csv(path)
    elif path.suffix in (".xlsx", ".xls"):
        df = pd.read_excel(path)
    else:
        raise SystemExit(f"Input must be .csv or .xlsx, got '{path.suffix}'")

    needed = [preprocessor.solute_col, preprocessor.solvents_col]
    if getattr(preprocessor, "solvent_fracs_col", None):
        needed.append(preprocessor.solvent_fracs_col)

    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise SystemExit(
            f"The model was trained on column(s) {missing}, which the input does not "
            f"have.\nColumns present: {list(df.columns)}"
        )
    return df


def add_domain_of_validity(df, mm, preprocessor):
    """Count Morgan substructures in each query that never appear in training."""
    from prekd.domain_of_validity import DoV

    train_frames = [
        m.df_train for m in mm.models if getattr(m, "df_train", None) is not None
    ]
    if not train_frames:
        print(
            "Warning: the saved models carry no training data, so the domain of "
            "validity cannot be computed. Skipping --dov."
        )
        return df

    df_train = pd.concat(train_frames).drop_duplicates(subset=[preprocessor.solute_col])
    dov = DoV(
        fingerprint_col=preprocessor.solute_col,
        solvent_col=preprocessor.solvents_col,
        solvent_frac_col=getattr(preprocessor, "solvent_fracs_col", None),
    )
    print(f"Scoring domain of validity against {len(df_train)} training solutes")
    train_fps = dov.get_fps(df_train)
    df = dov.get_fps_overlap(df, train_fps)
    df["outside_domain"] = df["fps_notin_train"] > DOV_CUTOFF
    n_out = int(df["outside_domain"].sum())
    if n_out:
        print(
            f"{n_out} of {len(df)} queries have more than {DOV_CUTOFF} unseen "
            "substructures and are outside the model's domain of validity"
        )
    return df


def main(args):
    print(f"Loading the model ensemble from {args.models}")
    mm = MultiModel.load_models(args.models, nmodels=args.nmodels)
    if not mm.models:
        raise SystemExit(f"No models found in {args.models}")
    print(f"Loaded {len(mm.models)} models")

    preprocessor = mm.models[0].preprocessor
    df = load_queries(args.input, preprocessor)
    print(f"Predicting for {len(df)} rows")

    # One row per (query, fold); collapse to per-query mean and spread
    per_model = mm.make_predictions(df)
    pred_cols = [f"{col}_pred" for col in mm.models[0].prediction_columns]

    out = df.copy()
    for col in pred_cols:
        grouped = per_model.groupby(per_model.index)[col]
        out[col] = grouped.mean()
        out[f"{col}_std"] = grouped.std()

    if args.dov:
        out = add_domain_of_validity(out, mm, preprocessor)

    if args.per_model_out:
        per_model.to_csv(args.per_model_out, index=False)
        print(f"Wrote per-fold predictions to {args.per_model_out}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)
    print(f"Wrote predictions to {out_path}")

    for col in pred_cols:
        print(f"  {col}: mean {out[col].mean():.3f}, mean ensemble std {out[f'{col}_std'].mean():.3f}")


if __name__ == "__main__":
    parser = ArgumentParser(
        description=__doc__, formatter_class=RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--models", required=True, help="Folder holding the trained model_* subfolders"
    )
    parser.add_argument("--input", required=True, help="Query .csv or .xlsx")
    parser.add_argument("--out", default="predictions.csv", help="Where to write predictions")
    parser.add_argument(
        "--nmodels",
        type=int,
        default=None,
        help="Use only the first N folds (default: every model in the folder)",
    )
    parser.add_argument(
        "--dov",
        action="store_true",
        help="Also report how far each query sits outside the training data",
    )
    parser.add_argument(
        "--per_model_out", default=None, help="Optionally write the per-fold predictions"
    )
    main(parser.parse_args())
