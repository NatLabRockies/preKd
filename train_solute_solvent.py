"""  Train MPNN 
"""

import pickle as pk
from argparse import ArgumentParser
from pathlib import Path
from shutil import copy
import pandas as pd
import numpy as np
import networkx as nx

import tensorflow as tf
#from tensorflow.keras import Model
#from tensorflow.keras.layers import (Add, Concatenate, Dense, Dropout,
#                                     Embedding, GlobalAveragePooling1D, Input,
#                                     Reshape)

# TODO these should be imported from the base folder of the package
from prekd.model_handler import MultiModel
from prekd.parameters import Parameters

from nfp.preprocessing.features import atom_features_v2, bond_features_v1
#from src.models.base_model import atom_features_v2, bond_features_v1
from prekd.models.solute_solvent_weighted_model import build_train_model, build_train_model_hybrid
from prekd.preprocessor import SolventFeaturesPreprocessor


pwd = Path(__file__).parent

print(f"{tf.__version__ = }")
print(f"{pd.__version__ = }")
print(f"{np.__version__ = }")


gpus = tf.config.list_physical_devices('GPU')
print("GPUs:", gpus)
# Currently, memory growth needs to be the same across GPUs
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)


########################################################################################
# Training Code
def main(dump_fname, arg_values, kfolds, save_folder):
    values = arg_values

    dump_fname = Path(dump_fname)

    print(f"Loading data from {dump_fname}")
    mm = MultiModel().load_training_data(dump_fname)
    print(f"{len(mm.df_input)} rows in input df")
    print(mm.df_input.head(2))
    #feat_cols = mm.solv_feat_cols
    #print(f"{mm.prediction_columns = }, {feat_cols = }")
    #df_train = mm.models[kfolds[0]].df_train
    #df_val = mm.models[kfolds[0]].df_validate

    parameters = Parameters(
        epochs=int(values.epochs),
        learning_rate=float(values.learning_rate),
        decay=float(values.decay),
        atom_features=int(values.af),
        bond_features=int(values.bf),
        mol_features=int(values.mf),
        num_messages=int(values.n_messages),
        dropout=float(values.dropout),
        batch_size=int(values.batch_size),
        no_mol_frac_weights=values.no_mol_frac_weights,
        prediction_columns=values.pred_cols,
        solute_col=values.solute_col,
        solvents_col=values.solvents_col,
        solvent_fracs_col=values.solvent_fracs_col if values.solvent_fracs_col != 'None' else None,
        #solute_feature_cols=mm.solute_feature_cols,
        # TODO Changing the value here doesn't update what's in the mm object
        #solvent_feature_cols=mm.solvent_feature_cols,
    )

    if save_folder:
        save_folder = Path(save_folder)
        copy(dump_fname, save_folder)
        with open(save_folder / "parameters.pk", "wb") as f:
            pk.dump(parameters.to_dict(), f)

    print(f"Training with {parameters.to_dict()}")
    
    # # Generate the preprocessors for each model
    # Here we use a preprocessor that uses just smiles
    # just generate the preprocessor for the kfolds that will use it
#    mm.generate_preprocessors(
#        preprocessor=SolventFeaturesPreprocessor,
#        atom_features=atom_features_v2,
#        bond_features=bond_features_v1,
#        solute_col=parameters.smiles_col,
#        solvent_cols=parameters.solvent_cols,
#        solute_feature_cols=parameters.solute_feature_cols,
#        solvent_feature_cols=parameters.solvent_feature_cols,
#        solvent_feature_df=mm.solvent_feature_df,
#    )

    for i in kfolds:
        # Train the models
        print("=" * 40)
        print(f"Training Kfold {i}")
        print("=" * 40)

        mm.models[i].generate_preprocessor(
            preprocessor=SolventFeaturesPreprocessor,
            atom_features=atom_features_v2,
            bond_features=bond_features_v1,
            solute_col=parameters.solute_col,
            solvents_col=parameters.solvents_col,
            solvent_fracs_col=parameters.solvent_fracs_col,
            batch_size=parameters.batch_size,
        )

        if save_folder:
            mm._save_model_state(save_folder, parameters.to_dict(), True)

        print("\nTraining")
        mm.train_model(
            model_i=i,
            modelbuilder=build_train_model,
            model_params=parameters.to_dict(),
            save_folder=save_folder,
            save_training=True,
            save_report_log=True,
            verbose=1,
        )
    return mm



if __name__ == "__main__":
    default_params = Parameters()
    parser = ArgumentParser()
    parser.add_argument("--kfolds", type=str, default="0")
    parser.add_argument("--save_folder", default=None)
    # Training data
    parser.add_argument("--mm_dump", default=None)
    parser.add_argument("--n_messages", type=int, default=default_params.num_messages)
    parser.add_argument("--af", type=int, default=default_params.atom_features)
    parser.add_argument("--bf", type=int, default=default_params.atom_features)
    parser.add_argument("--mf", type=int, default=default_params.mol_features)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--decay", type=float, default=1e-5)
    parser.add_argument("--solute_col", type=str, default=default_params.solute_col)
    parser.add_argument("--solvents_col", type=str, default=default_params.solvents_col)
    parser.add_argument("--solvent_fracs_col", type=str, default=default_params.solvent_fracs_col)
    parser.add_argument("--pred_cols", type=str, default=default_params.prediction_columns)
    parser.add_argument("--no_mol_frac_weights", action="store_true", default=False)
    values = parser.parse_args()
    values.kfolds = list(map(int, values.kfolds.split(","))) if ',' in values.kfolds else [int(values.kfolds)]
    values.pred_cols = values.pred_cols.split(",")
    
    save_folder = values.save_folder
    kfolds = values.kfolds
    print(values)

    # Make a new folder if it doesn't exist.
    try:
        Path.mkdir(Path(save_folder), parents=True)
    except:
        pass

    main(
        dump_fname = values.mm_dump,
        arg_values=values,
        kfolds=[int(i) for i in values.kfolds],
        save_folder=save_folder,
    )
