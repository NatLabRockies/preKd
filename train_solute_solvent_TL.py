"""  Train MPNN 
"""

import pickle as pk
from argparse import ArgumentParser
from pathlib import Path
from shutil import copy
import pandas as pd
import numpy as np
import networkx as nx

# TODO these should be imported from the base folder of the package
from prekd.model_handler import MultiModel
from prekd.parameters import Parameters
from prekd import utils

import tensorflow as tf
#from tensorflow.keras import Model
#from tensorflow.keras.layers import (Add, Concatenate, Dense, Dropout,
#                                     Embedding, GlobalAveragePooling1D, Input,
#                                     Reshape)

from nfp.preprocessing.features import atom_features_v2, bond_features_v1
#from src.models.base_model import atom_features_v2, bond_features_v1
from prekd.models.solute_solvent_weighted_model_TL import build_train_tl_model, build_train_tl_model_hybrid
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

    # Load the baseinal model
    loc_base_model = values.base_model_dir
    print(f"Loading original model from {loc_base_model}")
    base_mm = MultiModel().load_models(loc_base_model, nmodels=[0])
    # TODO train a single model with all the pretraining data(?)
    base_model = base_mm.models[0].model
    #print(f"Model has "
    #      f"{len(base_model.layers)} layers, "
    #      f"{len(base_model.trainable_weights)} trainable weights, "
    #      f"{len(base_model.inputs)} inputs, "
    #      f"{len(base_model.outputs)} outputs")
    print(base_model.summary)

    dump_fname = Path(dump_fname)
    loc_models = dump_fname.parent

    print(f"Loading data from {dump_fname}")
    mm = MultiModel().load_training_data(dump_fname)
    print(f"{len(mm.df_input)} rows in input df")
    print(mm.df_input.head(2))
    #feat_cols = mm.solv_feat_cols
    #print(f"{mm.prediction_columns = }, {feat_cols = }")
    #df_train = mm.models[kfolds[0]].df_train
    #df_val = mm.models[kfolds[0]].df_validate

    # TODO update to use passed in parameters
    params_file = loc_models / 'parameters_object.pk'
    print(f"reading parameters from {params_file}")
    parameters = utils.pickle_read(params_file)
    parameters.use_hybrid_loss = bool(values.use_hybrid_loss)
    parameters.hybrid_cutoff = float(values.hybrid_cutoff)
    parameters.hybrid_bce_weight = float(values.hybrid_bce_weight)

    # parameters = Parameters(
    #     epochs=int(values.epochs),
    #     learning_rate=float(values.learning_rate),
    #     decay=float(values.decay),
    #     atom_features=int(values.af),
    #     bond_features=int(values.bf),
    #     mol_features=int(values.mf),
    #     num_messages=int(values.n_messages),
    #     dropout=float(values.dropout),
    #     batch_size=int(values.batch_size),
    #     no_mol_frac_weights=values.no_mol_frac_weights,
    #     prediction_columns=values.pred_cols,
    #     solute_col=values.solute_col,
    #     solvents_col=values.solvents_col,
    #     solvent_fracs_col=values.solvent_fracs_col,
    #     #solute_feature_cols=mm.solute_feature_cols,
    #     # TODO Changing the value here doesn't update what's in the mm object
    #     #solvent_feature_cols=mm.solvent_feature_cols,
    # )

    # if save_folder:
    #     save_folder = Path(save_folder)
    #     copy(dump_fname, save_folder)
    #     with open(save_folder / "parameters.pk", "wb") as f:
    #         pk.dump(parameters.to_dict(), f)

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

        # mm.models[i].generate_preprocessor(
        #     preprocessor=SolventFeaturesPreprocessor,
        #     atom_features=atom_features_v2,
        #     bond_features=bond_features_v1,
        #     solute_col=parameters.solute_col,
        #     solvents_col=parameters.solvents_col,
        #     solvent_fracs_col=parameters.solvent_fracs_col,
        #     batch_size=parameters.batch_size,
        # )

        #  use the same preprocessor for this model as the baseinal
        print("Copying preprocessor")
        # Use the preprocessor from the baseinal model
        mm.models[i].preprocessor = base_mm.models[0].preprocessor
        mm.models[i].preprocessor.solute_col = parameters.solute_col
        mm.models[i].preprocessor.solvents_col = parameters.solvents_col
        mm.models[i].preprocessor.solvent_fracs_col = parameters.solvent_fracs_col 
        mm.models[i].solute_col = parameters.solute_col
        mm.models[i].solvents_col = parameters.solvents_col
        mm.models[i].solvent_fracs_col = parameters.solvent_fracs_col
        #mm.models[case_k].generate_preprocessor(preprocessor=PolymerPreprocessor, 
        #        atom_features=atom_features_v1, 
        #        bond_features=bond_features_v1, 
        #        batch_size=params["batch_size"])
                # Check if train/test data exist and create generators
        # need to create the generators still
        mm.models[i].train_generator = mm.models[i]._create_generator(
            mm.models[i].df_train, batch_size=parameters["batch_size"]
        )
        mm.models[i].validate_generator = mm.models[i]._create_generator(
            mm.models[i].df_validate, batch_size=parameters["batch_size"]
        )

        if save_folder:
            mm._save_model_state(save_folder, parameters.to_dict(), True)

        print("\nTraining")
        mm.train_model(
            model_i=i,
            modelbuilder=build_train_tl_model_hybrid,
            model_params=parameters.to_dict(),
            save_folder=save_folder,
            save_training=True,
            save_report_log=True,
            verbose=1,  # 0: no output, 1: progress bar per epoch, 2: only info per epoch is printed
            orig_model=base_model,
        )
    return mm



if __name__ == "__main__":
    default_params = Parameters()
    parser = ArgumentParser()
    parser.add_argument("--kfolds", type=str, default="0")
    parser.add_argument("--save_folder", default=None)
    # Training data
    parser.add_argument("--mm_dump", default=None)
    parser.add_argument("--base_model_dir", dest="base_model_dir", type=Path,
                        help="Directory of model to transfer from",
                        )
    parser.add_argument("--n_messages", type=int, default=default_params.num_messages)
    parser.add_argument("--af", type=int, default=default_params.atom_features)
    parser.add_argument("--bf", type=int, default=default_params.atom_features)
    parser.add_argument("--mf", type=int, default=default_params.mol_features)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--decay", type=float, default=1e-5)
    parser.add_argument("--use_hybrid_loss", action="store_true", default=True)
    parser.add_argument("--hybrid_cutoff", type=float, default=1.5)
    parser.add_argument("--hybrid_bce_weight", type=float, default=0.5)
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
