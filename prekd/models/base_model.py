"""  Train MPNN where the solvent features are included as 
a global feature vector in the solute/solvents graph
"""

import pickle as pk
from argparse import ArgumentParser
from pathlib import Path
from shutil import copy
import pandas as pd
import numpy as np
import networkx as nx

import keras
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import (Add, Concatenate, Dense, Dropout,
                                     Embedding, GlobalAveragePooling1D, Input,
                                     Reshape)

from nfp import (EdgeUpdate, GlobalUpdate, NodeUpdate,
                 masked_mean_absolute_error, RBFExpansion)
from nfp.preprocessing.mol_preprocessor import SmilesPreprocessor

from .losses import hybrid_mae_bce_loss, mae_loss_cutoff, bce_loss_cutoff



########################################################################################
# Model Code
def message_passing(atom_state, bond_state, global_state, connectivity, 
                    num_messages, num_mol_features):
    for _ in range(num_messages):
        new_bond_state = EdgeUpdate()(
                [atom_state, bond_state, connectivity, global_state]
        )
        bond_state = Add()([bond_state, new_bond_state])

        new_atom_state = NodeUpdate()(
            [atom_state, bond_state, connectivity, global_state]
        )
        atom_state = Add()([atom_state, new_atom_state])

        new_global_state = GlobalUpdate(units=num_mol_features, num_heads=1)(
            [atom_state, bond_state, connectivity, global_state]
        )
        global_state = Add()([global_state, new_global_state])

    return atom_state, bond_state, global_state


def embedding_to_output(embedding, prediction_name):
    linear = Dense(1, name=f"{prediction_name}_linear")(embedding)
    output = GlobalAveragePooling1D(name=f"{prediction_name}_Prediction")(linear)

    return output


def dense_series(dense_input, dense_size, name):
    dense = Dense(dense_size[0], name=f"{name}_1", activation="relu")(dense_input)

    for i, size in enumerate(dense_size[1:]):
        dense = Dense(size, name=f"{name}_{i+2}", activation="relu")(dense)

    return dense


def build_model(preprocessor, model_summary, prediction_columns, params):
    atom_input = Input(shape=[None], dtype=tf.int64, name="atom")
    bond_input = Input(shape=[None], dtype=tf.int64, name="bond")
    connectivity = Input(shape=[None, 2], dtype=tf.int64, name="connectivity")
    global_features = Input(shape=[None], dtype=tf.float32, name="global")

    ########################## Atom State
    # Embed Atom/Bond inputs into vectors of atom/bond feature length
    atom_state = Embedding(
        preprocessor.atom_classes,
        int(params["atom_features"]),
        name="atom_embedding",
        mask_zero=True,
    )(atom_input)

    ########################## Bond State
    bond_state = Embedding(
        preprocessor.bond_classes,
        int(params["bond_features"]),
        name="bond_embedding",
        mask_zero=True,
    )(bond_input)

    ########################## Global State
    # Input values and generate the global state
    global_features_state = Reshape((len(preprocessor.feat_cols),))(global_features)
    global_features_state = Dense(params["mol_features"], name="global_features")(global_features_state)
    global_state = GlobalUpdate(units=params["mol_features"], num_heads=1)(
        [atom_state, bond_state, connectivity, global_features_state]
    )
    global_state = Add()([global_state, global_features_state])

    ########################## Message Passing
    atom_state, bond_state, global_state = message_passing(
        atom_state, bond_state, global_state, connectivity, 
        params["num_messages"], params["mol_features"]
    )

    ########################## Output Layers
    output_layers = []

    dense_output = dense_series(bond_state, [64, 16], "Dense_After_MP")

    for prediction_column in prediction_columns:
        output_layers.append(embedding_to_output(dense_output, prediction_column))

    outputs = keras.ops.concatenate(output_layers, axis=-1)

    model = Model([atom_input, bond_input, connectivity, global_features], outputs)

    return model


def train_model(model, params):
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=params["learning_rate"], 
                                           weight_decay=params["decay"],
                                           clipnorm=1.0,  # Add gradient clipping
                                           ),
        loss=[masked_mean_absolute_error],
    )
    return model


def train_model_hybrid(model, params):
    """
    Above and blow 1.5 and -1.5 are more analytical errors, limitations of the instrument
    Try removing them from the MAE and try relabeling as categorical
    """
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=params["learning_rate"], 
                                           weight_decay=params["decay"],
                                           clipnorm=1.0,  # Add gradient clipping
                                           ),
        loss=[hybrid_mae_bce_loss],
        metrics=[mae_loss_cutoff, bce_loss_cutoff],
    )
    return model


def build_train_model(preprocessor, model_summary, prediction_columns, params):
    model = build_model(preprocessor, model_summary, prediction_columns, params)
    model = train_model(model, params)

    return model


