"""  Train MPNN where the solute and solvent graphs are combined into a singel (disconnected) graph
and the solvent fractions are included as node and edge weights
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
                                     Multiply, Reshape)

from nfp import (EdgeUpdate, GlobalUpdate, NodeUpdate,
                 masked_mean_absolute_error, RBFExpansion)
from nfp.preprocessing.mol_preprocessor import SmilesPreprocessor

from .base_model import (message_passing, embedding_to_output,
                                    dense_series, build_model as base_build_model,
                                    train_model as base_train_model,
                                    train_model_hybrid
                                    )


@keras.saving.register_keras_serializable(package="prekd")
class MaskedMultiply(tf.keras.layers.Layer):
    """Multiply two tensors while preserving the first input mask."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.supports_masking = True

    def call(self, inputs, **kwargs):
        left, right = inputs
        return left * right

    def compute_mask(self, inputs, mask=None):
        if mask is None:
            return None
        return mask[0]

    def get_config(self):
        return super().get_config()

    @classmethod
    def from_config(cls, config):
        return cls(**config)


def build_weighted_model(preprocessor, model_summary, prediction_columns, params):

    num_mol_features = params["mol_features"]
    atom_input = Input(shape=[None], dtype=tf.int64, name="atom")
    bond_input = Input(shape=[None], dtype=tf.int64, name="bond")
    connectivity = Input(shape=[None, 2], dtype=tf.int64, name="connectivity")
    atom_weights_input = Input(shape=[None], dtype=tf.float32, name='atom_weight')
    bond_weights_input = Input(shape=[None], dtype=tf.float32, name='bond_weight')
    if len(preprocessor.solute_feature_cols) > 0:
        global_features = Input(shape=[None], dtype=tf.float32, name="global")
    if preprocessor.num_solv_feat_cols > 0:
        atom_extra_features = Input(shape=[None, preprocessor.num_solv_feat_cols], dtype=tf.float32, name="atom_feature_vec")
        bond_extra_features = Input(shape=[None, preprocessor.num_solv_feat_cols], dtype=tf.float32, name="bond_feature_vec")
        num_mol_features += preprocessor.num_solv_feat_cols
    
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

    atom_weights = keras.ops.expand_dims(atom_weights_input, axis=-1)
    bond_weights = keras.ops.expand_dims(bond_weights_input, axis=-1)

    # scale the atom and bond features by the solvent fractions (solute should be 1)
    atom_state = MaskedMultiply(name="atom_weighted_embedding")([atom_state, atom_weights])
    bond_state = MaskedMultiply(name="bond_weighted_embedding")([bond_state, bond_weights])

    # Add the extra features to the atom and bond states
    if preprocessor.num_solv_feat_cols > 0:
        atom_state = Concatenate(axis=-1)([atom_state, atom_extra_features])
        bond_state = Concatenate(axis=-1)([bond_state, bond_extra_features])

    ########################## Global State
    # Input values and generate the global state
    if len(preprocessor.solute_feature_cols) > 0:
        global_features_state = Reshape((len(preprocessor.solute_feature_cols),))(global_features)
        global_features_state = Dense(num_mol_features, name="global_features")(global_features_state)
        global_state = GlobalUpdate(units=num_mol_features, num_heads=1)(
            [atom_state, bond_state, connectivity, global_features_state]
        )
        global_state = Add()([global_state, global_features_state])
    else:
        global_state = GlobalUpdate(units=num_mol_features, num_heads=1)(
            [atom_state, bond_state, connectivity]
        )

    ########################## Message Passing
    atom_state, bond_state, global_state = message_passing(
        atom_state, bond_state, global_state, connectivity, 
        params["num_messages"], num_mol_features,
    )

    # scale the atom and bond features by the solvent fractions (solute should be 1)
    atom_state = MaskedMultiply(name="atom_weighted_post_mp")([atom_state, atom_weights])
    bond_state = MaskedMultiply(name="bond_weighted_post_mp")([bond_state, bond_weights])

    ########################## Output Layers
    output_layers = []

    dense_output = dense_series(bond_state, [64, 16], "Dense_After_MP")

    for prediction_column in prediction_columns:
        output_layers.append(embedding_to_output(dense_output, prediction_column))

    outputs = keras.ops.concatenate(output_layers, axis=-1)

    inputs = [atom_input, bond_input, connectivity, atom_weights_input, bond_weights_input]
    if preprocessor.num_solv_feat_cols > 0:
        inputs += [atom_extra_features, bond_extra_features]
    if len(preprocessor.solute_feature_cols) > 0:
        inputs += [global_features]

    model = Model(inputs, outputs)

    return model


def build_train_model(preprocessor, model_summary, prediction_columns, params):
    model = build_weighted_model(preprocessor, model_summary, prediction_columns, params)
    model = base_train_model(model, params)
    return model


def build_train_model_hybrid(preprocessor, model_summary, prediction_columns, params):
    model = build_weighted_model(preprocessor, model_summary, prediction_columns, params)
    model = train_model_hybrid(model, params)
    return model

