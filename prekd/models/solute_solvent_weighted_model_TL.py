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


def get_last_graph_layers(base_model, num_messages):
    # get the last atom, bond, and global feature vectors (after the residual connection)
    # TODO automatically extract the last bond, atom, and global layers
    last_bond_layer = base_model.get_layer(f"edge_update_{num_messages-1}")
    last_bond_res_layer = base_model.layers[base_model.layers.index(last_bond_layer) + 1]
    print(f"{last_bond_layer.name = }, {last_bond_res_layer.name = }")

    last_atom_layer = base_model.get_layer(f"node_update_{num_messages-2}")
    last_atom_res_layer = base_model.layers[base_model.layers.index(last_atom_layer) + 1]
    print(f"{last_atom_layer.name = }, {last_atom_res_layer.name = }")

    last_mol_layer = base_model.get_layer(f"global_update_{num_messages-1}")
    last_mol_res_layer = base_model.layers[base_model.layers.index(last_mol_layer) + 1]
    print(f"{last_mol_layer.name = }, {last_mol_res_layer.name = }")

    return last_atom_res_layer.output, last_bond_res_layer.output, last_mol_res_layer.output


def build_transfer_learning_model(base_model, preprocessor, model_summary=False, prediction_columns=None, params=None):
    """ Starting from a model trained on base data, freeze and/or add layers and train on the polymer data
    """

    # keep the inputs the same 
    connectivity_layer = base_model.get_layer("connectivity")
    connectivity = connectivity_layer.output

    # atom_weights_input = Input(shape=[None], dtype=tf.float32, name='atom_weight')
    # bond_weights_input = Input(shape=[None], dtype=tf.float32, name='bond_weight')

    if params.get("freeze_to_message_block") is not None:
        # freeze all layers up to the nth message block
        print(f"Freezing layers up to {params['freeze_to_message_block']} message blocks")
        # TODO is the edge, node, or global update layer the last layer of the message block?
        bond_layer = base_model.get_layer(f"edge_update_{params['freeze_to_message_block'] - 1}")
        print(f"{params['freeze_to_message_block'] = }, {bond_layer.name = }, "
              f"{base_model.layers.index(bond_layer) = }")
        for i in range(base_model.layers.index(bond_layer) + 1):
            base_model.layers[i].trainable = False
        
        print(f"Number of trainable parameters: {np.sum([np.prod(v.get_shape()) for v in base_model.trainable_weights])}")
        print(f"Number of frozen parameters: {np.sum([np.prod(v.get_shape()) for v in base_model.non_trainable_weights])}")


    atom_state, bond_state, global_state = get_last_graph_layers(base_model, params["num_messages"])
    # if params.get("add_messages") is not None:
    #     # first rename the original layers to avoid name collisions
    #     for layer in base_model.layers:
    #         if layer.name in base_model.input_names:
    #             continue
    #         layer._name = f"base_{layer.name}"

    #     print(f"Adding {params['add_messages']} message passing layers")
    #     # add message passing layers
    #     for i in range(params["add_messages"]):
    #         message_idx = params["num_messages"] + i
    #         atom_state, bond_state, global_state = message_block(
    #             params, atom_state, bond_state, global_state, connectivity, message_idx,
    #         )


    # atom_weights = tf.expand_dims(atom_weights_input, axis=-1)
    # bond_weights = tf.expand_dims(bond_weights_input, axis=-1)
    ########################## Message Passing
    # scale the atom and bond features by the solvent fractions (solute should be 1)
    # atom_state = Multiply()([atom_state, atom_weights])
    # bond_state = Multiply()([bond_state, bond_weights])

    ########################## Output Layers
    # Add the new prediction layers
    output_layers = []

    dense_output = dense_series(bond_state, [64, 16], "Dense_After_MP")

    for prediction_column in prediction_columns:
        output_layers.append(embedding_to_output(dense_output, prediction_column))

    outputs = tf.concat(output_layers, axis=-1)

    model = Model(base_model.inputs, outputs)

    return model


def build_train_tl_model(base_model, preprocessor, model_summary, prediction_columns, params):
    model = build_transfer_learning_model(base_model, preprocessor, model_summary, prediction_columns, params)
    print(f"Total number of trainable parameters: {np.sum([np.prod(v.get_shape()) for v in model.trainable_weights])}")
    model = base_train_model(model, params)
    return model


def build_train_tl_model_hybrid(base_model, preprocessor, model_summary, prediction_columns, params):
    model = build_transfer_learning_model(base_model, preprocessor, model_summary, prediction_columns, params)
    print(f"Total number of trainable parameters: {np.sum([np.prod(v.get_shape()) for v in model.trainable_weights])}")
    model = train_model_hybrid(model, params)
    return model

