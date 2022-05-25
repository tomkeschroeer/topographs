"""Keras model of the DIPS tagger."""
import os

import h5py
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras import activations  # pylint: disable=import-error
from tensorflow.keras.callbacks import ModelCheckpoint  # pylint: disable=import-error
from tensorflow.keras.layers import (  # pylint: disable=import-error
    Activation,
    Dense,
    Masking,
    TimeDistributed,
    Layer,
    Dot,
    Input,
    Flatten
)
from tensorflow.keras.backend import batch_dot
from tensorflow.keras.models import Model  # pylint: disable=import-error
from tensorflow.keras.optimizers import Adam  # pylint: disable=import-error


class trks_layers(Layer):
    """Keras model definition of DIPS.

    Parameters
    ----------
    train_config : object
        training config
    input_shape : tuple
        dataset input shape
    continue_training : bool, optional
        Decide, if the training is continued using the latest
        model file, by default False

    Returns
    -------
    keras model
        Dips keras model
    int
        Number of epochs
    int
        Starting epoch number
    """
    def __init__(self, nodes, output_nodes, net_name):
        self.nodes = nodes
        self.output_nodes = output_nodes
        self.net_name = net_name

    def __call__(self, input_shape):
        # Set the track input
        input = Input(shape = input_shape)
        masked_inputs = Masking(mask_value=0)(input)
        tdd = masked_inputs

        # Define the TimeDistributed layers for the different tracks
        for i, phi_nodes in enumerate(self.nodes):

            tdd = TimeDistributed(Dense(phi_nodes), name=f"{self.net_name}_Phi{i}_Dense")(tdd)

            tdd = TimeDistributed(Activation(activations.relu), name=f"{self.net_name}_Phi{i}_ReLU")(
                tdd
            )

        # Set output and activation function
        output = TimeDistributed(Dense(self.output_nodes, activation="softmax"), name=self.net_name)(tdd)

        return input, output

class dense_network(Layer):
    def __init__(
        self,
        nodes,
        output_nodes,
    ):
        self.nodes = nodes
        self.output_nodes = output_nodes

    def __call__(self, dense_ntw):
        for node in self.nodes:
            dense_ntw = Dense(node)(dense_ntw)
        output = Dense(self.output_nodes, activation="softmax", name="Jet_class")(dense_ntw)

class DotProduct(Layer):
    def __init__(
        self,
        layer1,
        layer2
    ):
        self.layer1 = layer1
        self.layer2 = layer2
    def __call__(self):
        pool = Dot(axes = 1)([self.layer1, self.layer2])
        pool = Flatten()(pool)
        # print(f"pool vorher = {pool}")
        # pool = K.squeeze(pool, -2)
        # print(f"pool danach = {pool}")
        return pool
    

