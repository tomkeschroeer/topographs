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

class TrksLayers(Layer):
    def __init__(self, nodes, net_name):
        self.nodes = nodes
        self.net_name = net_name

    def __call__(self, input_shape):
        # Set the track input
        input = Input(shape = input_shape)
        masked_inputs = Masking(mask_value=0)(input)
        tdd = masked_inputs

        # Define the TimeDistributed layers for the different tracks
        for i, phi_nodes in enumerate(self.nodes[:-1]):

            tdd = TimeDistributed(Dense(phi_nodes), name=f"{self.net_name}_Phi{i}_Dense")(tdd)

            tdd = TimeDistributed(Activation(activations.relu), name=f"{self.net_name}_Phi{i}_ReLU")(
                tdd
            )

        # Set output and activation function
        output = TimeDistributed(Dense(self.nodes[-1], activation="softmax"), name=self.net_name)(tdd)

        return input, output

class DenseNetwork(Layer):
    def __init__(
        self,
        nodes,
    ):
        self.nodes = nodes

    def __call__(self, dense_ntw):
        for node in self.nodes[:-1]:
            dense_ntw = Dense(node)(dense_ntw)
        output = Dense(self.nodes[-1], activation="softmax", name="Jet_class")(dense_ntw)
        return output

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
        return pool
    

