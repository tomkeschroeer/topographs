"""Keras model of the DIPS tagger."""

from tensorflow.keras import activations  # pylint: disable=import-error
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
    """
    Define a TrksLayers as a layer
    """
    def __init__(self, nodes, net_name):
        """
        Init for TrksLayers

        Parameters
        ----------
        nodes: list
            list of the number of nodes for all hidden layers
        net_name: str
            name of the network

        Returns
        -------
        input : object
            returns input of TrksLayer layer
        output : object
            returns output layer of DenseNetwork
        """
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
    """
    Define a DenseNetwork as a layer
    """
    def __init__(
        self,
        nodes,
    ):
        """
        Init for DenseNetwork

        Parameters
        ----------
        nodes: list
            list of the number of nodes for all hidden layers

        Returns
        -------
        output : object
            returns output layer of DenseNetwork
        """
        self.nodes = nodes

    def __call__(self, dense_ntw):
        for node in self.nodes[:-1]:
            dense_ntw = Dense(node)(dense_ntw)
        output = Dense(self.nodes[-1], activation="softmax", name="Jet_class")(dense_ntw)
        return output

class DotProduct(Layer):
    """
    Define a DotProduct as a layer
    """
    def __init__(
        self,
        layer1,
        layer2
    ):
        """
        Init for DotProduct

        Parameters
        ----------
        layer1: Layer object
            first layer used for dot product
        layer1: Layer object
            second layer used for dot product
        
        Returns
        -------
        pool : object
            returns the dot product of two layers
        """
        self.layer1 = layer1
        self.layer2 = layer2
    def __call__(self):
        pool = Dot(axes = 1)([self.layer1, self.layer2])
        pool = Flatten()(pool)
        return pool
    

