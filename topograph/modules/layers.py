import tensorflow.keras.backend as K
from tensorflow import cast, greater
from tensorflow.keras import activations  # pylint: disable=import-error
from tensorflow.keras.layers import (  # pylint: disable=import-error
    Activation,
    Dense,
    Layer,
    TimeDistributed,
)
from tensorflow.math import exp


class Sigmoid(Layer):
    def __init__(self, **kwargs):
        super(Sigmoid, self).__init__(**kwargs)
        self.c1 = self.add_weight(shape=(1,), trainable=True, name="c1_sigmoid")
        self.c2 = self.add_weight(shape=(1,), trainable=True, name="c2_sigmoid")

    def call(self, x):
        return 1 / (1 + exp(-self.c1 * (x - self.c2)))


class ShiftRelu(Layer):
    def __init__(self, **kwargs):
        super(ShiftRelu, self).__init__(**kwargs)
        self.shift = self.add_weight(shape=(1,), trainable=True, name="relu_shift")
        self.slope = self.add_weight(shape=(1,), trainable=True, name="relu_slope")

    def call(self, x):
        return (
            self.slope * (x - self.shift) * cast(greater(x, self.shift), dtype=x.dtype)
        )


class EdgeLayers(Layer):
    """
    Define a TrksLayers as a layer
    """

    def __init__(self, nodes, net_name, **kwargs):
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
        super(EdgeLayers, self).__init__(name=net_name)
        self.nodes = nodes
        self.net_name = net_name

        # self.fac = self.add_weight(shape=(1,), trainable=True, name="new_relu_factor")
        self.layers = []
        for i, phi_nodes in enumerate(self.nodes[:-1]):
            self.layers.append(
                TimeDistributed(Dense(phi_nodes), name=f"{self.net_name}_Phi{i}_Dense")
            )
            self.layers.append(
                TimeDistributed(
                    Activation(activations.relu), name=f"{self.net_name}_Phi{i}_ReLU"
                )
            )
        self.layers.append(
            TimeDistributed(
                Dense(self.nodes[-1], activation="sigmoid"), name=f"{self.net_name}"
            )
        )

    def call(self, input_layer):
        # Set the track input
        tdd = self.layers[0](input_layer)
        # Define the TimeDistributed layers for the different tracks
        for layer in self.layers[1:]:
            tdd = layer(tdd)
        return tdd

    def get_config(self):
        config = super().get_config()
        config.update({"nodes": self.nodes, "net_name": self.net_name})
        return config


class FeatLayers(Layer):
    """
    Define a TrksLayers as a layer
    """

    def __init__(self, nodes, net_name, **kwargs):
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
        super(FeatLayers, self).__init__(name=net_name)  # , **kwargs)
        self.nodes = nodes
        self.net_name = net_name
        self.layers = []
        for i, phi_nodes in enumerate(self.nodes[:-1]):
            self.layers.append(
                TimeDistributed(Dense(phi_nodes), name=f"{self.net_name}_Phi{i}_Dense")
            )
            self.layers.append(
                TimeDistributed(
                    Activation(activations.relu), name=f"{self.net_name}_Phi{i}_ReLU"
                )
            )

        # Set output and activation function
        self.layers.append(
            TimeDistributed(
                Dense(self.nodes[-1], activation="linear"),
                name=f"{self.net_name}_Phi{len(self.nodes)}_Dense",
            )
        )

    def call(self, input_layer):
        # Set the track input
        tdd = self.layers[0](input_layer)
        # Define the TimeDistributed layers for the different tracks
        for layer in self.layers[1:]:
            tdd = layer(tdd)
        return tdd

    def get_config(self):
        config = super().get_config()
        config.update(
            {"nodes": self.nodes, "net_name": self.net_name, "name": self.net_name}
        )
        return config


# def get_dense_network_layers(nodes, net_name):
#     """
#     Define a DenseNetwork as a layer
#     """
#     nodes = nodes
#     net_name = net_name
#     layers = []
#     for i, node in enumerate(nodes[:-1]):
#         layers.append(Dense(node, name=f"vertex_network_dense_{i}"))
#         layers.append(Activation(activations.relu, name=f"{net_name}_ReLu_{i}"))
#     layers.append(Dense(nodes[-1], activation="linear", name=net_name))
#     return layers


class DenseNetwork(Layer):
    """
    Define a DenseNetwork as a layer
    """

    def __init__(self, nodes, net_name, **kwargs):
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
        super(DenseNetwork, self).__init__(name=net_name)
        self.nodes = nodes
        self.net_name = net_name
        self.layers = []
        for i, node in enumerate(self.nodes[:-1]):
            self.layers.append(Dense(node, name=f"{self.net_name}_layer_{i}"))
            self.layers.append(
                Activation(activations.relu, name=f"{self.net_name}_ReLu_{i}")
            )
        self.layers.append(
            Dense(self.nodes[-1], activation="linear", name=self.net_name)
        )
        # self.name = net_name

    def call(self, dense_ntw):
        dense_ntw = self.layers[0](dense_ntw)
        for layer in self.layers[1:]:
            dense_ntw = layer(dense_ntw)
        return dense_ntw

    def get_config(self):
        config = super().get_config()
        config.update(
            {"nodes": self.nodes, "net_name": self.net_name, "name": self.net_name}
        )
        return config


class DotProduct(Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, inputs):
        feat_layer, edge_layer = inputs[:2]
        pool = K.batch_dot(K.permute_dimensions(edge_layer, (0, 2, 1)), feat_layer)
        pool = K.squeeze(pool, -2)
        return pool
