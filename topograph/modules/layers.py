import tensorflow.keras.backend as K
from tensorflow.keras import activations  # pylint: disable=import-error
from tensorflow.keras.layers import (  # pylint: disable=import-error
    Activation,
    Dense,
    Layer,
    TimeDistributed,
)


class ShiftRelu(Layer):
    def __init__(self):
        super(ShiftRelu, self).__init__()
        self.fac = self.add_weight(shape=(1,), trainable=True, name="new_relu_factor")

    def call(self, x):
        shifted_relu = K.switch(x >= self.fac, (x + self.fac), (x - self.fac))
        # shifted_relu = K.squeeze(shifted_relu, axis=2)
        return shifted_relu


class EdgeLayers(Layer):
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
        super(EdgeLayers, self).__init__(name=net_name)
        self.nodes = nodes
        self.net_name = net_name
        # self.fac = self.add_weight(shape=(1,), trainable=True, name="new_relu_factor")

        self.slope_vals = []

    # def shifted_relu_activation_train(self, x):
    #     return K.switch(x >= self.fac, (x + self.fac), (x - self.fac))

    def call(self, input_layer):
        # Set the track input
        tdd = TimeDistributed(Dense(self.nodes[0]), name=f"{self.net_name}_Phi0_Dense")(
            input_layer
        )
        # Define the TimeDistributed layers for the different tracks
        for i, phi_nodes in enumerate(self.nodes[1:-1]):

            tdd = TimeDistributed(
                Dense(phi_nodes), name=f"{self.net_name}_Phi{i}_Dense"
            )(tdd)

            tdd = TimeDistributed(
                Activation(activations.relu), name=f"{self.net_name}_Phi{i}_ReLU"
            )(tdd)

        # Set output and activation function
        output = TimeDistributed(
            Dense(self.nodes[-1], activation="sigmoid"), name=f"{self.net_name}"
        )(tdd)
        output = K.squeeze(output, -1)
        print(output)
        output = ShiftRelu()(output)

        # output = Activation(self.shifted_relu_activation_train, name=self.net_name)(output_prev)
        return output


class FeatLayers(Layer):
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
        super(FeatLayers, self).__init__(name=net_name)
        self.nodes = nodes
        self.net_name = net_name

    def call(self, input_layer):
        # Set the track input
        tdd = TimeDistributed(Dense(self.nodes[0]), name=f"{self.net_name}_Phi0_Dense")(
            input_layer
        )
        # Define the TimeDistributed layers for the different tracks
        for i, phi_nodes in enumerate(self.nodes[1:-1]):

            tdd = TimeDistributed(
                Dense(phi_nodes), name=f"{self.net_name}_Phi{i}_Dense"
            )(tdd)

            tdd = TimeDistributed(
                Activation(activations.relu), name=f"{self.net_name}_Phi{i}_ReLU"
            )(tdd)

        # Set output and activation function
        output = TimeDistributed(
            Dense(self.nodes[-1], activation="linear"),
            name=f"{self.net_name}_Phi{len(self.nodes)}_Dense",
        )(tdd)

        return output


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

    def __init__(self, nodes, net_name):
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
        # self.name = net_name

    def call(self, dense_ntw):
        for i, node in enumerate(self.nodes[:-1]):
            dense_ntw = Dense(node, name=f"{self.net_name}_layer_{i}")(dense_ntw)
            dense_ntw = Activation(activations.relu, name=f"{self.net_name}_ReLu_{i}")(
                dense_ntw
            )
        dense_ntw = Dense(self.nodes[-1], activation="linear", name=self.net_name)(
            dense_ntw
        )
        return dense_ntw


class DotProduct(Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, inputs):
        feat_layer, edge_layer = inputs[:2]
        # assert (len(attention.shape) == 2) & (len(features.shape) == 3), "Please provide attention tensor as first argument (rank 2), followed by feature tensor (rank 3)"
        pool = K.batch_dot(feat_layer, K.expand_dims(edge_layer, 1))
        print(pool)
        # pool = K.squeeze(pool,-1)
        return pool


# class DotProduct(Layer):
#     """
#     Define a DotProduct as a layer
#     """

#     def __init__(self):
#         """
#         Init for DotProduct

#         Parameters
#         ----------
#         layer1: Layer object
#             first layer used for dot product
#         layer1: Layer object
#             second layer used for dot product

#         Returns
#         -------
#         pool : object
#             returns the dot product of two layers
#         """
#         super().__init__()

#     def call(self, layers):
#         # print(layer[0].shape)
#         # print(layer[1].shape)
#         pool = Dot(axes=1)(layers)
#         print(pool.shape)
#         # pool = Reshape((pool.shape[1]))(pool)
#         print(pool.shape)
#         pool = Flatten()(pool)
#         print(pool.shape)
#         return pool
