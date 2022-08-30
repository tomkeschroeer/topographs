import tensorflow.keras.backend as K
from tensorflow import cast, greater
from tensorflow.keras import activations, initializers  # pylint: disable=import-error
from tensorflow.keras.layers import (  # pylint: disable=import-error
    Activation,
    Dense,
    Layer,
    TimeDistributed,
)
from tensorflow.math import exp


class Sigmoid(Layer):
    """
    class for the Sigmoid Layer to be used as an Activation for the edge weights.
    """

    def __init__(self, **kwargs):
        """
        Init of the Sigmoid Layer.
        """
        super(Sigmoid, self).__init__(**kwargs)
        self.c1 = self.add_weight(shape=(), trainable=True, name="c1_sigmoid")
        self.c2 = self.add_weight(shape=(), trainable=True, name="c2_sigmoid")

    def call(self, x):
        """
        Define what happens when layer is called.

        Parameters
        ----------
        x : tf.Tensor
            output of previous layer.

        Returns
        -------
        x with Sigmoid activation applied.
        """
        return 1 / (1 + exp(-self.c1 * (x - self.c2)))


class ShiftRelu(Layer):
    """
    class for the Shifted ReLu Layer to be used as an Activation for the edge weights.
    """

    def __init__(self, **kwargs):
        """
        Init of the ShiftReLu Layer.
        """
        super(ShiftRelu, self).__init__(**kwargs)
        shift_initializer = initializers.Constant(0.8)
        slope_initializer = initializers.Constant(1.0)
        self.shift = self.add_weight(
            shape=(), trainable=True, name="relu_shift", initializer=shift_initializer
        )
        self.slope = self.add_weight(
            shape=(), trainable=True, name="relu_slope", initializer=slope_initializer
        )

    def call(self, x):
        """
        Define what happens when layer is called.

        Parameters
        ----------
        x : tf.Tensor
            output of previous layer.

        Returns
        -------
        x with Shifted ReLu activation applied.
        """
        return (
            self.slope * (x - self.shift) * cast(greater(x, self.shift), dtype=x.dtype)
        )


class EdgeLayers(Layer):
    """
    class for the layer to predict the edge weights.
    """

    def __init__(self, nodes, net_name, **kwargs):
        """
        Init for EdgeLayers

        Parameters
        ----------
        nodes: list
            list of the number of nodes for all hidden layers.
        net_name: str
            name of the network.
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
        """
        Define what happens when layer is called.

        Parameters
        ----------
        input_layer : object
            input of EdgeLayer.

        Returns
        -------
        tdd : object
            output layer of EdgeLayer.
        """
        # Set the track input
        tdd = self.layers[0](input_layer)
        # Define the TimeDistributed layers for the different tracks
        for layer in self.layers[1:]:
            tdd = layer(tdd)
        return tdd

    def get_config(self):
        """
        function to add parameters to class config.

        Returns
        -------
        config : dict
            modified config.
        """
        config = super().get_config()
        config.update({"nodes": self.nodes, "net_name": self.net_name})
        return config


class FeatLayers(Layer):
    """
    class for the layer to predict the edge weights.
    """

    def __init__(self, nodes, net_name, **kwargs):
        """
        Init for FeatLayers.

        Parameters
        ----------
        nodes: list
            list of the number of nodes for all hidden layers.
        net_name: str
            name of the network.
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
        """
        Define what happens when layer is called.

        Parameters
        ----------
        input_layer : object
            input of FeatLayer.

        Returns
        -------
        tdd : object
            output layer of FeatLayer.
        """
        tdd = self.layers[0](input_layer)
        for layer in self.layers[1:]:
            tdd = layer(tdd)
        return tdd

    def get_config(self):
        """
        function to add parameters to class config.

        Returns
        -------
        config : dict
            modified config.
        """
        config = super().get_config()
        config.update(
            {"nodes": self.nodes, "net_name": self.net_name, "name": self.net_name}
        )
        return config


class DenseNetwork(Layer):
    """
    class for the dense layer to predict the vertex feature.
    """

    def __init__(self, nodes, net_name, **kwargs):
        """
        Init for DenseNetwork.

        Parameters
        ----------
        nodes: list
            list of the number of nodes for all hidden layers.
        net_name: str
            name of the network.
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

    def call(self, input_layer):
        """
        Define what happens when layer is called.

        Parameters
        ----------
        input_layer : object
            input of DenseNetwork, output from the Dot product of the
            EdgeLayer and FeatLayer.

        Returns
        -------
        dense_ntw : object
            output layer of FeatLayer.
        """
        dense_ntw = self.layers[0](input_layer)
        for layer in self.layers[1:]:
            dense_ntw = layer(dense_ntw)
        return dense_ntw

    def get_config(self):
        """
        function to add parameters to class config.

        Returns
        -------
        config : dict
            modified config.
        """
        config = super().get_config()
        config.update(
            {"nodes": self.nodes, "net_name": self.net_name, "name": self.net_name}
        )
        return config


class DotProduct(Layer):
    """
    class for the dense layer to predict the vertex feature.
    """

    def __init__(self, **kwargs):
        """
        Init for DotProduct.
        """
        super().__init__(**kwargs)

    def call(self, inputs):
        """
        Define what happens when layer is called.

        Parameters
        ----------
        inputs : list
            list of layers to calculate the dot product from.

        Returns
        -------
        pool : object
            dot product of the inputs.
        """
        feat_layer, edge_layer = inputs[:2]
        pool = K.batch_dot(K.permute_dimensions(edge_layer, (0, 2, 1)), feat_layer)
        pool = K.squeeze(pool, -2)
        return pool
