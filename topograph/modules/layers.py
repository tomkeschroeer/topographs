from torch.nn import (
    ReLU,
    Softmax,
    Linear,
    Module,
    init
)

from torch import(
    bmm,
    squeeze,
    empty,
    greater,
    exp
)

class Sigmoid(Module):
    """
    class for the Sigmoid Layer to be used as an Activation for the edge weights.
    """

    def __init__(self, **kwargs):
        """
        Init of the Sigmoid Layer.
        """
        super(Sigmoid, self).__init__(**kwargs)
        self.c1 = empty(1)
        init.constant_(self.c1, 1)
        self.c2 = empty(1)
        init.constant_(self.c2, 1)

    def forward(self, x):
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


class ShiftRelu(Module):
    """
    class for the Shifted ReLu Layer to be used as an Activation for the edge weights.
    """

    def __init__(self, **kwargs):
        """
        Init of the ShiftReLu Layer.
        """
        super(ShiftRelu, self).__init__(**kwargs)
        self.shift = empty(1)
        init.constant_(self.shift, 0.8)
        self.slope = empty(1)
        init.constant_(self.slope, 1.0)

    def forward(self, x):
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
            self.slope * (x - self.shift) * greater(x, self.shift)
        )


class EdgeLayers(Module):
    """
    class for the layer to predict the edge weights.
    """

    def __init__(self, nodes):
        """
        Init for EdgeLayers

        Parameters
        ----------
        nodes: list
            list of the number of nodes for all hidden layers.
        net_name: str
            name of the network.
        """
        super().__init__()
        self.nodes = nodes
        self.layers = []
        for i in range(1,len(self.nodes)-1):
            self.layers.append(
                Linear(self.nodes[i-1], self.nodes[i])
            )
            self.layers.append(
                ReLU()
            )
        self.layers.append(
            Linear(self.nodes[-2], self.nodes[-1])
        )
        self.layers.append(
            Softmax(dim=2)
        )

    def forward(self, input_layer):
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


class FeatLayers(Module):
    """
    class for the layer to predict the edge weights.
    """

    def __init__(self, nodes):
        """
        Init for FeatLayers.

        Parameters
        ----------
        nodes: list
            list of the number of nodes for all hidden layers.
        net_name: str
            name of the network.
        """
        super().__init__()
        self.nodes = nodes
        self.layers = []
        for i in range(1,len(self.nodes)-1):
            self.layers.append(
                Linear(self.nodes[i-1], self.nodes[i])
            )
            self.layers.append(
                ReLU()
            )

        # Set output and activation function
        self.layers.append(
            Linear(self.nodes[-2], self.nodes[-1])
        )

    def forward(self, input_layer):
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


class VertexNetwork(Module):
    """
    class for the dense layer to predict the vertex feature.
    """

    def __init__(self, nodes, **kwargs):
        """
        Init for DenseNetwork.

        Parameters
        ----------
        nodes: list
            list of the number of nodes for all hidden layers.
        """
        super().__init__()
        self.nodes = nodes
        self.layers = []
        for i in range(1,len(self.nodes)-1):
            self.layers.append(
                Linear(self.nodes[i-1], self.nodes[i])
            )
            self.layers.append(
                ReLU()
            )

        self.layers.append(
            Linear(self.nodes[-2], self.nodes[-1])
        )

    def forward(self, input_layer):
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


class DotProduct(Module):
    """
    class for the dense layer to predict the vertex feature.
    """

    def __init__(self, **kwargs):
        """
        Init for DotProduct.
        """
        super().__init__(**kwargs)

    def forward(self, inputs):
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
        edge_shape = edge_layer.size()
        edge_shape = (edge_shape[0], edge_shape[2], edge_shape[1])
        pool = bmm(edge_layer.view(edge_shape), feat_layer)
        pool = squeeze(pool, -2)
        return pool

class TopographModel(Module):
    """
    class building the topograph model
    """

    def __init__(
        self,
        nodes_feat: list = [128, 30, 30, 30],
        nodes_weight : list = [128, 30, 30, 1],
        nodes_vertex : list = [30, 50, 50, 50, 1],
        activation_name: str = None,
    ):
        """
        Init of TopographModel class

        Parameters
        ----------
        config : object
            GetConfiguration object including information about the model
            architecture, train parameter etc.
        metadata_dict: dict
            dictionary giving the number of jets, tracks, features, etc.
        edge_weight_layer_name : str
            name of the layer predicting the edge weights.
        edge_feat_layer_name : str
            name of the layer predicting the edge features.
        vertex_network_layer_name : str
            name of the layer predicting the vertex features.
        input_weight_layer_name : str
            name of the input layer for the edge weight layer
        input_feat_layer_name : str
            name of the input layer for the edge feature layer
        """
        super().__init__()
        self.activation_name = activation_name

        self.nodes_feat = nodes_feat
        self.nodes_weight = nodes_weight
        self.nodes_vertex = nodes_vertex

        self.feat_layer = FeatLayers(nodes=self.nodes_feat)
        self.edge_layer = EdgeLayers(nodes=self.nodes_weight)

        if self.activation_name == "shifted_relu":
            self.add_activation = ShiftRelu()
        elif self.activation_name == "sigmoid":
            self.add_activation = Sigmoid()
        elif self.activation_name is None:
            self.add_activation = False
        else:
            raise KeyError(
                f"Undefined additional actrivation: {self.activation_name}. Please select one"
                ' of the following: ["shifted_relu", "sigmoid"] or leave empty/remove'
                " option."
            )
        self.dot_product = DotProduct()
        self.vertex_network = VertexNetwork(nodes=self.nodes_vertex)