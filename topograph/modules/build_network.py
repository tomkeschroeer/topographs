"""script to build topograph network."""
# from tensorflow.keras import Model

from topograph.modules.layers import (
    VertexNetwork,
    DotProduct,
    EdgeLayers,
    FeatLayers,
    ShiftRelu,
    Sigmoid,
)
from torch.nn import (
    Module,
    BCELoss,
    MSELoss
) 


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

        self.loss_edges = BCELoss()
        self.loss_vertex = MSELoss()

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

    def forward(self, input_feat, input_weight):
        """
        function to build and return the topograph model

        Parameters
        ----------
        input_feat : tensorflow.keras.layers.Input
            Input for the layer for the edge feature prediction.
        input_weight : tensorflow.keras.layers.Input
            Input for the layer for the edge weights prediction.

        Returns
        -------
        model
            topograph model ready to be trained.
        """
        edge_wt_out = self.edge_layer(input_weight)
        edge_feat_out = self.feat_layer(input_feat)
        if self.add_activation:
            add_activation = self.add_activation(edge_wt_out)
        else:
            add_activation = edge_wt_out
        dt_product = self.dot_product([edge_feat_out, add_activation])
        dense_vertex_out = self.vertex_network(dt_product)
        return dense_vertex_out, edge_wt_out
