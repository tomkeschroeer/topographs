"""script to build topograph network."""
from tensorflow.keras import Model

from topograph.modules.layers import (
    DenseNetwork,
    DotProduct,
    EdgeLayers,
    FeatLayers,
    ShiftRelu,
    Sigmoid,
)


class TopographModel:
    """
    class building the topograph model
    """

    def __init__(
        self,
        config: object,
        metadata_dict: dict,
        edge_weight_layer_name: str = "edge_weight",
        edge_feat_layer_name: str = "edge_feat",
        vertex_network_layer_name: str = "vertex_network",
        input_weight_layer_name: str = "input_1",
        input_feat_layer_name: str = "input_2",
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
        self.config = config
        self.metadata_dict = metadata_dict
        self.edge_weight_layer_name = edge_weight_layer_name
        self.edge_feat_layer_name = edge_feat_layer_name
        self.vertex_network_layer_name = vertex_network_layer_name
        self.input_weight_layer_name = input_weight_layer_name
        self.input_feat_layer_name = input_feat_layer_name

        self.nodes_feat = self.config.edge_feature_network["nodes"]
        self.nodes_weight = self.config.edge_weight_network["nodes"]
        self.nodes_vertex = self.config.vertex_network["nodes"]
        self.feat_layer = FeatLayers(
            nodes=self.nodes_feat, net_name=self.edge_feat_layer_name
        )
        self.edge_layer = EdgeLayers(
            nodes=self.nodes_weight, net_name=self.edge_weight_layer_name
        )
        add_activation = self.config.edge_weight_network.get("add_activation", None)
        if add_activation == "shifted_relu":
            self.add_activation = ShiftRelu()
        elif add_activation == "sigmoid":
            self.add_activation = Sigmoid()
        elif add_activation is None:
            self.add_activation = False
        else:
            raise KeyError(
                f"Undefined additional actrivation: {add_activation}. Please select one"
                ' of the following: ["shifted_relu", "sigmoid"] or leave empty/remove'
                " option."
            )
        self.dot_product = DotProduct()
        self.dense_vertex_out = DenseNetwork(
            nodes=self.nodes_vertex, net_name=self.vertex_network_layer_name
        )

    def get_model(self, input_feat, input_weight):
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
        edge_wt_out = self.edge_layer(input_feat)
        edge_feat_out = self.feat_layer(input_weight)
        if self.add_activation:
            add_activation = self.add_activation(edge_wt_out)
        else:
            add_activation = edge_wt_out
        dt_product = self.dot_product([edge_feat_out, add_activation])
        dense_vertex_out = self.dense_vertex_out(dt_product)

        model = Model(
            inputs=[input_feat, input_weight],
            outputs=[edge_wt_out, dense_vertex_out],
        )
        model.summary()
        model.compile(
            optimizer="Adam",
            run_eagerly=True,
            loss={
                self.edge_weight_layer_name: "binary_crossentropy",
                self.vertex_network_layer_name: "mean_squared_error",
            },
            loss_weights={
                self.edge_weight_layer_name: 100,
                self.vertex_network_layer_name: 1,
            },
            weighted_metrics=["accuracy"],
        )
        return model
