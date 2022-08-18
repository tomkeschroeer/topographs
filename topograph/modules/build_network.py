"""script to build topograph network."""
from tensorflow.keras import Model

from topograph.modules.layers import (
    DenseNetwork,
    DotProduct,
    EdgeLayers,
    FeatLayers,
    ShiftRelu,
)


class TopographModel:
    def __init__(self, config, metadata_dict):
        # super(TopographModel, self).__init__()
        self.config = config
        self.metadata_dict = metadata_dict
        self.nodes_feat = self.config.edge_feature_network["nodes"]
        self.nodes_weight = self.config.edge_weight_network["nodes"]
        self.nodes_vertex = self.config.vertex_network["nodes"]
        self.feat_layer = FeatLayers(nodes=self.nodes_feat, net_name="edge_feat")
        self.edge_layer = EdgeLayers(nodes=self.nodes_weight, net_name="edge_weight")
        self.shiftrelu = ShiftRelu()
        self.dot_product = DotProduct()
        self.dense_vertex_out = DenseNetwork(
            nodes=self.nodes_vertex, net_name="vertex_network"
        )

    def get_model(self, input_feat, input_weight):
        edge_wt_out = self.edge_layer(input_feat)
        edge_feat_out = self.feat_layer(input_weight)
        shiftrelu = self.shiftrelu(edge_wt_out)
        dt_product = self.dot_product([edge_feat_out, shiftrelu])
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
                "edge_weight": "binary_crossentropy",
                "vertex_network": "mean_squared_error",
            },
            loss_weights={
                "edge_weight": 100,
                "vertex_network": 1,
            },
            weighted_metrics=["accuracy"],
        )
        return model
