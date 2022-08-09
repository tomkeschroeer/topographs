"""script to build topograph network."""

from tensorflow.keras import Model
from tensorflow.keras.layers import Layer

from topograph.modules.layers import DenseNetwork, DotProduct, EdgeLayers, FeatLayers


class TopographModel(Layer):
    def __init__(self, config):
        self.config = config

    def __call__(self, input_feat, input_weight):
        nodes_feat = self.config.edge_feature_network["nodes"]
        nodes_weight = self.config.edge_weight_network["nodes"]
        nodes_vertex = self.config.vertex_network["nodes"]

        edge_feat_input, edge_feat_out = FeatLayers(
            nodes=nodes_feat, net_name="edge_feat"
        )(input_feat)
        edge_wt_input, edge_wt_out_prev, edge_wt_out = EdgeLayers(
            nodes=nodes_weight, net_name="edge_weight"
        )(
            input_weight
        )  # EdgeLayers(nodes = nodes_weight, net_name="edge_weight")(input_weight)
        dt_product = DotProduct(edge_wt_out, edge_feat_out)()

        dense_vertex_out = DenseNetwork(nodes=nodes_vertex, net_name="vertex_network")(
            dt_product
        )

        return (
            edge_feat_input,
            edge_wt_input,
            edge_feat_out,
            edge_wt_out_prev,
            dense_vertex_out,
        )


def get_model(input_feat, input_weight, config):
    """
    build graph network model for b tagging

    Parameters
    ----------
    input_feat: tuple
        Size of the input for the network for the edge features
    input_weight: tuple
        Size of the input for the network for the edge weights
    config: object
        config defining the network parameters

    Returns
    -------
    model : Model object
        returns graph network model
    """
    (
        edge_feat_input,
        edge_wt_input,
        edge_feat_out,
        edge_wt_out,
        dense_vertex_out,
    ) = TopographModel(config=config)(input_feat, input_weight)
    model = Model(
        inputs=[edge_feat_input, edge_wt_input],
        outputs=[edge_wt_out, dense_vertex_out],
    )
    model.summary()
    model.compile(
        optimizer="Adam",
        loss={
            "edge_weight_sigmoid": "binary_crossentropy",
            "vertex_network": "mean_squared_error",
        },
        loss_weights={
            "edge_weight_sigmoid": 100,
            "vertex_network": 1,
        },
        metrics=["accuracy"],
    )
    return model
