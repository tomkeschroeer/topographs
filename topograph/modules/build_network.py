from tensorflow.keras import Model

from topograph.modules.layers import (
    EdgeLayers,
    FeatLayers,
    DenseNetwork,
    DotProduct
)

from tensorflow.keras.layers import Layer

class TopographModel(Layer):
    def __init__(self, config):
        self.config = config
    
    def __call__(self, input_feat, input_weight):
        nodes_feat = self.config.edge_feature_network["nodes"]
        nodes_weight = self.config.edge_weight_network["nodes"]
        nodes_vertex = self.config.vertex_network["nodes"]

        edge_feat_input, edge_feat_output = FeatLayers(nodes = nodes_feat, net_name="edge_feat")(input_feat) 
        edge_weight_input, edge_weight_output = FeatLayers(nodes = nodes_weight, net_name="edge_weight")(input_weight) #EdgeLayers(nodes = nodes_weight, net_name="edge_weight")(input_weight)
        dt_product = DotProduct(edge_weight_output,edge_feat_output)()

        dense_vertex_output = DenseNetwork(nodes=nodes_vertex, net_name="vertex_network")(dt_product)

        return edge_feat_input, edge_weight_input, edge_feat_output, edge_weight_output, dense_vertex_output
        

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
    edge_feat_input, edge_weight_input, edge_feat_output, edge_weight_output, dense_vertex_output = TopographModel(config=config)(input_feat, input_weight)
    model = Model(inputs = [edge_feat_input, edge_weight_input], outputs = [edge_feat_output, edge_weight_output, dense_vertex_output])
    model.summary()
    model.compile(optimizer="Adam", loss={"edge_feat": "categorical_crossentropy", "edge_weight": "binary_crossentropy", "vertex_network":"categorical_crossentropy"}, loss_weights={"edge_feat": 1, "edge_weight": 1, "vertex_network":1})
    return model
