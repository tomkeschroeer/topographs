from tensorflow.keras import Model

from modules.layers import (
    TrksLayers,
    DenseNetwork,
    DotProduct
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
    nodes_feat = config.edge_feature_network["nodes"]
    nodes_weight = config.edge_weight_network["nodes"]
    nodes_vertex = config.vertex_network["nodes"]

    edge_feat_input, edge_feat_output = TrksLayers(nodes = nodes_feat, net_name="edge_feat")(input_feat) 
    edge_weight_input, edge_weight_output = TrksLayers(nodes = nodes_weight, net_name="edge_weight")(input_weight)
    dt_product = DotProduct(edge_weight_output,edge_feat_output)()

    dense_vertex_output = DenseNetwork(nodes=nodes_vertex)(dt_product)

    model = Model(inputs = [edge_feat_input, edge_weight_input], outputs = dense_vertex_output)
    #model.summary()
    model.compile(optimizer="Adam", loss="BinaryCrossentropy")
    return model
