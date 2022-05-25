from tensorflow.keras import Model

import argparse as pars
from layers import (
    TrksLayers,
    DenseNetwork,
    DotProduct
)
from Helpermodules import GetConfiguration

#from Helpermodules import GetConfiguration
import numpy as np

parser = pars.ArgumentParser()
parser.add_argument('--config', '-c', nargs='+', help='config file giving the network parameters')
args = parser.parse_args()
config = GetConfiguration(args.config[0])

x = np.arange(60000000).reshape(100000,40,15)
y = np.arange(1800000).reshape(100000,18)

input_feat = (40,15)
input_weight = (40,15)

nodes_feat = config.edge_feature_network["nodes"]
nodes_weight = config.edge_weight_network["nodes"]
nodes_vertex = config.vertex_network["nodes"]

edge_feat_input, edge_feat_output = TrksLayers(nodes = nodes_feat, net_name="edge_feat")(input_feat) 
edge_weight_input, edge_weight_output = TrksLayers(nodes = nodes_weight, net_name="edge_weight")(input_weight)
dt_product = DotProduct(edge_weight_output,edge_feat_output)()

dense_vertex_output = DenseNetwork(nodes=nodes_vertex)(dt_product)

model = Model(inputs = [edge_feat_input, edge_weight_input], outputs = dense_vertex_output)
model.summary()
model.compile(optimizer="Adam", loss="BinaryCrossentropy")

model.fit([x,x],y)
