from pickletools import optimize
from tensorflow.keras.layers import(
    Dense,
    Dot,
    Input,
    Concatenate
) 
from tensorflow.keras import Model

import argparse as pars
from layers import (
    trks_layers,
    dense_network,
    DotProduct
)
#from Helpermodules import GetConfiguration
import numpy as np

parser = pars.ArgumentParser()
parser.add_argument('--config', '-c', nargs='+', help='config file giving the network parameters')
args = parser.parse_args()
#config = GetConfiguration(args.config_file[0])

x = np.arange(60000000).reshape(100000,40,15)
y = np.arange(1800000).reshape(100000,18)

input_feat = (40,15)
input_weight = (40,15)
nodes_feat = [70,70,70,30]
nodes_weight = [40,40,40,1]

edge_feat_input, edge_feat_output = trks_layers(nodes = nodes_feat, output_nodes=30, net_name="edge_feat")(input_feat) 
edge_weight_input, edge_weight_output = trks_layers(nodes = nodes_weight, output_nodes=1, net_name="edge_weight")(input_weight)
dt_product = DotProduct(edge_weight_output,edge_feat_output)()

dense_vertex_output = dense_network(nodes=[50,50,50], output_nodes=18)(dt_product)

model = Model(inputs = [edge_feat_input, edge_weight_input], outputs = dense_vertex_output)
model.summary()
model.compile(optimizer="Adam", loss="BinaryCrossentropy")

model.fit([x,x],y)
