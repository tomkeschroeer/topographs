from torch.nn import (
    ReLU,
    Softmax,
    Linear,
    Module,
    init,
    ModuleList,
    parameter,
    Softplus,
    Sigmoid,
    MSELoss
)

from torch import(
    bmm,
    squeeze,
    empty,
    greater,
    exp,
    reshape,
    tensor,
    Tensor,
    log,
    stack,
    from_numpy,
    sum,
)

import numpy as np

class Softplus_norm(Module):
    """
    class for the Softplus Layer to be used as an Activation for the edge weights.
    """

    def __init__(self,norm, **kwargs):
        """
        Init of the Sigmoid Layer.
        """
        super(Softplus_norm, self).__init__(**kwargs)
        self.norm = norm
        
    def forward(self, x):
        """
        Define what happens when layer is called.

        Parameters
        ----------
        x : tf.Tensor
            output of previous layer.

        Returns
        -------
        x with normalised Softmax activation applied.
        """
        out = log(1+exp(x))/self.norm
        return log(1+exp(x))/self.norm
        
class Sigmoid_tr(Module):
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
        self.shift = parameter.Parameter(Tensor(tensor(0.2)), requires_grad=True)
        self.slope = parameter.Parameter(Tensor(tensor(1.0)), requires_grad=True)

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
        self.layers = ModuleList()
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
            Sigmoid()
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
        # print(input_layer)
        tdd = self.layers[0](input_layer)
        for layer in self.layers[1:]:
            tdd = layer(tdd)
        return tdd


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
        self.layers = ModuleList()
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
        # print(input_layer.shape)
        tdd = self.layers[0](input_layer)
        for layer in self.layers[1:]:
            tdd = layer(tdd)
        return tdd


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
        self.layers = ModuleList()
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
        dense_ntw = self.layers[0](tensor(input_layer))
        for layer in self.layers[1:]:
            dense_ntw = layer(dense_ntw)
        return dense_ntw


class DotProduct(Module):
    """
    class for the dense layer to predict the vertex feature.
    """

    def __init__(self, **kwargs):
        """
        Init for DotProduct.
        """
        super().__init__(**kwargs)

    def forward(self, feat_layer, edge_layer, mask):
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
        # sum(features * edges,axis=1)
        pool = sum(feat_layer * edge_layer, axis=1)  #* mask.unsqueeze(-1)
        # pool = pool.sum()
        # pool = squeeze(pool,0)
        return pool


class MultipleMSELoss(Module):
    def __init__(self):
        super().__init__()
        self.mse_per_var = MSELoss(reduction="none")
        
    def forward(self, output, target):
        if target.size()[-1] == 1: return self.mse_per_var(target,output).mean()
        average = target.abs().mean(dim=0)+1e-8
        loss = (self.mse_per_var(target, output)/average)
        loss = loss.mean()
        return loss
    