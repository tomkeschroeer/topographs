"""script to build topograph network."""
# from tensorflow.keras import Model

from pathlib import Path
from typing import Union

import numpy as np
import pytorch_lightning as pl
import torch as T
import torch.optim as optim
import wandb
from h5py import File
from torch.autograd import grad
from torch.nn import BCELoss, BCEWithLogitsLoss, Module, MSELoss, Softplus, ModuleList
from torch.nn.functional import binary_cross_entropy_with_logits

from topograph.modules.layers import (
    DotProduct,
    EdgeLayers,
    FeatLayers,
    MultipleMSELoss,
    ShiftRelu,
    Sigmoid,
    Softplus_norm,
    VertexNetwork,
)


class TopographModel(pl.LightningModule):
    """
    A PyTorch-based model for building and training topographical networks.

    The model consists of three branches, on for each type of particle within the jet.
    The tracks or particles considered comprise decay products b-quarks, c-quarks and 
    all remaining.

    It outputs a binary classification for each type of track deciding whether the track 
    originates from the corresponding particle (1 if yes, 0 otherwise).

    Secondly, a regression is performed seperately based on the tracks classified as 
    originating from b-quarks and c-quarks. The regression gives the transverse momentum
    of the hadron containing the quark

    Parameters
    ----------
    nodes_feat : list
        Number of nodes for the layers of the Deep sets extracting 
        the features of the tracks
    nodes_weight : list
        Number of nodes for the layers of the Deep sets used for an
        internatal represenation for the jets and the final binary 
        classification 
    nodes_vertex : list
        Number of nodes for the layers used for the regression task.
    activation_name : str
        Name of the activation function to be used in the model.
    lr : float
        Learning rate for the optimizer.
    loss_fac_edge : float
        Loss factor for edge-related losses.
    loss_fac_vert : float
        Loss factor for vertex-related losses.
    small_net : dict
        Dict with booleans deciding whether for a certain  jet type
        only the binary classification should be performed (set to True) 
        or also the regression (set to False)
    jet_types : list
        List of jet types to be considered in the model.

    Attributes
    ----------
    nodes_feat_prime : list
        Modified node features after applying transformations.
    nodes_weight_prime : list
        Modified node weights after applying transformations.
    feat_layers : torch.nn.ModuleList
        Layers for processing node features.
    edge_layers : torch.nn.ModuleList
        Layers for processing edges.
    dot_products : torch.nn.ModuleList
        Layers for computing dot products between nodes.
    feat_layers_prime : torch.nn.ModuleList
        Layers for processing transformed node features.
    edge_layers_prime : torch.nn.ModuleList
        Layers for processing transformed edges.
    """

    def __init__(
        self,
        nodes_feat: list = [128, 30, 30, 30],
        nodes_weight: list = [128, 30, 30, 1],
        nodes_vertex: list = [30, 50, 50, 50, 1],
        activation_name: str = None,
        lr: float = 1e-3,
        loss_fac_edge: float = 100,
        loss_fac_vert: float = 1,
        small_net: dict = {},
        jet_types: list = ["b", "c", "light"]
    ):
        """
        Init of TopographModel class

        Parameters
        ----------


        """
        super().__init__()
        self.save_hyperparameters()
        self.activation_name = activation_name
        self.lr = lr
        self.loss_fac_edge = loss_fac_edge
        self.loss_fac_vert = loss_fac_vert
        self.small_net = small_net
        self.jet_types = jet_types

        self.nodes_feat = nodes_feat
        self.nodes_weight = nodes_weight
        self.nodes_feat_prime = nodes_feat.copy()
        self.nodes_weight_prime = nodes_weight.copy()
        self.nodes_feat_prime[0] = self.nodes_feat_prime[0] + self.nodes_feat_prime[-1]
        self.nodes_weight_prime[0] = self.nodes_weight_prime[0] + self.nodes_feat_prime[-1]

        self.nodes_vertex = nodes_vertex

        self.feat_layers = ModuleList()
        self.edge_layers = ModuleList()
        self.dot_products = ModuleList()

        self.feat_layers_prime = ModuleList()
        self.edge_layers_prime = ModuleList()
        self.dot_products_prime = ModuleList()

        self.vertex_networks = ModuleList()

        for jet_type in self.jet_types:
            if self.small_net[jet_type]:
                self.vertex_networks.append(None)
                self.feat_layers_prime.append(None)
            else:
                self.vertex_networks.append(VertexNetwork(nodes=self.nodes_vertex))
                self.feat_layers_prime.append(FeatLayers(nodes=self.nodes_feat_prime))
            self.feat_layers.append(FeatLayers(nodes=self.nodes_feat))
            self.edge_layers.append(EdgeLayers(nodes=self.nodes_weight))
            self.dot_products.append(DotProduct())
            self.edge_layers_prime.append(EdgeLayers(nodes=self.nodes_weight_prime))
            self.dot_products_prime.append(DotProduct())
        # Define the loss funcitons
        self.loss_fn_vertex = MSELoss(reduction='none')  # MultipleMSELoss()

    def on_fit_start(self):
        """
        define metrics monitored for weights and biases (wandb)
        """
        if wandb.run:
            wandb.define_metric("train/edge", summary="min")
            wandb.define_metric("valid/edge", summary="min")
            wandb.define_metric("train/total", summary="min")
            wandb.define_metric("train/vertex", summary="min")
            wandb.define_metric("valid/total", summary="min")
            wandb.define_metric("valid/vertex", summary="min")

    def forward(self, inputs):
        """
        function to build and return the topograph model

        Parameters
        ----------
        inputs : torch.Tensor
            Input for the layer for the topograph model

        Returns
        -------
        model
            topograph model ready to be trained.
        """
        edge_wt_outs = [] #ModuleList()
        edge_feat_outs = [] #ModuleList()
        edge_wt_outs_prime = [] #ModuleList()
        edge_feat_outs_prime = [] #ModuleList()
        dot_products = [] #ModuleList()
        dot_products_prime = [] #ModuleList()
        dense_vertex_outs = [] #ModuleList()
        for i, jet_type in enumerate(self.jet_types):
            edge_wt_outs.append(self.edge_layers[i](inputs))
            edge_feat_outs.append(self.feat_layers[i](inputs))
            dot_products.append(self.dot_products[i](edge_wt_outs[i], edge_feat_outs[i]))
            dt_shape = dot_products[i].size()
            inputs_shape = inputs.size()
            input_to_concat = dot_products[i].reshape(dt_shape[0], 1, dt_shape[1])
            input_to_concat = input_to_concat.expand(dt_shape[0],inputs_shape[1], dt_shape[1])
            concat_inputs = T.cat((inputs, input_to_concat), axis = -1)
            edge_wt_outs_prime.append(self.edge_layers_prime[i](concat_inputs))
            if self.small_net[jet_type]:
                dense_vertex_outs.append(None)
            else:
                edge_feat_outs_prime.append(self.feat_layers_prime[i](concat_inputs))
                dot_products_prime.append(self.dot_products_prime[i](edge_wt_outs_prime[i], edge_feat_outs_prime[i]))
                dense_vertex_outs.append(self.vertex_networks[i](dot_products_prime[i]))
        return dense_vertex_outs, edge_wt_outs

    def basis_step(self, sample, _batch_idx):
        """
        basis step used for training and validation step
        Get the inputs and calculate all the losses
        The losses get calculated for each binary prediction of each 
        jet type defined in the config file. The total edge loss gets added.
        Additionally a MSE for the regression task is calculated and added.
        Afterwards the added losses get weighted in case the different weighting
        factors for the loss terms was defined.
        """
        inputs, labels, mask, mask_vertex = sample
        cal_loss_edge_start = True
        cal_loss_vertex_start = True
        loss_vertex_cal = None
        loss_vertex_per_fl = {}
        loss_edge_per_fl = {}
        vertex_outs, edge_outs = self.forward(inputs=inputs)
        for i, jet_type in enumerate(self.jet_types):
            labels_edge = labels[f"Y_edge_{jet_type}"]
            sample_weights = labels[f"sample_weights_{jet_type}"]
            if not self.small_net[jet_type]:
                labels_vertex =  labels[f"Y_vertex_features_{jet_type}"]
            labels_shape = labels_edge.size()
            labels_edge = labels_edge.reshape(labels_shape[0], labels_shape[1], 1)
            sample_weight_shape = sample_weights.size()
            sample_weights = sample_weights.reshape(
                sample_weight_shape[0], sample_weight_shape[1], 1
            )
            mask_vert = mask_vertex[f"vertex_mask_{jet_type}"]

            if cal_loss_edge_start:
                loss_edge_cal = binary_cross_entropy_with_logits(
                    edge_outs[i], labels_edge, sample_weights, reduction='none',
                )[mask].mean()
                loss_edge_per_fl[jet_type] = loss_edge_cal
                cal_loss_edge_start = False
            else:
                loss_edge_cal = loss_edge_cal + binary_cross_entropy_with_logits(
                    edge_outs[i], labels_edge, sample_weights, reduction='none',
                )[mask].mean()
                loss_edge_per_fl[jet_type] = loss_edge_cal
            if cal_loss_vertex_start and not self.small_net[jet_type]:
                loss_vertex_cal = self.loss_fn_vertex(
                    vertex_outs[i], labels_vertex
                )[mask_vert].mean()
                loss_vertex_per_fl[jet_type] = loss_vertex_cal
                cal_loss_vertex_start = False
            elif not cal_loss_vertex_start and not self.small_net[jet_type]:
                loss_vertex_cal = loss_vertex_cal + self.loss_fn_vertex(
                    vertex_outs[i], labels_vertex
                )[mask_vert].mean()
                loss_vertex_per_fl[jet_type] = loss_vertex_cal
        
        if loss_vertex_cal is not None:
            total = (
                self.loss_fac_edge * loss_edge_cal + self.loss_fac_vert * loss_vertex_cal
            )
        else:
            total = loss_edge_cal
        return loss_edge_cal, loss_vertex_cal, total, loss_edge_per_fl, loss_vertex_per_fl

    def training_step(self, sample: tuple, _batch_idx: int):
        """
        perform one training step
        calculate the loss terms for each network part and return total loss 
        for backpropagation, log the losses
        """
        loss_edge_cal, loss_vertex_cal, total, loss_edge_per_fl, loss_vertex_per_fl = self.basis_step(sample, _batch_idx)
        self.log("train/edge", loss_edge_cal)
        self.log("train/total", total)
        self.log("train/vertex", loss_vertex_cal)
        for jet_type in self.jet_types:
            self.log(f"train/edge_{jet_type}", loss_edge_per_fl[jet_type])
            if not self.small_net[jet_type]:
                self.log(f"train/vertex_{jet_type}", loss_vertex_per_fl[jet_type])
        return total

    def validation_step(self, sample: tuple, _batch_idx: int):
        """
        perform one validation step
        calculate the loss terms for each network part and log them to check if
        stopping criteria are met
        """
        loss_edge_cal, loss_vertex_cal, total, loss_edge_per_fl, loss_vertex_per_fl = self.basis_step(
            sample, _batch_idx
        )
        self.log("valid/edge", loss_edge_cal)
        self.log("valid/total", total)
        self.log("valid/vertex", loss_vertex_cal)
        for jet_type in self.jet_types:
            self.log(f"valid/edge_{jet_type}", loss_edge_per_fl[jet_type])
            if not self.small_net[jet_type]:
                self.log(f"valid/vertex_{jet_type}", loss_vertex_per_fl[jet_type])
        return total

    def configure_optimizers(self):
        """
        define the optimiser for the 
        """
        optimizer = optim.Adam(self.parameters(), lr=self.lr)
        scheduler = optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=self.lr,
            total_steps=self.trainer.estimated_stepping_batches,
        )
        return [optimizer], [scheduler]
