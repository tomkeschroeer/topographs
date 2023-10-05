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
    class building the topograph model
    """

    def __init__(
        self,
        nodes_feat: list = [128, 30, 30, 30],
        nodes_weight: list = [128, 30, 30, 1],
        nodes_vertex: list = [30, 50, 50, 50, 1],
        activation_name: str = None,
        save_dir: str = None,
        name: str = None,
        device: str = "gpu",
        lr: float = 1e-3,
        save: bool = False,
        loss_fac_edge: float = 100,
        loss_fac_vert: float = 1,
        tr_jet_type: str = "b",
        small_net: dict = {},
        jet_types: list = ["b", "c", "light"]
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
        self.save_hyperparameters()
        self.activation_name = activation_name
        self.loss_names = ["total", "edge_loss", "vertex_loss"]
        self.lr = lr
        self.full_name = Path(save_dir, name)
        self.loss_fac_edge = loss_fac_edge
        self.loss_fac_vert = loss_fac_vert
        self.tr_jet_type = tr_jet_type
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
                self.feat_layers_prime.append(FeatLayers(nodes=self.nodes_feat))
            self.feat_layers.append(FeatLayers(nodes=self.nodes_feat))
            self.edge_layers.append(EdgeLayers(nodes=self.nodes_weight))
            self.dot_products.append(DotProduct())
            self.edge_layers_prime.append(EdgeLayers(nodes=self.nodes_weight_prime))
            self.dot_products_prime.append(DotProduct())
        # Define the loss funcitons
        self.loss_fn_vertex = MSELoss(reduction='none')  # MultipleMSELoss()

    def on_fit_start(self):
        if wandb.run:
            wandb.define_metric("train/edge", summary="min")
            wandb.define_metric("valid/edge", summary="min")
            if not self.small_net[self.tr_jet_type]:
                wandb.define_metric("train/total", summary="min")
                wandb.define_metric("train/vertex", summary="min")
                wandb.define_metric("valid/total", summary="min")
                wandb.define_metric("valid/vertex", summary="min")

    def forward(self, inputs, mask):
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
        edge_wt_outs = {}
        edge_feat_outs = {}
        edge_wt_outs_prime = {}
        edge_feat_outs_prime = {}
        dot_products = {}
        dot_products_prime = {}
        dense_vertex_outs = {}
        for i, jet_type in enumerate(self.jet_types):
            edge_wt_outs[jet_type] = self.edge_layers[i](inputs)
            edge_feat_outs[jet_type] = self.feat_layers[i](inputs)
            dot_products[jet_type] = self.dot_products[i](edge_wt_outs[jet_type], edge_feat_outs[jet_type], mask)
            dt_shape = dot_products[jet_type].size()
            inputs_shape = inputs.size()
            input_to_concat = dot_products[jet_type].reshape(dt_shape[0], 1, dt_shape[1])
            input_to_concat = input_to_concat.expand(dt_shape[0],inputs_shape[1], dt_shape[1])
            concat_inputs = T.cat((inputs, input_to_concat), axis = -1)
            edge_wt_outs_prime[jet_type] = self.edge_layers_prime[i](concat_inputs)
            if self.small_net[jet_type]:
                dense_vertex_outs[jet_type] = None
            else:
                edge_feat_outs_prime[jet_type] = self.edge_layers_prime[i](concat_inputs)
                dot_products_prime[jet_type] = self.dot_products_prime[i](edge_wt_outs_prime[jet_type], edge_feat_outs_prime[jet_type], mask)
                dense_vertex_outs[jet_type] = self.vertex_networks[i](dot_products[jet_type])
        return dense_vertex_outs, edge_wt_outs

    def basis_step(self, sample, _batch_idx):
        inputs, labels, mask, mask_vertex, jet_type_per_jet = sample
        cal_loss_edge_start = True
        cal_loss_vertex_start = True
        loss_vertex_cal = None
        for jet_type in self.jet_types:
            vertex_outs, edge_outs = self.forward(inputs=inputs, mask=mask)
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
            # jet_type_mask = (jet_type_per_jet == self.jet_types.index(jet_type))
            # labels_edge = labels[f"Y_edge_{self.tr_jet_type}"]
            # mask_jt = mask[jet_type_mask]
            # input_jt = inputs[jet_type_mask]
            # vertex_outs, edge_outs = self.forward(inputs=input_jt, mask=mask_jt)
            # labels_edge = labels[f"Y_edge_{jet_type}"][jet_type_mask]
            # sample_weights = labels[f"sample_weights_{jet_type}"][jet_type_mask]
            # if not self.small_net[jet_type]:
            #     labels_vertex =  labels[f"Y_vertex_features_{jet_type}"][jet_type_mask]
            # labels_shape = labels_edge.size()
            # labels_edge = labels_edge.reshape(labels_shape[0], labels_shape[1], 1)
            # sample_weight_shape = sample_weights.size()
            # sample_weights = sample_weights.reshape(
            #     sample_weight_shape[0], sample_weight_shape[1], 1
            # )
            # mask_vert = mask_vertex[f"vertex_mask_{jet_type}"]
        # labels_vertex_shape = labels_vertex.size()
        # labels_vertex = labels_vertex.reshape(labels_vertex_shape[1], labels_vertex_shape[2])
        # inputs_shape = inputs.size()
        # inputs = inputs.reshape(inputs_shape[1], inputs_shape[2], inputs_shape[3])
        # mask_shape = mask.size()
        # mask = mask.reshape(mask_shape[1], mask_shape[2])
        # mask_vertex_shape = mask_vertex.size()
        # mask_vertex = mask_vertex.reshape(mask_vertex_shape[1])

            if cal_loss_edge_start:
                loss_edge_cal = binary_cross_entropy_with_logits(
                    edge_outs[jet_type], labels_edge, sample_weights, reduction='none',
                )[mask].mean()
                cal_loss_edge_start = False
            else:
                loss_edge_cal = loss_edge_cal + binary_cross_entropy_with_logits(
                    edge_outs[jet_type], labels_edge, sample_weights, reduction='none',
                )[mask].mean()
            if cal_loss_vertex_start and not self.small_net[jet_type]:
                loss_vertex_cal = self.loss_fn_vertex(
                    vertex_outs[jet_type], labels_vertex
                )[mask_vert].mean()
                cal_loss_vertex_start = False
            elif not cal_loss_vertex_start and not self.small_net[jet_type]:
                loss_vertex_cal = loss_vertex_cal + self.loss_fn_vertex(
                    vertex_outs[jet_type], labels_vertex
                )[mask_vert].mean()
        
        if loss_vertex_cal is not None:
            total = (
                self.loss_fac_edge * loss_edge_cal + self.loss_fac_vert * loss_vertex_cal
            )
        else:
            total = loss_edge_cal
        # total = loss_edge_cal
        return loss_edge_cal, loss_vertex_cal, total

    def training_step(self, sample: tuple, _batch_idx: int):
        loss_edge_cal, loss_vertex_cal, total = self.basis_step(sample, _batch_idx)
        self.log("train/edge", loss_edge_cal)
        # if self.small_net[self.tr_jet_type]: return loss_edge_cal
        self.log("train/total", total)
        self.log("train/vertex", loss_vertex_cal)
        return total

    def validation_step(self, sample: tuple, _batch_idx: int):
        loss_edge_cal, loss_vertex_cal, total = self.basis_step(
            sample, _batch_idx
        )
        self.log("valid/edge", loss_edge_cal)
        # if self.small_net[self.tr_jet_type]: return loss_edge_cal
        self.log("valid/total", total)
        self.log("valid/vertex", loss_vertex_cal)
        return total

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.lr)
        scheduler = optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=self.lr,
            total_steps=self.trainer.estimated_stepping_batches,
        )
        return [optimizer], [scheduler]
