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
from torch.nn import BCELoss, BCEWithLogitsLoss, Module, MSELoss, Softplus
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
        small_net: dict = {}
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

        self.nodes_feat = nodes_feat
        self.nodes_weight = nodes_weight
        self.nodes_vertex = nodes_vertex

        self.feat_layer_b = FeatLayers(nodes=self.nodes_feat)
        self.edge_layer_b = EdgeLayers(nodes=self.nodes_weight)
        self.feat_layer_c = FeatLayers(nodes=self.nodes_feat)
        self.edge_layer_c = EdgeLayers(nodes=self.nodes_weight)

        if self.activation_name == "shifted_relu":
            self.add_activation = ShiftRelu()
        elif self.activation_name == "sigmoid":
            self.add_activation = Sigmoid()
        elif self.activation_name == "softplus":
            norm = T.tensor(np.log(1 + np.exp(1)))
            self.add_activation = Softplus_norm(norm=norm)
        elif self.activation_name is None:
            self.add_activation = False
        else:
            raise KeyError(
                f"Undefined additional actrivation: {self.activation_name}. Please"
                ' select one of the following: ["shifted_relu", "sigmoid", "softplus"]'
                " or leave empty/remove option."
            )
        self.dot_product_c = DotProduct()
        self.dot_product_b = DotProduct()
        self.vertex_network_c = VertexNetwork(nodes=self.nodes_vertex)
        self.vertex_network_b = VertexNetwork(nodes=self.nodes_vertex)
        # Define the loss funcitons
        self.loss_fn_vertex = MSELoss(reduction='none')  # MultipleMSELoss()

        if save:
            with File(
                "/home/users/s/schroeer/scratch/PhD/Topograph_repos/output/inputs.h5",
                "w",
            ) as inputs_file:
                inputs_file.create_dataset(
                    name="inputs",
                    shape=(0, 40, 20),
                    chunks=True,
                    maxshape=(None, 40, 20),
                )
                inputs_file.create_dataset(
                    name="labels", shape=(0, 40, 1), chunks=True, maxshape=(None, 40, 1)
                )

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
        edge_wt_out_b = self.edge_layer_b(inputs)
        edge_feat_out_b = self.feat_layer_b(inputs)
        edge_wt_out_c = self.edge_layer_c(inputs)
        edge_feat_out_c = self.feat_layer_c(inputs)
        if self.small_net[self.tr_jet_type]: return None, None, edge_wt_out_b, edge_wt_out_c
        if self.activation_name is not None:
            add_activation_b = self.add_activation(edge_wt_out_b)
            dt_product_b = self.dot_product(edge_feat_out_b, add_activation_b, mask)
            add_activation_c = self.add_activation(edge_wt_out_c)
            dt_product_c = self.dot_product(edge_feat_out_c, add_activation_c, mask)
        else:
            dt_product_b = self.dot_product_b(edge_feat_out_b, edge_wt_out_b, mask)
            dt_product_c = self.dot_product_c(edge_feat_out_c, edge_wt_out_c, mask)
        dense_vertex_out_b = self.vertex_network_b(dt_product_b)
        dense_vertex_out_c = self.vertex_network_c(dt_product_c)
        return dense_vertex_out_b, dense_vertex_out_c, edge_wt_out_b, edge_wt_out_c

    def basis_step(self, sample, _batch_idx):
        inputs, labels, mask, mask_vertex = sample
        # labels_edge = labels[f"Y_edge_{self.tr_jet_type}"]
        labels_edge_b = labels[f"Y_edge_b"]
        labels_edge_c = labels[f"Y_edge_c"]
        sample_weights_b = labels[f"sample_weights_b"]
        sample_weights_c = labels[f"sample_weights_c"]
        # if not self.small_net[self.tr_jet_type]:
        labels_vertex_b =  labels[f"Y_vertex_features_b"]
        labels_vertex_c =  labels[f"Y_vertex_features_c"]

        labels_shape_c = labels_edge_c.size()
        labels_edge_c = labels_edge_c.reshape(labels_shape_c[0], labels_shape_c[1], 1)
        sample_weight_shape_c = sample_weights_c.size()
        sample_weights_c = sample_weights_c.reshape(
            sample_weight_shape_c[0], sample_weight_shape_c[1], 1
        )
        labels_shape_b = labels_edge_b.size()
        labels_edge_b = labels_edge_b.reshape(labels_shape_b[0], labels_shape_b[1], 1)
        sample_weight_shape_b = sample_weights_b.size()
        sample_weights_b = sample_weights_b.reshape(
            sample_weight_shape_b[0], sample_weight_shape_b[1], 1
        )
        # labels_vertex_shape = labels_vertex.size()
        # labels_vertex = labels_vertex.reshape(labels_vertex_shape[1], labels_vertex_shape[2])
        # inputs_shape = inputs.size()
        # inputs = inputs.reshape(inputs_shape[1], inputs_shape[2], inputs_shape[3])
        # mask_shape = mask.size()
        # mask = mask.reshape(mask_shape[1], mask_shape[2])
        # mask_vertex_shape = mask_vertex.size()
        # mask_vertex = mask_vertex.reshape(mask_vertex_shape[1])

        vertex_out_b, vertex_out_c, edge_out_b, edge_out_c = self.forward(inputs=inputs, mask=mask)
        loss_edge_cal = binary_cross_entropy_with_logits(
            edge_out_c, labels_edge_c, sample_weights_c, reduction='none',
        )[mask].mean() + binary_cross_entropy_with_logits(
            edge_out_b, labels_edge_b, sample_weights_b, reduction='none',
        )[mask].mean()

        if self.small_net[self.tr_jet_type]: return loss_edge_cal, None, None
        loss_vertex_cal = self.loss_fn_vertex(
            vertex_out_b, labels_vertex_b
        )[mask_vertex[f"vertex_mask_b"]].mean() + self.loss_fn_vertex(
            vertex_out_c, labels_vertex_c
        )[mask_vertex[f"vertex_mask_c"]].mean()
        total = (
            self.loss_fac_edge * loss_edge_cal + self.loss_fac_vert * loss_vertex_cal
        )
        # total = loss_edge_cal
        return loss_edge_cal, loss_vertex_cal, total

    def training_step(self, sample: tuple, _batch_idx: int):
        loss_edge_cal, loss_vertex_cal, total = self.basis_step(sample, _batch_idx)
        self.log("train/edge", loss_edge_cal)
        if self.small_net[self.tr_jet_type]: return loss_edge_cal
        self.log("train/total", total)
        self.log("train/vertex", loss_vertex_cal)
        return total

    def validation_step(self, sample: tuple, _batch_idx: int):
        loss_edge_cal, loss_vertex_cal, total = self.basis_step(
            sample, _batch_idx
        )
        self.log("valid/edge", loss_edge_cal)
        if self.small_net[self.tr_jet_type]: return loss_edge_cal
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
