"""script to build topograph network."""
# from tensorflow.keras import Model

from topograph.modules.layers import (
    VertexNetwork,
    DotProduct,
    EdgeLayers,
    FeatLayers,
    ShiftRelu,
    Sigmoid,
    Softplus_norm,
    MultipleMSELoss
)
from torch.nn import (
    Module,
    BCELoss,
    MSELoss,
    Softplus,
    BCEWithLogitsLoss
) 

from torch.nn.functional import binary_cross_entropy_with_logits

from torch.autograd import grad

import torch.optim as optim
import pytorch_lightning as pl
import torch as T
import numpy as np
from pathlib import Path
from typing import Union
import wandb

from h5py import File

class TopographModel(pl.LightningModule):
    """
    class building the topograph model
    """

    def __init__(
        self,
        nodes_feat: list = [128, 30, 30, 30],
        nodes_weight : list = [128, 30, 30, 1],
        nodes_vertex : list = [30, 50, 50, 50, 1],
        activation_name: str = None,
        save_dir: str = None,
        name: str = None,
        device: str = "gpu",
        lr: float = 1e-3,
        save: bool = False,
        loss_fac_edge: float = 100,
        loss_fac_vert: float = 1,
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

        self.nodes_feat = nodes_feat
        self.nodes_weight = nodes_weight
        self.nodes_vertex = nodes_vertex

        self.feat_layer = FeatLayers(nodes=self.nodes_feat)
        self.edge_layer = EdgeLayers(nodes=self.nodes_weight)
        if self.activation_name == "shifted_relu":
            self.add_activation = ShiftRelu()
        elif self.activation_name == "sigmoid":
            self.add_activation = Sigmoid()
        elif self.activation_name == "softplus":
            norm = T.tensor(np.log(1+np.exp(1)))
            self.add_activation = Softplus_norm(norm=norm)
        elif self.activation_name is None:
            self.add_activation = False
        else:
            raise KeyError(
                f"Undefined additional actrivation: {self.activation_name}. Please select one"
                ' of the following: ["shifted_relu", "sigmoid", "softplus"] or leave empty/remove'
                " option."
            )
        self.dot_product = DotProduct()
        self.vertex_network = VertexNetwork(nodes=self.nodes_vertex)
        # Define the loss funcitons
        self.loss_fn_vertex = MSELoss() #MultipleMSELoss()

        if save:
            with File("/home/users/s/schroeer/scratch/PhD/Topograph_repos/output/inputs.h5", "w") as inputs_file:
                inputs_file.create_dataset(name="inputs",shape=(0,40,20),chunks=True, maxshape=(None,40,20))
                inputs_file.create_dataset(name="labels",shape=(0,40,1), chunks=True, maxshape=(None,40,1))
            
    def on_fit_start(self):
        if wandb.run:
            wandb.define_metric("train/total", summary="min")
            wandb.define_metric("train/edge", summary="min")
            wandb.define_metric("train/vertex", summary="min")
            wandb.define_metric("valid/total", summary="min")
            wandb.define_metric("valid/edge", summary="min")
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
        edge_wt_out = self.edge_layer(inputs)
        edge_feat_out = self.feat_layer(inputs)
        if self.activation_name is not None:
            add_activation = self.add_activation(edge_wt_out)
            dt_product = self.dot_product(edge_feat_out, add_activation, mask)
        else:
            dt_product = self.dot_product(edge_feat_out, edge_wt_out, mask)
        dense_vertex_out = self.vertex_network(dt_product)
        return dense_vertex_out, edge_wt_out

    def basis_step(self, sample, _batch_idx, save=False):
        inputs, labels_edge, labels_vertex, sample_weights, mask, mask_vertex = sample
        if save:
            inputs_save = inputs.reshape((1024, 40, 20))
            labels_save = labels_edge.reshape((1024, 40, 1))
            with File("/home/users/s/schroeer/scratch/PhD/Topograph_repos/output/inputs.h5", "a") as inputs_file:
                len_inp = len(inputs_save)
                inputs_file["inputs"].resize((inputs_file["inputs"].shape[0]+len_inp), axis=0)
                inputs_file["inputs"][-len_inp:] = inputs_save
                inputs_file["labels"].resize((inputs_file["labels"].shape[0]+len_inp), axis=0)
                inputs_file["labels"][-len_inp:] = labels_save
        labels_shape = labels_edge.size()
        labels_edge = labels_edge.reshape(labels_shape[0],labels_shape[1], labels_shape[2], 1)
        vertex_out, edge_out = self.forward(inputs=inputs, mask=mask)
        loss_edge_cal = binary_cross_entropy_with_logits(edge_out, labels_edge, sample_weights)
        loss_vertex_cal = self.loss_fn_vertex(
            vertex_out[mask_vertex], labels_vertex[mask_vertex]
        )
        total = self.loss_fac_edge*loss_edge_cal + self.loss_fac_vert*loss_vertex_cal
        # total = loss_edge_cal
        return loss_edge_cal, loss_vertex_cal, total

    def training_step(self, sample: tuple, _batch_idx: int):
        loss_edge_cal, loss_vertex_cal, total = self.basis_step(sample, _batch_idx)
        self.log("train/total", total)
        self.log("train/vertex", loss_vertex_cal)
        self.log("train/edge", loss_edge_cal)
        return total

    def validation_step(self, sample: tuple, _batch_idx: int):
        loss_edge_cal, loss_vertex_cal, total = self.basis_step(sample, _batch_idx, save=False)
        self.log("valid/total", total)
        self.log("valid/vertex", loss_vertex_cal)
        self.log("valid/edge", loss_edge_cal)

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.lr)
        scheduler = optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=self.lr,
            total_steps=self.trainer.estimated_stepping_batches,
        )
        return [optimizer], [scheduler]
