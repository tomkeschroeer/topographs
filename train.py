"""Training script to perform topograph training."""
import argparse as pars
import random as rd
import sys
from os import makedirs

import numpy as np
import pytorch_lightning as pl
from h5py import File

sys.path.insert(0, "/home/users/s/schroeer/scratch/PhD/Topograph_repos/flavour_tagging")
import shutil

import torch.nn as nn
import torch.optim as optim
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers.wandb import WandbLogger
from torch import from_numpy, save, tensor
from torch.utils.data import DataLoader, Dataset, IterableDataset, TensorDataset

from topograph.modules import (
    GetConfiguration,
    IterableFlavourTaggingDataset,
    TopographModel,
    Topographs_dataset,
    get_sample_weights,
)


def get_parser():
    """
    Argument parser for the train script

    Returns
    -------
    args: parse_args
    """
    parser = pars.ArgumentParser()
    parser.add_argument(
        "--config",
        "-c",
        type=str,
        required=True,
        help="config file giving the network parameters",
    )
    parser.add_argument(
        "--vars",
        "-v",
        type=str,
        required=False,
        default=None,
        help="index of vars used for training",
    )

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = get_parser()
    config = GetConfiguration(args.config)
    vars = args.vars
    if vars is not None:
        vars = np.array(vars.split(",")).astype(int)
    n_epochs = config.epochs
    stepsize = config.stepsize
    lr = getattr(config, "lr", 1e-3)

    loss_fac_edge = getattr(config, "loss_fac_edge", 100)
    loss_fac_vert = getattr(config, "loss_fac_vert", 1)
    if loss_fac_edge is None:
        loss_fac_edge = 100
    if loss_fac_vert is None:
        loss_fac_vert = 1

    used_vertex_properties = getattr(config, "used_vertex_properties", None)

    config_file_name = args.config
    input_file_name = config_file_name.split("/")[-1].replace(".yaml", "")
    makedirs(config.output_training, exist_ok=True)
    shutil.copyfile(
        config_file_name,
        f"{config.output_training}/{input_file_name}.yaml".replace("//", "/"),
    )

    edge_weight_layer_name = "edge_weight"
    edge_feat_layer_name = "edge_feat"
    vertex_network_layer_name = "vertex_network"
    input_weight_layer_name = "input_1"
    input_feat_layer_name = "input_2"

    train_file = f"{config.output}/{config.training_file_name}".replace("//", "/")

    edge_feat_nodes = config.edge_feature_network["nodes"]
    edge_weight_nodes = config.edge_weight_network["nodes"]
    if vars is not None:
        edge_feat_nodes[0] = len(vars)
        edge_weight_nodes[0] = len(vars)

    str_vars = ""
    if vars is not None:
        for var in vars:
            str_vars += f"_{var}"

    training_output_folder = config.output_training
    training_output_folder = (
        training_output_folder[:-1]
        if training_output_folder[-1] == "/"
        else training_output_folder
    )
    training_output_folder += str_vars

    nodes_vertex = config.vertex_network["nodes"]
    if isinstance(used_vertex_properties, int):
        nodes_vertex[-1] = 1
    elif isinstance(used_vertex_properties, (list, np.ndarray)):
        nodes_vertex[-1] = len(used_vertex_properties)

    topomodel = TopographModel(
        nodes_feat=edge_feat_nodes,
        nodes_weight=edge_weight_nodes,
        nodes_vertex=nodes_vertex,
        # save_dir=config.output_training,
        activation_name=config.edge_weight_network["add_activation"],
        lr=lr,
        loss_fac_edge=loss_fac_edge,
        loss_fac_vert=loss_fac_vert,
        small_net=config.small_net,
        jet_types=config.jet_types,
    )

    makedirs(f"{training_output_folder}/modelfiles", exist_ok=True)
    training_file = f"{config.output}/{config.training_file_name}".replace("//", "/")
    val_file = f"{config.output}/{config.validation_file_name}".replace("//", "/")

    topomodel.train()
    topograph_loss_edges = nn.BCELoss()
    topograph_loss_vertex = nn.MSELoss()

    njets = getattr(config, "njets", -1)
    njets = -1 if njets is None else njets

    tracks_dataset = Topographs_dataset(
        filename=training_file,
        batch_size=min(njets, 1024),
        n_samples=njets,
        jet_types=config.jet_types,
    )

    tracks_loader = DataLoader(tracks_dataset, batch_size=None)

    njets_val = getattr(config, "njets_val", -1)
    njets_val = -1 if njets_val is None else njets_val

    valid_dataset = Topographs_dataset(
        filename=val_file, batch_size=min(njets_val, 1024), n_samples=njets_val, jet_types=config.jet_types, train=False
    )
    valid_loader = DataLoader(valid_dataset, batch_size=None)

    makedirs(f"{training_output_folder}/checkpoints", exist_ok=True)
    checkpoint = ModelCheckpoint(
        monitor="valid/vertex", # ""valid/edge" if config.small_net else "valid/total",
        filename="checkpoint_train_{epoch}",
        dirpath=f"{training_output_folder}/checkpoints",
        save_top_k=-1,
    )

    logger = WandbLogger(
        name=config.model_name,
        save_dir=training_output_folder,
        project="pytorch_runs",
    )

    trainer = pl.Trainer(
        max_epochs=config.epochs,
        callbacks=[checkpoint],
        logger=logger,
        accelerator="auto",
        log_every_n_steps=10,
    )

    trainer.fit(
        model=topomodel,
        train_dataloaders=tracks_loader,
        val_dataloaders=valid_loader,
    )
