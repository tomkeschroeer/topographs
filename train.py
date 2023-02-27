"""Training script to perform topograph training."""
import argparse as pars
import pytorch_lightning as pl
import numpy as np
import random as rd
from h5py import File
from os import makedirs
import sys
sys.path.insert(0, "/home/users/s/schroeer/scratch/PhD/Topograph_repos/flavour_tagging")
import shutil

import torch.nn as nn
from torch import tensor, save
import torch.optim as optim
from pytorch_lightning.loggers.wandb import WandbLogger
from torch.utils.data import Dataset, DataLoader, TensorDataset
from pytorch_lightning.callbacks import ModelCheckpoint
from lightning_lite.utilities.exceptions import MisconfigurationException

from topograph.modules import (
    GetConfiguration,
    TopographModel,
    get_sample_weights,
)

from topograph.modules import IterableFlavourTaggingDataset


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

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = get_parser()
    config = GetConfiguration(args.config)
    n_epochs = config.epochs
    stepsize = config.stepsize
    lr = config.lr

    config_file_name = args.config
    input_file_name = config_file_name.split("/")[-1].replace(".yaml","")
    makedirs(config.output_training, exist_ok=True)
    shutil.copyfile(config_file_name, f"{config.output_training}/{input_file_name}.yaml".replace("//","/"))

    edge_weight_layer_name = "edge_weight"
    edge_feat_layer_name = "edge_feat"
    vertex_network_layer_name = "vertex_network"
    input_weight_layer_name = "input_1"
    input_feat_layer_name = "input_2"

    metadata_dict = {}
    train_file = f"{config.output}/{config.training_file_name}".replace("//", "/")
    with File(train_file, "r") as f:
        (
            metadata_dict["n_jets"],
            metadata_dict["n_trks"],
            metadata_dict["n_trk_features"],
        ) = f[f"{config.tracks_name}"].shape
        _, metadata_dict["n_vertex_feat"] = f[f"{config.vertex_feat_name}"].shape
        _, _, metadata_dict["n_edge_y"] = f[f"{config.edge_name}"].shape

    topomodel = TopographModel(
        nodes_feat=config.edge_feature_network["nodes"],
        nodes_weight=config.edge_weight_network["nodes"],
        nodes_vertex=config.vertex_network["nodes"],
        save_dir=config.output_training,
        name=config.model_name,
        activation_name=config.edge_weight_network["add_activation"],
        lr=config.lr
    )
    
    makedirs(f"{config.output_training}/modelfiles", exist_ok=True)
    training_file = f"{config.output}/{config.training_file_name}".replace("//","/")
    val_file = f"{config.output}/{config.validation_file_name}".replace("//","/")

    topomodel.train()
    topograph_loss_edges = nn.BCELoss()
    topograph_loss_vertex = nn.MSELoss()
    
    njets = getattr(config, "njets", -1)
    njets = -1 if njets is None else njets
    tracks_dataset = IterableFlavourTaggingDataset(
        dset="train",
        buffer_shuffle=True,
        file_name = training_file,
        batch_size = 1024,
        drop_last = True,
        buffer_size = 10_000,
        njets = getattr(config, "njets", -1)
    )
    tracks_loader = DataLoader(tracks_dataset)
    
    njets_val = getattr(config, "njets_val", -1)
    njets_val = -1 if njets_val is None else njets_val
    valid_dataset = IterableFlavourTaggingDataset(
        dset="valid",
        buffer_shuffle=False,
        file_name = val_file,
        batch_size = 1024,
        drop_last = True,
        buffer_size = 100_000,
        njets = njets_val
    )
    valid_loader = DataLoader(valid_dataset)

    makedirs(f"{config.output_training}/checkpoints".replace("//","/"), exist_ok=True)
    checkpoint = ModelCheckpoint(
        monitor="valid/total",
        filename="checkpoint_train_{epoch}",
        dirpath=f"{config.output_training}/checkpoints",
        save_top_k=-1
    )
    logger = WandbLogger(
        name = config.model_name,
        save_dir=config.output_training,
        project="pytorch_runs",
    )
    try:
        trainer = pl.Trainer(
            max_epochs=config.epochs,
            callbacks=[checkpoint],
            logger=logger,
            accelerator="auto",
            log_every_n_steps=10
        )
    except MisconfigurationException:
        print("No gpu found!")
        trainer = pl.Trainer(
            max_epochs=config.epochs,
            callbacks=[checkpoint],
            logger=logger
        )

    trainer.fit(
        model=topomodel,
        train_dataloaders=tracks_loader,
        val_dataloaders=valid_loader,
    )