"""Training script to perform topograph training."""
import argparse as pars
import numpy as np
import random as rd
from h5py import File
from os import makedirs

import torch.nn as nn
from torch import tensor, save
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from topograph.modules import (
    GetConfiguration,
    TopographModel
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

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = get_parser()
    config = GetConfiguration(args.config)
    n_epochs = config.epochs
    stepsize = config.stepsize
    lr = config.lr

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
        nodes_feat=[20, 70, 70, 70, 30],
        nodes_weight=[20, 70, 70, 70, 1],
        nodes_vertex=[30, 50, 50, 50, 1]
    )

    n_jets = metadata_dict['n_jets']
    n_steps = n_jets // stepsize
    if n_jets % stepsize != 0:
        n_steps += 1

    makedirs(f"{config.output_training}/modelfiles", exist_ok=True)
    training_file = f"{config.output}/{config.training_file_name}".replace("//","/")
    with File(training_file, "r") as f:
        tracks_all = f["X_train_tracks"][:]
        Y_edge_all = f["Y_edge"][:]
        Y_vertex_all = f["Y_vertex_features"][:]
    ind = np.linspace(0,len(tracks_all)-1, len(tracks_all), dtype=int)

    optimiser = optim.Adam(topomodel.parameters(), lr=lr)

    for epoch in range(1,n_epochs+1):
        print(f"start epoch {epoch}")
        rd.shuffle(ind)
        tracks_all = tracks_all[ind]
        Y_edge_all = Y_edge_all[ind]
        Y_vertex_all = Y_vertex_all[ind]
        topomodel.train()
        topograph_loss_edges = nn.BCELoss()
        topograph_loss_vertex = nn.MSELoss()
        for step in range(n_steps):
            tracks = tensor(tracks_all[step*50:(step+1)*50])
            Y_edge = tensor(np.float32(Y_edge_all[step*50:(step+1)*50]))
            Y_vertex = tensor(np.float32(Y_vertex_all[step*50:(step+1)*50]))

            model, model_edge = topomodel(tracks, tracks)
            loss_edges = (
                topograph_loss_edges(
                    model_edge,
                    Y_edge,
                )
            )
            loss_vertex = (
                topograph_loss_vertex(
                    model,
                    Y_vertex
                )
            )
            loss = loss_vertex + 100*loss_edges
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()

        modelfile_dir = (
        config.output_training + f"/modelfiles/model_epoch{epoch:03d}.h5"
        ).replace("//", "/")

        save({
            'epoch': epoch,
            'model_state_dict': topomodel.state_dict(),
            'optimiser_state_dict': optimiser.state_dict(),
            'loss': loss,
            }, modelfile_dir)            
