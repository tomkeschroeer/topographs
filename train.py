"""Training script to perform topograph training."""
import argparse as pars

from h5py import File
import torch.nn as nn
from torch import tensor
import numpy as np

from topograph.modules import (
    DataLoader,
    GetConfiguration,
    TopographModel,
    load_tfrecords_train_dataset,
)

import torch as T


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

    edge_weight_layer_name = "edge_weight"
    edge_feat_layer_name = "edge_feat"
    vertex_network_layer_name = "vertex_network"
    input_weight_layer_name = "input_1"
    input_feat_layer_name = "input_2"

    metadata_dict = {}
    val_file = f"{config.output}/{config.training_file_name}".replace("//", "/")
    with File(val_file, "r") as f:
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

    training_file = f"{config.output}/{config.training_file_name}".replace("//","/")
    with File(training_file, "r") as f:
        tracks_all = f["X_train_tracks"][:500]
        Y_edge_all = f["Y_edge"][:500]
        Y_vertex_all = f["Y_vertex_features"][:500]

    for step in range(0,10):
        tracks = tensor(tracks_all[step*50:(step+1)*50])
        Y_edge = tensor(np.float32(Y_edge_all[step*50:(step+1)*50]))
        Y_vertex = tensor(np.float32(Y_vertex_all[step*50:(step+1)*50]))
        topomodel.train()

        model, model_edge = topomodel(tracks, tracks)
        topograph_loss_edges = nn.BCELoss()
        topograph_loss_vertex = nn.MSELoss()
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
        loss.backward()
        print(loss)
