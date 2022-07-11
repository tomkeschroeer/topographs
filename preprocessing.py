import argparse as pars
from importlib_metadata import metadata
import numpy as np
import tensorflow as tf
import pathlib
from h5py import File
from tensorflow import (
    TensorShape,
    float32,
    int32
)
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.data import Dataset

from topograph.modules import (
    GetConfiguration,
    get_model,
    DataLoader,
    Matcher
)

from topograph.modules.tools import DatasetCreater, GlobalConfig

def get_parser():
    """
    Argument parser for the train script

    Returns
    -------
    args: parse_args
    """
    parser = pars.ArgumentParser()
    parser.add_argument(
        '--config', 
        '-c', 
        type=str,
        required=True, 
        help='config file giving the network parameters'
    )

    args = parser.parse_args()
    return args

if __name__ == "__main__":

    args = get_parser()
    config = GetConfiguration(args.config)
    input_files = config.get_all_input_files()
    metadata_dict = {}
    stepsize = 5_000
    global_conf = GlobalConfig()
    input_files = config.get_all_input_files()
    for input_file_ind in range(len(input_files)):
        with File(input_files[input_file_ind], "r") as f:
            njets = len(f["/jets"][:])
        n_steps = njets//stepsize
        for step in range(n_steps):
            datasets = DatasetCreater(input_file=input_files[input_file_ind], step=step, stepsize=stepsize)
            if step == 0 and input_file_ind == 0:
                with File(f"{config.output}/training_topographs.h5", "w") as train_file:
                    train_file.create_dataset("Y_vertex_features", data = datasets.get_vertex_feat_y(), chunks=True, maxshape=(None,len(global_conf.vertex_features)))
                    train_file.create_dataset("Y_edge_features", data = datasets.get_edge_feat_y(), chunks=True, maxshape=(None,40,len(global_conf.edge_features)))
                    train_file.create_dataset("Y_edge", data = datasets.get_edge_y(), chunks=True, maxshape=(None,40,1))
                    train_file.create_dataset("X_train_tracks", data = datasets.get_track_input(), chunks=True, maxshape=(None,40,len(global_conf.track_inputs)))
            else:
                njets_step = datasets.get_n_valid_jets()
                with File(f"{config.output}/training_topographs.h5", "a") as train_file:
                    train_file["Y_vertex_features"].resize((train_file["Y_vertex_features"].shape[0] + njets_step), axis=0)
                    train_file["Y_vertex_features"][-njets_step:] = datasets.get_vertex_feat_y()
                    train_file["Y_edge_features"].resize((train_file["Y_edge_features"].shape[0] + njets_step), axis=0)
                    train_file["Y_edge_features"][-njets_step:] = datasets.get_edge_feat_y()
                    train_file["Y_edge"].resize((train_file["Y_edge"].shape[0] + njets_step), axis=0)
                    train_file["Y_edge"][-njets_step:] = datasets.get_edge_y()
                    train_file["X_train_tracks"].resize((train_file["X_train_tracks"].shape[0] + njets_step), axis=0)
                    train_file["X_train_tracks"][-njets_step:] = datasets.get_track_input()
