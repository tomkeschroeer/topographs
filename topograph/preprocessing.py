import argparse as pars
from importlib_metadata import metadata
import numpy as np
import tensorflow as tf
from h5py import File
from tensorflow import (
    TensorShape,
    float32,
    int32
)
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.data import Dataset

from modules import (
    GetConfiguration,
    get_model,
    DataLoader,
    Matcher
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
    metadata_dict = {}
    with File(config.input, "r") as f:
        reco = f["tracks_loose"][0]
        truth = f["truth_fromBC"][:, 0]
        jet = f["jets"][0]

    MatchTruthReco = Matcher(
        dR_truth=truth["dr"],
        dR_reco=reco["dr"],
        phi_truth=truth["phi"],
        eta_truth=truth["eta"],
        eta_reco=reco["eta"],
        eta_jet=list([jet["eta"]])*40
    )

    MatchTruthReco.matching_truth_tracks()