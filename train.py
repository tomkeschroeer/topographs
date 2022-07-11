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

from topograph.modules import (
    GetConfiguration,
    get_model,
    DataLoader
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
    with File(config.training_input, "r") as f:
        metadata_dict["n_jets"], metadata_dict["n_trks"], metadata_dict["n_trk_features"] = f[f"{config.track_name}"].shape
        metadata_dict["n_edge_y"] = 1
        _, _, metadata_dict["n_edge_feat"] = f[f"{config.edge_feat_name}"].shape
        _, metadata_dict["n_vertex_feat"] = f[f"{config.vertex_feat_name}"].shape

    types = ({
            "input_1": float32,
            "input_2": float32
        },
        {
            "edge_feat": float32,
            "edge_weight": int32,
            "vertex_network": float32
        }
    )
    shapes = ({
            "input_1": TensorShape((None, metadata_dict["n_trks"], metadata_dict["n_trk_features"])),
            "input_2": TensorShape((None, metadata_dict["n_trks"], metadata_dict["n_trk_features"]))
        },
        {
            "edge_feat": TensorShape((None, metadata_dict["n_trks"], metadata_dict["n_edge_feat"])),
            "edge_weight": TensorShape((None, metadata_dict["n_trks"], metadata_dict["n_edge_y"])),
            "vertex_network": TensorShape((None, metadata_dict["n_vertex_feat"]))

        }
    )

    tf_dataset = (Dataset.from_generator(
            DataLoader(
                input=config.training_input,
                metadata_dict=metadata_dict,
                savetracks=True,
                track_name=config.track_name,
                edge_name=config.edge_name,
                edge_feat_name=config.edge_feat_name,
                vertex_feat_name=config.vertex_feat_name
            ),
            types,
            shapes
        )
         .repeat()
         .prefetch(tf.data.AUTOTUNE)
    )

    model_checkpoint = ModelCheckpoint(
        config.output + "/model_epoch{epoch:03d}.h5",
        verbose=True,
        save_best_only=False,
        save_weights_only=False,
    )

    model = get_model(input_feat=(metadata_dict["n_trks"], metadata_dict["n_trk_features"]), input_weight=(metadata_dict["n_trks"], metadata_dict["n_trk_features"]), config=config)
    #print(len(list(tf_dataset)))
    callbacks = [model_checkpoint]
    model.fit(tf_dataset, epochs = config.epochs, steps_per_epoch = 5, callbacks=callbacks)

