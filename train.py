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
    # with File(config.input, "r") as f:
    #     metadata_dict["n_jets"], metadata_dict["n_trks"], metadata_dict["n_trk_features"] = f[f"{config.track_name}"].shape
    #     _, metadata_dict["n_dim"] = f[config.target_name].shape

    # types = ({
    #         "input_1": float32,
    #         "input_2": float32
    #     },
    #     int32
    # )
    # shapes = ({
    #         "input_1": TensorShape((None, metadata_dict["n_trks"], metadata_dict["n_trk_features"])),
    #         "input_2": TensorShape((None, metadata_dict["n_trks"], metadata_dict["n_trk_features"]))
    #     },
    #     TensorShape((None, metadata_dict["n_dim"]))
    # )

    # tf_dataset = (Dataset.from_generator(
    #         DataLoader(
    #             input=config.input,
    #             metadata_dict=metadata_dict,
    #             savetracks=True,
    #             track_name=config.track_name,
    #             jets_name=config.jets_name,
    #             target_name=config.target_name
    #         ),
    #         types,
    #         shapes
    #     )
    #      .repeat()
    #      .prefetch(tf.data.AUTOTUNE)
    # )

    x = np.arange(6000).reshape(10,40,15)
    y_weight = np.random.random(400).reshape(10,40,1)
    y_feat = np.arange(12000).reshape(10,40,30)
    y = np.arange(30).reshape(10,3)

    model_checkpoint = ModelCheckpoint(
        config.output + "/model_epoch{epoch:03d}.h5",
        verbose=True,
        save_best_only=False,
        save_weights_only=False,
    )

    model = get_model(input_feat=(40,15), input_weight=(40,15), config=config)
    #print(len(list(tf_dataset)))
    callbacks = [model_checkpoint]
    model.fit([x,x], [y_feat, y_weight,y], epochs = config.epochs, steps_per_epoch = 5, callbacks=callbacks)

