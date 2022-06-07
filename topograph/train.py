import argparse as pars
from importlib_metadata import metadata
import numpy as np
from h5py import File
from tensorflow import (
    TensorShape,
    float32,
    int32
)
from tensorflow.data import Dataset

from modules import (
    GetConfiguration,
    get_model,
    DataGenerator
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
    with File(config.input, "r") as f:
        metadata["n_jets"], metadata["n_trks"], metadata["n_trk_features"] = f[f"X_{tracks_name}_train"].shape
        _, metadata["n_dim"] = f["Y_train"].shape

    types = ({
            "input_1": float32,
            "input_2": float32
        },
        int32
    )
    shapes = ({
            "input_1": TensorShape(None, metadata["ntracks"], metadata["n_trk_features"]),
            "input_2": TensorShape(None, metadata["ntracks"], metadata["n_trk_features"])
        },
        TensorShape(Nonemetadata["n_dim"])
    )

    tf_dataset = (Dataset.from_generator(
            DataGenerator(
                input=config.input,
                metadata=metadata
                ),
            types,
            shapes
        )
        .repeat()
        .prefetch(tf.data.AUTOTUNE)
    )


    model = get_model(input_feat=(40,15), input_weight=(40,15), config=config)
    model.fit(tf_dataset)