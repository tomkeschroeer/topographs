import argparse as pars
from h5py import File

import tensorflow as tf
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
    step_activation
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
    with File(f"{config.output}/{config.training_file_name}", "r") as f:
        metadata_dict["n_jets"], metadata_dict["n_trks"], metadata_dict["n_trk_features"] = f[f"{config.tracks_name}"].shape
        _, metadata_dict["n_vertex_feat"] = f[f"{config.vertex_feat_name}"].shape

    DatasetGenerator = DataLoader(
            input=f"{config.output}/{config.training_file_name}".replace("//","/"),
            get_labels=True,
            get_inputs=True,
            metadata_dict=metadata_dict,
            savetracks=True,
            track_name=config.tracks_name,
            edge_name=config.edge_name,
            edge_feat_name=config.edge_feat_name,
            vertex_feat_name=config.vertex_feat_name
        )

    types, shapes = DatasetGenerator.get_types_shapes()

    tf_dataset = (Dataset.from_generator(
            DatasetGenerator,
            types,
            shapes
        )
         .repeat()
         .prefetch(tf.data.AUTOTUNE)
    )

    model_checkpoint = ModelCheckpoint(
        config.output + "modelfiles/model_epoch{epoch:03d}.h5",
        verbose=True,
        save_best_only=False,
        save_weights_only=False,
    )

    callbacks = [model_checkpoint]
    
    model = get_model(
        input_feat=(metadata_dict["n_trks"], metadata_dict["n_trk_features"]), 
        input_weight=(metadata_dict["n_trks"], metadata_dict["n_trk_features"]), 
        config=config)

    model.fit(
        tf_dataset, 
        epochs = config.epochs, 
        steps_per_epoch = metadata_dict["n_jets"]/config.stepsize, 
        callbacks=callbacks
    )
