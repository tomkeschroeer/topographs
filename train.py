"""Training script to perform topograph training."""
import argparse as pars

import tensorflow as tf
from h5py import File
from tensorflow.data import Dataset
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.layers import Input

from topograph.modules import (
    DataLoader,
    GetConfiguration,
    TopographModel,
    load_tfrecords_train_dataset,
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

    edge_weight_layer_name = "edge_weight"
    edge_feat_layer_name = "edge_feat"
    vertex_network_layer_name = "vertex_network"
    input_weight_layer_name = "input_1"
    input_feat_layer_name = "input_2"

    if config.tfrecords["use_tfrecords_to_train"]:
        # types, shapes = get_types_shapes(
        #     get_labels=True,
        #     get_weight_labels=True,
        #     get_inputs=self.config.use_sample_weights,
        #     metadata_dict=metadata_dict,
        #     input_feat_layer_name=input_weight_layer_name,
        #     input_weight_layer_name=input_weight_layer_name,
        #     edge_feat_layer_name=edge_feat_layer_name,
        #     edge_weight_layer_name=edge_feat_layer_name,
        #     vertex_network_layer_name=vertex_network_layer_name
        # )
        tf_dataset, metadata_dict = load_tfrecords_train_dataset(config=config)

    else:
        DatasetGenerator = DataLoader(
            input=train_file,
            get_labels=True,
            get_weight_labels=True,
            get_inputs=True,
            get_sample_weights=config.use_sample_weights,
            metadata_dict=metadata_dict,
            savetracks=True,
            track_name=config.tracks_name,
            edge_name=config.edge_name,
            edge_feat_name=config.edge_feat_name,
            vertex_feat_name=config.vertex_feat_name,
            edge_weight_layer_name=edge_weight_layer_name,
            edge_feat_layer_name=edge_feat_layer_name,
            vertex_network_layer_name=vertex_network_layer_name,
            input_weight_layer_name=input_weight_layer_name,
            input_feat_layer_name=input_feat_layer_name,
        )

        types, shapes = DatasetGenerator.get_types_shapes()
        tf_dataset = (
            Dataset.from_generator(DatasetGenerator, types, shapes)
            .repeat()
            .prefetch(tf.data.AUTOTUNE)
        )

        metadata_dict_val = {}
        val_file = f"{config.output}/{config.validation_file_name}".replace("//", "/")
        with File(val_file, "r") as f:
            (
                metadata_dict_val["n_jets"],
                metadata_dict_val["n_trks"],
                metadata_dict_val["n_trk_features"],
            ) = f[f"{config.tracks_name}"].shape
            _, metadata_dict_val["n_vertex_feat"] = f[
                f"{config.vertex_feat_name}"
            ].shape
            _, _, metadata_dict_val["n_edge_y"] = f[f"{config.edge_name}"].shape

        DatasetGeneratorVal = DataLoader(
            input=val_file,
            get_labels=True,
            get_weight_labels=True,
            get_inputs=True,
            get_sample_weights=False,
            metadata_dict=metadata_dict_val,
            savetracks=True,
            track_name=config.tracks_name,
            edge_name=config.edge_name,
            edge_feat_name=config.edge_feat_name,
            vertex_feat_name=config.vertex_feat_name,
            edge_weight_layer_name=edge_weight_layer_name,
            edge_feat_layer_name=edge_feat_layer_name,
            vertex_network_layer_name=vertex_network_layer_name,
            input_weight_layer_name=input_weight_layer_name,
            input_feat_layer_name=input_feat_layer_name,
        )

        types_val, shapes_val = DatasetGeneratorVal.get_types_shapes()
        tf_dataset_val = Dataset.from_generator(
            DatasetGeneratorVal, types_val, shapes_val
        ).prefetch(tf.data.AUTOTUNE)

    modelfile_dir = (config.output + "/modelfiles/model_epoch{epoch:03d}.h5").replace(
        "//", "/"
    )

    model_checkpoint = ModelCheckpoint(
        modelfile_dir,
        verbose=True,
        monitor="accuracy",
        save_best_only=False,
        save_weights_only=False,
    )

    callbacks = [model_checkpoint]
    input_feat = Input(shape=(metadata_dict["n_trks"], metadata_dict["n_trk_features"]))
    input_weight = Input(
        shape=(metadata_dict["n_trks"], metadata_dict["n_trk_features"])
    )

    model_builder = TopographModel(
        config=config,
        metadata_dict=metadata_dict,
        edge_weight_layer_name=edge_weight_layer_name,
        edge_feat_layer_name=edge_feat_layer_name,
        vertex_network_layer_name=vertex_network_layer_name,
        input_weight_layer_name=input_weight_layer_name,
        input_feat_layer_name=input_feat_layer_name,
    )

    model = model_builder.get_model(input_feat, input_weight)

    model.fit(
        tf_dataset,
        epochs=config.epochs,
        # validation_data=tf_dataset_val,
        steps_per_epoch=metadata_dict["n_jets"] // config.stepsize,
        callbacks=callbacks,
    )
