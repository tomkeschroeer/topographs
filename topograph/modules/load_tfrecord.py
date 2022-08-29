"""Reader for tf records datasets."""
import json
import os

import tensorflow as tf


def load_tfrecords_train_dataset(
    train_file_folder: str,
    nfiles: int = 5,
    batch_size: int = 5000,
    get_edge_labels: bool = True,
    get_edge_feat_labels: bool = False,
    get_vertex_labels: bool = True,
    get_sample_weights: bool = False,
    get_inputs: bool = True,
    track_name: str = "X_train_tracks",
    edge_name: str = "Y_edge",
    edge_feat_name: str = "Y_edge_features",
    vertex_feat_name: str = "Y_vertex_features",
    edge_weight_layer_name: str = "edge_weight",
    edge_feat_layer_name: str = "edge_feat",
    vertex_network_layer_name: str = "vertex_network",
    input_weight_layer_name: str = "input_1",
    input_feat_layer_name: str = "input_2",
):
    """
    Load the train dataset from tfrecords files.

    Parameters
    ----------
    config : object
        Loaded train config.
    edge

    Returns
    -------
    train_dataset : tfrecord.Dataset
        Loaded train dataset from tfrecords
    metadata : dict
        Dict with the metadata infos of the train dataset.

    Raises
    ------
    ValueError
        If one of the given input files is not a tfrecords file.
    KeyError
        If no metadata file could be found in tfrecords directory.
    """
    train_file_names = os.listdir(train_file_folder)

    for train_file_name in train_file_names:

        # Check if file is tfrecords or .h5
        if not (".tfrecord" in train_file_name) and not (
            train_file_name == "metadata.json"
        ):
            raise ValueError(
                f"Input file {train_file_name} is neither a "
                ".h5 file nor a directory with TF Record Files. "
                "You should check this."
            )

    if "metadata.json" not in train_file_names:
        raise KeyError("No metadata file in directory.")

    tfrecord_reader = TFRecordReader(
        path=train_file_folder,
        batch_size=batch_size,
        nfiles=nfiles,
        track_name=track_name,
        edge_name=edge_name,
        edge_feat_name=edge_feat_name,
        vertex_feat_name=vertex_feat_name,
        get_edge_labels=get_edge_labels,
        get_edge_feat_labels=get_edge_feat_labels,
        get_vertex_labels=get_vertex_labels,
        get_sample_weights=get_sample_weights,
        get_inputs=get_inputs,
        edge_weight_layer_name=edge_weight_layer_name,
        edge_feat_layer_name=edge_feat_layer_name,
        vertex_network_layer_name=vertex_network_layer_name,
        input_weight_layer_name=input_weight_layer_name,
        input_feat_layer_name=input_feat_layer_name,
    )

    train_dataset = tfrecord_reader.load_dataset()
    metadata_name = (train_file_folder + "/metadata.json").replace("//", "/")
    with open(metadata_name, "r") as metadata_file:
        metadata = json.load(metadata_file)
    return train_dataset, metadata


class TFRecordReader:
    """Reader for tf records datasets."""

    def __init__(
        self,
        path: str,
        batch_size: int,
        nfiles: int,
        track_name: str = "X_train_tracks",
        edge_name: str = "Y_edge",
        edge_feat_name: str = "Y_edge_features",
        vertex_feat_name: str = "Y_vertex_features",
        get_edge_labels: bool = True,
        get_edge_feat_labels: bool = False,
        get_vertex_labels: bool = True,
        get_sample_weights: bool = False,
        get_inputs: bool = True,
        edge_weight_layer_name: str = "edge_weight",
        edge_feat_layer_name: str = "edge_feat",
        vertex_network_layer_name: str = "vertex_network",
        input_weight_layer_name: str = "input_1",
        input_feat_layer_name: str = "input_2",
    ):
        """
        Reads the tf records dataset.

        Parameters
        ----------
        path : str
            path where TFRecord is saved
        batch_size : int
            size of batches for the training
        nfiles : int
            number of tf record files loaded in parallel
        tagger_name : str
            Name of the tagger that is used
        tracks_name : str, optional
            Name of the track collection that is loaded,
            by default None
        use_track_labels : bool, optional
            Decide if you want to use track labels, by default
            False.
        sample_weights : bool, optional
            decide wether or not the sample weights should
            be returned, by default False
        n_cond : int, optional
            number of additional variables used for attention,
            by default None
        """
        self.path = path
        self.batch_size = batch_size
        self.nfiles = nfiles

        self.track_name = track_name
        self.edge_name = edge_name
        self.edge_feat_name = edge_feat_name
        self.vertex_feat_name = vertex_feat_name

        self.get_edge_labels = get_edge_labels
        self.get_edge_feat_labels = get_edge_feat_labels
        self.get_vertex_labels = get_vertex_labels
        self.get_sample_weights = get_sample_weights
        self.get_inputs = get_inputs

        self.edge_weight_layer_name = edge_weight_layer_name
        self.edge_feat_layer_name = edge_feat_layer_name
        self.vertex_network_layer_name = vertex_network_layer_name
        self.input_weight_layer_name = input_weight_layer_name
        self.input_feat_layer_name = input_feat_layer_name

    def load_dataset(self):
        """
        Load TFRecord and create Dataset for training

        Returns
        -------
        tf_Dataset
        """
        data_files = tf.io.gfile.glob((self.path + "/*.tfrecord").replace("//", "/"))
        dataset_shards = tf.data.Dataset.from_tensor_slices([data_files])
        dataset_shards.shuffle(tf.cast(tf.shape(data_files)[0], tf.int64))
        tf_dataset = dataset_shards.interleave(
            tf.data.TFRecordDataset,
            num_parallel_calls=tf.data.AUTOTUNE,
            cycle_length=self.nfiles,
        )
        tf_dataset = (
            tf_dataset.shuffle(self.batch_size * 10)
            .batch(self.batch_size)
            .map(
                self.decode_fn,
                num_parallel_calls=tf.data.experimental.AUTOTUNE,
            )
            .repeat()
            .prefetch(3)
        )
        return tf_dataset

    def decode_fn(self, record_bytes):
        """
        Convert serialised Dataset to dictionary and return inputs and labels

        Parameters
        ----------
        record_bytes : serialised object
            serialised Dataset

        Returns
        -------
        inputs : dict
            Dictionary of tf_data of jet and track inputs
        labels : tf_data
            tf data stream of labels

        Raises
        ------
        KeyError
            If given track selection not in metadata.
        KeyError
            If no conditional info is found in metadata.
        ValueError
            If tagger type is not supported.
        """

        # Get metadata file and load it
        metadata_name = (self.path + "/metadata.json").replace("//", "/")
        with open(metadata_name, "r") as metadata_file:
            metadata = json.load(metadata_file)

        features = {}
        shapes = {}

        if self.get_inputs:
            shapes[self.track_name] = (metadata["n_trks"], metadata["n_trk_features"])
            features[self.track_name] = tf.io.FixedLenFeature(
                shape=shapes[self.track_name], dtype=tf.float32
            )

        if self.get_edge_labels:
            shapes[self.edge_name] = (metadata["n_trks"], metadata["n_edge_y"])
            features[self.edge_name] = tf.io.FixedLenFeature(
                shape=shapes[self.edge_name], dtype=tf.int64
            )

        if self.get_edge_feat_labels:
            shapes[self.edge_feat_name] = (
                metadata["n_trks"],
                metadata["n_edge_feat_y"],
            )
            features[self.edge_feat_name] = tf.io.FixedLenFeature(
                shape=shapes[self.edge_feat_name], dtype=tf.float32
            )

        if self.get_vertex_labels:
            shapes[self.vertex_feat_name] = metadata["n_vertex_feat"]
            features[self.vertex_feat_name] = tf.io.FixedLenFeature(
                shape=shapes[self.vertex_feat_name], dtype=tf.float32
            )

        if self.get_sample_weights:
            shapes["sample_weight"] = (metadata["n_trks"], metadata["n_edge_y"])
            features["sample_weight"] = tf.io.FixedLenFeature(
                shape=shapes["sample_weight"], dtype=tf.float32
            )

        parse_ex = tf.io.parse_example(record_bytes, features)  # pylint: disable=E1120

        input_dir = {
            self.input_weight_layer_name: parse_ex[self.track_name],
            self.input_feat_layer_name: parse_ex[self.track_name],
        }

        label_dir = {}
        if self.get_edge_labels:
            label_dir[self.edge_weight_layer_name] = parse_ex[self.edge_name]
        if self.get_edge_feat_labels:
            label_dir[self.edge_feat_layer_name] = parse_ex[self.edge_feat_name]
        if self.get_vertex_labels:
            label_dir[self.vertex_network_layer_name] = parse_ex[self.vertex_feat_name]

        if self.get_sample_weights:
            sample_weights = {self.edge_weight_layer_name: parse_ex["sample_weight"]}

        return input_dir, label_dir, sample_weights
