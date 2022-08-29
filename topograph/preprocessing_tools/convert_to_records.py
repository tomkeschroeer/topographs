"""Module converting h5 to tf records."""

import json
import os

import h5py
import numpy as np
import tensorflow as tf
import tqdm

from topograph.modules import get_sample_weights


class H5toTfrecordsConverter:
    """h5 converter to tf records."""

    def __init__(self, config):
        self.config = config
        self.file_name = f"{self.config.output}/{self.config.training_file_name}"
        self.chunk_size = 5_000

        self.tracks_name = config.tracks_name
        self.edge_name = config.edge_name
        # self.edge_feat_name = config.edge_feat_name
        self.vertex_feat_name = config.vertex_feat_name
        self.get_sample_weights = config.use_sample_weights

    def load_h5File_Train(self):
        """
        load the numbers of entries given by the chunk size for the jets,
        tracks and labels from train file.

        Yields
        ------
        X_jets : array_like
            Training jets
        X_trks : array_like
            Training tracks
        Y_jets : array_like
            Training jet labels
        Y_trks : array_like
            Training track labels
        Weights : array_like
            Training weights
        X_Add_Vars : array_like
            Conditional variables for CADS and Umami Cond Att.
        """

        # Open the h5 output file
        with h5py.File(self.file_name, "r") as hFile:

            # Get the number of jets in the file
            length_dataset = len(hFile[self.tracks_name])

            # Get the number of loads that needs to be done
            total_loads = length_dataset // self.chunk_size

            # Ensure that the loads are enough
            if length_dataset % self.chunk_size != 0:
                total_loads += 1

            for i in tqdm.tqdm(range(total_loads)):

                # Get start and end chunk index
                start = i * self.chunk_size
                end = (i + 1) * self.chunk_size

                # Get the jets
                tracks = hFile[self.tracks_name][start:end]
                edges = hFile[self.edge_name][start:end]
                # edge_feat = hFile[self.edge_feat_name][start:end]
                vertex_feat = hFile[self.vertex_feat_name][start:end]
                sample_weights = np.array(list(map(get_sample_weights, edges)))

                yield tracks, edges, vertex_feat, sample_weights

    def save_parameters(self, record_dir):
        """
        write metadata into metadata.json and save it with tf record files

        Parameters
        ----------
        record_dir : str
            directory where metadata should be saved
        """

        # Open h5 file
        metadata_dict = {}
        with h5py.File(self.file_name) as h5file:
            (
                metadata_dict["n_jets"],
                metadata_dict["n_trks"],
                metadata_dict["n_trk_features"],
            ) = h5file[f"{self.tracks_name}"].shape
            _, metadata_dict["n_vertex_feat"] = h5file[f"{self.vertex_feat_name}"].shape
            _, _, metadata_dict["n_edge_y"] = h5file[f"{self.edge_name}"].shape

        # Get filepath for the metadata file
        metadata_filename = record_dir + "/metadata.json"

        # Write the metadata (dim. values) to file
        with open(metadata_filename, "w") as metadata:
            json.dump(metadata_dict, metadata)

    def write_tfrecord(self):
        """
        write inputs and labels of train file into a TFRecord
        """

        # Get the path to h5 file and make a dir with that name
        record_dir = self.file_name.replace(".h5", "")
        os.makedirs(record_dir, exist_ok=True)

        # Get filename
        tf_filename_start = record_dir.split("/")[-1]
        n = 0

        # Iterate over chunks
        for (tracks, edges, vertex_feat, sample_weight) in self.load_h5File_Train():
            n += 1

            # Get filename of the chunk
            filename = (
                record_dir
                + "/"
                + tf_filename_start
                + "_"
                + str(n).zfill(4)
                + ".tfrecord"
            )

            with tf.io.TFRecordWriter(filename) as file_writer:
                for iterator, _ in enumerate(tracks):

                    # Get record bytes example
                    record_bytes = tf.train.Example()

                    # Add jets
                    record_bytes.features.feature[
                        self.tracks_name
                    ].float_list.value.extend(tracks[iterator].reshape(-1))

                    # Add labels
                    # record_bytes.features.feature[self.edge_feat_name].int64_list.value.extend(
                    #     edges_feat[iterator]
                    # )

                    # Add weights
                    record_bytes.features.feature[
                        self.edge_name
                    ].int64_list.value.extend(edges[iterator].reshape(-1))
                    record_bytes.features.feature[
                        self.vertex_feat_name
                    ].float_list.value.extend(vertex_feat[iterator].reshape(-1))

                    if self.get_sample_weights:
                        record_bytes.features.feature[
                            "sample_weight"
                        ].float_list.value.extend(sample_weight[iterator].reshape(-1))

                    # Write to file
                    file_writer.write(record_bytes.SerializeToString())
        self.save_parameters(record_dir=record_dir)

    def Run(self):
        self.write_tfrecord()
