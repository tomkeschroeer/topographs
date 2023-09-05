"""
Classes for loading the flavour tagging datasets for pytorch
"""
# pylint: disable=no-member

import math
from math import isnan
from pathlib import Path
from typing import Union

import h5py
import numpy as np
import torch as T
from torch.utils.data import Dataset, IterableDataset, TensorDataset, get_worker_info

from topograph.modules.tools import get_sample_weights

class Topographs_dataset(IterableDataset):
    def __init__(self, filename, n_samples, jet_types, batch_size=50_000, train=True):
        IterableDataset.__init__(self)
        
        self.filename = filename
        # self.batch_size = batch_size
        self.n_samples = n_samples
        self.jet_types = jet_types
        self.batch_size = n_samples
        if train:
            r = self.n_samples%self.batch_size
            if r != 0:
                self.n_samples = self.n_samples-r
            self.batch_size = min(batch_size, n_samples)
            

    def open(self):
        self.file = h5py.File(self.filename)
        self.tracks = self.file["X_train_tracks"]
        self.jet_type_per_jet = self.file["jet_type"]
        self.labels_open = {}
        for jet_type in self.jet_types:
            self.labels_open[f"Y_edge_{jet_type}"] = self.file[f"Y_edge_{jet_type}"]
            self.labels_open[f"Y_vertex_features_{jet_type}"]= self.file[f"Y_vertex_features_{jet_type}"]

    def get_indeces(self):
        # if self.n_samples == -1: 
        #     self.n_samples = len(self.tracks)
        self.n_batches = np.ceil(self.n_samples/self.batch_size)
        starts = np.linspace(0,(self.n_batches-1)*self.batch_size, int(self.n_batches), endpoint=True, dtype=int)
        ends = np.linspace(self.batch_size, self.n_batches*self.batch_size, int(self.n_batches), endpoint=True, dtype=int)
        return zip(starts, ends)

    def __iter__(self):
        indices = self.get_indeces()
        self.open()
        self.labels = {}
        self.vertex_masks = {}
        for inds in indices:
            self.tracks_batch = self.tracks[inds[0] : inds[1]].astype(np.float32)
            self.mask_batch = ~np.all(self.tracks_batch[..., :3] == 0, axis=-1)
            self.jet_types_batch = self.jet_type_per_jet[inds[0] : inds[1]].astype(int)
            for jet_type in self.jet_types:
                self.labels[f"Y_edge_{jet_type}"] = self.labels_open[f"Y_edge_{jet_type}"][inds[0] : inds[1]].astype(np.float32)
                self.labels[f"Y_vertex_features_{jet_type}"] = self.labels_open[f"Y_vertex_features_{jet_type}"][inds[0] : inds[1]].astype(np.float32)
                self.labels[f"sample_weights_{jet_type}"] = np.array(
                    list(map(get_sample_weights, np.stack((self.labels[f"Y_edge_{jet_type}"], self.mask_batch), axis=1))), dtype=np.float32
                )
                self.vertex_masks[f"vertex_mask_{jet_type}"] = (self.labels[f"Y_vertex_features_{jet_type}"] != -999.0)
            yield self.tracks_batch, self.labels, self.mask_batch, self.vertex_masks, self.jet_types_batch

    def __len__(self) -> int:
        num_sampels = self.n_samples if self.n_samples != -1 else len(self.tracks)
        num_batches = num_sampels / self.batch_size
        return math.ceil(num_batches)

    def on_epoch_start(self):
        """Reopen HDF file before each epoch to have fresh cache"""
        self.open()

    def on_epoch_end(self):
        """Close the HDF file at the end on epoch to free up the cache"""
        self.file.close()


class FlavourTaggingCommon:
    """Parent class to collect the common attributes and methods for the two types
    of flavour tagging datasets

    Points to an HDF file containing track information for Geant4 simulated jets
    along with their labels.
    Each event is described by up to 40 tracks, each with 21 attributes.

    For more information see:
    http://cds.cern.ch/record/2811135/files/ATL-PHYS-PUB-2022-027.pdf

    This type of dataset returns batches, not samples, as it improves stream speed by an
    order of magnitude
    """

    def __init__(
        self,
        dset: str = "train",
        file_name: str = "/set/me/please.h5",
        batch_size: int = 1024,
        drop_last: bool = True,
        start: Union[int, float] = 0,
        end: Union[int, float] = 0,
        vars: list = [],
        dtype: np.dtype = np.float32,
        buffer_size: int = 100_000,
        buffer_shuffle: bool = False,
        track_name="X_train_tracks",
        Y_edge_name="Y_edge",
        Y_vertex_name="Y_vertex_features",
        njets=-1,
        used_vertex_properties=None,
    ):
        """
        kwargs:
            dset: Either train or test
            file_name: The full path+name of the file to load
            batch_size: Size of each chunk to load from the dataset
            drop_last: If the final incomplete batch will be dropped
            start: Index of the firt readable element in the file
                float -> fraction of file size
            end: Index of the final readable element in the file
                float -> fraction of file size
            batches_per_buff: The number of batches in each buffer
            buffer_shuffle: If the buffers are shuffled when loaded
            dtype: The dtype to return the numpy arrays as
            buffer_size: Size of worker buffer
                Only applicable for Iterable variant
            buffer_shuffle: If the buffers are shuffled when loaded
                Only applicable for Iterable variant
        """
        print(f"Creating a {dset} flavour tagging dataset")

        ## Set basic class attributes
        self.dset = dset
        self.file_name = Path(file_name)
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.dtype = dtype
        self.buffer_size = buffer_size
        self.buffer_shuffle = buffer_shuffle
        self.track_name = track_name
        self.Y_edge_name = Y_edge_name
        self.Y_vertex_name = Y_vertex_name
        self.njets = njets
        self.vars = vars
        self.used_vertex_properties = used_vertex_properties

        ## Get the data from the file and save the number of samples
        print(f"Loading file: {str(self.file_name)}")
        self._open_file()
        self.num_file_samples = len(self.tracks)

        ## Save the number of accesible samples
        self.start = start
        self.end = end or self.num_file_samples
        if 0 < start < 1:
            self.start = math.floor(start * self.num_file_samples)
        if 0 < end < 1:
            self.end = math.floor(end * self.num_file_samples)
        print(f" - number of samples = {self.num_samples}")

    @property
    def num_samples(self):
        """The number of accesible samples in the file"""
        return self.end - self.start

    def _open_file(self):
        """Open the HDF files and keep the tables open"""
        self.file = h5py.File(self.file_name, "r")
        self.tracks = self.file[self.track_name][: self.njets]

        ## Truth information is not available in test files
        # if self.dset != "test":
        self.vertex_labels = self.file[self.Y_vertex_name][: self.njets]
        self.edge_label = self.file[self.Y_edge_name][: self.njets]
        self.sample_weights = np.array(list(map(get_sample_weights, self.edge_label)))

    def __len__(self) -> int:
        num_batches = self.num_samples / self.batch_size
        if self.drop_last:
            return math.floor(num_batches)
        return math.ceil(num_batches)

    def on_epoch_start(self):
        """Reopen HDF file before each epoch to have fresh cache"""
        self._open_file()

    def on_epoch_end(self):
        """Close the HDF file at the end on epoch to free up the cache"""
        self.file.close()


class IterableFlavourTaggingDataset(FlavourTaggingCommon, IterableDataset):
    """Uses a distributed streaming method
    - Slower but allows for buffer shuffling
    - Should be only used for training!
    """

    def __init__(self, **kwargs) -> None:
        IterableDataset.__init__(self)
        FlavourTaggingCommon.__init__(self, **kwargs)

        ## Determine the buffer size such that it is divisible by batch size
        r = self.buffer_size % self.batch_size
        if r != 0:
            self.buffer_size = self.buffer_size - r
            print("Buffer size must be EXACTLY divisible by batch size!")
            print(f" - changing buffer size to {self.buffer_size}")

    def __iter__(self):
        """Called seperately for each worker
        - Divides up the readable portions of the file into workers
        - Divides up the worker's samples into buffers (which are shuffled)
        - Divides up the buffers into batches
        - Returns each batch
        """

        ## Single-process vs multi process data loading
        worker_info = get_worker_info()
        worker_id = 0 if worker_info is None else worker_info.id
        num_workers = 1 if worker_info is None else worker_info.num_workers

        ## Calculate the bounds of the worker
        per_worker = self.num_samples // num_workers
        worker_start = self.start + worker_id * per_worker
        worker_end = min(worker_start + per_worker, self.end)

        ## Calculate the number of buffers required for the worker
        num_buffers = math.ceil(per_worker / self.buffer_size)

        ## Cycle through the buffers
        for buf_id in range(num_buffers):
            buf_start = worker_start + buf_id * self.buffer_size
            buf_end = min(buf_start + self.buffer_size, worker_end)

            ## Load the seperate buffer for each of the data fields
            if self.vars is None:
                buf_tracks = self.tracks[buf_start:buf_end, :, :].astype(self.dtype)
            else:
                buf_tracks = self.tracks[buf_start:buf_end, :, self.vars].astype(
                    self.dtype
                )
            buf_edge_labels = self.edge_label[buf_start:buf_end].astype("f")
            if self.used_vertex_properties is not None:
                buf_vertex_labels = self.vertex_labels[
                    buf_start:buf_end, self.used_vertex_properties
                ].astype("f")
            else:
                buf_vertex_labels = self.vertex_labels[buf_start:buf_end]
            buf_sample_weights = self.sample_weights[buf_start:buf_end].astype("f")

            ## Calculate the number of batches required for the buffer
            this_buff_size = len(buf_tracks)
            num_batches = math.ceil(this_buff_size / self.batch_size)

            ## Cycle through the buffer pulling out each batch
            for batch_id in range(num_batches):
                batch_start = batch_id * self.batch_size
                batch_end = batch_start + self.batch_size

                ## If the batch is incomplete it means that we have reached end of file
                if self.drop_last and batch_end > this_buff_size:
                    return

                ## Yield the batch from each of the buffers
                tracks = buf_tracks[batch_start:batch_end]
                edge_labels = buf_edge_labels[batch_start:batch_end]
                # edge_labels[edge_labels==0] = 1
                vertex_labels = buf_vertex_labels[batch_start:batch_end]
                sample_weights = buf_sample_weights[batch_start:batch_end]

                ## Generate the mask based on the first three features of the tracks
                mask = ~np.all(tracks[..., :3] == 0, axis=-1)
                # tracks[mask] = -T.inf
                # tracks = T.Tensor(tracks).masked_fill(mask, T.inf)
                mask_vertex_labels = np.all(~np.isnan(vertex_labels), axis=-1)
                yield tracks, edge_labels, vertex_labels, sample_weights, mask, mask_vertex_labels

    def __getitem__(self, *_args) -> None:
        """Implemented just to quiet the pylance error even though you shouldnt need
        this method for an iterable datset!
        """
        return None
