from pathlib import Path
import math
from typing import Union

import h5py
import numpy as np

import torch as T
from torch.utils.data import Dataset, IterableDataset, get_worker_info


def get_sample_weights(x):
    # print(f"x in get sample weights = {x}")
    x = x.flatten()
    length = len(x)
    # print(f"Therefore, the length is {length}")
    n_b = sum(x)
    n_nonb = length - n_b
    fac_b = length / (n_b + 1e-5)
    fac_nonb = length / (n_nonb + 1e-5)
    weights = np.ones(len(x))
    weights[x == 1] = fac_b
    weights[x == 0] = fac_nonb
    # print(f"Let's check the weights: what do they look like?! {weights}")
    return weights.reshape((length, 1))


class FlavourTaggingCommon:
    """Parent class to collect the common attributes and methods for the two types
    of flavour tagging datasets

    Points to an HDF file containing track information for Geant4 simulated jets
    along with their labels.
    Each event is described by up to 40 tracks, each with 21(22) attributes.

    For more information see:
    http://cds.cern.ch/record/2811135/files/ATL-PHYS-PUB-2022-027.pdf

    Information about the truth labelling can be found here:
    https://ftag.docs.cern.ch/algorithms/labelling/

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
        dtype: np.dtype = np.float32,
        buffer_size: int = 100_000,
        buffer_shuffle: bool = False,
        trk_inpt_name: str = "X_train_tracks",
        vertex_labels: str = "Y_vertex_features",
        edge_labels: str = "Y_edge",
        use_lep_ID: bool = False,
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
                trk_inpt_name
            trk_inpt_name: The name of the table containing the track features
            trk_outp_name: The name of the table containing the track and vertex labels
            jet_inpt_name: The name of the table containing the jet features (pt, eta)
            jet_outp_name: The name of the table containing the jet labels (flavour)
            use_lep_ID: If the lepton ID input should be used if available
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
        self.trk_inpt_name = trk_inpt_name
        self.vertex_labels = vertex_labels
        self.edge_labels = edge_labels

        ## Get the data from the file and save the number of samples
        print(f"Loading file: {str(self.file_name)}")
        self._open_file()
        self.num_file_samples = len(self.tracks)
        self.track_dim = self.tracks.shape[-1]
        self.num_flavours = (
            self.jet_labels.shape[-1] if hasattr(self, "jet_labels") else 3
        )

        ## If we are using the lepton ID input
        if use_lep_ID:
            if self.track_dim == 21:
                print("Warning! Use lep ID was set but file has no such variable!")
                print("-Changing to false")
        elif self.track_dim == 22:
            print("Dropping the lepton ID input from the table")
            self.track_dim = 21

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
        self.tracks = self.file[self.trk_inpt_name]
        # self.jets = self.file[self.jet_inpt_name]

        ## Truth information is not available in test files
        if self.dset != "test":
            self.edge_labels = self.file[self.edge_labels]
            self.vertex_labels = self.file[self.vertex_labels]
            # self.jet_labels = self.file[self.jet_outp_name]

    def __len__(self) -> int:
        num_batches = self.num_samples / self.batch_size
        if self.drop_last:
            return math.floor(num_batches)
        return math.ceil(num_batches)

    def on_epoch_start(self, _epoch_num: int)->None:
        """Reopen HDF file before each epoch to have fresh cache"""
        self._open_file()

    def on_epoch_end(self, _epoch_num: int)->None:
        """Close the HDF file at the end on epoch to free up the cache"""
        self.file.close()

    def get_track_weights(self):
        """Return the track weights based on how frequent the classes occur
        and then normalise to one
        """
        # class_counts = T.from_numpy(
        #     np.unique(self.track_labels[..., 0], return_counts=True)[1]
        # )[1:]
        class_counts = T.tensor(get_track_counts(str(self.file_name)))
        return T.sum(class_counts) / class_counts / len(class_counts)


class MappableFlavourTaggingDataset(FlavourTaggingCommon, Dataset):
    """A mappable type dataset
    - Quick but no shuffling
    - Good for testing and exporting
    """

    def __init__(self, **kwargs) -> None:
        Dataset.__init__(self)
        FlavourTaggingCommon.__init__(self, **kwargs)

    def __getitem__(self, idx: int) -> tuple:
        """Returns an entire batch of data"""

        ## Because this type of dataset pre batches our data we need to get bounds
        beg = idx * self.batch_size + self.start
        end = beg + self.batch_size
        end = min(end, self.end)

        ## Load the input data
        tracks = self.tracks[beg:end, :, : self.track_dim].astype(self.dtype)
        jets = self.jets[beg:end, :2].astype(self.dtype)  # eta phi only

        ## Load the truth data (track labels are category IDXs)
        if self.dset != "test":  # Not available in our test sets
            labels = self.labels[beg:end].astype(self.dtype)
            track_labels = self.track_labels[beg:end].astype("l")
        else:
            labels = None
            track_labels = None

        ## Generate the mask based on the first three features of the tracks
        mask = ~np.all(tracks[..., :3] == 0, axis=-1)

        return tracks, jets, mask, labels, track_labels


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
            buf_edge_labels = self.edge_labels[buf_start:buf_end, :2].astype(self.dtype)  # eta phi
            buf_vertex_labels = self.vertex_labels[buf_start:buf_end].astype(self.dtype)
            buf_tracks = self.tracks[buf_start:buf_end, :, : self.track_dim].astype(
                self.dtype
            )

            ## Calculate the number of batches required for the buffer
            this_buff_size = len(buf_tracks)
            num_batches = math.ceil(this_buff_size / self.batch_size)

            ## Cycle through the buffer pulling out each batch
            for batch_id in range(num_batches):
                batch_start = batch_id * self.batch_size
                batch_end = batch_start + self.batch_size

                ## If the batch is incomplete it means that we have reached end of file
                ## This is always the case due to ensuring that the buff size is
                ## divisible by the batch size
                if self.drop_last and batch_end > this_buff_size:
                    return

                ## Yeild the batch from each of the buffers
                tracks = buf_tracks[batch_start:batch_end]
                labels_edge = buf_edge_labels[batch_start:batch_end]
                labels_vertex = buf_vertex_labels[batch_start:batch_end]
                sample_weights = get_sample_weights(labels_edge)

                ## Generate the mask based on the first three features of the tracks
                mask = ~np.all(tracks[..., :3] == 0, axis=-1)
                

                yield tracks, tracks, labels_edge, labels_vertex, sample_weights
