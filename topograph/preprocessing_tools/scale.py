import numpy as np
from h5py import File
import os
import json

from topograph.modules.tools import (
    get_logger,
    GlobalConfig,
    get_track_mask
)

class Scaler:
    def __init__(self, config):
        self.config = config
        self.global_conf = GlobalConfig()
        self.var_list = self.global_conf.track_inputs
        self.scale_dict_path = f"{self.config.output}/{self.config.scale_dict}"

    def Run(self):
        logger = get_logger()
        logger.info("scale inputs...")

        # Extract the correct variables
        chunk_size = 1e5

        # Get the file_length
        file_length = len(File(f"{self.config.output}/{self.config.one_file_name}", "r")[f"/{self.config.input_tracks_name}"].fields(self.var_list[0])[:])

        # Get the number of chunks we need to load
        n_chunks = int(np.ceil(file_length / chunk_size))

        logger.info("Calculating scaling and shifting values for the track variables")

        # Init a empty scale dict for the tracks
        scale_dict_trk = {}

        # Loop over all tracks selections
        scale_dict_trk_selection = {}
        # Load generator
        trks_scaling_generator = self.get_scaling_tracks_generator(
            input_file=f"{self.config.output}/{self.config.one_file_name}",
            nJets=file_length,
            tracks_name=self.config.input_tracks_name,
            chunk_size=chunk_size,
        )

        # Loop over chunks
        for chunk_counter in range(n_chunks):
            logger.info(f"Using chunk {chunk_counter+1} from {n_chunks}")
            # Check if this is the first time loading from the generator
            if chunk_counter == 0:
                # Get the first chunk of scales from the generator
                scale_dict_trk_selection, nTrks_loaded = next(
                    trks_scaling_generator
                )
            else:
                # Get the next chunk of scales from the generator
                tmp_dict_trk, tmp_nTrks_loaded = next(trks_scaling_generator)

                # Combine the scale dicts coming from the generator
                (
                    scale_dict_trk_selection,
                    nTrks_loaded,
                ) = self.join_scale_dicts_trks(
                    first_scale_dict=scale_dict_trk_selection,
                    second_scale_dict=tmp_dict_trk,
                    first_nTrks=nTrks_loaded,
                    second_nTrks=tmp_nTrks_loaded,
                )

        scale_dict_trk.update({self.config.input_tracks_name: scale_dict_trk_selection})

            # Add scale dict for given tracks selection to the more general one
        # TODO: change in python 3.9
        # save scale/shift dictionary to json file
        os.makedirs(os.path.dirname(self.scale_dict_path), exist_ok=True)
        with open(self.scale_dict_path, "w") as outfile:
            json.dump(scale_dict_trk, outfile, indent=4)
        logger.info(f"Saved scale dictionary as {self.scale_dict_path}")

    def get_scaling_tracks_generator(
        self,
        input_file: str,
        nJets: int,
        tracks_name: str,
        chunk_size: int = int(10000),
    ):
        """
        Set up a generator that loads the tracks in chunks and calculates the mean/std.


        Parameters
        ----------
        input_file : str
            File which is to be scaled.
        nJets : int
            Number of jets which are to be scaled.
        tracks_name : str
            Name of the tracks
        chunk_size : int, optional
            The number of jets which are loaded and scaled/shifted per step,
            by default int(10000)

        Yields
        ------
        scale_dict_trk : dict
            Dict with the scale/shift values for each variable.
        nTrks : int
            Number of tracks used for scaling/shifting.
        """

        # Load the variables which are scaled/shifted

        # Open h5 file
        with File(input_file, "r") as infile_all:

            # Get the indices
            start_ind = 0
            tupled_indices = []

            while start_ind < nJets:
                end_ind = int(start_ind + chunk_size)
                end_ind = min(end_ind, nJets)
                tupled_indices.append((start_ind, end_ind))
                start_ind = end_ind

            for index_tuple in tupled_indices:

                trks = np.asarray(
                    infile_all[f"/{tracks_name}"][index_tuple[0] : index_tuple[1]]
                )

                track_mask = get_track_mask(trks)

                X_trk_train = np.stack(
                    [np.nan_to_num(trks[v]) for v in self.var_list], axis=-1
                )

                scale_dict_trk, nTrks = self.get_scaling_tracks(
                    data=X_trk_train[:, :, :],
                    var_names=self.var_list,
                    track_mask=track_mask,
                )

                # Yield the scale dict and the number jets
                yield scale_dict_trk, nTrks

    def get_scaling_tracks(
        self,
        data: np.ndarray,
        var_names: list,
        track_mask: np.ndarray,
    ):
        """
        Calculate the scale dict for the tracks and return the dict.

        Parameters
        ----------
        data : np.ndarray
            Loaded tracks with shape (nJets, nTrks, nTrkFeatures)
        var_names : list
            List of variables which are to be scaled
        track_mask : np.ndarray
            Boolen array where False denotes padded tracks,
            with shape (nJets, nTrks)

        Returns
        -------
        scale_dict : dict
            Scale dict with scaling/shifting values for each variable
        nTrks : int
            Number of tracks used to calculate the scaling/shifting
        """
        # TODO add weight support for tracks

        # Initalise scale dict
        scale_dict = {}

        # For each track variable
        for v, name in enumerate(var_names):
            f = data[:, :, v]

            # Get tracks
            slc = f[track_mask]
            nTrks = len(slc)

            # Caculate normalisation parameters
            m, s = slc.mean(), slc.std()
            scale_dict[name] = {"shift": float(m), "scale": float(s)}

        return scale_dict, nTrks

    def join_scale_dicts_trks(
        self,
        first_scale_dict: dict,
        second_scale_dict: dict,
        first_nTrks: int,
        second_nTrks: int,
    ):
        """
        Combining the scale dicts of two track chunks.

        Parameters
        ----------
        first_scale_dict : dict
            First scale dict to join.
        second_scale_dict : dict
            Second scale dict to join.
        first_nTrks : int
            Number of tracks used for the first scale dict.
        second_nTrks : int
            Number of tracks used for the second scale dict.

        Returns
        -------
        combined_scale_dict : dict
            The combined scale dict.
        combined_nTrks : int
            The combined number of tracks.
        """

        # Init a new combined scale dict
        combined_scale_dict = {}

        for var in first_scale_dict:
            # Add var to combined dict
            combined_scale_dict[var] = {}

            # Combine the means
            (
                combined_scale_dict[var]["shift"],
                combined_scale_dict[var]["scale"],
            ) = self.join_mean_scale(
                first_scale_dict=first_scale_dict,
                second_scale_dict=second_scale_dict,
                variable=var,
                first_N=first_nTrks,
                second_N=second_nTrks,
            )

        # Sum of nTrks corresponding to combined scale dict
        combined_nTrks = first_nTrks + second_nTrks

        return combined_scale_dict, combined_nTrks

    def join_mean_scale(
        self,
        first_scale_dict: dict,
        second_scale_dict: dict,
        variable: str,
        first_N: int,
        second_N: int,
    ):
        """
        Combine the mean and scale of the two input scale dict.

        Parameters
        ----------
        first_scale_dict : dict
            First scale dict with the variable and
            their respective mean/std inside.
        second_scale_dict : dict
            Second scale dict with the variable and
            their respective mean/std inside.
        variable : str
            Variable which is to be combined.
        first_N : int
            Number of tracks/jets used to calculate mean/std for first dict.
        second_N : int
            Number of tracks/jets used to calculate mean/std for second dict.

        Returns
        -------
        combined_mean : float
            Combined mean/shift.
        combined_std : float
            Combined std/scale.
        """

        # Get the values in variables
        mean = first_scale_dict[variable]["shift"]
        tmp_mean = second_scale_dict[variable]["shift"]
        std = first_scale_dict[variable]["scale"]
        tmp_std = second_scale_dict[variable]["scale"]

        # Combine the means
        combined_mean = (mean * first_N + tmp_mean * second_N) / (first_N + second_N)

        # Combine the std
        combined_std = np.sqrt(
            (
                (
                    (((mean - combined_mean) ** 2 + std**2) * first_N)
                    + (((tmp_mean - combined_mean) ** 2 + tmp_std**2) * second_N)
                )
            )
            / (first_N + second_N)
        )

        return combined_mean, combined_std