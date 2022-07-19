import chunk
import numpy as np
from h5py import File
import json

from topograph.modules import (
    GlobalConfig,
    get_logger,
    get_track_mask
)

class Apply_Scaler:
    def __init__(self, config):
        self.config = config
        self.scale_dict_path = f"{self.config.output}/{self.config.scale_dict}"
        self.global_conf = GlobalConfig()
        self.var_list = self.global_conf.track_inputs
        self.input_tracks_name = self.config.input_tracks_name

    def Run(self):
        logger = get_logger()
        # logger.info(f"Scale/Shift jets from {self.config.input}")
        # logger.info(f"Using scales in {self.scale_dict_path}")

        # chunk_size = 1e5

        # file_length = len(File(f"{self.config.output}/{self.config.one_file_name}", "r")[f"/{self.input_tracks_name}"][self.var_list[0]][:])

        # n_chunks = int(np.ceil(file_length / chunk_size))

        # # Check if tracks are used
        # tracks_scale_dict = {}
        # # Get the scale dict for tracks
        # with open(self.scale_dict_path, "r") as infile:
        #     full_scale_dict = json.load(infile)
        #     tracks_scale_dict[self.input_tracks_name] = full_scale_dict[f"{self.input_tracks_name}"]


        # logger.info("Applying scaling and shifting.")
        self.out_file = f"{self.config.output}/{self.config.preprocessing_file_name}"
        # logger.info(f"Save scaled inputs in file {self.out_file}")

        # scale_generator = self.scale_generator(
        #     input_file=f"{self.config.output}/{self.config.one_file_name}",
        #     nJets = file_length,
        #     tracks_scale_dict=tracks_scale_dict,
        #     chunk_size=chunk_size,
        # )
        # with File(self.out_file, "w") as h5file:

        #     # Set up chunk counter and start looping
        #     chunk_counter = 0
        #     for chunk_counter in range(n_chunks):
        #         logger.info(
        #             f"Applying scales for chunk {chunk_counter+1} of {n_chunks}."
        #         )
        #         try:
        #             tracks = next(scale_generator)

        #             if chunk_counter == 0:
        #                 h5file.create_dataset(
        #                     self.input_tracks_name,
        #                     data=tracks[0],
        #                   #  compression="lzf",
        #                     chunks=((100,) + tracks[0].shape[1:]),
        #                     maxshape=(
        #                         None,
        #                         tracks[0].shape[1],
        #                         tracks[0].shape[2],
        #                     ),
        #                 )

        #             else:
        #                 h5file[self.input_tracks_name].resize(
        #                     (h5file[self.input_tracks_name].shape[0] + tracks[0].shape[0]),
        #                     axis=0,
        #                 )
                        
        #                 h5file[self.input_tracks_name][-tracks[0].shape[0] :] = tracks[0]

        #         except StopIteration:
        #             break

        #         chunk_counter += 1

        self.save_remaining_dt(logger)

    def scale_generator(
        self,
        input_file: str,
        nJets: int,
        tracks_scale_dict: dict = None,
        chunk_size: int = int(10000),
    ):
        """
        Set up a generator who applies the scaling/shifting for the given
        jet variables.

        Parameters
        ----------
        input_file : str
            File which is to be scaled.
        nJets : int
            Number of jets which are to be scaled.
        tracks_scale_dict : dict, optional
            Scale dict of the track variables., by default None
        chunk_size : int, optional
            The number of jets which are loaded and scaled/shifted per step,
            by default int(10000)

        Yields
        ------
        jets : np.ndarray
            Yielded jets
        tracks : np.ndarray
            Yielded tracks
        labels : np.ndarray
            Yielded labels
        tracks_labels : np.ndarray
            Yielded track labels
        flavour : np.ndarray
            Yielded flavours

        Raises
        ------
        ValueError
            If scale is found to be 0 or inf for any jet variable.
        """

        # Open the file and load the jets
        with File(input_file, "r") as f:

            # Get the indices
            start_ind = 0
            tupled_indices = []
            while start_ind < nJets:
                end_ind = int(start_ind + chunk_size)
                end_ind = min(end_ind, nJets)
                tupled_indices.append((start_ind, end_ind))
                start_ind = end_ind
                end_ind = int(start_ind + chunk_size)

            for index_tuple in tupled_indices:
                tracks = []
                # Loop on each track selection
                trk_scale_dict = tracks_scale_dict[f"{self.input_tracks_name}"]
                # Load tracks
                trks = np.asarray(
                    f[f"/{self.input_tracks_name}"][
                        index_tuple[0] : index_tuple[1]
                    ]
                )

                # Apply scaling to the tracks
                trks = self.apply_scaling_trks(
                    trks=trks,
                    var_list=self.var_list,
                    scale_dict=trk_scale_dict
                )
                tracks.append(trks)
                # Yield jets, labels and tracks
                yield tracks

    def apply_scaling_trks(
        self,
        trks: np.ndarray,
        var_list: dict,
        scale_dict: dict
    ):
        """
        Apply the scaling/shifting to the tracks.

        Parameters
        ----------
        trks : np.ndarray
            Loaded tracks as numpy array.
        variable_config : dict
            Loaded variable config.
        scale_dict : dict
            Loaded scale dict.
        tracks_name : str
            Name of the tracks.

        Returns
        -------
        scaled_trks : np.ndarray
            The tracks scaled and shifted.
        trk_labels : np.ndarray
            The track labels, if defined in the variable config.

        Raises
        ------
        ValueError
            If scale is found to be 0 or inf for any track variable.
        """
        var_arr_list = []
        # Get track mask
        #track_mask = get_track_mask(trks)
        track_mask = get_track_mask(trks)
        # Iterate over variables and scale/shift it
        for var in var_list:
            x = trks[var]
            
        # Stack the results for new dataset
            shift = np.float32(scale_dict[var]["shift"])
            scale = np.float32(scale_dict[var]["scale"])
            if scale == 0 or np.isinf(scale):
                raise ValueError(f"Scale parameter for track var {var} is {scale}.")
            x = np.where(
                track_mask,
                x - shift,
                x,
            )
            x = np.where(
                track_mask,
                x / scale,
                x,
            )
            var_arr_list.append(np.nan_to_num(x))
        scaled_trks = np.stack(var_arr_list, axis=-1)

        # Return the scaled and tracks and, if defined, the track labels
        return scaled_trks

    def save_remaining_dt(self, logger):
        stepsize = 500_000
        with File(f"{self.config.output}/{self.config.one_file_name}", "r") as f:
            fulllen = len(f[self.config.input_tracks_name])
            stepsize = min(fulllen, stepsize)
            n_steps = fulllen // stepsize +1
            with File(self.out_file, "a") as o:
                for step in range(n_steps):
                    logger.info(f"Appending remaining dataset... step {step + 1} from {n_steps}")
                    if step == 0:
                        if self.config.input_jet_name in o.keys():
                            del o[f"/{self.config.input_jet_name}"]
                        if self.config.input_truth_name in o.keys():
                            del o[f"/{self.config.input_truth_name}"]
                        if "edge_features" in o.keys():
                            del o["/edge_features"]
                        jet_data = f[f"/{self.config.input_jet_name}"][:stepsize]
                        n_entries = len(jet_data)
                        logger.info(f"Appending {n_entries} entries to dataset...")
                        o.create_dataset(data=jet_data, name=self.config.input_jet_name, chunks=True, maxshape=(None,))
                        truth_data = f[f"/{self.config.input_truth_name}"][:stepsize]
                        o.create_dataset(data=truth_data, name = self.config.input_truth_name, chunks=True, maxshape=(None, truth_data.shape[0],))
                        edge_features = f[f"/{self.config.input_tracks_name}"][:stepsize][self.global_conf.edge_features]
                        o.create_dataset(data=edge_features, name="edge_features", chunks=True, maxshape=(None, edge_features.shape[0],))
                    else:
                        n_entries = len(f[f"/{self.config.input_jet_name}"][step*stepsize:(step+1)*stepsize])
                        logger.info(f"Appending {n_entries} entries to dataset...")
                        o[self.config.input_jet_name].resize((o[self.config.input_jet_name].shape[0] + n_entries), axis=0)
                        o[self.config.input_jet_name][-n_entries:] = f[f"/{self.config.input_jet_name}"][step*stepsize:(step+1)*stepsize]
                        o[self.config.input_truth_name].resize((o[self.config.input_truth_name].shape[0] + n_entries), axis=0)
                        o[self.config.input_truth_name][-n_entries:] = f[f"/{self.config.input_truth_name}"][step*stepsize:(step+1)*stepsize]
                        o["edge_features"].resize((o["edge_features"].shape[0] + n_entries), axis=0)
                        o["edge_features"][-n_entries:] = f[f"/{self.config.input_tracks_name}"][step*stepsize:(step+1)*stepsize][self.global_conf.edge_features]
        logger.info("Appending done.")
