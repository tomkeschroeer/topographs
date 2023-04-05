import chunk
import numpy as np
from h5py import File
import json

from topograph.modules import (
    GlobalConfig,
    get_logger,
    get_mask
)

class Apply_Scaler:
    def __init__(self, config, dataset_types):
        self.config = config
        self.dataset_types = dataset_types 
        self.scale_dict_path_basic = f"{self.config.output}/{self.config.scale_dict}".replace(".json", "")
        self.global_conf = GlobalConfig()
        self.tracks_name = self.config.tracks_name
        self.vert_prop_name = self.config.vertex_feat_name
        self.var_list = {
            self.tracks_name: self.global_conf.track_inputs,
            self.vert_prop_name: self.global_conf.vertex_features
        }
        self.file_names = {
            self.config.training_file_name: self.config.njets,
            self.config.validation_file_name: self.config.njets_val,
            self.config.testing_file_name: self.config.njets_test
        }
        self.out_file = None

    def Run(self):
        logger = get_logger()

        chunk_size = 1e5
        self.scale_dict_path = f"{self.scale_dict_path_basic}.json"
        input_file=f"{self.config.output}/{self.config.preprocessing_file_name}".replace(".h5","") + ".h5"
        logger.info(f"Scale/Shift jets from {input_file}")
        logger.info(f"Using scales in {self.scale_dict_path}")
        file_length = len(File(input_file, "r")[f"/{self.tracks_name}"][self.var_list[self.tracks_name][0]][:])
        n_chunks = int(np.ceil(file_length / chunk_size))

        # Check if tracks are used
        scale_dict = {}
        # Get the scale dict for tracks
        with open(self.scale_dict_path, "r") as infile:
            scale_dict = json.load(infile)
            # scale_dict[self.tracks_name] = full_scale_dict[self.tracks_name]
            # scale_dict[self.vert_prop_name] = full_scale_dict[self.vert_prop_name]

        logger.info("Applying scaling and shifting.")
        
        for file_name, njets_per_file in self.file_names.items():
            self.out_file = f"{self.config.output}/{file_name}".replace(".h5.h5", ".h5")
            logger.info(f"Save scaled inputs in file {self.out_file}")
            with File(self.out_file, "w") as h5file:
                for keyname in [self.tracks_name, self.vert_prop_name]:
                    scale_generator = self.scale_generator(
                        input_file=input_file,
                        nJets = njets_per_file,
                        keyname=keyname,
                        scale_dict=scale_dict[keyname],
                        chunk_size=chunk_size,
                    )
                    # Set up chunk counter and start looping
                    chunk_counter = 0
                    for chunk_counter in range(n_chunks):
                        logger.info(
                            f"Applying scales for chunk {chunk_counter+1} of {n_chunks}."
                        )
                        try:
                            data = next(scale_generator)

                            if chunk_counter == 0:
                                h5file.create_dataset(
                                    keyname,
                                    data=data[0],
                                #  compression="lzf",
                                    chunks=((100,) + data[0].shape[1:]),
                                    maxshape=(
                                        None,
                                        *(data[0].shape[1:]),
                                    ),
                                )

                            else:
                                h5file[keyname].resize(
                                    (h5file[keyname].shape[0] + data[0].shape[0]),
                                    axis=0,
                                )
                                
                                h5file[keyname][-data[0].shape[0] :] = data[0]

                        except StopIteration:
                            break

                        chunk_counter += 1

            self.save_remaining_dt(logger, input_file, n_entries_total=njets_per_file)

    def scale_generator(
        self,
        input_file: str,
        nJets: int,
        keyname: str,
        scale_dict: dict = None,
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
                scaled_data = []
                # Load tracks
                data = np.asarray(
                    f[keyname][
                        index_tuple[0] : index_tuple[1]
                    ]
                )

                # Apply scaling to the tracks
                data = self.apply_scaling(
                    data=data,
                    var_list=self.var_list[keyname],
                    scale_dict=scale_dict
                )
                scaled_data.append(data)
                yield scaled_data

    def apply_scaling(
        self,
        data: np.ndarray,
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
        mask = get_mask(data)
        # Iterate over variables and scale/shift it
        for var in var_list:
            x = data[var]
            
        # Stack the results for new dataset
            shift = np.float32(scale_dict[var]["shift"])
            scale = np.float32(scale_dict[var]["scale"])
            if scale == 0 or np.isinf(scale):
                raise ValueError(f"Scale parameter for track var {var} is {scale}.")
            x = np.where(
                mask,
                x - shift,
                x,
            )
            x = np.where(
                mask,
                x / scale,
                x,
            )
            var_arr_list.append(np.nan_to_num(x))
        scaled_data = np.stack(var_arr_list, axis=-1)

        # Return the scaled and tracks and, if defined, the track labels
        return scaled_data

    def save_remaining_dt(self, logger, input_file, n_entries_total):
        chunk_size = 1000
        start_ind = 0
        tupled_indices = []

        while start_ind < n_entries_total:
            end_ind = int(start_ind + chunk_size)
            end_ind = min(end_ind, n_entries_total)
            tupled_indices.append((start_ind, end_ind))
            start_ind = end_ind
            
        with File(input_file, "r") as f:
            for step, indices in enumerate(tupled_indices):
                with File(self.out_file, "a") as o:
                    logger.info(f"Appending remaining dataset... step {step + 1} from {len(tupled_indices)}")
                    if step == 0:
                        if self.config.edge_name in o.keys():
                            del o[self.config.edge_name]
                        edges = f[self.config.edge_name][indices[0] : indices[1]]
                        o.create_dataset(data=edges, name=self.config.edge_name, chunks=True, maxshape=(None, edges.shape[1]))
                    else:
                        n_entries = indices[1] - indices[0]
                        o[self.config.edge_name].resize((o[self.config.edge_name].shape[0] + n_entries), axis=0)
                        o[self.config.edge_name][-n_entries:] = f[self.config.edge_name][indices[0] : indices[1]]
        logger.info("Appending done.")
