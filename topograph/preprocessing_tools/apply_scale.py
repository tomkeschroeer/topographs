import chunk
import json

import numpy as np
from h5py import File

from topograph.modules import GlobalConfig, get_logger, get_mask, scary_shuffle


class Apply_Scaler:
    def __init__(self, config, dataset_types):
        self.config = config
        self.dataset_types = dataset_types
        self.scale_dict_path_basic = (
            f"{self.config.output}/{self.config.scale_dict}".replace(".json", "")
        )
        self.global_conf = GlobalConfig()
        self.tracks_name = self.config.tracks_name
        self.vert_prop_name = self.config.vertex_feat_name
        self.var_list = {f"{self.vert_prop_name}_{jet_type}": self.global_conf.vertex_features for jet_type in self.config.jet_types}
        self.var_list[self.tracks_name] = self.global_conf.track_inputs
        self.file_names = {
            self.config.training_file_name: self.config.njets,
            self.config.validation_file_name: self.config.njets_val,
            self.config.testing_file_name: self.config.njets_test,
        }
        self.njets = self.config.njets + self.config.njets_val + self.config.njets_test
        self.out_file = None
        self.jet_types = self.config.jet_types

    def Run(self):
        logger = get_logger()

        chunk_size = 1e5
        self.scale_dict_path = f"{self.scale_dict_path_basic}.json"
        input_file = (
            f"{self.config.output}/{self.config.preprocessing_file_name}".replace(
                ".h5", ""
            )
            + ".h5"
        )
        logger.info(f"Scale/Shift jets from {input_file}")
        logger.info(f"Using scales in {self.scale_dict_path}")
        file_length = len(
            File(input_file, "r")[f"/{self.tracks_name}"][
                self.var_list[self.tracks_name][0]
            ][:]
        )
        n_chunks = int(np.ceil(file_length / chunk_size))

        # Check if tracks are used
        scale_dict = {}
        # Get the scale dict for tracks
        with open(self.scale_dict_path, "r") as infile:
            scale_dict = json.load(infile)
            # scale_dict[self.tracks_name] = full_scale_dict[self.tracks_name]
            # scale_dict[self.vert_prop_name] = full_scale_dict[self.vert_prop_name]

        logger.info("Applying scaling and shifting.")
        dict_names = scale_dict.keys()
        # for file_name, njets_per_file in self.file_names.items():
        self.out_file = input_file.replace(".h5", "_scaled.h5")
        logger.info(f"Save scaled inputs in file {self.out_file}")
        with File(self.out_file, "w") as h5file:
            for keyname in dict_names:
                scale_generator = self.scale_generator(
                    input_file=input_file,
                    nJets=self.njets,
                    keyname=keyname,
                    scale_dict=scale_dict[keyname],
                    chunk_size=chunk_size,
                )
                # Set up chunk counter and start looping
                chunk_counter = 0
                for chunk_counter in range(n_chunks):
                    logger.info(
                        f"Applying scales for chunk {chunk_counter+1} of"
                        f" {n_chunks}."
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

        self.save_remaining_dt(logger, input_file, n_entries_total=self.njets)
        chunk_size = np.min((chunk_size, int(self.config.njets_test/len(self.jet_types)))).astype(int)
        print(chunk_size)
        starting_point = 0
        n_total = len(File(self.out_file,"r")[f"/{self.config.tracks_name}"])
        ind_array = np.linspace(0, n_total-1, n_total).astype(int)
        scary_shuffle(ind_array)
        n_jets_per_jettype = {jettype: 0 for jettype in self.jet_types}
        total_njets_per_jettype = int(self.njets/len(self.jet_types))
        starting_point = 0
        for file_name, njets in self.file_names.items():
            input_file = (
                f"{self.config.output}/{file_name}".replace(
                    ".h5", ""
                )
                + ".h5"
            )
            with File(input_file, "w") as f:
                nsteps = int(njets//chunk_size)
                r = njets % chunk_size
                if r != 0:
                    nsteps += 1
                # starting_point = 0
                inds = np.array([[(step*chunk_size + starting_point), ((step+1)*chunk_size + starting_point)] for step in range(nsteps)], dtype=int)
                if inds[-1,1] > njets + starting_point: inds[-1,1] = njets + starting_point
                starting_point = inds[-1,1]
                create_file = True
                with File(self.out_file, "r") as o:
                    keys=o.keys()
                    for ind in inds:
                        for jet_type in self.jet_types:
                            jet_type_in_jet = o["jet_type"][ind[0]:ind[1]]
                            # print(self.jet_types.index(jet_type))
                            mask = (jet_type_in_jet == self.jet_types.index(jet_type))
                            njets_step = sum(mask)
                            n_jets_per_jettype[jet_type] = n_jets_per_jettype[jet_type] + njets_step
                            # if n_jets_per_jettype[jet_type] >= total_njets_per_jettype:
                            if create_file:
                                for key in keys:
                                    data_chunk = o[key][ind[0]:ind[1]][mask]
                                    shape = data_chunk.shape
                                    shape = (None,) if len(shape) == 1 else (None, *shape[1:])
                                    f.create_dataset(key, data=data_chunk, chunks=True, maxshape=shape)
                                create_file = False
                            else:
                                # print(o[key][ind[0]:ind[1]])
                                if njets_step >0:
                                    for key in keys: #ind[1]-ind[0]
                                        f[key].resize((f[key].shape[0] + njets_step), axis=0)
                                        f[key][-njets_step:] = o[key][ind[0]:ind[1]][mask]
                            # if n_jets_per_jettype[jet_type] >=

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
                data = np.asarray(f[keyname][index_tuple[0] : index_tuple[1]])

                # Apply scaling to the tracks
                data = self.apply_scaling(
                    data=data, var_list=self.var_list[keyname], scale_dict=scale_dict
                )
                scaled_data.append(data)
                yield scaled_data

    def apply_scaling(self, data: np.ndarray, var_list: dict, scale_dict: dict):
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
        # track_mask = get_track_mask(trks)
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
        chunk_size = min(50_000, n_entries_total)
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
                    logger.info(
                        f"Appending remaining dataset... step {step + 1} from"
                        f" {len(tupled_indices)}"
                    )
                    if step == 0:
                        if self.config.edge_name in o.keys():
                            del o[self.config.edge_name]
                        if "edge_origin" in o.keys():
                            del o["edge_origin"]
                        if "track_extra" in o.keys():
                            del o["track_extra"]
                        if "track_extra_truth" in o.keys():
                            del o["track_extra_truth"]
                        if "track_extra" in o.keys():
                            del o["track_extra"]
                        if "unscaled_pt" in o.keys():
                            del o["unscaled_pt"]
                        if "jet_pt" in o.keys():
                            del o["jet_pt"]
                        if "HadronTruthLabel" in o.keys():
                            del o["HadronTruthLabel"]
                        if "jet_type" in o.keys():
                            del o["jet_type"]
                        for jet_type in self.config.jet_types:
                            edges = f[f"{self.config.edge_name}_{jet_type}"][indices[0] : indices[1]]
                            o.create_dataset(
                                data=edges,
                                name=f"{self.config.edge_name}_{jet_type}",
                                chunks=True,
                                maxshape=(None, edges.shape[1]),
                            )
                        edge_origin = f["edge_origin"][indices[0] : indices[1]]
                        track_extra = f["track_extra"][indices[0] : indices[1]]
                        track_extra_truth = f["track_extra_truth"][
                            indices[0] : indices[1]
                        ]
                        # unscaled_pt = f["unscaled_pt"][indices[0] : indices[1]]
                        HadronTruthLabel = f["HadronTruthLabel"][indices[0] : indices[1]]
                        jet_type = f["jet_type"][indices[0] : indices[1]]
                        o.create_dataset(
                            data=edge_origin,
                            name="edge_origin",
                            chunks=True,
                            maxshape=(None, edge_origin.shape[1]),
                        )
                        o.create_dataset(
                            data=track_extra,
                            name="track_extra",
                            chunks=True,
                            maxshape=(None, track_extra.shape[1]),
                        )
                        o.create_dataset(
                            data=track_extra_truth,
                            name="track_extra_truth",
                            chunks=True,
                            maxshape=(None, track_extra.shape[1]),
                        )
                        # o.create_dataset(
                        #     data=unscaled_pt,
                        #     name="unscaled_pt",
                        #     chunks=True,
                        #     maxshape=(None,),
                        # )
                        o.create_dataset(
                            data=HadronTruthLabel,
                            name="HadronTruthLabel",
                            chunks=True,
                            maxshape=(None,),
                        )
                        o.create_dataset(
                            data=jet_type,
                            name="jet_type",
                            chunks=True,
                            maxshape=(None,),
                        )
                    else:
                        n_entries = indices[1] - indices[0]
                        for jet_type in self.config.jet_types:
                            o[f"{self.config.edge_name}_{jet_type}"].resize(
                                (o[f"{self.config.edge_name}_{jet_type}"].shape[0] + n_entries), axis=0
                            )
                            o[f"{self.config.edge_name}_{jet_type}"][-n_entries:] = f[
                                f"{self.config.edge_name}_{jet_type}"
                            ][indices[0] : indices[1]]
                        o["edge_origin"].resize(
                            (o["edge_origin"].shape[0] + n_entries), axis=0
                        )
                        o["edge_origin"][-n_entries:] = f["edge_origin"][
                            indices[0] : indices[1]
                        ]
                        o["track_extra"].resize(
                            (o["track_extra"].shape[0] + n_entries), axis=0
                        )
                        o["track_extra"][-n_entries:] = f["track_extra"][
                            indices[0] : indices[1]
                        ]
                        o["track_extra_truth"].resize(
                            (o["track_extra_truth"].shape[0] + n_entries), axis=0
                        )
                        o["track_extra_truth"][-n_entries:] = f["track_extra_truth"][
                            indices[0] : indices[1]
                        ]
                        # o["unscaled_pt"].resize(
                        #     (o["unscaled_pt"].shape[0] + n_entries), axis=0
                        # )
                        # o["unscaled_pt"][-n_entries:] = f["unscaled_pt"][
                        #     indices[0] : indices[1]
                        # ]
                        o["HadronTruthLabel"].resize((o["HadronTruthLabel"].shape[0] + n_entries), axis=0)
                        o["HadronTruthLabel"][-n_entries:] = f["HadronTruthLabel"][indices[0] : indices[1]]
                        o["jet_type"].resize((o["jet_type"].shape[0] + n_entries), axis=0)
                        o["jet_type"][-n_entries:] = f["jet_type"][indices[0] : indices[1]]

        logger.info("Appending done.")
