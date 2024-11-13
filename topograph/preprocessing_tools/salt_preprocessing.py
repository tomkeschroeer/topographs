import os
from glob import glob

import numpy as np
import numpy.ma as ma
from h5py import File
import pandas as pd 

from topograph.modules.tools import DatasetCreater, GlobalConfig, get_logger



class Salt_preprocess:
    def __init__(self, config, dataset_types):
        self.config = config
        self.dataset_types = dataset_types
        self.create_file = True

    def Run(self):
        self.logger = get_logger()
        self.logger.info("Starting Salt_preprocess")
        self.logger.info("Creating datasets") 
        jet_types = self.config.jet_types
        njets = int(
            self.dataset_types[""]["njets"]
            + self.dataset_types["_val"]["njets"]
            + self.dataset_types["_test"]["njets"]
        )
        stepsize = min(50_000, int(njets / 2))
        # stepsize = 1000
        n_steps = njets // stepsize if njets % stepsize == 0 else njets // stepsize + 1
        input_file = (
            f"{self.config.output}/{self.config.preprocessing_file_name}".replace(
                "//", "/"
            ).replace(".h5","") + ".h5"
        )

        output_file = input_file.replace(".h5", "") + "_salt.h5"
        keys = [key for key in File(input_file).keys()]

        with File(output_file, "w") as h5fw:
            with File(input_file, "r") as f:
                data_jets_keys = []
                data_tracks_keys = []
                data_neutrals_keys = []
                newkey_track = True
                newkey_jet = True
                newkey_neutral = True
                for key in keys:
                    final_dset = self.check_mergeing_keys(key)
                    data = f[key][:1]
                    if final_dset == "jets":
                        self.logger.info(f"Creating {final_dset} dataset")
                        data_jets_keys.append(key)
                        if data.dtype.names is None:
                            dtype_j = np.dtype([(key, data.dtype.type)])                        
                        else:
                            dtype_j = data.dtype
                        if dtype_j.names[0] == "pt":
                            dtype_j.names=(f"pt_{key}",)
                        if newkey_jet:
                            newkey_jet = False
                            dtype_all_j = dtype_j
                        else:
                            dtype_all_j = np.dtype(dtype_all_j.descr + dtype_j.descr)

                    elif final_dset == "tracks":
                        self.logger.info(f"Creating {final_dset} dataset")
                        if key == "track_extra":
                            continue
                        data_tracks_keys.append(key)
                        if data.dtype.names is None:
                            dtype_tr = np.dtype([(key, data.dtype.type)])
                        else:
                            dtype_tr = data.dtype
                            if "valid" in dtype_tr.names:
                                dtype_tr_ar = np.array(dtype_tr.descr)[np.array(dtype_tr.names) != "valid"]
                                dtype_tr = np.dtype([tuple(i) for i in dtype_tr_ar])

                        if newkey_track:
                            newkey_track = False
                            dtype_all_tr = dtype_tr
                        else:
                            dtype_all_tr = np.dtype(dtype_all_tr.descr + dtype_tr.descr)
                    elif final_dset == "neutrals":
                        self.logger.info(f"Creating {final_dset} dataset")
                        data_neutrals_keys.append(key)
                        if data.dtype.names is None:
                            dtype_n = np.dtype([(key, data.dtype.type)])
                        else:
                            dtype_n = data.dtype
                        if newkey_neutral:
                            newkey_neutral = False
                            dtype_all_n = dtype_n
                        else:
                            dtype_all_n = np.dtype(dtype_all_n.descr + dtype_n.descr)
                for step in range(n_steps):
                    for j, key_j in enumerate(data_jets_keys):
                        data_j = pd.DataFrame(f[key_j][step * stepsize : (step + 1) * stepsize]) #, dtype=np.void)
                        if j == 0:
                            data_merged_j = data_j
                        else:
                            data_merged_j = pd.concat((data_merged_j, data_j), axis=1)
                    data_merged_j = data_merged_j.to_numpy()
                    data_merged_j = np.array([tuple(c) for c in data_merged_j], dtype=dtype_all_j)

                    tracks_shape = f[data_tracks_keys[0]][step * stepsize : (step + 1) * stepsize].shape
                    for j, key_tr in enumerate(data_tracks_keys):
                        if key_tr == "valid": continue
                        temp_data = f[key_tr][step * stepsize : (step + 1) * stepsize].flatten()
                        data_tr = pd.DataFrame(temp_data).drop("valid", axis=1, errors="ignore") #, dtype=np.void)
                        if temp_data.dtype.names is None:
                            data_tr = data_tr.rename(columns={0: key_tr})
                        if j == 0:
                            data_merged_tr = data_tr
                        else:
                            data_merged_tr = pd.concat((data_merged_tr, data_tr), axis=1)
                    data_merged_tr = data_merged_tr.to_numpy().reshape((tracks_shape[0]*tracks_shape[1], len(data_merged_tr.columns)))
                    data_merged_tr[np.isnan(data_merged_tr)] = 0.
                    data_merged_tr = np.array([tuple(c) for c in data_merged_tr], dtype=dtype_all_tr).reshape(tracks_shape)
                    for j, key_n in enumerate(data_neutrals_keys):
                        data_n = f[key_n][step * stepsize : (step + 1) * stepsize]
                        if j == 0:
                            data_merged_n = data_n
                        else:
                            data_merged_n = np.concatenate((data_merged_n, data_n), axis=1)
                    shape_j = data_merged_j.shape
                    shape_tr = data_merged_tr.shape
                    shape_n = data_merged_n.shape
                    maxshape_j = (None,) if len(shape_j) == 1 else (None, *shape_j[1:])
                    maxshape_tr = (None,) if len(shape_tr) == 1 else (None, *shape_tr[1:])
                    maxshape_n = (None,) if len(shape_n) == 1 else (None, *shape_n[1:])
                    if step == 0:
                        h5fw.create_dataset(self.config.input_jet_name, data=data_merged_j, chunks=True, maxshape=maxshape_j)
                        h5fw.create_dataset(self.config.input_tracks_name, data=data_merged_tr, chunks=True, maxshape=maxshape_tr)
                        h5fw.create_dataset(self.config.input_neutral_name, data=data_merged_n, chunks=True, maxshape=maxshape_n)
                    else:
                        data_jets = data_merged_j
                        data_tracks = data_merged_tr
                        data_neutrals = data_merged_n
                        h5fw[self.config.input_jet_name].resize((h5fw[self.config.input_jet_name].shape[0] + data_jets.shape[0]), axis=0)
                        h5fw[self.config.input_tracks_name].resize((h5fw[self.config.input_tracks_name].shape[0] + data_tracks.shape[0]), axis=0)
                        h5fw[self.config.input_neutral_name].resize((h5fw[self.config.input_neutral_name].shape[0] + data_neutrals.shape[0]), axis=0)
                        h5fw[self.config.input_jet_name][-data_jets.shape[0]:] = data_jets
                        h5fw[self.config.input_tracks_name][-data_tracks.shape[0]:] = data_tracks
                        h5fw[self.config.input_neutral_name][-data_neutrals.shape[0]:] = data_neutrals
                # data_merged = np.array([tuple(c) for c in data_merged], dtype=dtype_all)
                # data_merged = np.array(data_merged, dtype=dtype_all)

            # data_j = np.array([[np.array(c) for c in co] for co in data_j])

        self.logger.info("Finished Salt_preprocess")

    def check_mergeing_keys(self, dataset_name):
        jets = [
            "HadronConeExclTruthLabelID",
            "flavour_label",
            "HadronConeExclTruthLabelPt",
            "HadronConeExclTruthLabelLxy",
            "n_tracks",
            "Y_vertex_features_b",
            "Y_vertex_features_light",
            "Y_vertex_features_c",
            "HadronTruthLabel",
            "jet_inputs",
            "jets",
            "mask_jets",
        ]
        tracks = [
            "X_train_tracks",
            "Y_edge_light",
            "Y_edge_b",
            "Y_edge_c",
            "edge_origin",
            "track_extra",
            "track_extra_truth",
            "tracks_loose",
            "Y_edge_weight_b",
            "Y_edge_weight_c",
            "Y_edge_weight_light",
            "mask_tracks",
        ]
        neutrals = [
            "neutrals"
        ]

        if dataset_name in jets:
            return "jets"
        elif dataset_name in tracks:
            return "tracks"
        elif dataset_name in neutrals:
            return "neutrals"
        return "none"