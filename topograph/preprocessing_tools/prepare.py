import os
from glob import glob

import numpy as np
import numpy.ma as ma
from h5py import File

from topograph.modules.tools import DatasetCreater, GlobalConfig, get_logger


class Prepare:
    def __init__(self, config, dataset_types, jet_type):
        self.config = config
        self.dataset_types = dataset_types
        self.jet_type = jet_type

    def Run(self):
        logger = get_logger()
        global_conf = GlobalConfig()
        output_file = (
            f"{self.config.output}/{self.config.preprocessing_file_name}".replace(
                "//", "/"
            ).replace(".h5","")
        )
        jet_types = self.config.jet_types
        njets = int((
            self.dataset_types[""]["njets"]
            + self.dataset_types["_val"]["njets"]
            + self.dataset_types["_test"]["njets"]
        )/len(jet_types))
        stepsize = min(50_000, int(njets / 2))

        # stepsize = 200
        # input_file = f"{output_dir}/{self.config.one_file_name}.h5".replace(".h5.h5",".h5").replace("//","/")
        updated_filelist = glob(self.config.input)
        os.makedirs(self.config.output, exist_ok=True)
        create_file = True
        continue_loading = True
        output_file_jettype = f"{output_file}_{self.jet_type}.h5" #if len(jet_types) > 1 else f"{output_file}.h5"
        for ninput, input_file in enumerate(updated_filelist):
            if continue_loading == False:
                break
            with File(input_file, "r") as f:
                njets_file = len(f["/jets"][:])
            n_steps = njets_file // stepsize + 1
            for step in range(n_steps):
                logger.info(f"Process file {input_file}, step {step+1}/{n_steps} for jet type {self.jet_type}.")
                datasets = DatasetCreater(
                    config=self.config,
                    input_file=input_file,
                    step=step,
                    stepsize=stepsize,
                    replace_invalid=True,
                    jet_type=self.jet_type,
                )
                vertex_feat = datasets.get_vertex_feat_y()
                edge_y = datasets.get_edge_y()
                edge_origin = datasets.get_edge_origin()
                track_inputs = datasets.get_track_input()
                tracks_extra = datasets.get_extra_track()
                tracks_extra_true = datasets.get_extra_track_truth()
                unscaled_pt = datasets.get_unscaled_pt()
                jet_pt = datasets.get_jet_pt()
                if create_file:
                    with File(output_file_jettype, "w") as train_file:
                        n_tracks = track_inputs.shape[1]
                        if vertex_feat is not None:
                            train_file.create_dataset(f"{self.config.vertex_feat_name}_{self.jet_type}", data = vertex_feat, chunks=True, maxshape=(None,)) #len(global_conf.vertex_features))) # dtype=datasets.vertex_feat_dtypes
                        # train_file.create_dataset(self.config.edge_feat_name, data = datasets.get_edge_feat_y(), chunks=True, maxshape=(None,40,len(global_conf.edge_features)))
                        train_file.create_dataset(f"{self.config.edge_name}_{self.jet_type}", data = edge_y, chunks=True, maxshape=(None,n_tracks,))
                        train_file.create_dataset("edge_origin", data = edge_origin, chunks=True, maxshape=(None,n_tracks,))
                        train_file.create_dataset(f"{self.config.tracks_name}", data = np.array(track_inputs , dtype=datasets.reco_dtypes), chunks=True, maxshape=(None,n_tracks))
                        train_file.create_dataset("track_extra", data = tracks_extra, chunks=True, maxshape=(None,n_tracks,))
                        train_file.create_dataset("track_extra_truth", data = tracks_extra_true, chunks=True, maxshape=(None,n_tracks,))
                        if unscaled_pt is not None:
                            train_file.create_dataset("unscaled_pt", data = unscaled_pt, chunks=True, maxshape=(None,))
                        if jet_pt is not None:
                            train_file.create_dataset("jet_pt", data = jet_pt, chunks=True, maxshape=(None,))
                        create_file = False
                else:
                    njets_step = datasets.get_n_valid_jets()
                    logger.info(f"loading {njets_step} valid jets")
                    if njets_step > 0:
                        with File(output_file_jettype, "a") as train_file:
                            if vertex_feat is not None:
                                train_file[f"{self.config.vertex_feat_name}_{self.jet_type}"].resize(
                                    (
                                        train_file[f"{self.config.vertex_feat_name}_{self.jet_type}"].shape[0]
                                        + njets_step
                                    ),
                                    axis=0,
                                )
                                train_file[f"{self.config.vertex_feat_name}_{self.jet_type}"][
                                    -njets_step:
                                ] = vertex_feat
                            # train_file[self.config.edge_feat_name].resize((train_file[self.config.edge_feat_name].shape[0] + njets_step), axis=0)
                            # train_file[self.config.edge_feat_name][-njets_step:] = datasets.get_edge_feat_y()
                            train_file[f"{self.config.edge_name}_{self.jet_type}"].resize(
                                (
                                    train_file[f"{self.config.edge_name}_{self.jet_type}"].shape[0]
                                    + njets_step
                                ),
                                axis=0,
                            )
                            train_file[f"{self.config.edge_name}_{self.jet_type}"][
                                -njets_step:
                            ] = edge_y
                            train_file["edge_origin"].resize(
                                (train_file["edge_origin"].shape[0] + njets_step),
                                axis=0,
                            )
                            train_file["edge_origin"][
                                -njets_step:
                            ] = edge_origin
                            train_file[self.config.tracks_name].resize(
                                (
                                    train_file[self.config.tracks_name].shape[0]
                                    + njets_step
                                ),
                                axis=0,
                            )
                            train_file[self.config.tracks_name][
                                -njets_step:
                            ] = datasets.get_track_input()
                            train_file["track_extra"].resize(
                                (train_file["track_extra"].shape[0] + njets_step),
                                axis=0,
                            )
                            train_file["track_extra"][
                                -njets_step:
                            ] = tracks_extra
                            train_file["track_extra_truth"].resize(
                                (train_file["track_extra_truth"].shape[0] + njets_step),
                                axis=0,
                            )
                            train_file["track_extra_truth"][
                                -njets_step:
                            ] = tracks_extra_true
                            if unscaled_pt is not None:
                                train_file["unscaled_pt"].resize(
                                    (train_file["unscaled_pt"].shape[0] + njets_step),
                                    axis=0,
                                )
                                train_file["unscaled_pt"][
                                    -njets_step:
                                ] = unscaled_pt
                            if jet_pt is not None:
                                train_file["jet_pt"].resize(
                                    (train_file["jet_pt"].shape[0] + njets_step), axis=0
                                )
                                train_file["jet_pt"][-njets_step:] = jet_pt
                            logger.info(
                                "loaded "
                                + str(len(train_file[f"{self.config.tracks_name}"]))
                                + " jets in total"
                            )
                            if len(train_file[f"{self.config.tracks_name}"]) >= int(
                                njets
                            ):
                                continue_loading = False
                if continue_loading == False:
                    logger.info(f"Loaded {njets} jets, stop loading.")
                    break
