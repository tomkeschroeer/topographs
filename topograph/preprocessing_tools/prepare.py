import os
from glob import glob

import numpy as np
import numpy.ma as ma
from h5py import File

from topograph.modules.tools import DatasetCreater, GlobalConfig, get_logger


class Prepare:
    def __init__(self, config, dataset_types):
        self.config = config
        self.dataset_types = dataset_types
        self.create_file = True

    def Run(self):
        logger = get_logger()
        logger.info("start preparing")
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
        stepsize = min(50_000,int(njets / 2))

        # input_file = f"{output_dir}/{self.config.one_file_name}.h5".replace(".h5.h5",".h5").replace("//","/")
        updated_filelist = glob(self.config.input)
        os.makedirs(self.config.output, exist_ok=True)
        continue_loading = np.array([True]*len(jet_types)).astype(bool)
        load_jet_types = np.array(jet_types)[continue_loading]
        n_jets_loaded = np.array([0]*len(jet_types))
        # {jet_type: True for jet_type in jet_types}
        # output_file_jettype = f"{output_file}.h5" #f"{output_file}_{self.jet_type}.h5" #if len(jet_types) > 1 else f"{output_file}.h5"
        for ninput, input_file in enumerate(updated_filelist):
            if not np.any(continue_loading):
                break
            with File(input_file, "r") as f:
                njets_file = len(f["/jets"][:])
            n_steps = njets_file // stepsize + 1
            for step in range(n_steps):
                logger.info(f"Process file {input_file}, step {step+1}/{n_steps}.")# jet type {self.jet_type}.")
                datasets = DatasetCreater(
                    config=self.config,
                    input_file=input_file,
                    step=step,
                    stepsize=stepsize,
                    replace_invalid=True,
                    jet_types=jet_types,
                    small_net=self.config.small_net,
                    load_jet_types=load_jet_types
                )
                vertex_feat = datasets.get_vertex_feat_y()
                edge_y = datasets.get_edge_y()
                edge_origin = datasets.get_edge_origin()
                track_inputs = datasets.get_track_input()
                jet_types_to_save = datasets.get_jet_types_to_save()
                # tracks_extra = datasets.get_extra_track()
                tracks_extra_true = datasets.get_extra_track_truth()
                # unscaled_pt = datasets.get_unscaled_pt()
                jet_pt = datasets.get_jet_pt()
                # overall_inds = datasets.get_overall_inds()
                HadrTruth = datasets.get_HadrLabel()
                flavour_label = datasets.get_flavour_label()
                print(np.unique(flavour_label))
                jet_inputs = datasets.get_jet_input()
                if self.config.solo_topo:
                    self.preprocess_for_solo_topo(
                        jet_types=jet_types,
                        datasets=datasets,
                        edge_y=edge_y,
                        edge_origin=edge_origin,
                        HadrTruth=HadrTruth,
                        jet_pt=jet_pt,
                        jet_types_to_save=jet_types_to_save,
                        load_jet_types=load_jet_types,
                        logger=logger,
                        n_jets_loaded=n_jets_loaded,
                        njets=njets,
                        output_file=output_file,
                        track_inputs=track_inputs,
                        # tracks_extra=tracks_extra,
                        tracks_extra_true=tracks_extra_true,
                        vertex_feat=vertex_feat
                    )
                else:
                    self.preprocess_for_salt_topo(
                        jet_types=jet_types,
                        datasets=datasets,
                        edge_y=edge_y,
                        edge_origin=edge_origin,
                        HadrTruth=HadrTruth,
                        jet_pt=jet_pt,
                        jet_types_to_save=jet_types_to_save,
                        load_jet_types=load_jet_types,
                        logger=logger,
                        n_jets_loaded=n_jets_loaded,
                        njets=njets,
                        output_file=output_file,
                        track_inputs=track_inputs,
                        # tracks_extra=tracks_extra,
                        tracks_extra_true=tracks_extra_true,
                        vertex_feat=vertex_feat,
                        flavour_label=flavour_label,
                        jet_inputs=jet_inputs
                    )
                continue_loading = np.array(n_jets_loaded < int(njets))
                n_jets_loaded += [datasets.get_n_valid_jets_type(jet_type) for jet_type in jet_types]
                load_jet_types=np.array(jet_types)[continue_loading]
                logger.info(
                    "loaded "
                    + str(sum(n_jets_loaded))
                    + " jets in total"
                )
                string = f"that includes"
                for n_jets_jettype, jettype in zip(n_jets_loaded, jet_types):
                    string += f"{n_jets_jettype} {jettype}-jets and "
                string = string[:-4]
                logger.info(string)
                if not np.any(continue_loading):
                    logger.info(f"Loaded {njets} jets per jet flavour, stop loading.")
                    break

                    
    def preprocess_for_solo_topo(
            self, 
            jet_types, 
            datasets,
            edge_y, 
            edge_origin, 
            HadrTruth, 
            jet_pt, 
            jet_types_to_save, 
            load_jet_types, 
            logger, 
            n_jets_loaded, 
            njets,
            output_file, 
            track_inputs, 
            # tracks_extra, 
            tracks_extra_true, 
            vertex_feat
        ):
        if self.create_file:
            for jet_type in jet_types:
                mask = jet_types_to_save == jet_types.index(jet_type)
                n_tracks = track_inputs.shape[1]
                # if vertex_feat is not None:
                # train_file.create_dataset(self.config.edge_feat_name, data = datasets.get_edge_feat_y(), chunks=True, maxshape=(None,40,len(global_conf.edge_features)))
                output_file_jettype = f"{output_file}_{jet_type}.h5"
                with File(output_file_jettype, "w") as train_file:
                    for snd_jet_type in jet_types:
                        train_file.create_dataset(f"{self.config.edge_name}_{snd_jet_type}", data = edge_y[snd_jet_type][mask], chunks=True, maxshape=(None,n_tracks,))
                        train_file.create_dataset(f"{self.config.vertex_feat_name}_{snd_jet_type}", data = vertex_feat[snd_jet_type][mask], chunks=True, maxshape=(None,)) #len(global_conf.vertex_features))) # dtype=datasets.vertex_feat_dtypes
                    train_file.create_dataset("edge_origin", data = edge_origin[mask], chunks=True, maxshape=(None,n_tracks,))
                    train_file.create_dataset(f"{self.config.tracks_name}", data = np.array(track_inputs , dtype=datasets.reco_dtypes)[mask], chunks=True, maxshape=(None,n_tracks))
                    train_file.create_dataset(f"jet_type", data = np.array(jet_types_to_save)[mask], chunks=True, maxshape=(None,))
                    # train_file.create_dataset("track_extra", data = tracks_extra[mask], chunks=True, maxshape=(None,n_tracks,))
                    train_file.create_dataset("track_extra_truth", data = tracks_extra_true[mask], chunks=True, maxshape=(None,n_tracks,))
                    train_file.create_dataset("HadronTruthLabel", data = HadrTruth[mask], chunks=True, maxshape=(None,))
                # if unscaled_pt is not None:
                #     train_file.create_dataset("unscaled_pt", data = unscaled_pt, chunks=True, maxshape=(None,))
                    if jet_pt is not None:
                        train_file.create_dataset("jet_pt", data = jet_pt[mask], chunks=True, maxshape=(None,))
            n_jets_loaded += [datasets.get_n_valid_jets_type(jet_type) for jet_type in jet_types]
            continue_loading = np.array(n_jets_loaded < int(njets))
            load_jet_types=np.array(jet_types)[continue_loading]
            logger.info(
                    "loaded "
                    + str(sum(n_jets_loaded))
                    + " jets in total"
                )
            logger.info(f"that includes")
            for n_jets_jettype, jettype in zip(n_jets_loaded, jet_types):
                logger.info(f"{n_jets_jettype} {jettype}-jets and")
            self.create_file = False
        else:
            for jet_type in jet_types:
                mask = jet_types_to_save == jet_types.index(jet_type)
                output_file_jettype = f"{output_file}_{jet_type}.h5"
                njets_step = sum(mask) #datasets.get_n_valid_jets()
                logger.info(f"loading {njets_step} valid jets")
                if njets_step > 0:
                    with File(output_file_jettype, "a") as train_file:
                        for jet_type in jet_types:
                            train_file[f"{self.config.vertex_feat_name}_{jet_type}"].resize(
                                (
                                    train_file[f"{self.config.vertex_feat_name}_{jet_type}"].shape[0]
                                    + njets_step
                                ),
                                axis=0,
                            )
                            train_file[f"{self.config.vertex_feat_name}_{jet_type}"][
                                -njets_step:
                            ] = vertex_feat[jet_type][mask]
                        # train_file[self.config.edge_feat_name].resize((train_file[self.config.edge_feat_name].shape[0] + njets_step), axis=0)
                        # train_file[self.config.edge_feat_name][-njets_step:] = datasets.get_edge_feat_y()
                            train_file[f"{self.config.edge_name}_{jet_type}"].resize(
                                (
                                    train_file[f"{self.config.edge_name}_{jet_type}"].shape[0]
                                    + njets_step
                                ),
                                axis=0,
                            )
                            train_file[f"{self.config.edge_name}_{jet_type}"][
                                -njets_step:
                            ] = edge_y[jet_type][mask]
                        train_file["HadronTruthLabel"].resize((train_file["HadronTruthLabel"].shape[0] + njets_step), axis=0)
                        train_file["HadronTruthLabel"][-njets_step:] = HadrTruth[mask]
                        train_file["edge_origin"].resize(
                            (train_file["edge_origin"].shape[0] + njets_step),
                            axis=0,
                        )
                        train_file["edge_origin"][
                            -njets_step:
                        ] = edge_origin[mask]
                        train_file[self.config.tracks_name].resize(
                            (
                                train_file[self.config.tracks_name].shape[0]
                                + njets_step
                            ),
                            axis=0,
                        )
                        train_file[self.config.tracks_name][
                            -njets_step:
                        ] = datasets.get_track_input()[mask]
                        train_file["jet_type"].resize(
                            (
                                train_file["jet_type"].shape[0]
                                + njets_step
                            ),
                            axis=0,
                        )
                        train_file["jet_type"][
                            -njets_step:
                        ] = jet_types_to_save[mask]
                        # train_file["track_extra"].resize(
                        #     (train_file["track_extra"].shape[0] + njets_step),
                        #     axis=0,
                        # )
                        # train_file["track_extra"][
                        #     -njets_step:
                        # ] = tracks_extra[mask]
                        train_file["track_extra_truth"].resize(
                            (train_file["track_extra_truth"].shape[0] + njets_step),
                            axis=0,
                        )
                        train_file["track_extra_truth"][
                            -njets_step:
                        ] = tracks_extra_true[mask]
                        # if unscaled_pt is not None:
                        #     train_file["unscaled_pt"].resize(
                        #         (train_file["unscaled_pt"].shape[0] + njets_step),
                        #         axis=0,
                        #     )
                        #     train_file["unscaled_pt"][
                        #         -njets_step:
                        #     ] = unscaled_pt
                        if jet_pt is not None:
                            train_file["jet_pt"].resize(
                                (train_file["jet_pt"].shape[0] + njets_step), axis=0
                            )
                            train_file["jet_pt"][-njets_step:] = jet_pt[mask]
                        # train_file["overall_inds"].resize(
                        #         (train_file["overall_inds"].shape[0] + njets_step), axis=0
                        #     )
                        # train_file["overall_inds"][-njets_step:] = overall_inds
                            
            
    def preprocess_for_salt_topo(
            self, 
            jet_types, 
            datasets,
            edge_y, 
            edge_origin, 
            HadrTruth, 
            jet_pt, 
            jet_types_to_save, 
            load_jet_types, 
            logger, 
            n_jets_loaded, 
            njets,
            output_file, 
            track_inputs, 
            # tracks_extra, 
            tracks_extra_true, 
            vertex_feat,
            flavour_label,
            jet_inputs
        ):
        if self.create_file:
            for jet_type in jet_types:
                mask = jet_types_to_save == jet_types.index(jet_type)
                n_tracks = track_inputs.shape[1]
                # if vertex_feat is not None:
                # train_file.create_dataset(self.config.edge_feat_name, data = datasets.get_edge_feat_y(), chunks=True, maxshape=(None,40,len(global_conf.edge_features)))
                output_file_jettype = f"{output_file}_{jet_type}.h5"
                with File(output_file_jettype, "w") as train_file:
                    for snd_jet_type in jet_types:
                        train_file.create_dataset(f"{self.config.edge_name}_{snd_jet_type}", data = edge_y[snd_jet_type][mask], chunks=True, maxshape=(None,n_tracks,))
                        train_file.create_dataset(f"{self.config.vertex_feat_name}_{snd_jet_type}", data = vertex_feat[snd_jet_type][mask], chunks=True, maxshape=(None,)) #len(global_conf.vertex_features))) # dtype=datasets.vertex_feat_dtypes
                    train_file.create_dataset("edge_origin", data = edge_origin[mask], chunks=True, maxshape=(None,n_tracks,))
                    train_file.create_dataset(f"{self.config.tracks_name}", data = np.array(track_inputs , dtype=datasets.reco_dtypes)[mask], chunks=True, maxshape=(None,n_tracks))
                    train_file.create_dataset(f"jet_type", data = np.array(jet_types_to_save)[mask], chunks=True, maxshape=(None,))
                    # train_file.create_dataset("track_extra", data = tracks_extra[mask], chunks=True, maxshape=(None,n_tracks,))
                    train_file.create_dataset("track_extra_truth", data = tracks_extra_true[mask], chunks=True, maxshape=(None,n_tracks,))
                    train_file.create_dataset("HadronConeExclTruthLabelID", data = HadrTruth[mask], chunks=True, maxshape=(None,))
                    train_file.create_dataset("flavour_label", data = flavour_label[mask], chunks=True, maxshape=(None,))
                    train_file.create_dataset("jet_inputs", data=jet_inputs[mask], chunks=True, maxshape=(None,))
                # if unscaled_pt is not None:
                #     train_file.create_dataset("unscaled_pt", data = unscaled_pt, chunks=True, maxshape=(None,))
                    if jet_pt is not None:
                        train_file.create_dataset("jet_pt", data = jet_pt[mask], chunks=True, maxshape=(None,))
            n_jets_loaded += [datasets.get_n_valid_jets_type(jet_type) for jet_type in jet_types]
            continue_loading = np.array(n_jets_loaded < int(njets))
            load_jet_types=np.array(jet_types)[continue_loading]
            logger.info(
                    "loaded "
                    + str(sum(n_jets_loaded))
                    + " jets in total"
                )
            logger.info(f"that includes")
            for n_jets_jettype, jettype in zip(n_jets_loaded, jet_types):
                logger.info(f"{n_jets_jettype} {jettype}-jets and")
            self.create_file = False
        else:
            for jet_type in jet_types:
                mask = jet_types_to_save == jet_types.index(jet_type)
                output_file_jettype = f"{output_file}_{jet_type}.h5"
                njets_step = sum(mask) #datasets.get_n_valid_jets()
                logger.info(f"loading {njets_step} valid jets for jet type {jet_type}")
                if njets_step > 0:
                    with File(output_file_jettype, "a") as train_file:
                        for jet_type in jet_types:
                            train_file[f"{self.config.vertex_feat_name}_{jet_type}"].resize(
                                (
                                    train_file[f"{self.config.vertex_feat_name}_{jet_type}"].shape[0]
                                    + njets_step
                                ),
                                axis=0,
                            )
                            train_file[f"{self.config.vertex_feat_name}_{jet_type}"][
                                -njets_step:
                            ] = vertex_feat[jet_type][mask]
                        # train_file[self.config.edge_feat_name].resize((train_file[self.config.edge_feat_name].shape[0] + njets_step), axis=0)
                        # train_file[self.config.edge_feat_name][-njets_step:] = datasets.get_edge_feat_y()
                            train_file[f"{self.config.edge_name}_{jet_type}"].resize(
                                (
                                    train_file[f"{self.config.edge_name}_{jet_type}"].shape[0]
                                    + njets_step
                                ),
                                axis=0,
                            )
                            train_file[f"{self.config.edge_name}_{jet_type}"][
                                -njets_step:
                            ] = edge_y[jet_type][mask]
                        train_file["HadronConeExclTruthLabelID"].resize((train_file["HadronConeExclTruthLabelID"].shape[0] + njets_step), axis=0)
                        train_file["HadronConeExclTruthLabelID"][-njets_step:] = HadrTruth[mask]
                        train_file["flavour_label"].resize((train_file["flavour_label"].shape[0] + njets_step), axis=0)
                        train_file["flavour_label"][-njets_step:] = flavour_label[mask]
                        train_file["jet_inputs"].resize((train_file["jet_inputs"].shape[0] + njets_step), axis=0)
                        train_file["jet_inputs"][-njets_step:] = flavour_label[mask]
                        train_file["edge_origin"].resize(
                            (train_file["edge_origin"].shape[0] + njets_step),
                            axis=0,
                        )
                        train_file["edge_origin"][
                            -njets_step:
                        ] = edge_origin[mask]
                        train_file[self.config.tracks_name].resize(
                            (
                                train_file[self.config.tracks_name].shape[0]
                                + njets_step
                            ),
                            axis=0,
                        )
                        train_file[self.config.tracks_name][
                            -njets_step:
                        ] = datasets.get_track_input()[mask]
                        train_file["jet_type"].resize(
                            (
                                train_file["jet_type"].shape[0]
                                + njets_step
                            ),
                            axis=0,
                        )
                        train_file["jet_type"][
                            -njets_step:
                        ] = jet_types_to_save[mask]
                        # train_file["track_extra"].resize(
                        #     (train_file["track_extra"].shape[0] + njets_step),
                        #     axis=0,
                        # )
                        # train_file["track_extra"][
                        #     -njets_step:
                        # ] = tracks_extra[mask]
                        train_file["track_extra_truth"].resize(
                            (train_file["track_extra_truth"].shape[0] + njets_step),
                            axis=0,
                        )
                        train_file["track_extra_truth"][
                            -njets_step:
                        ] = tracks_extra_true[mask]
                        # if unscaled_pt is not None:
                        #     train_file["unscaled_pt"].resize(
                        #         (train_file["unscaled_pt"].shape[0] + njets_step),
                        #         axis=0,
                        #     )
                        #     train_file["unscaled_pt"][
                        #         -njets_step:
                        #     ] = unscaled_pt
                        if jet_pt is not None:
                            train_file["jet_pt"].resize(
                                (train_file["jet_pt"].shape[0] + njets_step), axis=0
                            )
                            train_file["jet_pt"][-njets_step:] = jet_pt[mask]
                        # train_file["overall_inds"].resize(
                        #         (train_file["overall_inds"].shape[0] + njets_step), axis=0
                        #     )
                        # train_file["overall_inds"][-njets_step:] = overall_inds
