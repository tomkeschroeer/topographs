from glob import glob
import os
import numpy as np
from h5py import File
from time import time
import random

from topograph.modules import (
    get_logger,
    GlobalConfig
)

class OneFileMaker:
    def __init__(self, config, dataset_types):
        self.config = config
        self.dataset_types = dataset_types
    def Run(self):
        updated_filelist = glob(self.config.input)
        output_file = f"{self.config.output}/{self.config.one_file_name}".replace(".h5","")
        logger = get_logger()
        metadata = {}
        continue_loading = True
        global_conf = GlobalConfig()
        vertex_features = list(global_conf.vertex_features.keys())
        os.makedirs(self.config.output, exist_ok=True)
        for dataset_type, dataset_feat in self.dataset_types.items():
            logger.info(f"producing file for {dataset_type}")
            njets_datatype = dataset_feat["njets"]
            stepsize = min(500_000, int(njets_datatype/2))
            continue_loading = True
            input_files = updated_filelist
            for input_file_ind, input_file in enumerate(input_files):
                if continue_loading == False: break
                logger.info(f"Process file {input_file}")
                with File(input_file, "r") as f:
                    njets = len(f[f"/{self.config.input_jet_name}"][:])
                    _, metadata["ntracks"] = f[f"/{self.config.input_tracks_name}"].shape
                # _,_,metadata["ntruth_var"] = f[f"/{self.config.input_truth_name}"].shape
                    n_steps = njets//stepsize + 1
                    for step in range(n_steps):
                        logger.info(f"Process file number {input_file_ind+1} from {len(input_files)}, step {step+1}/{n_steps}")
                        if step == 0 and input_file_ind == 0:
                            with File(f"{output_file}{dataset_type}.h5", "w") as out_file:
                                out_file.create_dataset(self.config.input_tracks_name, data = f[f"/{self.config.input_tracks_name}"].fields(global_conf.track_inputs)[step*stepsize:(step+1)*stepsize], chunks=True, maxshape=(None, metadata["ntracks"]))
                                out_file.create_dataset(self.config.input_truth_name, data = f[f"/{self.config.input_truth_name}"].fields(vertex_features + ["flavour"])[step*stepsize:(step+1)*stepsize], chunks=True, maxshape=(None ,metadata["ntracks"] ))
                                out_file.create_dataset(self.config.input_jet_name, data = f[f"/{self.config.input_jet_name}"].fields(["HadronConeExclExtendedTruthLabelID"])[step*stepsize:(step+1)*stepsize], chunks=True, maxshape=(None,))
                                out_file.create_dataset("edge_features", data=f[f"/{self.config.input_tracks_name}"].fields(["truthOriginLabel"])[step*stepsize:(step+1)*stepsize], chunks=True, maxshape=(None, metadata["ntracks"]))
                        else:
                            with File(f"{output_file}{dataset_type}.h5", "a") as out_file:
                                n_entries = len(f[f"/{self.config.input_jet_name}"][step*stepsize:(step+1)*stepsize])
                                if n_entries > 0:
                                    out_file[self.config.input_tracks_name].resize((out_file[self.config.input_tracks_name].shape[0] + n_entries), axis=0)
                                    out_file[self.config.input_tracks_name][-n_entries:] = f[f"/{self.config.input_tracks_name}"].fields(global_conf.track_inputs)[step*stepsize:(step+1)*stepsize]
                                    out_file[self.config.input_truth_name].resize((out_file[self.config.input_truth_name].shape[0] + n_entries), axis=0)
                                    out_file[self.config.input_truth_name][-n_entries:] = f[f"/{self.config.input_truth_name}"].fields(vertex_features + ["flavour"])[step*stepsize:(step+1)*stepsize]
                                    out_file[self.config.input_jet_name].resize((out_file[self.config.input_jet_name].shape[0] + n_entries), axis=0)
                                    out_file[self.config.input_jet_name][-n_entries:] = f[f"/{self.config.input_jet_name}"].fields(["HadronConeExclExtendedTruthLabelID"])[step*stepsize:(step+1)*stepsize]
                                    out_file["edge_features"].resize((out_file["edge_features"].shape[0] + n_entries), axis=0)
                                    out_file["edge_features"][-n_entries:] = f[f"/{self.config.input_tracks_name}"].fields(["truthOriginLabel"])[step*stepsize:(step+1)*stepsize]           
                                if len(out_file[self.config.input_jet_name]) >= int(njets_datatype): continue_loading = False
                        if continue_loading == False: 
                            logger.info(f"Loaded {njets_datatype} jets, stop loading.")
                            break
                    updated_filelist.remove(input_file)
