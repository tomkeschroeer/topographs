from glob import glob
import os
from h5py import File
from time import time

from topograph.modules import (
    get_logger,
    GlobalConfig
)

class OneFileMaker:
    def __init__(self, config):
        self.config = config
    def Run(self):
        input_files = glob(self.config.input)
        logger = get_logger()
        stepsize = 500_000
        metadata = {}
        continue_loading = True
        global_conf = GlobalConfig()
        os.makedirs(self.config.output, exist_ok=True)
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
                        with File(f"{self.config.output}/{self.config.one_file_name}", "w") as out_file:
                            out_file.create_dataset(self.config.input_tracks_name, data = f[f"/{self.config.input_tracks_name}"].fields(global_conf.track_inputs + ["truthOriginLabel"])[step*stepsize:(step+1)*stepsize], chunks=True, maxshape=(None, metadata["ntracks"]))
                            out_file.create_dataset(self.config.input_truth_name, data = f[f"/{self.config.input_truth_name}"].fields(global_conf.vertex_features + ["flavour"])[step*stepsize:(step+1)*stepsize], chunks=True, maxshape=(None ,metadata["ntracks"] ))
                            out_file.create_dataset(self.config.input_jet_name, data = f[f"/{self.config.input_jet_name}"].fields(["HadronConeExclExtendedTruthLabelID"])[step*stepsize:(step+1)*stepsize], chunks=True, maxshape=(None,))
                           # out_file.create_dataset("edge_features", data=f[f"/{self.config.input_tracks_name}"].fields(["truthOriginLabel"])[step*stepsize:(step+1)*stepsize], chunks=True, maxshape=(None, metadata["ntracks"]))
                    else:
                        with File(f"{self.config.output}/{self.config.one_file_name}", "a") as out_file:
                            n_entries = len(f[f"/{self.config.input_jet_name}"][step*stepsize:(step+1)*stepsize])
                            out_file[self.config.input_tracks_name].resize((out_file[self.config.input_tracks_name].shape[0] + n_entries), axis=0)
                            out_file[self.config.input_tracks_name][-n_entries:] = f[f"/{self.config.input_tracks_name}"].fields(global_conf.track_inputs + ["truthOriginLabel"])[step*stepsize:(step+1)*stepsize]
                            out_file[self.config.input_truth_name].resize((out_file[self.config.input_truth_name].shape[0] + n_entries), axis=0)
                            out_file[self.config.input_truth_name][-n_entries:] = f[f"/{self.config.input_truth_name}"].fields(global_conf.vertex_features + ["flavour"])[step*stepsize:(step+1)*stepsize]
                            out_file[self.config.input_jet_name].resize((out_file[self.config.input_jet_name].shape[0] + n_entries), axis=0)
                            out_file[self.config.input_jet_name][-n_entries:] = f[f"/{self.config.input_jet_name}"].fields(["HadronConeExclExtendedTruthLabelID"])[step*stepsize:(step+1)*stepsize]
                            #out_file["edge_features"].resize((out_file[self.config.input_jet_name].shape[0] + n_entries), axis=0)
                            #out_file["edge_features"][-n_entries:] = f[f"/{self.config.input_tracks_name}"].fields(["truthOriginLabel"])[step*stepsize:(step+1)*stepsize]           
                            if len(out_file[self.config.input_jet_name]) >= int(self.config.njets): continue_loading = False
                    if continue_loading == False: 
                        logger.info(f"Loaded {self.config.njets} jets, stop loading.")
                        break
