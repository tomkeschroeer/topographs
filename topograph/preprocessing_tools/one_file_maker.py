from glob import glob
import os
from h5py import File

from topograph.modules import (
    get_logger
)

class OneFileMaker:
    def __init__(self, config):
        self.config = config
    def Run(self):
        input_files = glob(self.config.input)
        logger = get_logger()
        stepsize = 500_000
        metadata = {}
        os.makedirs(self.config.output, exist_ok=True)
        for input_file_ind, input_file in enumerate(input_files):
            logger.info(f"Process file {input_file}")
            with File(input_file, "r") as f:
                njets = len(f[f"/{self.config.input_jet_name}"][:])
                print(f[f"/{self.config.input_tracks_name}"].shape)
                _, metadata["ntracks"] = f[f"/{self.config.input_tracks_name}"].shape
               # _,_,metadata["ntruth_var"] = f[f"/{self.config.input_truth_name}"].shape
                n_steps = njets//stepsize
                for step in range(n_steps):
                    logger.info(f"Process file number {input_file_ind+1} from {len(input_files)}, step {step+1}/{n_steps}")
                    if step == 0 and input_file_ind == 0:
                        print("create")
                        with File(f"{self.config.output}/{self.config.one_file_name}", "w") as out_file:
                            print(f.keys())
                            out_file.create_dataset(self.config.input_tracks_name, data = f[f"/{self.config.input_tracks_name}"][step*stepsize:(step+1)*stepsize], chunks=True, maxshape=(None, metadata["ntracks"]))
                            out_file.create_dataset(self.config.input_truth_name, data = f[f"/{self.config.input_truth_name}"][step*stepsize:(step+1)*stepsize], chunks=True, maxshape=(None ,metadata["ntracks"] ))
                            out_file.create_dataset(self.config.input_jet_name, data = f[f"/{self.config.input_jet_name}"]["HadronConeExclExtendedTruthLabelID"][step*stepsize:(step+1)*stepsize], chunks=True, maxshape=None)
                    else:
                        with File(f"{self.config.output}/{self.config.one_file_name}", "a") as out_file:
                            out_file[self.config.input_tracks_name].resize(((step + 1) * stepsize), axis=0)
                            out_file[self.config.input_tracks_name][-stepsize:] = f[f"/{self.config.input_tracks_name}"][step*stepsize:(step+1)*stepsize]
                            out_file[self.config.input_truth_name].resize(((step + 1) * stepsize), axis=0)
                            out_file[self.config.input_truth_name][-stepsize:] = f[f"/{self.config.input_truth_name}"][step*stepsize:(step+1)*stepsize]
                            out_file[self.config.input_jet_name].resize(((step + 1) * stepsize), axis=0)
                            out_file[self.config.input_jet_name][-stepsize:] = f[f"/{self.config.input_jet_name}"]["HadronConeExclExtendedTruthLabelID"][step*stepsize:(step+1)*stepsize]