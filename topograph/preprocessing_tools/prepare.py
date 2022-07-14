import os
from h5py import File
import numpy as np
import numpy.ma as ma

from topograph.modules.tools import (
    GlobalConfig,
    get_logger,
    DatasetCreater
)

class Prepare:
    def __init__(self, config):
        self.config = config
    
    def Run(self):
        logger = get_logger()
        stepsize = 300_000
        global_conf = GlobalConfig()
        input_files = self.config.get_all_input_files()
        training_file_dir = f"{self.config.output}/training_files"
        os.makedirs(training_file_dir, exist_ok=True)
        for input_file_ind in range(len(input_files)):
            with File(input_files[input_file_ind], "r") as f:
                njets = len(f["/jets"][:])
            n_steps = njets//stepsize
            for step in range(n_steps):
                logger.info(f"Process file number {input_file_ind+1} from {len(input_files)}, step {step+1}/{n_steps}")
                datasets = DatasetCreater(input_file=input_files[input_file_ind], step=step, stepsize=stepsize, replace_invalid=True)
                if step == 0 and input_file_ind == 0:
                    with File(f"{training_file_dir}/{self.config.training_file_name}", "w") as train_file:
                        train_file.create_dataset("Y_vertex_features", data = datasets.get_vertex_feat_y(), chunks=True, maxshape=(None,len(global_conf.vertex_features)))
                        train_file.create_dataset("Y_edge_features", data = datasets.get_edge_feat_y(), chunks=True, maxshape=(None,40,len(global_conf.edge_features)))
                        train_file.create_dataset("Y_edge", data = datasets.get_edge_y(), chunks=True, maxshape=(None,40,1))
                        train_file.create_dataset("X_train_tracks", data = datasets.get_track_input(), chunks=True, maxshape=(None,40,len(global_conf.track_inputs)))
                else:
                    njets_step = datasets.get_n_valid_jets()
                    with File(f"{training_file_dir}/{self.config.training_file_name}", "a") as train_file:
                        train_file["Y_vertex_features"].resize((train_file["Y_vertex_features"].shape[0] + njets_step), axis=0)
                        train_file["Y_vertex_features"][-njets_step:] = datasets.get_vertex_feat_y()
                        train_file["Y_edge_features"].resize((train_file["Y_edge_features"].shape[0] + njets_step), axis=0)
                        train_file["Y_edge_features"][-njets_step:] = datasets.get_edge_feat_y()
                        train_file["Y_edge"].resize((train_file["Y_edge"].shape[0] + njets_step), axis=0)
                        train_file["Y_edge"][-njets_step:] = datasets.get_edge_y()
                        train_file["X_train_tracks"].resize((train_file["X_train_tracks"].shape[0] + njets_step), axis=0)
                        train_file["X_train_tracks"][-njets_step:] = datasets.get_track_input()
