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
    def __init__(self, config, dataset_types):
        self.config = config
        self.dataset_types = dataset_types
    
    def Run(self):
        logger = get_logger()
        stepsize = 500_000
        global_conf = GlobalConfig()
        output_dir = f"{self.config.output}"
        for dataset_type, dataset_feat in self.dataset_types.items():
            input_file = f"{output_dir}/{self.config.preprocessing_file_name}".replace(".h5","").replace("//","/")
            input_file = input_file + f"{dataset_type}.h5"
            with File(input_file, "r") as f:
                njets = len(f["/jets"][:])
            n_steps = njets//stepsize +1
            for step in range(n_steps):
                logger.info(f"Process file {input_file}, step {step+1}/{n_steps}")
                datasets = DatasetCreater(config=self.config, input_file=input_file, step=step, stepsize=stepsize, replace_invalid=True)
                if step == 0:
                    with File(dataset_feat["final_filename"], "w") as train_file:
                        train_file.create_dataset(self.config.vertex_feat_name, data = datasets.get_vertex_feat_y(), chunks=True, maxshape=(None,len(global_conf.vertex_features)))
                        #train_file.create_dataset(self.config.edge_feat_name, data = datasets.get_edge_feat_y(), chunks=True, maxshape=(None,40,len(global_conf.edge_features)))
                        train_file.create_dataset(self.config.edge_name, data = datasets.get_edge_y(), chunks=True, maxshape=(None,40,1))
                        train_file.create_dataset(self.config.tracks_name, data = datasets.get_track_input(), chunks=True, maxshape=(None,40,len(global_conf.track_inputs)))
                else:
                    njets_step = datasets.get_n_valid_jets()
                    logger.info(f"loading {njets_step} valid jets")
                    if njets_step >0:
                        with File(dataset_feat["final_filename"], "a") as train_file:
                            train_file[self.config.vertex_feat_name].resize((train_file[self.config.vertex_feat_name].shape[0] + njets_step), axis=0)
                            train_file[self.config.vertex_feat_name][-njets_step:] = datasets.get_vertex_feat_y()
                            #train_file[self.config.edge_feat_name].resize((train_file[self.config.edge_feat_name].shape[0] + njets_step), axis=0)
                            #train_file[self.config.edge_feat_name][-njets_step:] = datasets.get_edge_feat_y()
                            train_file[self.config.edge_name].resize((train_file[self.config.edge_name].shape[0] + njets_step), axis=0)
                            train_file[self.config.edge_name][-njets_step:] = datasets.get_edge_y()
                            train_file[self.config.tracks_name].resize((train_file[self.config.tracks_name].shape[0] + njets_step), axis=0)
                            train_file[self.config.tracks_name][-njets_step:] = datasets.get_track_input()
