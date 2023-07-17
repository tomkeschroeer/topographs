import os
from h5py import File
import numpy as np
import numpy.ma as ma
from glob import glob

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
        global_conf = GlobalConfig()
        output_file = f"{self.config.output}/{self.config.preprocessing_file_name}".replace("//","/")
        njets = self.dataset_types[""]["njets"] +  self.dataset_types["_val"]["njets"] +  self.dataset_types["_test"]["njets"]
        stepsize = min(500_000, int(njets/2))
        # stepsize = 200
        # input_file = f"{output_dir}/{self.config.one_file_name}.h5".replace(".h5.h5",".h5").replace("//","/")
        updated_filelist = glob(self.config.input)
        continue_loading = True
        os.makedirs(self.config.output, exist_ok=True)
        for ninput, input_file in enumerate(updated_filelist):
            if continue_loading == False: break
            with File(input_file, "r") as f:
                njets_file = len(f["/jets"][:])
            n_steps = njets_file//stepsize +1
            for step in range(n_steps):
                logger.info(f"Process file {input_file}, step {step+1}/{n_steps}")
                datasets = DatasetCreater(config=self.config, input_file=input_file, step=step, stepsize=stepsize, replace_invalid=True)
                if step == 0 and ninput == 0:
                    with File(output_file, "w") as train_file:
                        train_file.create_dataset(self.config.vertex_feat_name, data = datasets.get_vertex_feat_y(), chunks=True, maxshape=(None,)) #len(global_conf.vertex_features))) # dtype=datasets.vertex_feat_dtypes
                        #train_file.create_dataset(self.config.edge_feat_name, data = datasets.get_edge_feat_y(), chunks=True, maxshape=(None,40,len(global_conf.edge_features)))
                        train_file.create_dataset(self.config.edge_name, data = datasets.get_edge_y(), chunks=True, maxshape=(None,20,))
                        train_file.create_dataset("edge_origin", data = datasets.get_edge_origin(), chunks=True, maxshape=(None,20,))
                        train_file.create_dataset(self.config.tracks_name, data = np.array(datasets.get_track_input() , dtype=datasets.reco_dtypes), chunks=True, maxshape=(None,20))
                        train_file.create_dataset("track_extra", data = datasets.get_extra_track(), chunks=True, maxshape=(None,20,))
                        train_file.create_dataset("track_extra_truth", data = datasets.get_extra_track_truth(), chunks=True, maxshape=(None,20,))
                        train_file.create_dataset("unscaled_pt", data = datasets.get_unscaled_pt(), chunks=True, maxshape=(None,))
                        train_file.create_dataset("jet_pt", data = datasets.get_jet_pt(), chunks=True, maxshape=(None,))
                        train_file.create_dataset("truthOriginLabel", data = datasets.get_truthOriginLabel(), chunks=True, maxshape=(None,))
                else:
                    njets_step = datasets.get_n_valid_jets()
                    logger.info(f"loading {njets_step} valid jets")
                    if njets_step >0:
                        with File(output_file, "a") as train_file:
                            train_file[self.config.vertex_feat_name].resize((train_file[self.config.vertex_feat_name].shape[0] + njets_step), axis=0)
                            train_file[self.config.vertex_feat_name][-njets_step:] = datasets.get_vertex_feat_y()
                            #train_file[self.config.edge_feat_name].resize((train_file[self.config.edge_feat_name].shape[0] + njets_step), axis=0)
                            #train_file[self.config.edge_feat_name][-njets_step:] = datasets.get_edge_feat_y()
                            train_file[self.config.edge_name].resize((train_file[self.config.edge_name].shape[0] + njets_step), axis=0)
                            train_file[self.config.edge_name][-njets_step:] = datasets.get_edge_y()
                            train_file["edge_origin"].resize((train_file["edge_origin"].shape[0] + njets_step), axis=0)
                            train_file["edge_origin"][-njets_step:] = datasets.get_edge_origin()
                            train_file[self.config.tracks_name].resize((train_file[self.config.tracks_name].shape[0] + njets_step), axis=0)
                            train_file[self.config.tracks_name][-njets_step:] = datasets.get_track_input()
                            train_file["track_extra"].resize((train_file["track_extra"].shape[0] + njets_step), axis=0)
                            train_file["track_extra"][-njets_step:] = datasets.get_extra_track()
                            train_file["track_extra_truth"].resize((train_file["track_extra"].shape[0] + njets_step), axis=0)
                            train_file["track_extra_truth"][-njets_step:] = datasets.get_extra_track_truth()
                            train_file["unscaled_pt"].resize((train_file["unscaled_pt"].shape[0] + njets_step), axis=0)
                            train_file["unscaled_pt"][-njets_step:] = datasets.get_unscaled_pt()
                            train_file["jet_pt"].resize((train_file["jet_pt"].shape[0] + njets_step), axis=0)
                            train_file["jet_pt"][-njets_step:] = datasets.get_jet_pt()
                            logger.info("loaded " + str(len(train_file[self.config.vertex_feat_name])) + " jets in total")
                            if len(train_file[self.config.vertex_feat_name]) >= int(njets): continue_loading = False
                if continue_loading == False: 
                    logger.info(f"Loaded {njets} jets, stop loading.")
                    break
