import os
from h5py import File
import numpy as np
import numpy.ma as ma

from topograph.modules.tools import (
    GlobalConfig,
    get_logger,
)

class Prepare:
    def __init__(self, config):
        self.config = config
    
    def Run(self):
        logger = get_logger()
        #stepsize = 300_000
        stepsize = 100
        global_conf = GlobalConfig()
        input_files = self.config.get_all_input_files()
        training_file_dir = f"{self.config.output}/training_files"
        os.makedirs(training_file_dir, exist_ok=True)
        for input_file_ind in range(len(input_files)):
            with File(input_files[input_file_ind], "r") as f:
                njets = len(f["/jets"][:])
            n_steps = njets//stepsize
            for step in range(100):#n_steps):
                logger.info(f"Process file number {input_file_ind+1} from {len(input_files)}, step {step+1}/{n_steps}")
                datasets = DatasetCreater(input_file=input_files[input_file_ind], step=step, stepsize=stepsize)
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

class DatasetCreater:
    def __init__(self, input_file, step, stepsize):
        self.global_conf = GlobalConfig()
        self.input_file = input_file
        self.step = step
        self.stepsize = stepsize
        self.ind_truthflav = None

        with File(self.input_file, "r") as f:
            self.truth = f["/truth_hadrons"][self.step*self.stepsize:(self.step+1)*self.stepsize :]
            self.HadrConeTruth = f["/jets"][self.step*self.stepsize:(self.step+1)*self.stepsize]["HadronConeExclExtendedTruthLabelID"]
            self.reco = f["/tracks_loose"][self.step*self.stepsize:(self.step+1)*self.stepsize, :]

        self.ind_truthflav = self.get_b_indeces()
        self.truth = self.truth[self.ind_truthflav]
        self.reco = self.reco[self.ind_truthflav]

    def get_b_indeces(self):
        hadronflavour = self.truth["flavour"]
        return np.logical_and(self.HadrConeTruth == 5, [sum(hf == 5)==1 for hf in hadronflavour])
   
    def get_n_valid_jets(self): 
        return sum(self.ind_truthflav)

    def get_edge_y(self):
        truthOriginLabel = self.reco["truthOriginLabel"]
        tOL_fromB = [[OL == 3 for OL in tracklabels] for tracklabels in truthOriginLabel]
        tOL_fromBC = [[OL == 4 for OL in tracklabels] for tracklabels in truthOriginLabel]
        tOL = np.array([[np.array([edge_y]).astype(int) for edge_y in (np.logical_or(fromB, fromBC))] for fromB, fromBC in zip(tOL_fromB, tOL_fromBC)])
        return tOL
    
    def get_edge_feat_y(self):
        edge_feat_y = np.array([[list(feat_track) for feat_track in feat_jet] for feat_jet in self.reco[self.global_conf.edge_features]])
        return edge_feat_y

    def get_vertex_feat_y(self):
        vertex_feat = np.array([list(vertex_feat[hf == 5][0]) for hf, vertex_feat in zip(self.truth["flavour"], self.truth[self.global_conf.vertex_features])])
        return vertex_feat
       
    def get_track_input(self):
        track_input = ma.masked_invalid([[list(inputs_track) for inputs_track in input_jet] for input_jet in self.reco[self.global_conf.track_inputs]])
        mask = track_input.mask
        track_input[mask] = -999
        return track_input