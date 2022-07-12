from h5py import File
import numpy as np
import numpy.ma as ma
import yaml
import pathlib

import tensorflow.keras.backend as K
from tensorflow import constant

def step_activation(x):
    return K.switch(x >= 0.7, constant([[1]], dtype=x.dtype), constant([[0]], dtype=x.dtype))

def Mask_invalid(x):
    return K.equal(x, np.nan)

class GlobalConfig:
    def __init__(self):
        global_config_path = f"{pathlib.Path(__file__).parent.absolute()}/general_config.yaml"
        with open(global_config_path) as global_conf_file:
            global_conf = yaml.load(global_conf_file, Loader=yaml.FullLoader)
            self.track_inputs = global_conf.get("track_inputs")
            self.edge_features = global_conf.get("edge_features")
            self.vertex_features = global_conf.get("vertex_features")

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