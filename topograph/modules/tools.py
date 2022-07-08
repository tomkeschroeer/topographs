import tensorflow.keras.backend as K
from tensorflow import constant
from h5py import File
import numpy as np
import yaml
import pathlib

def step_activation(x):
    return K.switch(x >= 0.7, constant([[1]], dtype=x.dtype), constant([[0]], dtype=x.dtype))

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
        print(sum(self.ind_truthflav))
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
        tOL = np.array([(np.logical_or(fromB, fromBC)) for fromB, fromBC in zip(tOL_fromB, tOL_fromBC)]).astype(int)
        return tOL
    
    def get_edge_feat_y(self):
        return np.array(self.reco[self.global_conf.edge_features])

    def get_vertex_feat_y(self):
        return np.array([vertex_feat[hf == 5][0] for hf, vertex_feat in zip(self.truth["flavour"], self.truth[self.global_conf.vertex_features])]) 
       
    def get_track_input(self):
        return self.reco[self.global_conf.track_inputs]