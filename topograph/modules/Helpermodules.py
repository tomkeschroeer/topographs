import yaml
from h5py import File
from glob import glob

class GetConfiguration:
    def __init__(self, config_file):
        self.getConfFile(config_file)
        self.getParameters()

    def getConfFile(self, config_file):
        with open(config_file, "r") as conf_file:
            self.conf = yaml.load(conf_file, Loader=yaml.FullLoader)
  
    def getParameters(self):
        # if "input" in self.conf:
        #     for input in self.conf["input"].keys():
        #         self.conf["input = self.conf["inputs"][input]
        # else:
        #     raise KeyError("You need to specify inputs in your config file")
        
        # setattr(self, "inputs", self.conf["inputs"])

        config_items = [
            "input",
            "output",
            "track_name",
            "epochs",
            "edge_feature_network",
            "edge_weight_network",
            "vertex_network",
            "training_input",
            "edge_feat_name",
            "edge_name",
            "vertex_feat_name"
        ]

        for item in config_items:
            if item in self.conf:
                setattr(self, item, self.conf[item])
            else:
                raise KeyError(f"You need to specify {item} in your config file")
    
    def get_all_input_files(self):
        try:
            return glob(self.input)
        except KeyError:
            raise KeyError(f"No input file defined.")

class DataGenerator:
    def __init__(
        self, 
        input : str, 
        metadata_dict : dict,
        stepsize : int = 5_000,
        savejets : bool = False,
        savetracks : bool = True,
        track_name : str = "tracks",
        edge_name : str ="edge",
        edge_feat_name : str = "edge_feat",
        vertex_feat_name : str = "vertex_feat"
        ):
        """
        class to get dataset for topograph training

        Parameters
        ----------
        input: 
            input file containing the jet and track information
        metadata_dict:
            dictionary containing the total number of jets and tracks
        stepsize:
            the number of samples returned per generator step. Default: 5_000
        savejets:
            bool defining if jets are supposed to be saved
        """
        self.input = input
        self.metadata_dict = metadata_dict
        self.stepsize = stepsize
        self.savejets = savejets
        self.savetracks = savetracks
        self.track_name = track_name
        self.edge_feat_name = edge_feat_name
        self.edge_name = edge_name
        self.vertex_feat_name = vertex_feat_name

    def load_in_memory(self, step : int = 0):
        with File(self.input) as f:
            if self.savejets:
                self.jets_batch = f[self.jets_name][step*self.stepsize : (step+1)*self.stepsize]
            if self.savetracks:
                self.track_batch = f[self.track_name][step*self.stepsize : (step+1)*self.stepsize]
            self.edge_feat_batch = f[self.edge_feat_name][step*self.stepsize : (step+1)*self.stepsize]
            self.edge_batch = f[self.edge_name][step*self.stepsize : (step+1)*self.stepsize]
            self.vertex_feat_batch = f[self.vertex_feat_name][step*self.stepsize : (step+1)*self.stepsize]
    
class DataLoader(DataGenerator):
    def __call__(self):
        n_samples = self.metadata_dict["n_jets"]
        n_steps = n_samples//self.stepsize
        for step in range(n_steps):
            self.load_in_memory(step=step)
            yield {"input_1":self.track_batch, "input_2":self.track_batch},{"edge_feat":self.edge_feat_batch,"edge_weight":self.edge_batch,"vertex_network":self.vertex_feat_batch}


        
        
