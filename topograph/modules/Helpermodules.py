import yaml
from h5py import File

class GetConfiguration:
    def __init__(self, config_file):
        self.getConfFile(config_file)
        self.getParameters()

    def getConfFile(self, config_file):
        with open(config_file, "r") as conf_file:
            self.conf = yaml.load(conf_file, Loader=yaml.FullLoader)
  
    def getParameters(self):
        if "inputs" in self.conf:
            for input in self.conf["inputs"].keys():
                if not isinstance(self.conf["inputs"][input], list):
                    self.conf["inputs"][input] = [self.conf["inputs"][input]]
                else:
                    self.conf["inputs"][input] = self.conf["inputs"][input]
        else:
            raise KeyError("You need to specify inputs in your config file")
        
        setattr(self, "inputs", self.conf["inputs"])

        config_items = [
            "output",
            "jets_name",
            "track_name",
            "target_name",
            "edge_feature_network",
            "edge_weight_network",
            "vertex_network"
        ]

        for item in config_items:
            if item in self.conf:
                setattr(self, item, self.conf[item])
            else:
                raise KeyError(f"You need to specify {item} in your config file")

class DataGenerator:
    def __init__(
        self, 
        input : str, 
        metadata : dict,
        stepsize : int = 5_000,
        savejets : bool = False,
        savetracks : bool = True,
        track_name : str = "tracks",
        jet_name : str = "jets",
        target_name: str = "y"
        ):
        """
        class to get dataset for topograph training

        Parameters
        ----------
        input: 
            input file containing the jet and track information
        metadata:
            dictionary containing the total number of jets and tracks
        stepsize:
            the number of samples returned per generator step. Default: 5_000
        savejets:
            bool defining if jets are supposed to be saved
        """
        self.input = input
        self.metadata = metadata
        self.stepsize = stepsize
        self.savejets = savejets
        self.savetracks = savetracks
        self.track_name = track_name
        self.jet_name = jet_name
        self.target_name = target_name
    
    def __call__():
        self.get_Dataset()

    def load_in_memory(self, step : int = 0):
        with File(input) as f:
            if self.savejets:
                self.jets_batch = f[self.jet_name][step*self.stepsize, (step+1)*self.stepsize]
            if self.savetracks:
                self.track_batch = f[self.track_name][step*self.stepsize, (step+1)*self.stepsize]
            self.y = f[self.target_name][step*self.stepsize, (step+1)*self.stepsize]
            
        
    def get_Dataset(self):
        n_samples = self.metadata["n_samples"]
        n_steps = n_samples//self.stepsize
        for step in range(n_steps):
            self.load_in_memory(step=step)
            if self.savetracks and self.savejets:
                yield {"input_1":self.track_batch, "input_2":self.track_batch, "input_3":self.jets_batch}, self.y
            elif self.savetracks:
                yield {"input_1":self.track_batch, "input_2":self.track_batch}, self.y
            elif self.savejets:
                yield self.jets_batch, self.y
    
    
        