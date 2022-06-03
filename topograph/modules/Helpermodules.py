import yaml

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
            raise KeyError("You need to specify inputs in your config file")
        
        setattr(self, "inputs", self.conf["inputs"])

        config_items = [
            "output",
            "edge_feature_network",
            "edge_weight_network",
            "vertex_network"
        ]

        for item in config_items:
            if item in self.conf:
                setattr(self, item, self.conf[item])
            else:
                raise KeyError(f"You need to specify {item} in your config file")
    
        