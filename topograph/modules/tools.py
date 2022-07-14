from h5py import File
import numpy as np
from glob import glob
import yaml
import pathlib
import logging

import tensorflow.keras.backend as K
from tensorflow import constant

def step_activation(x):
    return K.switch(x >= 0.7, constant([[1]], dtype=x.dtype), constant([[0]], dtype=x.dtype))

def Mask_invalid(x):
    return K.equal(x, np.nan)

def get_logger():
    """Set DebugLevel for logging.

    Returns
    -------
    object
        Topograph logger.
    """

    log_levels = {
        "CRITICAL": logging.CRITICAL,
        "ERROR": logging.ERROR,
        "WARNING": logging.WARNING,
        "INFO": logging.INFO,
        "DEBUG": logging.DEBUG,
        "NOTSET": logging.NOTSET,
    }
    topo_logger = logging.getLogger("Topograph")
    topo_logger.setLevel(log_levels["INFO"])
    ch_handler = logging.StreamHandler()
    ch_handler.setLevel(log_levels["INFO"])
    ch_handler.setFormatter(CustomFormatter())

    topo_logger.addHandler(ch_handler)
    topo_logger.propagate = False
    return topo_logger

class CustomFormatter(logging.Formatter):
    """Logging Formatter to add colors and count warning / errors
    using implementation from
    https://stackoverflow.com/questions/384076/how-can-i-color-python-logging-output"""  # noqa # pylint: disable=C0301

    grey = "\x1b[38;21m"
    yellow = "\x1b[33;21m"
    green = "\x1b[32;21m"
    red = "\x1b[31;21m"
    bold_red = "\x1b[31;1m"
    reset = "\x1b[0m"
    debugformat = (
        "%(asctime)s - %(levelname)s:%(name)s: %(message)s (%(filename)s:%(lineno)d)"
    )
    date_format = "%(levelname)s:%(name)s: %(message)s"

    FORMATS = {
        logging.DEBUG: grey + debugformat + reset,
        logging.INFO: green + date_format + reset,
        logging.WARNING: yellow + date_format + reset,
        logging.ERROR: red + debugformat + reset,
        logging.CRITICAL: bold_red + debugformat + reset,
    }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)

class GlobalConfig:
    def __init__(self):
        global_config_path = f"{pathlib.Path(__file__).parent.absolute()}/general_config.yaml"
        with open(global_config_path) as global_conf_file:
            global_conf = yaml.load(global_conf_file, Loader=yaml.FullLoader)
            self.track_inputs = global_conf.get("track_inputs")
            self.edge_features = global_conf.get("edge_features")
            self.vertex_features = global_conf.get("vertex_features")

class GetConfiguration:
    def __init__(self, config_file):
        self.getConfFile(config_file)
        self.getParameters()

    def getConfFile(self, config_file):
        with open(config_file, "r") as conf_file:
            self.conf = yaml.load(conf_file, Loader=yaml.FullLoader)
  
    def getParameters(self):
        config_items = [
            "input",
            "output",
            "track_name",
            "epochs",
            "steps_per_epoch",
            "edge_feature_network",
            "edge_weight_network",
            "vertex_network",
            "training_file_name",
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
        stepsize : int = 5,
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
