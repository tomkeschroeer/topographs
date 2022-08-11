"""Script containing different tools."""
import logging
import pathlib
from glob import glob

import numpy as np
import tensorflow.keras.backend as K
import yaml
from h5py import File
from tensorflow import TensorShape, constant, float32


def step_activation(x):
    return K.switch(
        x >= 0.7, constant([[1]], dtype=x.dtype), constant([[0]], dtype=x.dtype)
    )


def shifted_relu_activation(x):
    return K.switch(x >= 0.8, (x - 0.8), constant([[0]], dtype=x.dtype))


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


def get_track_mask(trks):
    for var, dtype in trks.dtype.fields.items():
        if "f" in dtype[0].str:
            track_mask = ~np.isnan(trks[var])
            return track_mask


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
        global_config_path = (
            f"{pathlib.Path(__file__).parent.absolute()}/general_config.yaml"
        )
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
        # config_items = [
        #     "input",
        #     "output",
        #     "stepsize",
        #     "lr",
        #     "njets",
        #     "tracks_name",
        #     "input_tracks_name",
        #     "input_jet_name",
        #     "input_truth_name",
        #     "epochs",
        #     "steps_per_epoch",
        #     "edge_feature_network",
        #     "edge_weight_network",
        #     "vertex_network",
        #     "preprocessing_file_name",
        #     "one_file_name",
        #     "scale_dict",
        #     "training_file_name",
        #     "edge_feat_name",
        #     "edge_name",
        #     "vertex_feat_name",
        # ]

        for item in self.conf.keys():  # config_items:
            if item in self.conf:
                setattr(self, item, self.conf[item])
            else:
                raise KeyError(f"You need to specify {item} in your config file")

    def get_all_input_files(self):
        try:
            return glob(self.input)
        except KeyError:
            raise KeyError("No input file defined.")


class DatasetCreater:
    def __init__(self, config, input_file, step, stepsize, replace_invalid=False):
        self.global_conf = GlobalConfig()
        self.config = config
        self.input_file = input_file
        self.step = step
        self.stepsize = stepsize
        self.ind_truthflav = None
        self.replace_invalid = replace_invalid
        with File(self.input_file, "r") as f:
            self.truth = f[f"/{self.config.input_truth_name}"][
                self.step * self.stepsize : (self.step + 1) * self.stepsize
            ]
            self.HadrConeTruth = f[f"/{self.config.input_jet_name}"][
                "HadronConeExclExtendedTruthLabelID"
            ][self.step * self.stepsize : (self.step + 1) * self.stepsize][:]
            self.reco = f[f"/{self.config.input_tracks_name}"][
                self.step * self.stepsize : (self.step + 1) * self.stepsize, :
            ]
            self.edge_features = f["/edge_features"][
                self.step * self.stepsize : (self.step + 1) * self.stepsize, :
            ]

        self.ind_truthflav = self.get_b_indeces()
        self.truth = self.truth[self.ind_truthflav]
        self.reco = self.reco[self.ind_truthflav]
        self.edge_features = self.edge_features[self.ind_truthflav]

    def get_b_indeces(self):
        hadronflavour = self.truth["flavour"]
        return np.logical_and(
            self.HadrConeTruth == 5, [sum(hf == 5) == 1 for hf in hadronflavour]
        )

    def get_n_valid_jets(self):
        return sum(self.ind_truthflav)

    def get_edge_y(self):
        truthOriginLabel = self.edge_features["truthOriginLabel"]
        tOL_fromB = [
            [OL == 3 for OL in tracklabels] for tracklabels in truthOriginLabel
        ]
        tOL_fromBC = [
            [OL == 4 for OL in tracklabels] for tracklabels in truthOriginLabel
        ]
        tOL = np.array(
            [
                [
                    np.array([edge_y]).astype(int)
                    for edge_y in (np.logical_or(fromB, fromBC))
                ]
                for fromB, fromBC in zip(tOL_fromB, tOL_fromBC)
            ]
        )
        return tOL

    def get_edge_feat_y(self):
        edge_feat_y = np.array(
            [
                [list(feat_track) for feat_track in feat_jet]
                for feat_jet in self.edge_features[self.global_conf.edge_features]
            ]
        )
        return edge_feat_y

    def get_vertex_feat_y(self):
        vertex_feat = np.array(
            [
                list(vertex_feat[hf == 5][0])
                for hf, vertex_feat in zip(
                    self.truth["flavour"],
                    self.truth[self.global_conf.vertex_features],
                )
            ]
        )
        vertex_feat = np.log(vertex_feat)
        return vertex_feat

    def get_track_input(self):
        track_input = [
            [list(inputs_track) for inputs_track in input_jet]
            for input_jet in self.reco
        ]
        return track_input


class DataGenerator:
    def __init__(
        self,
        input: str,
        metadata_dict: dict,
        get_labels: bool = True,
        get_inputs: bool = True,
        get_weight_labels: bool = False,
        stepsize: int = 5000,
        savejets: bool = False,
        savetracks: bool = True,
        track_name: str = "tracks",
        edge_name: str = "edge",
        edge_feat_name: str = "edge_feat",
        vertex_feat_name: str = "vertex_feat",
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
        self.get_inputs = get_inputs
        self.get_labels = get_labels
        self.get_weight_labels = get_weight_labels
        self.metadata_dict = metadata_dict
        self.stepsize = stepsize
        self.savejets = savejets
        self.savetracks = savetracks
        self.track_name = track_name
        self.edge_feat_name = edge_feat_name
        self.edge_name = edge_name
        self.vertex_feat_name = vertex_feat_name

    def load_in_memory(self, step: int = 0):
        with File(self.input) as f:
            if self.get_inputs:
                self.track_batch = f[self.track_name][
                    step * self.stepsize : (step + 1) * self.stepsize
                ]
            if self.get_labels:
                self.vertex_feat_batch = f[self.vertex_feat_name][
                    step * self.stepsize : (step + 1) * self.stepsize
                ]
            if self.get_weight_labels:
                self.edge_batch = f[self.edge_name][
                    step * self.stepsize : (step + 1) * self.stepsize
                ]

    def get_types_shapes(self):
        if self.get_labels and self.get_inputs and self.get_weight_labels:
            types = (
                {"input_1": float32, "input_2": float32},
                {"edge_weight_sigmoid": float32, "vertex_network": float32},
            )
            shapes = (
                {
                    "input_1": TensorShape(
                        (
                            None,
                            self.metadata_dict["n_trks"],
                            self.metadata_dict["n_trk_features"],
                        )
                    ),
                    "input_2": TensorShape(
                        (
                            None,
                            self.metadata_dict["n_trks"],
                            self.metadata_dict["n_trk_features"],
                        )
                    ),
                },
                {
                    "edge_weight_sigmoid": TensorShape(
                        (
                            None,
                            self.metadata_dict["n_trks"],
                            self.metadata_dict["n_edge_y"],
                        )
                    ),
                    "vertex_network": TensorShape(
                        (None, self.metadata_dict["n_vertex_feat"])
                    ),
                },
            )
        elif self.get_labels and self.get_inputs:
            types = ({"input_1": float32, "input_2": float32}, float32)
            shapes = (
                {
                    "input_1": TensorShape(
                        (
                            None,
                            self.metadata_dict["n_trks"],
                            self.metadata_dict["n_trk_features"],
                        )
                    ),
                    "input_2": TensorShape(
                        (
                            None,
                            self.metadata_dict["n_trks"],
                            self.metadata_dict["n_trk_features"],
                        )
                    ),
                },
                TensorShape((None, self.metadata_dict["n_vertex_feat"])),
            )
        elif self.get_labels:
            types = float32
            shapes = TensorShape((None, self.metadata_dict["n_vertex_feat"]))
        elif self.get_inputs:
            types = {"input_1": float32, "input_2": float32}
            shapes = {
                "input_1": TensorShape(
                    (
                        None,
                        self.metadata_dict["n_trks"],
                        self.metadata_dict["n_trk_features"],
                    )
                ),
                "input_2": TensorShape(
                    (
                        None,
                        self.metadata_dict["n_trks"],
                        self.metadata_dict["n_trk_features"],
                    )
                ),
            }
        return types, shapes


class DataLoader(DataGenerator):
    def __call__(self):
        n_samples = self.metadata_dict["n_jets"]
        n_steps = n_samples // self.stepsize + 1
        for step in range(n_steps):
            self.load_in_memory(step=step)
            if self.get_inputs and self.get_labels and self.get_weight_labels:
                yield {"input_1": self.track_batch, "input_2": self.track_batch}, {
                    "edge_weight_sigmoid": self.edge_batch,
                    "vertex_network": self.vertex_feat_batch,
                }
            elif self.get_inputs and self.get_labels and not self.get_weight_labels:
                yield {
                    "input_1": self.track_batch,
                    "input_2": self.track_batch,
                }, self.vertex_feat_batch
            elif self.get_inputs:
                yield {"input_1": self.track_batch, "input_2": self.track_batch}
            elif self.get_labels:
                yield self.vertex_feat_batch
            elif self.get_weight_labels:
                yield self.get_weight_labels
