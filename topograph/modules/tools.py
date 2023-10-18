"""Script containing different tools."""
import logging
import pathlib
from glob import glob

import numpy as np

# import tensorflow.keras.backend as K
import yaml
from h5py import File

# from tensorflow import TensorShape, Variable, constant, float32
from torch.utils.data import IterableDataset


def get_sample_weights(x_n_mask):
    x = x_n_mask[0]
    mask = x_n_mask[1].astype(bool)
    shape = np.array(x).shape
    x_masked = x[mask]
    x_masked = x_masked.flatten()
    n_total = len(x_masked)
    n_b = sum(x)
    n_nonb = n_total - n_b
    fac_b = n_total / (n_b + 1e-5)
    fac_nonb = n_total / (n_nonb + 1e-5)
    weights = np.ones(len(x))
    weights[x == 1] = fac_b
    weights[x == 0] = fac_nonb
    return weights.reshape((shape[0]))


def step_activation(x):
    return K.switch(
        x >= 0.7, constant([[1]], dtype=x.dtype), constant([[0]], dtype=x.dtype)
    )


def shifted_relu_activation(x):
    return K.switch(x >= 0.8, (x - 0.8), constant([[0]], dtype=x.dtype))


def shifted_relu_activation_train(x):
    fac = Variable(0.8)
    return K.switch(x >= fac, (x - fac), constant([[0]], dtype=x.dtype))


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


def get_mask(trks):
    for var, dtype in trks.dtype.fields.items():
        if "f" in dtype[0].str:
            mask = np.logical_and(~np.isnan(trks[var]), ~(trks[var]==-999.))
            return mask

def scary_shuffle(*arrays):
    """?!Should?! shuffle a collection of arrays inplace in the exact same way"""
    assert all(len(a) == len(arrays[0]) for a in arrays)
    rng_state = np.random.get_state()
    for a in arrays:
        np.random.shuffle(a)
        np.random.set_state(rng_state)

def get_types_shapes(
    get_labels: bool = True,
    get_inputs: bool = True,
    get_weight_labels: bool = False,
    get_sample_weights: bool = False,
    input_weight_layer_name: str = None,
    input_feat_layer_name: str = None,
    edge_weight_layer_name: str = None,
    edge_feat_layer_name: str = None,
    vertex_network_layer_name: str = None,
    metadata_dict: dict = None,
):
    if get_labels and get_inputs and get_weight_labels and get_sample_weights:
        types = (
            {
                input_weight_layer_name: float32,
                input_feat_layer_name: float32,
            },
            {
                edge_weight_layer_name: float32,
                vertex_network_layer_name: float32,
            },
            {edge_weight_layer_name: float32},
        )
        shapes = (
            {
                input_weight_layer_name: TensorShape(
                    (
                        None,
                        metadata_dict["n_trks"],
                        metadata_dict["n_trk_features"],
                    )
                ),
                input_feat_layer_name: TensorShape(
                    (
                        None,
                        metadata_dict["n_trks"],
                        metadata_dict["n_trk_features"],
                    )
                ),
            },
            {
                edge_weight_layer_name: TensorShape(
                    (
                        None,
                        metadata_dict["n_trks"],
                        metadata_dict["n_edge_y"],
                    )
                ),
                vertex_network_layer_name: TensorShape(
                    (None, metadata_dict["n_vertex_feat"])
                ),
            },
            {
                edge_weight_layer_name: TensorShape(
                    (
                        None,
                        metadata_dict["n_trks"],
                        metadata_dict["n_edge_y"],
                    )
                ),
            },
        )
    elif get_labels and get_inputs and get_weight_labels:
        types = (
            {
                input_weight_layer_name: float32,
                input_feat_layer_name: float32,
            },
            {
                edge_weight_layer_name: float32,
                vertex_network_layer_name: float32,
            },
        )
        shapes = (
            {
                input_weight_layer_name: TensorShape(
                    (
                        None,
                        metadata_dict["n_trks"],
                        metadata_dict["n_trk_features"],
                    )
                ),
                input_feat_layer_name: TensorShape(
                    (
                        None,
                        metadata_dict["n_trks"],
                        metadata_dict["n_trk_features"],
                    )
                ),
            },
            {
                edge_weight_layer_name: TensorShape(
                    (
                        None,
                        metadata_dict["n_trks"],
                        metadata_dict["n_edge_y"],
                    )
                ),
                vertex_network_layer_name: TensorShape(
                    (None, metadata_dict["n_vertex_feat"])
                ),
            },
        )
    elif get_labels and get_inputs:
        types = (
            {
                input_weight_layer_name: float32,
                input_feat_layer_name: float32,
            },
            float32,
        )
        shapes = (
            {
                input_weight_layer_name: TensorShape(
                    (
                        None,
                        metadata_dict["n_trks"],
                        metadata_dict["n_trk_features"],
                    )
                ),
                input_feat_layer_name: TensorShape(
                    (
                        None,
                        metadata_dict["n_trks"],
                        metadata_dict["n_trk_features"],
                    )
                ),
            },
            TensorShape((None, metadata_dict["n_vertex_feat"])),
        )
    elif get_labels:
        types = float32
        shapes = TensorShape((None, metadata_dict["n_vertex_feat"]))
    elif get_inputs:
        types = {
            input_weight_layer_name: float32,
            input_feat_layer_name: float32,
        }
        shapes = {
            input_weight_layer_name: TensorShape(
                (
                    None,
                    metadata_dict["n_trks"],
                    metadata_dict["n_trk_features"],
                )
            ),
            input_feat_layer_name: TensorShape(
                (
                    None,
                    metadata_dict["n_trks"],
                    metadata_dict["n_trk_features"],
                )
            ),
        }
    return types, shapes


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
    def __init__(self, alternative_conf=None):
        if alternative_conf is None:
            global_config_path = (
                f"{pathlib.Path(__file__).parent.absolute()}/general_config.yaml"
            )
        else:
            global_config_path = (
                f"{pathlib.Path(__file__).parent.absolute()}/{alternative_conf}.yaml"
            )
        with open(global_config_path) as global_conf_file:
            global_conf = yaml.load(global_conf_file, Loader=yaml.FullLoader)
            self.track_inputs = global_conf.get("track_inputs")
            self.edge_features = global_conf.get("edge_features")
            self.vertex_features = list(global_conf.get("vertex_features", {}).keys())
            self.vertex_feat_dict = global_conf.get("vertex_features", {})
            self.flavour = global_conf.get("flavour", {})
            for key, fl in self.flavour.items():
                if fl == "None":
                    self.flavour[key] = None
            self.truthOriginLabel = global_conf.get("truthOriginLabel", {})
            self.hadron_cone_excl_label = global_conf.get("HadronConeExclLabel", {})


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
            "output_training",
            "model_name",
            "preprocessing_file_name",
            "scale_dict",
            "training_file_name",
            "validation_file_name",
            "testing_file_name",
            "njets",
            "njets_val",
            "njets_test",
            "used_vertex_properties",
            "input_tracks_name",
            "input_truth_name",
            "input_jet_name",
            "model_files",
            "jet_types",
            "small_net",
            "tracks_name",
            "edge_name",
            "edge_feat_name",
            "vertex_feat_name",
            "epochs",
            "stepsize",
            "lr",
            "use_sample_weights",
            "loss_fac_edge",
            "loss_fac_vert",
            "edge_feature_network",
            "edge_weight_network",
            "vertex_network",
            "evaluation",
            "train_together",
            "file_dirs"
        ]

        req_items = [
            "input",
            "output",
            "output_training",
            "model_name",
        ]

        for item in config_items:
            if item in self.conf:
                setattr(self, item, self.conf[item])
            elif item not in req_items:
                self.set_default(item)
            else:
                raise KeyError(f"You need to specify {item} in your config file")
        self.small_net = self.conf.get("small_net", False)
        self.train_jet_type = self.conf.get("train_jet_type", self.jet_types[0])

    def get_all_input_files(self):
        try:
            return glob(self.input)
        except KeyError:
            raise KeyError("No input file defined.")
    
    def set_default(self, item):
        defaults = {   
            "preprocessing_file_name": "preprocessed_ttbar_topograph.h5",
            "scale_dict": "scale_dict_ttbar.json",
            "training_file_name": "training_ttbar_topographs.h5",
            "validation_file_name": "validation_ttbar_topographs.h5",
            "testing_file_name": "testing_ttbar_topographs.h5",
            "njets": 1_000_000,
            "njets_val": 500_000,
            "njets_test": 50_000,
            "used_vertex_properties": None,
            "input_tracks_name": "tracks_loose",
            "input_truth_name": "truth_hadrons",
            "input_jet_name": "jets",
            "model_files": None,
            "jet_types": ["b"],
            "small_net": {"b": False},
            "tracks_name": "X_train_tracks",
            "edge_name": "Y_edge",
            "edge_feat_name": "Y_edge_features",
            "vertex_feat_name": "Y_vertex_features",
            "epochs": 200,
            "stepsize": 10_000,
            "lr": 0.01,
            "use_sample_weights": True,
            "loss_fac_edge": 1,
            "loss_fac_vert": 1,
            "edge_feature_network": {"nodes": [21, 20, 20, 30], "n_tracks_name": "tracks"},
            "edge_weight_network": {"nodes": [21, 20, 20, 30], "n_tracks_name": "tracks", "add_activation": None},
            "vertex_network": {"nodes": [30, 30, 30, 30, 1]},
            "evaluation": {"model_file_numbers": [199]},
            "train_together": [],
            "file_dirs": {},
        }
        setattr(self, item, defaults[item])


class DatasetCreater:
    def __init__(self, config, input_file, step, stepsize, replace_invalid=False, jet_types=["b"], small_net={"b":False}, load_jet_types=["b"]):
        self.global_conf = GlobalConfig()
        self.config = config
        self.input_file = input_file
        self.step = step
        self.stepsize = stepsize
        self.ind_truthflav = None
        self.replace_invalid = replace_invalid
        self.vertex_features = self.global_conf.vertex_features
        self.ntracks = 40
        self.jet_types = jet_types
        self.load_jet_types = load_jet_types
        self.small_net = small_net
        with File(self.input_file, "r") as f:
            self.truth = f[f"/{self.config.input_truth_name}"][
                self.step * self.stepsize : (self.step + 1) * self.stepsize
            ]
            self.HadrConeTruth = f[f"/{self.config.input_jet_name}"][
                "HadronConeExclExtendedTruthLabelID"
            ][self.step * self.stepsize : (self.step + 1) * self.stepsize]

            # dt = self.truth.dtype
            self.vertex_feat_dtypes = np.array(
                self.truth[self.global_conf.vertex_features].dtype
            )
            self.vertex_feat_dtypes = np.dtype(
                list(self.truth[self.global_conf.vertex_features].dtype.fields.items())
            )
            # self.vertex_feat_dtypes = np.stack(self.vertex_feat_dtypes)
            # self.vertex_feat_dtypes = np.dtype(list(zip(self.vertex_feat_dtypes.names, self.vertex_feat_dtypes["formats"])))
            # self.vertex_feat_dtypes = zip(self.vertex_feat_dtypes["names"], self.vertex_feat_dtypes["formats"])
            
            self.reco = f[f"/{self.config.input_tracks_name}"].fields(self.global_conf.track_inputs)[
                self.step * self.stepsize : (self.step + 1) * self.stepsize, :self.ntracks
            ]
            self.reco_dtypes = self.reco.dtype
            self.reco_jets = f["jets"].fields(["pt", "eventNumber"])[
                self.step * self.stepsize : (self.step + 1) * self.stepsize
            ]
            self.truthOriginLabel = f[f"/{self.config.input_tracks_name}"].fields("truthOriginLabel")[
                self.step * self.stepsize : (self.step + 1) * self.stepsize, :self.ntracks
            ]
            # self.trackExtraTruth = 
            # f["ConeExclFinalLabels"].fields(["pt", "phi"])[
              #
            # self.edge_features = f["/edge_features"][ConeExclFinalLabels
            #     self.step * self.stepsize : (self.step + 1) * self.stepsize, :
            # ]
            self.trackExtra = f[f"/{self.config.input_tracks_name}"].fields(["pt", "dphi"])[
                self.step * self.stepsize : (self.step + 1) * self.stepsize, :self.ntracks
            ]

        self.jet_types_to_save = np.full(shape=self.HadrConeTruth.shape, fill_value=-1)
        self.ind_truthflav_dict = {jet_type: self.get_indeces(jet_type) for jet_type in self.load_jet_types}
        self.ind_truthflav = self.ind_truthflav_dict[self.load_jet_types[0]]
        self.jet_types_to_save[self.ind_truthflav_dict[self.load_jet_types[0]]] = self.jet_types.index(self.jet_types[0])
        if len(self.load_jet_types)>1:
            for jet_type in self.load_jet_types[1:]:
                self.ind_truthflav = np.logical_or(self.ind_truthflav, self.ind_truthflav_dict[jet_type])
                self.jet_types_to_save[self.ind_truthflav_dict[jet_type]] = self.jet_types.index(jet_type)
        self.jet_types_to_save = self.jet_types_to_save[self.ind_truthflav]
        self.truth = self.truth[self.ind_truthflav]
        self.reco = self.reco[self.ind_truthflav]
        self.truthOriginLabel = self.truthOriginLabel[self.ind_truthflav]
        self.trackExtra = self.trackExtra[self.ind_truthflav]
        self.reco_jets = self.reco_jets[self.ind_truthflav]
        self.HadrConeTruth = self.HadrConeTruth[self.ind_truthflav]

    def get_indeces(self, jet_type):
        hadronflavour = self.truth["flavour"]
        hadrconemask = self.check_cases_and_return(
            self.HadrConeTruth,
            self.global_conf.hadron_cone_excl_label[jet_type],
        )
        hflavourmask = [self.check_cases_and_return(
            hf,
            self.global_conf.flavour[jet_type],
        ) for hf in hadronflavour]
        if None in hflavourmask[0]:
            return hadrconemask
        return np.logical_and(
            hadrconemask, np.sum(hflavourmask, axis = 1) == 1
        )

    def get_n_valid_jets(self):
        return sum(self.ind_truthflav)

    def get_jet_types_to_save(self):
        return self.jet_types_to_save

    def get_n_valid_jets_type(self, jet_type):
        try:
            return sum(self.ind_truthflav_dict[jet_type])
        except KeyError:
            return 0

    def get_edge_y(self):
        truthOriginLabel = self.truthOriginLabel
        edges = {}
        for jet_type in self.jet_types:
            edges[jet_type] = self.check_cases_and_return(truthOriginLabel,self.global_conf.truthOriginLabel[jet_type]).astype(int)
        return edges
    
    def get_HadrLabel(self):
        return self.HadrConeTruth

    def get_edge_origin(self):
        return self.truthOriginLabel

    def get_jet_pt(self):
        return self.reco_jets

    def get_extra_track(self):
        return self.trackExtra

    def get_extra_track_truth(self):
        return self.truth["flavour"] #self.trackExtraTruth

    def get_unscaled_pt(self):
        flavour = self.truth["flavour"]
        mask = self.check_cases_and_return(
            flavour,
            self.global_conf.flavour[self.jet_type],
        )
        if None in mask: return np.full(shape=mask.shape[0], fill_value=-999.)
        return self.truth["pt"][mask]

    def get_edge_feat_y(self):
        edge_feat_y = np.array(
            [
                [list(feat_track) for feat_track in feat_jet]
                for feat_jet in self.edge_features[self.global_conf.edge_features]
            ]
        )
        return edge_feat_y
    
    def get_vertex_feat_y(self):
        flavour = self.truth["flavour"]
        vertex_feat = self.truth[self.vertex_features]
        vertex_shape = vertex_feat.shape
        if len(self.vertex_features) > 1:
            shape = vertex_shape[:2]
        else:
            shape = (vertex_shape[0],)
        vert_dtype = vertex_feat.dtype
        vertex_feat_jet_types = {}
        for jet_type in self.jet_types:
            vertex_feat_jet_types[jet_type] = np.full(shape=shape, fill_value=-999., dtype=vert_dtype)
            if not self.small_net[jet_type]:
                mask = self.check_cases_and_return(
                    flavour,
                    self.global_conf.flavour[jet_type],
                )
                jet_mask = np.sum(mask, axis=1)==1
                mask[~jet_mask] = np.full(shape=mask.shape[1], fill_value=False)
                vertex_feat_jet_types[jet_type] = np.full(shape=shape, fill_value=-999., dtype=vert_dtype)

                vertex_feat_jet_types[jet_type][jet_mask] = vertex_feat[mask]
                if None in mask: return np.full(shape=(flavour.shape[0]), fill_value=-999., dtype=vert_dtype)
                for key in self.vertex_features:
                    if self.global_conf.vertex_feat_dict[key]["log"]:
                        vertex_feat_jet_types[jet_type][key][jet_mask] = np.log(vertex_feat_jet_types[jet_type][key][jet_mask])
        return vertex_feat_jet_types

    def get_track_input(self):
        return self.reco
    
    def get_overall_inds(self):
        return self.overall_inds

    def check_cases_and_return(self, data, jettype_inf):
        if isinstance(jettype_inf, int):
            return (data == jettype_inf)
        if isinstance(jettype_inf, list):
            x1 = (data == jettype_inf[0])
            for i in range(1,len(jettype_inf)):
                x1 = np.logical_or(x1, data == jettype_inf[i])
            return x1
        if jettype_inf is None:
            return np.full(shape=data.shape, fill_value=None)


class DataGenerator(IterableDataset):
    def __init__(
        self,
        input: str,
        metadata_dict: dict,
        get_labels: bool = True,
        get_inputs: bool = True,
        get_weight_labels: bool = False,
        get_sample_weights: bool = False,
        stepsize: int = 5000,
        savejets: bool = False,
        savetracks: bool = True,
        track_name: str = "tracks",
        edge_name: str = "edge",
        edge_feat_name: str = "edge_feat",
        vertex_feat_name: str = "vertex_feat",
        n_samples: int = None,
        edge_weight_layer_name: str = "edge_weight",
        edge_feat_layer_name: str = "edge_feat",
        vertex_network_layer_name: str = "vertex_network",
        input_weight_layer_name: str = "input_1",
        input_feat_layer_name: str = "input_2",
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
        IterableDataset.__init__(self)
        self.input = input
        self.get_inputs = get_inputs
        self.get_labels = get_labels
        self.get_weight_labels = get_weight_labels
        self.get_sample_weights = get_sample_weights
        self.metadata_dict = metadata_dict
        self.stepsize = stepsize
        self.savejets = savejets
        self.savetracks = savetracks
        self.track_name = track_name
        self.edge_feat_name = edge_feat_name
        self.edge_name = edge_name
        self.vertex_feat_name = vertex_feat_name
        self.n_samples = n_samples
        self.edge_weight_layer_name = edge_weight_layer_name
        self.edge_feat_layer_name = edge_feat_layer_name
        self.vertex_network_layer_name = vertex_network_layer_name
        self.input_weight_layer_name = input_weight_layer_name
        self.input_feat_layer_name = input_feat_layer_name

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
            if self.get_sample_weights:
                self.sample_weight_batch = np.array(
                    list(map(get_sample_weights, self.edge_batch))
                )

    def get_types_shapes(self):
        return get_types_shapes(
            get_labels=self.get_labels,
            get_inputs=self.get_inputs,
            get_weight_labels=self.get_weight_labels,
            get_sample_weights=self.get_sample_weights,
            input_weight_layer_name=self.input_weight_layer_name,
            input_feat_layer_name=self.input_feat_layer_name,
            edge_weight_layer_name=self.edge_weight_layer_name,
            vertex_network_layer_name=self.vertex_network_layer_name,
            metadata_dict=self.metadata_dict,
        )

    def __iter__(self):
        if self.n_samples is None:
            self.n_samples = self.metadata_dict["n_jets"]
        n_steps = self.n_samples // self.stepsize
        if self.n_samples % self.n_samples != 0:
            n_steps += 1
        for step in range(n_steps):
            self.load_in_memory(step=step)
            if (
                self.get_inputs
                and self.get_labels
                and self.get_weight_labels
                and self.get_sample_weights
            ):
                yield (
                    self.track_batch,
                    self.track_batch,
                    self.edge_batch,
                    self.vertex_feat_batch,
                    self.sample_weight_batch,
                )
            elif self.get_inputs and self.get_labels and self.get_weight_labels:
                yield (
                    self.track_batch,
                    self.track_batch,
                    self.edge_batch,
                    self.vertex_feat_batch,
                )
            elif self.get_inputs and self.get_labels and not self.get_weight_labels:
                yield (self.track_batch, self.track_batch, self.vertex_feat_batch)
            elif self.get_inputs:
                yield (self.track_batch, self.track_batch)
            elif self.get_labels:
                yield self.vertex_feat_batch
            elif self.get_weight_labels:
                yield self.get_weight_labels
