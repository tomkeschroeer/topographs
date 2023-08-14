import time
from glob import glob
from os import makedirs

import matplotlib.pyplot as plt
import numpy as np
import numpy.ma as ma
import torch.optim as optim
from h5py import File
from mlxtend.evaluate import confusion_matrix
from mlxtend.plotting import plot_confusion_matrix
from puma import Histogram, HistogramPlot, PlotBase
from pytorch_lightning.callbacks import ModelSummary
from sklearn.metrics import auc
from torch import device, load, ones_like
from torch.nn import Module
from torch.utils.data import DataLoader

from topograph.modules import (
    GlobalConfig,
    IterableFlavourTaggingDataset,
    Topographs_dataset,
    TopographModel,
    get_logger,
)
from topograph.plotting.plotting_tools import (
    calculate_binary_preds,
    calculate_efficiency,
    calculate_pT_diff,
)


def create_figure(plot):
    plot.set_title()
    plot.set_xlabel()
    plot.set_ylabel(plot.axis_top)
    plot.set_tick_params()
    plot.fig.tight_layout()
    plot.plotting_done = True
    return plot


def get_var_names(var, used_vertex_properties):
    vardict = {
        "pT": "log($p_T$)",
        "eta": "$\eta$",
        "dr": "$\Delta R$"
    }
    if used_vertex_properties is None:
        varlist = np.array(list(vardict.keys()))[:]
    else:
        varlist = np.array(list(vardict.keys()))[used_vertex_properties]
    varlist = list([varlist]) if isinstance(varlist, str) else list(varlist)
    try:
        ind = varlist.index(var)
    except ValueError:
        ind = -1
    return vardict[var], ind


def get_n_bins(dists, var):
    if var in [
        "numberOfPixelHits",
        "numberOfSCTHits",
        "numberOfInnermostPixelLayerHits",
        "numberOfNextToInnermostPixelLayerHits",
        "numberOfInnermostPixelLayerSharedHits",
        "numberOfInnermostPixelLayerSplitHits",
        "numberOfPixelSharedHits",
        "numberOfPixelSplitHits",
        "numberOfSCTSharedHits",
        "numberOfPixelHoles",
        "numberOfSCTHoles",
    ]:
        dist1 = np.unique(dists[0])
        dist2 = np.unique(dists[1])
        ticks = np.around(np.unique(list(dist1) + list(dist2)), 2)
        nbins = len(ticks)
        rangebins = (
            min(ticks) - np.abs(ticks[1] - ticks[0]) / 2,
            max(ticks) + np.abs(ticks[1] - ticks[0]) / 2,
        )
        return nbins, rangebins, ticks
    return 50, None, None


def get_point_styles(N):
    point_styles = [
        "r-",
        "b-",
        "g-",
        "c.",
        "m.",
        "rx",
        "bx",
        "gx",
        "cx",
        "mx",
        "yx",
        "kx",
        "r.",
        "b.",
        "g.",
        "c.",
        "m.",
        "y.",
        "k.",
        "rv",
        "bv",
        "gv",
        "cv",
        "mv",
        "yv",
        "kv",
    ]
    return point_styles[:N]


def get_colours(N):
    colours = [
        "#1f77b4",
        "#ff7f0e",
        "#2ca02c",
        "#7c5295",
        "#1f77b4",
        "#012F51",
        "#ff7f0e",
        "#1f77b4",
        "#B45F06",
        "#A300A3",
        "#38761D",
        "#9ed670",
    ]
    return colours[:N]


def load_topomodel(
    modelfiles=None,
    nodes_feat=[20, 70, 70, 70, 30],
    nodes_weight=[20, 70, 70, 70, 1],
    nodes_vertex=[30, 50, 50, 50, 1],
    activation_name=None,
    small_net={},
):
    if modelfiles is None:
        raise KeyError("Please provide vaild modelfiles.")
    topomodels = {jet_type: TopographModel.load_from_checkpoint(
                        checkpoint_path=modelfile,
                        save_dir="./",
                        name="name",
                        nodes_feat=nodes_feat,
                        nodes_weight=nodes_weight,
                        nodes_vertex=nodes_vertex,
                        activation_name=activation_name,
                        tr_jet_type=jet_type,
                        small_net=small_net
                    )
                    for jet_type, modelfile in modelfiles.items()}
    return topomodels


def get_predictions_and_labels(models, dataset, jet_types, small_net):
    preds, labels, grads = ({},{},{})
    for jet_type in jet_types:
        if not small_net[jet_type]:
            preds[f"preds_v_{jet_type}"] = []
            labels[f"labels_v_{jet_type}"] = [] 
        preds[f"preds_e_{jet_type}"] = []    
        labels[f"labels_e_{jet_type}"] = []
        grads[f"grads_{jet_type}"] = []
    masks = []
    for jet_type in jet_types:
        models[jet_type].eval()
    for sample in dataset:
        inputs, labels_dict, mask, _ = sample
        for jet_type in jet_types:
            tmp_dict = {}
            inputs.requires_grad_()
            tmp_dict["output_v"], tmp_dict["output_e"] = models[jet_type].forward(inputs, mask)
            tmp_dict["output_e"].backward(gradient=ones_like(tmp_dict["output_e"]))
            preds[f"preds_e_{jet_type}"].append(tmp_dict["output_e"].detach().numpy())
            labels[f"labels_e_{jet_type}"].append(labels_dict[f"Y_edge_{jet_type}"].detach().numpy())
            if not small_net[jet_type]:
                preds[f"preds_v_{jet_type}"].append(tmp_dict["output_v"].detach().numpy())
                labels[f"labels_v_{jet_type}"].append(labels_dict[f"Y_vertex_features_{jet_type}"].detach().numpy())
            grad = inputs.grad.data
            grads[f"grads_{jet_type}"].append(grad.detach().numpy())
        masks.append(mask.detach().numpy())
    for jet_type in jet_types:
        shape_labels_e = np.array(labels[f"labels_e_{jet_type}"]).shape
        shape_preds_e = np.array(preds[f"preds_e_{jet_type}"]).shape
        if not small_net[jet_type]:
            shape_preds_v = np.array(preds[f"preds_v_{jet_type}"]).shape
            shape_labels_v = np.array(labels[f"labels_v_{jet_type}"]).shape
        shape_grads = np.array(grads[f"grads_{jet_type}"]).shape
        preds[f"preds_e_{jet_type}"] = np.array(preds[f"preds_e_{jet_type}"]).reshape(
            shape_preds_e[0] * shape_preds_e[1], shape_preds_e[2]
        )  # ,*shape_preds_e[3:])
        labels[f"labels_e_{jet_type}"] = np.array(labels[f"labels_e_{jet_type}"]).reshape(
            shape_labels_e[0] * shape_labels_e[1], shape_labels_e[2]
        )
        if not small_net[jet_type]:
            preds[f"preds_v_{jet_type}"] = np.array(preds[f"preds_v_{jet_type}"]).reshape(
                shape_preds_v[0] * shape_preds_v[1], *shape_preds_v[2:]
            )  # ,*shape_preds_v[3:])
            labels[f"labels_v_{jet_type}"] = np.array(labels[f"labels_v_{jet_type}"]).reshape(
                shape_labels_v[0] * shape_labels_v[1], *shape_labels_v[2:]
            )
        grads[f"grads_{jet_type}"] = np.array(grads[f"grads_{jet_type}"]).reshape(shape_grads[0] * shape_grads[1], *shape_grads[2:])
    shape_masks = np.array(masks).shape
    masks = np.array(masks).reshape(shape_masks[0] * shape_masks[1], *shape_masks[2:])
    return preds, labels, masks, grads


class Plotter:
    def __init__(self, config, cut_val=None, vars=None):
        self.config = config
        self.cut_val = cut_val
        self.jet_types = config.jet_types
        self.small_net = config.small_net
        self.global_config = GlobalConfig(alternative_conf=None)
        if cut_val is not None:
            self.cut_val = cut_val if cut_val <= 1 else cut_val / 100
        self.logger = get_logger()
        self.test_file = (
            f"{self.config.output}/{self.config.testing_file_name}".replace("//", "/")
        )
        self.njet_test = self.config.njets_test
        self.njet_test = 500
        self.file_formats = [self.config.file_formats] if isinstance(self.config.file_formats, str) else self.config.file_formats
        datafilename = (
            "plotting_data_tr.h5"
            if (self.cut_val is None)
            else f"plotting_data_tr_cutval={self.cut_val}.h5"
        )
        self.used_vertex_properties = getattr(
            self.config, "used_vertex_properties", None
        )

        str_vars = ""
        if vars is not None:
            for var in vars:
                str_vars += f"_{var}"

        self.training_output_folder = config.output_training
        self.training_output_folder = (
            self.training_output_folder[:-1]
            if self.training_output_folder[-1] == "/"
            else self.training_output_folder
        )
        self.training_output_folder += str_vars

        self.plot_file = f"{self.training_output_folder}/{datafilename}"
        self.add_activation = self.config.edge_weight_network.get(
            "add_activation", None
        )
        self.model_file_numbers = self.config.evaluation.get("model_file_numbers", [1])

        self.plot_effs = self.config.evaluation.get("plot_efficiency", {}).get(
            "plot", False
        )
        self.plot_effs_zeros = self.config.evaluation.get(
            "plot_efficiency_zeros_only", {}
        ).get("plot", False)
        self.plot_effs_ones = self.config.evaluation.get(
            "plot_efficiency_ones_only", {}
        ).get("plot", False)
        self.plot_pt = self.config.evaluation.get("plot_pt", {}).get("plot", False)
        self.plot_eta = self.config.evaluation.get("plot_eta", {}).get("plot", False)
        self.plot_dr = self.config.evaluation.get("plot_dr", {}).get("plot", False)
        self.plot_loss = self.config.evaluation.get("plot_loss", {}).get("plot", False)

        self.plot_conf_matrix = self.config.evaluation.get("plot_conf_matrix", {}).get(
            "plot", False
        )
        self.plot_preds_per_epoch = self.config.evaluation.get(
            "plot_preds_per_epoch", {}
        ).get("plot", False)
        self.plot_preds_scatter = self.config.evaluation.get(
            "plot_preds_scatter", {}
        ).get("plot", False)
        self.plot_saliency = self.config.evaluation.get("plot_saliency", {}).get(
            "plot", False
        )
        self.plot_saliency_pertrack = self.config.evaluation.get(
            "plot_saliency_pertrack", {}
        ).get("plot", False)
        self.plot_saliency_pervar = self.config.evaluation.get(
            "plot_saliency_pervar", {}
        ).get("plot", False)
        self.plot_n_tracks_per_jet = self.config.evaluation.get(
            "plot_n_tracks_per_jet", {}
        ).get("plot", False)
        self.plot_vertex_labels = self.config.evaluation.get("vertex_labels", {}).get(
            "plot", False
        )
        self.plot_weights = self.config.evaluation.get("weights", {}).get("plot", False)
        self.plot_target_input_corr = self.config.evaluation.get(
            "plot_target_input_corr", {}
        ).get("plot", False)
        self.plot_inputs = self.config.evaluation.get("plot_inputs", {}).get(
            "plot", False
        )
        self.plot_track_origin = self.config.evaluation.get(
            "plot_track_origin", {}
        ).get("plot", False)
        self.plot_roc_curves = self.config.evaluation.get("plot_roc_curves", {}).get(
            "plot", False
        )
        self.plot_hadron_pt = self.config.evaluation.get("plot_hadron_pt", {}).get(
            "plot", False
        )
        self.plot_linear_fit = self.config.evaluation.get("plot_linear_fit", {}).get(
            "plot", False
        )
        self.recalculate_effs = self.config.evaluation.get("plot_efficiency", {}).get(
            "recalculate", False
        )
        self.recalculate_effs_zeros = self.config.evaluation.get(
            "plot_efficiency_zeros_only", {}
        ).get("recalculate", False)
        self.recalculate_effs_ones = self.config.evaluation.get(
            "plot_efficiency_ones_only", {}
        ).get("recalculate", False)
        self.recalculate_pt = self.config.evaluation.get("plot_pt", {}).get(
            "recalculate", True
        )
        self.recalculate_eta = self.config.evaluation.get("plot_eta", {}).get(
            "recalculate", True
        )
        self.recalculate_loss = self.config.evaluation.get("plot_loss", {}).get(
            "recalculate", False
        )
        self.recalculate_preds_scatter = self.config.evaluation.get(
            "plot_preds_scatter", {}
        ).get("recalculate", False)
        self.recalculate_saliency = self.config.evaluation.get("plot_saliency", {}).get(
            "recalculate", False
        )
        self.recalculate_target_input_corr = self.config.evaluation.get(
            "plot_target_input_corr", {}
        ).get("recalculate", False)
        self.recalculate_inputs = self.config.evaluation.get("plot_inputs", {}).get(
            "recalculate", False
        )
        self.plot_dir = f"{self.training_output_folder}/plots"
        self.model_pred_folder = f"{self.training_output_folder}/model_predictions"
        makedirs(self.plot_dir, exist_ok=True)

        self.metadata_dict = {}
        with File(self.test_file, "r") as f:
            (
                _,
                self.metadata_dict["n_trks"],
                self.metadata_dict["n_trk_features"],
            ) = f[f"{self.config.tracks_name}"].shape
        self.n_modelfiles = 0
        # self.effs, self.effs_zeros, self.effs_ones = [],[],[]

    def Run(self):
        nbins_scatter = 100
        self.n_modelfiles = len(glob(f"{self.model_pred_folder}/epoch_pred_*"))

        # 
        with File(self.test_file, "r") as f:
            jet_type_per_jet = f["/jet_type"][:self.njet_test]
        for jet_type in self.jet_types:
            self.get_all_values()
            if self.check_if_recalculate():
                for i in range(self.n_modelfiles):
                    with File(
                        f"{self.model_pred_folder}/epoch_pred_{i:03d}.h5", "r"
                    ) as model_data:
                        preds = model_data[f"pred_edge_{jet_type}"][:self.njet_test]
                        labels = model_data[f"labels_edge_{jet_type}"][:self.njet_test]
                        # grads = model_data["grads"][:self.njet_test]
                        slope, shift, c1, c2 = None, None, None, None
                        if self.add_activation == "shifted_relu":
                            slope = model_data["slope"][()]
                            shift = model_data["shift"][()]
                        elif self.add_activation == "sigmoid":
                            c1 = model_data["c1"][()]
                            c2 = model_data["c2"][()]

                    if self.plot_preds_scatter and self.recalculate_preds_scatter:
                        self.logger.info(
                            "getting predictions for a scatter plot for model"
                            f" model_epoch{i:03d}"
                        )
                        hist, _ = np.histogram(
                            preds,
                            bins=nbins_scatter,
                            range=(self.startpoint_scatter, self.endpoint_scatter),
                        )
                        self.preds_scatter.append(hist)

                    if self.recalculate_effs:
                        self.logger.info(f"getting efficiency for model model_epoch{i:03d}")
                        self.effs = self.get_efficiency(
                            preds=preds,
                            labels=labels,
                            effs=self.effs,
                            slope=slope,
                            shift=shift,
                            c1=c1,
                            c2=c2,
                            pos=i,
                        )
                        if len(self.jet_types)>1:
                            self.effs_only_zero_labels = self.get_efficiency(
                                preds=preds[jet_type_per_jet != self.jet_types.index(jet_type)],
                                labels=labels[jet_type_per_jet != self.jet_types.index(jet_type)],
                                effs=self.effs_only_zero_labels,
                                slope=slope,
                                shift=shift,
                                c1=c1,
                                c2=c2,
                                pos=i,
                            )
                            self.effs_only_ones_labels = self.get_efficiency(
                                preds=preds[jet_type_per_jet == self.jet_types.index(jet_type)],
                                labels=labels[jet_type_per_jet == self.jet_types.index(jet_type)],
                                effs=self.effs_only_ones_labels,
                                slope=slope,
                                shift=shift,
                                c1=c1,
                                c2=c2,
                                pos=i,
                            )


                    if self.recalculate_effs_ones:
                        self.logger.info(
                            f"getting efficiency, ones only, for model model_epoch{i:03d}"
                        )
                        self.effs_ones = self.get_efficiency(
                            preds=preds,
                            labels=labels,
                            effs=self.effs_ones,
                            slope=slope,
                            shift=shift,
                            c1=c1,
                            c2=c2,
                            pos=i,
                            ones_only=True,
                        )

                    if self.recalculate_effs_zeros:
                        self.logger.info(
                            f"getting efficiency, zeros only, for model model_epoch{i:03d}"
                        )
                        self.effs_zeros = self.get_efficiency(
                            preds=preds,
                            labels=labels,
                            effs=self.effs_zeros,
                            slope=slope,
                            shift=shift,
                            c1=c1,
                            c2=c2,
                            pos=i,
                            zeros_only=True,
                        )

                if self.plot_effs:
                    self.logger.info(f"plotting efficiencies...")
                    if self.effs is not None:
                        if len(self.jet_types) > 1:
                            self.plot_vals(
                                ylabel="efficiency",
                                xlabel="epoch",
                                plot_name=f"eff_per_epoch_{jet_type}"
                                if self.cut_val is None
                                else f"eff_per_epoch_cutval={self.cut_val}",
                                vals=[self.effs, self.effs_only_zero_labels, self.effs_only_ones_labels],
                                labels=["all tracks", f"only non-{jet_type} tracks", f"only {jet_type} tracks"],
                                point_styles=get_point_styles(3),
                            )
                        else:
                            self.plot_vals(
                                ylabel="efficiency",
                                xlabel="epoch",
                                plot_name=f"eff_per_epoch_{jet_type}"
                                if self.cut_val is None
                                else f"eff_per_epoch_cutval={self.cut_val}",
                                vals=[self.effs],
                                labels=["all tracks"],
                                point_styles=get_point_styles(1),
                            )

                if self.plot_effs_ones:
                    self.logger.info(f"plotting efficiencies, ones only...")
                    if self.effs_ones is not None:
                        self.plot_vals(
                            ylabel="efficiency",
                            xlabel="epoch",
                            plot_name=f"eff_per_epoch_ones_{jet_type}"
                            if self.cut_val is None
                            else f"eff_per_epoch_ones_cutval={self.cut_val}",
                            vals=[self.effs_ones],
                            labels=[""],
                            point_styles=get_point_styles(1),
                        )

                if self.plot_effs_zeros:
                    self.logger.info(f"plotting efficiencies, zeros only...")
                    if self.effs_zeros is not None:
                        self.plot_vals(
                            ylabel="efficiency",
                            xlabel="epoch",
                            plot_name=f"eff_per_epoch_zeros_{jet_type}"
                            if self.cut_val is None
                            else f"eff_per_epoch_zeros_cutval={self.cut_val}",
                            vals=[self.effs_zeros],
                            labels=[""],
                            point_styles=get_point_styles(1),
                        )
                    # effs_onlys_non

        if self.plot_pt:
            self.logger.info(f"plotting pT...")
            self.plotting_regression_scatter(
                model_file_numbers=self.model_file_numbers, var="pT"
            )

        if self.plot_eta:
            self.logger.info(f"plotting eta...")
            self.plotting_regression_scatter(
                model_file_numbers=self.model_file_numbers, var="eta"
            )

        if self.plot_dr and not self.small_net[jet_type]:
            self.logger.info(f"plotting dr...")
            self.plotting_regression_scatter(
                model_file_numbers=self.model_file_numbers, var="dr"
            )

        if self.plot_conf_matrix:
            self.logger.info("plotting confusion matrix...")
            self.plotting_confusion_matrix(
                model_file_numbers=self.model_file_numbers,
            )

        if self.plot_preds_per_epoch:
            self.logger.info("plotting predictions per epoch...")
            self.plotting_preds_per_epoch(model_file_numbers=self.model_file_numbers)

        if self.plot_preds_scatter:
            self.logger.info(f"plotting predictions in scatter plot...")
            self.plotting_scatter_vals(
                ylabel="predicition",
                xlabel="epoch",
                plot_name="predictions",
                xvals=list(range(0, len(self.preds_scatter))),
                yvals=np.linspace(
                    self.startpoint_scatter,
                    self.endpoint_scatter,
                    num=len(self.preds_scatter[0]),
                    endpoint=True,
                ),
                zvals=self.preds_scatter,
                title="predictions",
            )

        if self.recalculate_preds_scatter:
            self.save_vals(dataset_name="preds_scatter", data=self.preds_scatter)
            self.save_vals(dataset_name="endpoint_scatter", data=self.endpoint_scatter)
            self.save_vals(
                dataset_name="startpoint_scatter", data=self.startpoint_scatter
            )

        if self.plot_saliency:
            self.plotting_saliency_map(model_file_numbers=self.model_file_numbers)

        if self.plot_saliency_pertrack:
            self.plotting_saliency_map_pertrack(model_file_numbers=self.model_file_numbers)

        if self.plot_saliency_pervar:
            self.plotting_saliency_map_pervar(model_file_numbers=self.model_file_numbers)

        if self.plot_vertex_labels and not self.small_net[jet_type]:
            self.plotting_vertex_labels_per_epoch(
                model_file_numbers=self.model_file_numbers
            )

        if self.plot_weights:
            self.plotting_model_weights(model_file_numbers=self.model_file_numbers)

        if self.plot_target_input_corr and not self.small_net[jet_type]:
            self.plotting_target_input_correlation()

        if self.plot_n_tracks_per_jet:
            self.plotting_n_tracks(model_file_numbers=self.model_file_numbers)

        if self.plot_track_origin:
            self.plotting_track_origin()

        if self.plot_inputs:
            self.plotting_input(model_file_numbers=self.model_file_numbers)

        if self.plot_roc_curves:
            self.plotting_roc_curves(model_file_numbers=self.model_file_numbers)

        if self.plot_hadron_pt and not self.small_net[jet_type]:
            self.plotting_hadron_pt_from_tracks()
        
        if self.plot_linear_fit and not self.small_net[jet_type]:
            self.plotting_linear_fit(model_file_numbers=self.model_file_numbers)

        self.plotting_non_jet_tracks_pred(model_file_numbers=self.model_file_numbers)

    def get_all_values(self):
        if self.recalculate_effs is False and self.plot_effs:
            try:
                with File(self.plot_file, "r+") as f:
                    self.effs = f["efficiency"][:self.njet_test]
            except (KeyError, FileNotFoundError) as er:
                self.logger.warn(
                    "No efficiencies found in file or file not found. Recalculate"
                    " instead"
                )
                self.recalculate_effs = True
                self.effs = np.full(
                    shape=(self.n_modelfiles), fill_value=-1.0, dtype=float
                )
                self.effs_only_zero_labels = np.full(
                    shape=(self.n_modelfiles), fill_value=-1.0, dtype=float
                )
                self.effs_only_ones_labels = np.full(
                    shape=(self.n_modelfiles), fill_value=-1.0, dtype=float
                )
        else:
            self.effs = np.full(shape=(self.n_modelfiles), fill_value=-1.0, dtype=float)
            self.effs_only_zero_labels = np.full(
                    shape=(self.n_modelfiles), fill_value=-1.0, dtype=float
                )
            self.effs_only_ones_labels = np.full(
                    shape=(self.n_modelfiles), fill_value=-1.0, dtype=float
                )
        if self.recalculate_effs_ones is False and self.plot_effs_ones:
            try:
                with File(self.plot_file, "r+") as f:
                    self.effs_ones = f["efficiency_ones_only"][:self.njet_test]
            except (KeyError, FileNotFoundError) as er:
                self.logger.warn(
                    "No efficiencies (ones only) found in file or file not found."
                    " Recalculate instead"
                )
                self.recalculate_effs_ones = True
                self.effs_ones = np.full(
                    shape=(self.n_modelfiles), fill_value=-1.0, dtype=float
                )
        else:
            self.effs_ones = np.full(
                shape=(self.n_modelfiles), fill_value=-1.0, dtype=float
            )
        if self.recalculate_effs_zeros is False and self.plot_effs_zeros:
            try:
                with File(self.plot_file, "r+") as f:
                    self.effs_zeros = f["efficiency_zeros_only"][:self.njet_test]
            except (KeyError, FileNotFoundError) as er:
                self.logger.warn(
                    "No efficiencies (zeros only) found in file or file not found."
                    " Recalculate instead"
                )
                self.recalculate_effs_zeros = True
                self.effs_zeros = np.full(
                    shape=(self.n_modelfiles), fill_value=-1.0, dtype=float
                )
        else:
            self.effs_zeros = np.full(
                shape=(self.n_modelfiles), fill_value=-1.0, dtype=float
            )
        if self.recalculate_loss is False and self.plot_loss:
            try:
                with File(self.plot_file, "r+") as f:
                    self.loss = f["loss"][:self.njet_test]
            except (KeyError, FileNotFoundError) as er:
                self.logger.warn(
                    "No loss found in file or file not found. Recalculate instead"
                )
                self.recalculate_loss = True
        else:
            self.loss = []
        if self.recalculate_preds_scatter is False and self.plot_preds_scatter:
            try:
                with File(self.plot_file, "r+") as f:
                    self.preds_scatter = f["preds_scatter"][:self.njet_test]
                    self.endpoint_scatter = f["endpoint_scatter"][()]
                    self.startpoint_scatter = f["startpoint_scatter"][()]
            except (KeyError, FileNotFoundError) as er:
                self.logger.warn(
                    "No loss found in file or file not found. Recalculate instead"
                )
                self.recalculate_loss = True
        else:
            self.preds_scatter = []
            self.endpoint_scatter = 1.0
            self.startpoint_scatter = 0.0

    def check_if_recalculate(self):
        return (
            (self.recalculate_effs)
            or (self.recalculate_effs_zeros)
            or (self.recalculate_effs_ones)
            or (self.recalculate_loss)
            or (self.recalculate_preds_scatter)
            or (self.plot_effs)
            or (self.plot_effs_zeros)
            or (self.plot_effs_ones)
            or (self.plot_preds_scatter)
        )

    def get_efficiency(
        self,
        preds,
        labels,
        effs,
        pos,
        slope=None,
        shift=None,
        c1=None,
        c2=None,
        zeros_only=False,
        ones_only=False,
    ):
        Ntotal = self.metadata_dict["n_trks"] * len(preds)
        # Ntotal = len(preds)
        eff = calculate_efficiency(
            preds,
            labels,
            Ntotal,
            slope,
            shift,
            c1,
            c2,
            zeros_only=zeros_only,
            ones_only=ones_only,
            cut_val=self.cut_val,
        )()
        effs[pos] = eff
        return effs

    def plotting_regression_scatter(self, model_file_numbers, var):
        with File(self.test_file, "r") as f:
            jet_type_per_jet = f["jet_type"][:self.njet_test]
        for model_file_number in model_file_numbers:
            var_str, var_numb = get_var_names(var, self.used_vertex_properties)
            if var_numb == -1:
                self.logger.warning(f"Skipping plotting of {var}, not used in training")
                break
            for jet_type in self.jet_types:
                if self.small_net[jet_type]: 
                    self.logger.warning(f"Skipping plotting of {jet_type}-network, no regression trained.")
                    continue
                self.logger.info(f"plotting {var} regression for model {model_file_number} for {jet_type}-jets...")
                with File(
                    f"{self.model_pred_folder}/epoch_pred_{model_file_number:03d}.h5", "r"
                ) as f:
                    try:
                        preds = f[f"pred_vertex_features_{jet_type}"][:self.njet_test, var_numb]
                    except ValueError:
                        preds = f[f"pred_vertex_features_{jet_type}"][:self.njet_test]
                    try:
                        labels = f[f"labels_vertex_features_{jet_type}"][:self.njet_test, var_numb]
                    except ValueError:
                        labels = f[f"labels_vertex_features_{jet_type}"][:self.njet_test]
                jet_type_int = self.jet_types.index(jet_type)
                mask = (jet_type_per_jet == jet_type_int)
                plot_names = [f"{jet_type}_jets", f"non-{jet_type}_jets"]
                preds_masked = preds[mask]
                labels_masked = labels[mask]
                var_min = np.min(labels_masked[~np.isnan(labels_masked)])
                var_max = np.max(labels_masked[~np.isnan(labels_masked)])
                var_min_pred = np.min(preds_masked[~np.isnan(preds_masked)])
                var_max_pred = np.max(preds_masked[~np.isnan(preds_masked)])
                bins = np.linspace(
                    min(var_min, var_min_pred), max(var_max, var_max_pred), 60
                )
                bins = np.linspace(-2, 2, 60)
                hist = np.histogram2d(preds, labels, bins=[bins, bins])[0]
                self.plotting_scatter_vals(
                    ylabel=f"true {var_str}",
                    xlabel=f"predicted {var_str}",
                    plot_name=f"{var}_regression_model_{model_file_number}_{jet_type}_only_{jet_type}_jets",
                    xvals=bins,
                    yvals=bins,
                    zvals=hist,
                    title=None,
                    y_ticklabels=None,
                    swap_inputs=True,
                )

                regs = calculate_pT_diff(pred=preds_masked, label=labels_masked)()
                var_min = np.min(regs)  # [~np.isnan(regs)])
                var_max = np.max(regs)  # [~np.isnan(regs)])
                var_min_pred = np.min(preds_masked[~np.isnan(preds_masked)])
                var_max_pred = np.max(preds_masked[~np.isnan(preds_masked)])
                bins_x = np.linspace(var_min_pred, var_max_pred, 30)
                bins_x = np.linspace(-2, 2, 30)
                bins_y = np.linspace(var_min, var_max, 30)
                bins_y = np.linspace(-1.5, 1.5, 30)
                # bins = np.linspace(-2,2,30)
                hist_Delta = np.histogram2d(preds_masked, regs, bins=[bins_x, bins_y])[0]
                self.plotting_scatter_vals(
                    ylabel=f"Delta {var_str}",
                    xlabel=f"predicted {var_str}",
                    plot_name=f"Delta_{var}_model_{model_file_number}_{jet_type}_only_{jet_type}_jets",
                    xvals=bins_x,
                    yvals=bins_y,
                    zvals=hist_Delta,
                    title=None,
                    y_ticklabels=None,
                    swap_inputs=True,
                )
                mask_non = (jet_type_per_jet != jet_type_int)
                preds_non_masked = preds[mask_non]
                print(preds_non_masked)
                self.plot_hist(
                    ylabel="number of jets",
                    xlabel="log $p_T$",
                    plot_name=f"{var}_model_{model_file_number}_{jet_type}_only_non-{jet_type}_jets",
                    vals=[preds_non_masked],
                    labels="",
                    colours=get_colours(1),
                    nbins=50,
                    title=f"non-{jet_type} jets",
                    logy=False,
                    norm=False,
                    y_ticklabels=None,
                    x_ticklabels=None,
                    binrange=None,
                )


    def plotting_input(self, model_file_numbers):
        for jet_type in self.jet_types:
            with File(
                f"{self.model_pred_folder}/epoch_pred_{model_file_numbers[0]:03d}.h5", "r"
            ) as f:
                mask = f["mask"][: self.njet_test]
            with File(self.test_file, "r") as f:
                inputs = f[f"X_train_tracks"][: self.njet_test]
                labels = f[f"Y_edge_{jet_type}"][: self.njet_test]
                jet_types = f["jet_type"][: self.njet_test]
            inputs_jettype = inputs[np.logical_and(labels == 1, mask)]
            inputs_nonjettype_all = inputs[np.logical_and(labels == 0, mask)]
            if len(self.jet_types) > 1:
                inputs_nonjettype_fromjettype = inputs[
                    np.logical_and(
                        np.logical_and(labels == 0, mask), 
                        np.repeat((jet_types == self.jet_types.index(jet_type)).reshape(mask.shape[0], 1), mask.shape[1], axis = 1)
                    )
                ]
            input_vars = self.global_config.track_inputs
            for i in range(0, len(input_vars)):
                self.logger.info(f"plotting distribution for {input_vars[i]}")
                dist_jettype = inputs_jettype[:, i]
                dist_nonjettype_all = inputs_nonjettype_all[:, i]
                dists = [dist_jettype, dist_nonjettype_all]
                legend_labels = [f"{jet_type}-tracks", f"non-{jet_type} tracks"]
                if len(self.jet_types) > 1:
                    dist_nonjettype_fromjettype = inputs_nonjettype_fromjettype[:, i]
                    dists.append(dist_nonjettype_fromjettype)
                    legend_labels.append(f"non-{jet_type} tracks from {jet_type}-jets")
                nbins, binrange, ticks = get_n_bins(dists=dists, var=input_vars[i])
                self.plot_hist(
                    ylabel="normalised number of tracks",
                    xlabel=input_vars[i],
                    plot_name=f"Distr_{input_vars[i]}_{jet_type}",
                    colours=get_colours(len(legend_labels)),
                    vals=dists,
                    labels=legend_labels,
                    nbins=nbins,
                    binrange=binrange,
                    x_ticklabels=ticks,
                    logy=False,
                )

    def plotting_roc_curves(self, model_file_numbers):
        self.logger.info("plotting roc curves...")
        for model_file_number in model_file_numbers:
            for jet_type in self.jet_types:
                self.logger.info(f"plotting roc curve for model {model_file_number} and {jet_type}-jets")
                with File(
                    f"{self.model_pred_folder}/epoch_pred_{model_file_number:03d}.h5", "r"
                ) as f:
                    labels = f[f"labels_edge_{jet_type}"][: self.njet_test]
                    preds = f[f"pred_edge_{jet_type}"][: self.njet_test]
                with File(f"{self.test_file}", "r") as f:
                    edge_origin = f["edge_origin"][: self.njet_test]
                Pos = preds[labels == 1]
                Neg = preds[labels == 0]
                Pos_b = preds[np.logical_and(labels == 1, edge_origin == 3)]
                Neg_b = preds[np.logical_or(labels == 0, edge_origin == 4)]
                Pos_c = preds[np.logical_and(labels == 1, edge_origin == 4)]
                Neg_c = preds[np.logical_or(labels == 0, edge_origin == 3)]
                percentages = np.linspace(start=0, stop=100, num=100, endpoint=True)
                cut_vals_tpr = np.percentile(Pos, 100 - percentages)
                epsilon = 1e-3
                fpr = [sum(Neg > cut_val_tpr) / len(Neg) for cut_val_tpr in cut_vals_tpr]
                tpr = [sum(Pos > cut_val_tpr) / len(Pos) for cut_val_tpr in cut_vals_tpr]
                fpr_b = [
                    sum(Neg_b > cut_val_tpr) / len(Neg_b) for cut_val_tpr in cut_vals_tpr
                ]
                tpr_b = [
                    sum(Pos_b > cut_val_tpr) / len(Pos_b) for cut_val_tpr in cut_vals_tpr
                ]
                fpr_c = [
                    sum(Neg_c > cut_val_tpr) / len(Neg_c) for cut_val_tpr in cut_vals_tpr
                ]
                tpr_c = [
                    sum(Pos_c > cut_val_tpr) / len(Pos_c) for cut_val_tpr in cut_vals_tpr
                ]
                auc_val = np.round(auc(fpr, tpr), 2)
                auc_val_b = np.round(auc(fpr_b, tpr_b), 2)
                auc_val_c = np.round(auc(fpr_c, tpr_c), 2)
                plot, plotnames = self.plot_vals(
                    ylabel="TPR",
                    xlabel="FPR",
                    plot_name=f"ROC_curve_{jet_type}",
                    vals=[[fpr, tpr], [fpr_b, tpr_b], [fpr_c, tpr_c]],
                    labels=["b and bc", "b", "bc"],
                    title=(
                        f"ROC curve, AUC = {auc_val}, AUC_b = {auc_val_b}, AUC_bc ="
                        f" {auc_val_c}"
                    ),
                    y_values_given=True,
                    point_styles=get_point_styles(3),
                    return_plot=True,
                )
                plot.axis_top.plot([0, 1], [1, 1], "b", linestyle="dashed")
                plot.axis_top.plot([0, 0], [0, 1], "b", linestyle="dashed")
                for plotname in plotnames:
                    plot.savefig(plotname)
    
    def plotting_linear_fit(self, model_file_numbers):
        for model_file_number in model_file_numbers:
            for jet_type in self.jet_types:
                with File(
                    f"{self.model_pred_folder}/epoch_pred_{model_file_number:03d}.h5", "r"
                ) as f:
                    labels = f[f"labels_vertex_features_{jet_type}"][: self.njet_test]
                    preds = f[f"pred_vertex_features_{jet_type}"][: self.njet_test]
                for i in range(len(self.global_config.vertex_features)):
                    self.logger.info(f"plotting distance to linear regression for model {model_file_number} and variable {self.global_config.vertex_features[i]} for {jet_type}-jets")
                    pred_var = preds[:,i]
                    labels_var = labels[:,i]
                    pred_var = pred_var[labels_var != -999.]
                    labels_var = labels_var[labels_var != -999.]
                    bounds = min([-labels_var.min(), labels_var.max()])
                    var_bins = np.linspace(-bounds,bounds, 11)
                    slope, offset = np.polyfit(pred_var, labels_var, deg=1)
                    dist = slope*var_bins+offset - var_bins
                    slope_dist = np.round(dist[1]-dist[0]/(var_bins[1]-var_bins[0]),4)
                    offset_dist = np.round(dist[0]-slope_dist*var_bins[0],4)
                    self.plot_vals(
                        ylabel="mean of distance to linear fit",
                        xlabel=f"true {jet_type}-hadron {self.global_config.vertex_features[i]}",
                        plot_name=f"linear_fit_distance_model_{model_file_number}_{self.global_config.vertex_features[i]}_{jet_type}",
                        vals=[[var_bins, dist]],
                        labels=["$f(x^{true}) = $"+f"{slope_dist}" + "$x^{true} + $" + f"{offset_dist}"],
                        point_styles=["bo"],
                        y_values_given=True,
                        legend_loc="lower right"
                    )


    def plotting_track_origin(self):
        test_file = f"{self.config.output}/{self.config.testing_file_name}".replace(
            "//", "/"
        )
        for jet_type in self.jet_types:
            self.logger.info(f"plot the origin of tracks for {jet_type}-jets...")
            with File(test_file, "r") as test:
                edge_origin = test["edge_origin"][: self.config.njets_test]
                edge_label = test[f"Y_edge_{jet_type}"][: self.config.njets_test]
            track_origin = [
                "pile-up",
                "Fake",
                "Primary",
                "FromB",
                "FromBC",
                "FromC",
                "FromTau",
                "Oth. 2nd",
            ]

            jettype_tracks = edge_origin[np.logical_and(edge_label == 1, edge_origin != -1)]
            non_jettype_tracks = edge_origin[np.logical_and(edge_label == 0, edge_origin != -1)]

            self.plot_hist(
                ylabel="normalised number of tracks",
                xlabel="track origin",
                plot_name=f"origin_labels_{jet_type}",
                vals=[jettype_tracks, non_jettype_tracks],
                labels=[f"{jet_type}-tracks", f"non-{jet_type} tracks"],
                title="track origins",
                logy=False,
                norm=True,
                nbins=8,
                binrange=(-0.5, 7.5),
                x_ticklabels=track_origin,
                colours=get_colours(2),
            )

    def plotting_regression(self, model_file_numbers, var):
        for model_file_number in model_file_numbers:
            self.logger.info(f"plotting {var} regression for model {model_file_number}")
            var_str, var_numb = get_var_names(var, self.used_vertex_properties)
            if var_numb == -1:
                self.logger.warning(f"Skipping plotting of {var}, not used in training")
                break
            with File(
                f"{self.model_pred_folder}/epoch_pred_{model_file_number:03d}.h5", "r"
            ) as f:
                try:
                    preds = f["pred_vertex_features"][:, var_numb]
                    labels = f["labels_vertex_features"][:, var_numb]
                except ValueError:
                    preds = f["pred_vertex_features"][:self.njet_test]
                    labels = f["labels_vertex_features"][:self.njet_test]
            var_min = np.min(labels[~np.isnan(labels)])
            var_max = np.max(labels[~np.isnan(labels)])
            var_min_pred = np.min(preds[~np.isnan(preds)])
            var_max_pred = np.max(preds[~np.isnan(preds)])
            plot_var = PlotBase(
                ylabel=f"predicted {var_str}",
                xlabel=f"true {var_str}",
                n_ratio_panels=0,
                logy=False,
                ymin=var_min_pred,
                ymax=var_max_pred,
            )
            plot_var.initialise_figure()
            plot_var.axis_top.plot(labels, preds, "b.")
            plot_var.axis_top.plot([var_min, var_max], [var_min, var_max], "r-")
            plot_var.set_y_lim()
            plot_var = create_figure(plot=plot_var)
            for file_format in self.file_formats:
                plot_var.savefig(
                    f"{self.plot_dir}/{var}_regression_model_{model_file_number}.{file_format}"
                )

            plot_var_diff = PlotBase(
                ylabel=f"Delta {var_str}",
                xlabel=f"predicted {var_str}",
                n_ratio_panels=0,
                logy=False,
            )

            regs = calculate_pT_diff(pred=preds, label=labels)()
            plot_var_diff.initialise_figure()
            plot_var_diff.axis_top.plot(labels, regs, "b.")
            plot_var_diff.axis_top.plot([var_min, var_max], [0, 0], "r-")
            plot_var_diff = create_figure(plot=plot_var_diff)
            for file_format in self.file_formats:
                plot_var_diff.savefig(
                    f"{self.plot_dir}/Delta_{var}_model_{model_file_number}.{file_format}"
                )

    def plotting_n_tracks(self, model_file_numbers):
        with File(
                f"{self.model_pred_folder}/epoch_pred_{model_file_numbers[0]:03d}.h5", "r"
            ) as f:
            mask_inp_all = f["mask"][: self.njet_test]
        with File(self.test_file, "r") as f:
            tracks_extra_all = f["track_extra"][: self.njet_test]
            edge_origin_all = f["edge_origin"][: self.njet_test]
        for jet_type in self.jet_types:
            self.logger.info(f"plotting number of tracks, {jet_type}-jets...")
            with File(self.test_file, "r") as f:
                labels_e = f[f"Y_edge_{jet_type}"][: self.njet_test]
                mask_jettype = f["jet_type"][:self.njet_test] == self.jet_types.index(jet_type)
                # tracks_extra = tracks_extra_all[mask_jettype]
                # edge_origin = edge_origin_all[mask_jettype]
                labels_e = labels_e[mask_jettype]
                mask_inp = mask_inp_all[mask_jettype]
                # inputs = f[f"X_train_tracks"][: self.njet_test][mask_jettype]
            n_jettype_tracks = np.sum(labels_e, axis=1)
            n_tracks = np.sum(mask_inp, axis=1)
            n_non_jettype_tracks = n_tracks - n_jettype_tracks
            dists = [n_tracks, n_jettype_tracks, n_non_jettype_tracks]
            legend_labels = ["all tracks", f"{jet_type}-tracks", f"non-{jet_type} tracks"]
            self.plot_hist(
                ylabel=f"number of {jet_type}-jets",
                xlabel="number of tracks",
                vals=dists,
                labels=legend_labels,
                nbins=int(max(n_tracks)) + 1,
                binrange=(-0.5, max(n_tracks) + 0.5),
                plot_name=f"number_of_tracks_{jet_type}",
                colours=get_colours(len(legend_labels)),
                norm=False,
                logy=True,
            )

    def plotting_confusion_matrix(self, model_file_numbers):
        with File(self.test_file, "r") as f:
            jet_types_per_jet = f["jet_type"][:self.njet_test]
        for model_file_number in model_file_numbers:
            self.logger.info(f"plotting confusion matrix for model {model_file_number}")
            for jet_type in self.jet_types:
                with File(
                    f"{self.model_pred_folder}/epoch_pred_{model_file_number:03d}.h5", "r"
                ) as f:
                    preds = f[f"pred_edge_{jet_type}"][:self.njet_test]
                    labels = f[f"labels_edge_{jet_type}"][:self.njet_test]
                    if len(self.jet_types) > 1:
                        jet_type_int = self.jet_types.index(jet_type)
                        preds_ones = preds[jet_types_per_jet == jet_type_int].flatten()
                        labels_ones = labels[jet_types_per_jet == jet_type_int].flatten()
                        preds_zeros = preds[jet_types_per_jet != jet_type_int].flatten()
                        labels_zeros = labels[jet_types_per_jet != jet_type_int].flatten()
                    preds=preds.flatten()
                    labels=labels.flatten()
                preds = calculate_binary_preds(
                    preds=preds
                )()
                if len(self.jet_types) > 1:
                    preds_ones = calculate_binary_preds(
                        preds=preds_ones
                    )()
                    preds_zeros = calculate_binary_preds(
                        preds=preds_zeros
                    )()
                conf_mat = confusion_matrix(labels, preds, binary=True)
                plot_confusion_matrix(
                    conf_mat=conf_mat,
                    colorbar=True,
                    show_normed=True,
                    show_absolute=True,
                    class_names=[0, 1],
                )
                plt.tight_layout()
                for file_format in self.file_formats:
                    plt.savefig(f"{self.plot_dir}/conf_matrix_model_{model_file_number}_{jet_type}.{file_format}")
                plt.close()
                if len(self.jet_types) > 1:
                    conf_mat_ones = confusion_matrix(labels_ones, preds_ones, binary=True)
                    plot_confusion_matrix(
                        conf_mat=conf_mat_ones,
                        colorbar=True,
                        show_normed=True,
                        show_absolute=True,
                        class_names=[0, 1],
                    )
                    plt.tight_layout()
                    for file_format in self.file_formats:
                        plt.savefig(f"{self.plot_dir}/conf_matrix_model_{model_file_number}_ones_{jet_type}.{file_format}")
                    plt.close()
                    conf_mat_zeros = confusion_matrix(labels_zeros, preds_zeros, binary=True)
                    plot_confusion_matrix(
                        conf_mat=conf_mat_zeros,
                        colorbar=True,
                        show_normed=True,
                        show_absolute=True,
                        class_names=[0, 1],
                    )
                    plt.tight_layout()
                    for file_format in self.file_formats:
                        plt.savefig(f"{self.plot_dir}/conf_matrix_model_{model_file_number}_zeros_{jet_type}.{file_format}")
                    plt.close()

    def plotting_preds_per_epoch(self, model_file_numbers):
        for model_file_number in model_file_numbers:
            for jet_type in self.jet_types:
                self.logger.info(
                    f"plotting predictions per epoch for model {model_file_number} for {jet_type}-jets."
                )
                with File(
                    f"{self.model_pred_folder}/epoch_pred_{model_file_number:03d}.h5", "r"
                ) as f:
                    preds = f[f"pred_edge_{jet_type}"][:self.njet_test].flatten()
                    labels = f[f"labels_edge_{jet_type}"][:self.njet_test].flatten()
                nbins = 50
                binrange = (min(labels), max(labels))
                legend_labels = ["$b$ tracks", "non-$b$ tracks"]
                preds_one = preds[labels == 1]
                preds_zeros = preds[labels == 0]
                self.plot_hist(
                    ylabel="normalised number of tracks",
                    xlabel="prediction",
                    vals=[preds_one, preds_zeros],
                    labels=legend_labels,
                    colours=get_colours(len(legend_labels)),
                    title=f"predictions for epoch {model_file_number}",
                    nbins=nbins,
                    binrange=binrange,
                    plot_name=f"predicitions_split_epoch_{model_file_number}_{jet_type}",
                )

    def plotting_non_jet_tracks_pred(self, model_file_numbers):
        with File(self.test_file, "r") as f:
            jet_type_per_jet = f["jet_type"][:self.njet_test]
        point_styles = ["b_","r_", "g_", "c_", "m_"]
        for model_file_number in model_file_numbers:
            for jet_type in self.jet_types:
                jet_type_int = self.jet_types.index(jet_type)
                self.logger.info(
                    f"plotting track predictions for non-{jet_type} tracks for model {model_file_number}."
                )
                with File(
                    f"{self.model_pred_folder}/epoch_pred_{model_file_number:03d}.h5", "r"
                ) as f:
                    preds = f[f"pred_edge_{jet_type}"][:self.njet_test]
                    labels = f[f"labels_edge_{jet_type}"][:self.njet_test]
                    nbins = len(preds[0])
                    ntracks = len(preds)
                    binrange=(0,nbins)
                vals = []
                legend_labels = []
                for pred_jet_type in self.jet_types:
                    if pred_jet_type != jet_type:
                        pred_jet_type_int = self.jet_types.index(pred_jet_type)
                        preds_j = np.stack(
                            calculate_binary_preds(
                                preds=preds[jet_type_per_jet==pred_jet_type_int]
                            )(),
                            axis=1
                        )
                        preds_j = np.sum(preds_j, axis = 1).astype(float)
                        preds_j /= ntracks
                        vals.append(preds_j)
                        legend_labels.append(f"{pred_jet_type}-jets")
                self.plot_vals(
                    ylabel="normalised number of unblocked tracks",
                    xlabel="track number",
                    vals=vals,
                    labels=legend_labels,
                    # colours=get_colours(len(legend_labels)),
                    title=f"non {jet_type}-jets predictions for epoch {model_file_number}",
                    # nbins=nbins,
                    # binrange=binrange,
                    plot_name=f"non_jet_track_plots_model_{model_file_number}_{jet_type}",
                    point_styles=point_styles[:len(legend_labels)],
                )

    def plotting_vertex_labels_per_epoch(self, model_file_numbers):
        for model_file_number in model_file_numbers:
            for jet_type in self.jet_types:
                self.logger.info(
                    f"plotting predictions per epoch for model {model_file_number} and jet type {jet_type}"
                )
                with File(
                    f"{self.model_pred_folder}/epoch_pred_{model_file_number:03d}.h5", "r"
                ) as f:
                    labels = f[f"labels_vertex_features_{jet_type}"][:self.njet_test].flatten()
                nbins = 50
                binrange = (min(labels), max(labels))
                self.plot_hist(
                    ylabel="normalised number of tracks",
                    xlabel="prediction",
                    vals=[labels],
                    labels=[""],
                    colours=get_colours(1),
                    title=f"labels for epoch {model_file_number}, {jet_type}-jets",
                    nbins=nbins,
                    binrange=binrange,
                    plot_name=f"labels_split_epoch_{model_file_number}_{jet_type}",
                )

    def plotting_saliency_map(self, model_file_numbers):
            with File(self.test_file, "r") as f:
                jet_types = f[f"jet_type"][:self.njet_test]
            for model_file_number in model_file_numbers:
                for jet_type in self.jet_types:
                    self.logger.info(f"plotting saliency map for model {model_file_number} for {jet_type}-jets")
                    with File(
                        f"{self.model_pred_folder}/epoch_pred_{model_file_number:03d}.h5", "r"
                    ) as f:
                        grads = f[f"gradients_{jet_type}"][:self.njet_test]
                        grads_mask = f["mask"][:self.njet_test]
                        grads_shape = grads.shape
                        labels_edge = f[f"labels_edge_{jet_type}"][:self.njet_test]
                    rep_grad_mask = (
                        np.array(np.repeat(grads_mask, grads_shape[-1], axis=-1))
                        .astype(bool)
                        .reshape(grads_shape)
                    )
                    rep_grad_mask_jettype = (
                        np.repeat(
                            np.logical_and(grads_mask, labels_edge == 1),
                            grads_shape[-1],
                            axis=-1,
                        )
                        .astype(bool)
                        .reshape(grads.shape)
                    )
                    rep_grad_mask_nonjettype = (
                        np.repeat(
                            np.logical_and(grads_mask, labels_edge == 0),
                            grads_shape[-1],
                            axis=-1,
                        )
                        .astype(bool)
                        .reshape(grads.shape)
                    )

                    grads_all = ma.array(grads, mask=~rep_grad_mask).mean(axis=1)
                    grads_jettype = ma.array(grads, mask=~rep_grad_mask_jettype).mean(axis=1)
                    grads_nonjettype = ma.array(grads, mask=~rep_grad_mask_nonjettype).mean(axis=1)

                    jettype_mask = (jet_types == self.jet_types.index(jet_type))
                    grads_nonjettype_shape = grads_nonjettype.shape
                    grads_nonjettype_onlyjetttype = grads[jettype_mask]
                    grads_onlyjettype_mask = grads_mask[jettype_mask]
                    rep_grad_mask_nonjettype = (
                        np.repeat(
                            np.logical_and(grads_onlyjettype_mask, labels_edge[jettype_mask] == 0),
                            grads_nonjettype_shape[-1],
                            axis=-1
                        )
                    )        
                    grads_nonjettype_onlyjettype = ma.array(grads_nonjettype_onlyjetttype, mask=~rep_grad_mask_nonjettype).mean(axis=1)

                    track_vars = list(range(len(self.global_config.track_inputs)))

                    if len(track_vars) != grads_shape[-1]:
                        self.logger.warning(
                            "Number of track variables is not the same as the one indicated by"
                            " the saved gradients. Only use the numbers of variables as y-axis"
                        )
                        track_vars = list(range(grads_shape[-1]))

                    sal_bins_all = np.linspace(
                        min(grads_all.flatten()), max(grads_all.flatten()), num=25
                    )
                    hists_all = [
                        np.histogram(
                            grads_all[:, i][grads_all[:, i] != 0.0], bins=sal_bins_all
                        )[0]
                        / len(grads_all[:, i])
                        for i in track_vars
                    ]
                    minimal_perc = np.concatenate(
                        np.array([np.argwhere(hist > 0.1).flatten() for hist in hists_all])
                    )
                    minimum = min(minimal_perc)
                    maximum = max(minimal_perc)
                    # sal_bins_all = np.linspace(minimum, maximum, num=25)
                    grads_all[grads_all < sal_bins_all[minimum]] = sal_bins_all[minimum]
                    grads_all[grads_all > sal_bins_all[maximum]] = sal_bins_all[maximum]
                    sal_bins_all = np.linspace(
                        min(grads_all.flatten()), max(grads_all.flatten()), num=25
                    )
                    # sal_bins_all = np.linspace(-0.1,0.01,50)
                    hists_all = [
                        np.histogram(grads_all[:, i], bins=sal_bins_all)[0]
                        / len(grads_all[:, i])
                        for i in track_vars
                    ]
                    self.plotting_scatter_vals(
                        ylabel="input variable",
                        xlabel="gradient",
                        xvals=sal_bins_all[1:] - (sal_bins_all[1:] - sal_bins_all[0:-1]) / 2,
                        yvals=track_vars,
                        zvals=np.stack((hists_all)),
                        plot_name=f"saliency_map_alltracks_model_{model_file_number:03d}",
                        y_ticklabels=self.global_config.track_inputs,
                        swap_inputs=False,
                    )

                    sal_bins_jettype = np.linspace(
                        min(grads_jettype.flatten()), max(grads_jettype.flatten()), num=25
                    )
                    hists_jettype = [
                        np.histogram(grads_jettype[:, i][grads_jettype[:, i] != 0.0], bins=sal_bins_jettype)[0]
                        / len(grads_jettype[:, i])
                        for i in track_vars
                    ]
                    minimal_perc = np.concatenate(
                        np.array([np.argwhere(hist > 0.1).flatten() for hist in hists_jettype])
                    )
                    minimum = min(minimal_perc)
                    maximum = max(minimal_perc)
                    # sal_bins_b = np.linspace(minimum, maximum, num=25)
                    grads_jettype[grads_jettype < sal_bins_jettype[minimum]] = sal_bins_jettype[minimum]
                    grads_jettype[grads_jettype > sal_bins_jettype[maximum]] = sal_bins_jettype[maximum]
                    sal_bins_jettype = np.linspace(
                        min(grads_jettype.flatten()), max(grads_jettype.flatten()), num=25
                    )
                    self.plotting_scatter_vals(
                        ylabel="input variable",
                        xlabel="gradient",
                        xvals=sal_bins_jettype[1:] - (sal_bins_jettype[1:] - sal_bins_jettype[0:-1]) / 2,
                        yvals=track_vars,
                        zvals=np.stack((hists_jettype)),
                        plot_name=f"saliency_map_{jet_type}_tracks_model_{model_file_number:03d}",
                        y_ticklabels=self.global_config.track_inputs,
                        swap_inputs=False,
                    )

                    sal_bins_nonjettype = np.linspace(
                        min(grads_nonjettype.flatten()), max(grads_nonjettype.flatten()), num=25
                    )
                    hists_nonjettype = [
                        np.histogram(
                            grads_nonjettype[:, i][grads_nonjettype[:, i] != 0.0], bins=sal_bins_nonjettype
                        )[0]
                        / len(grads_nonjettype[:, i])
                        for i in track_vars
                    ]
                    minimal_perc = np.concatenate(
                        np.array([np.argwhere(hist > 0.1).flatten() for hist in hists_nonjettype])
                    )
                    minimum = min(minimal_perc)
                    maximum = max(minimal_perc)
                    # sal_bins_nonb = np.linspace(minimum, maximum, num=25)
                    grads_nonjettype[grads_nonjettype < sal_bins_nonjettype[minimum]] = sal_bins_nonjettype[minimum]
                    grads_nonjettype[grads_nonjettype > sal_bins_nonjettype[maximum]] = sal_bins_nonjettype[maximum]
                    sal_bins_nonjettype = np.linspace(
                        min(grads_nonjettype.flatten()), max(grads_nonjettype.flatten()), num=25
                    )
                    self.plotting_scatter_vals(
                        ylabel="input variable",
                        xlabel="gradient",
                        xvals=sal_bins_nonjettype[1:] - (sal_bins_nonjettype[1:] - sal_bins_nonjettype[0:-1]) / 2,
                        yvals=track_vars,
                        zvals=np.stack((hists_nonjettype)),
                        plot_name=f"saliency_map_non_{jet_type}_tracks_model_{model_file_number:03d}",
                        y_ticklabels=self.global_config.track_inputs,
                        swap_inputs=False,
                    )

                    sal_bins_nonjettype_onlyjettype = np.linspace(
                        min(grads_nonjettype_onlyjettype.flatten()), max(grads_nonjettype_onlyjettype.flatten()), num=25
                    )
                    hists_nonjettype_onlyjettype = [
                        np.histogram(
                            grads_nonjettype_onlyjettype[:, i][grads_nonjettype_onlyjettype[:, i] != 0.0], bins=sal_bins_nonjettype_onlyjettype
                        )[0]
                        / len(grads_nonjettype_onlyjettype[:, i])
                        for i in track_vars
                    ]
                    minimal_perc = np.concatenate(
                        np.array([np.argwhere(hist > 0.1).flatten() for hist in hists_nonjettype_onlyjettype])
                    )
                    minimum = min(minimal_perc)
                    maximum = max(minimal_perc)
                    # sal_bins_nonb = np.linspace(minimum, maximum, num=25)
                    grads_nonjettype_onlyjettype[grads_nonjettype_onlyjettype < sal_bins_nonjettype_onlyjettype[minimum]] = sal_bins_nonjettype_onlyjettype[minimum]
                    grads_nonjettype_onlyjettype[grads_nonjettype_onlyjettype > sal_bins_nonjettype_onlyjettype[maximum]] = sal_bins_nonjettype_onlyjettype[maximum]
                    sal_bins_nonjettype_onlyjettype = np.linspace(
                        min(grads_nonjettype_onlyjettype.flatten()), max(grads_nonjettype_onlyjettype.flatten()), num=25
                    )
                    self.plotting_scatter_vals(
                        ylabel="input variable",
                        xlabel="gradient",
                        xvals=sal_bins_nonjettype_onlyjettype[1:] - (sal_bins_nonjettype_onlyjettype[1:] - sal_bins_nonjettype_onlyjettype[0:-1]) / 2,
                        yvals=track_vars,
                        zvals=np.stack((hists_nonjettype_onlyjettype)),
                        plot_name=f"saliency_map_non_{jet_type}_only_{jet_type}_jets_tracks_model_{model_file_number:03d}",
                        y_ticklabels=self.global_config.track_inputs,
                        swap_inputs=False,
                    )

    def plotting_saliency_map_pertrack(self, model_file_numbers):
        for model_file_number in model_file_numbers:
            for jet_type in self.jet_types:
                self.logger.info(
                    f"plotting saliency map per track for model {model_file_number} for {jet_type}"
                )
                with File(
                    f"{self.model_pred_folder}/epoch_pred_{model_file_number:03d}.h5", "r"
                ) as f:
                    grads = f[f"gradients_{jet_type}"][:self.njet_test]
                    grads_mask = f["mask"][:self.njet_test]
                    grads_shape = grads.shape
                    labels_edge = f[f"labels_edge_{jet_type}"][:self.njet_test]
                rep_grad_mask = (
                    np.array(np.repeat(grads_mask, grads_shape[-1], axis=-1))
                    .astype(bool)
                    .reshape(grads_shape)
                )
                rep_grad_mask_b = (
                    np.repeat(
                        np.logical_and(grads_mask, labels_edge == 1),
                        grads_shape[-1],
                        axis=-1,
                    )
                    .astype(bool)
                    .reshape(grads.shape)
                )
                rep_grad_mask_nonb = (
                    np.repeat(
                        np.logical_and(grads_mask, labels_edge == 0),
                        grads_shape[-1],
                        axis=-1,
                    )
                    .astype(bool)
                    .reshape(grads.shape)
                )
                grads_all = ma.array(grads, mask=~rep_grad_mask).mean(axis=0)
                grads_jettype = ma.array(grads, mask=~rep_grad_mask_b).mean(axis=0)
                grads_nonjettype = ma.array(grads, mask=~rep_grad_mask_nonb).mean(axis=0)
                track_vars = list(range(len(self.global_config.track_inputs)))
                if len(track_vars) != grads_shape[-1]:
                    self.logger.warning(
                        "Number of track variables is not the same as the one indicated by"
                        " the saved gradients. Only use the numbers of variables as y-axis"
                    )
                    track_vars = list(range(grads_shape[-1]))
                ntracks = 8
                sal_bins = np.linspace(0, ntracks, ntracks + 1)
                self.plotting_scatter_vals(
                    ylabel="input variable",
                    xlabel="tracks",
                    xvals=sal_bins[1:] - (sal_bins[1:] - sal_bins[0:-1]) / 2,
                    yvals=track_vars,
                    zvals=grads_all[:ntracks],  # np.stack((hists)),
                    plot_name=(
                        f"saliency_map_alltracks_pertrack_model_{model_file_number:03d}_{jet_type}"
                    ),
                    y_ticklabels=self.global_config.track_inputs,
                    swap_inputs=True,
                )
                self.plotting_scatter_vals(
                    ylabel="input variable",
                    xlabel="tracks",
                    xvals=sal_bins[1:] - (sal_bins[1:] - sal_bins[0:-1]) / 2,
                    yvals=track_vars,
                    zvals=grads_jettype[:ntracks],  # np.stack((hists)),
                    plot_name=(
                        f"saliency_map_btracks_pertrack_model_{model_file_number:03d}_{jet_type}"
                    ),
                    y_ticklabels=self.global_config.track_inputs,
                    swap_inputs=True,
                )
                self.plotting_scatter_vals(
                    ylabel="input variable",
                    xlabel="tracks",
                    xvals=sal_bins[1:] - (sal_bins[1:] - sal_bins[0:-1]) / 2,
                    yvals=track_vars,
                    zvals=grads_nonjettype[:ntracks],  # np.stack((hists)),
                    plot_name=(
                        f"saliency_map_nonbtracks_pertrack_model_{model_file_number:03d}_{jet_type}"
                    ),
                    y_ticklabels=self.global_config.track_inputs,
                    swap_inputs=True,
                )

    def plotting_hadron_pt_from_tracks(self):
        with File(self.test_file, "r") as f:
            data_target = f["jet_pt"][: self.njet_test]
            labels_e = f["Y_edge"][: self.njet_test]
            tracks_extra = f["track_extra"][: self.njet_test]
            edge_origin = f["edge_origin"][: self.njet_test]
        phis = tracks_extra["dphi"]
        pt = tracks_extra["pt"]
        # pt = pt[~np.isnan(pt)]
        # phis = phis[~np.isnan(phis)]
        pt = ma.array(pt, mask=~(labels_e == 1))
        phis = ma.array(phis, mask=~(labels_e == 1))
        pt_bs_coll = []
        pt_true = []
        pi = np.pi
        pt_shape = pt.shape
        for i in range(0, pt_shape[0]):
            pt_bs = pt.data[i][~pt.mask[i]]
            phi_bs = phis.data[i][~phis.mask[i]]
            if len(pt_bs) > 0:
                pt_b = pt_bs[0]
                phi_b = phi_bs[0]
            else:
                continue
            if len(pt_bs) > 1:
                for j in range(1, len(pt_bs)):
                    phi_b = np.abs(phi_b - phi_bs[j])
                    if phi_b > pi:
                        phi_b = pi - phi_b
                    pt_b = np.sqrt(
                        pt_b * pt_b
                        + pt_bs[j] * pt_bs[j]
                        - 2 * pt_b * pt_bs[j] * np.cos(phi_b)
                    )
            pt_true.append(data_target[i])
            pt_bs_coll.append(pt_b)
        true_min = min(pt_true)
        cal_min = min(pt_bs_coll)
        true_max = max(pt_true)
        cal_max = max(pt_bs_coll)
        bins_true = np.linspace(0, 70000, 20)
        bins_cal = np.linspace(0, 20000, 20)
        # bins_true = np.linspace(true_min, true_max, 50)
        # bins_cal = np.linspace(cal_min, cal_max, 50)
        hist = np.histogram2d(pt_bs_coll, pt_true, bins=[bins_cal, bins_true])[0]
        self.plotting_scatter_vals(
            xvals=bins_cal,
            yvals=bins_true,
            zvals=hist,
            ylabel="True b-hadron p_T",
            xlabel="p_T based on b-hadron tracks",
            plot_name="calculated_bhadron_pt",
        )

    def plotting_saliency_map_pervar(self, model_file_numbers):
        track_vars = list(range(len(self.global_config.track_inputs)))
        for model_file_number in model_file_numbers:
            for jet_type in self.jet_types:
                with File(
                    f"{self.model_pred_folder}/epoch_pred_{model_file_number:03d}.h5", "r"
                ) as f:
                    grads = f[f"gradients_{jet_type}"][:self.njet_test]
                    grads_mask = f["mask"][:self.njet_test]
                    grads_shape = grads.shape
                    labels_edge = f[f"labels_edge_{jet_type}"][:self.njet_test]
                rep_grad_mask_wozero = np.logical_and(
                    (np.repeat(grads_mask, grads_shape[-1], axis=-1))
                    .astype(bool)
                    .reshape(grads_shape),
                    grads != 0,
                )
                rep_grad_mask_b_wozero = np.logical_and(
                    np.repeat(
                        np.logical_and(grads_mask, labels_edge == 1),
                        grads_shape[-1],
                        axis=-1,
                    )
                    .astype(bool)
                    .reshape(grads.shape),
                    grads != 0,
                )
                rep_grad_mask_nonb_wozero = np.logical_and(
                    np.repeat(
                        np.logical_and(grads_mask, labels_edge == 0),
                        grads_shape[-1],
                        axis=-1,
                    )
                    .astype(bool)
                    .reshape(grads.shape),
                    grads != 0,
                )
                rep_grad_mask = (
                    np.repeat(grads_mask, grads_shape[-1], axis=-1)
                    .astype(bool)
                    .reshape(grads_shape)
                )
                rep_grad_mask_b = (
                    np.repeat(
                        np.logical_and(grads_mask, labels_edge == 1),
                        grads_shape[-1],
                        axis=-1,
                    )
                    .astype(bool)
                    .reshape(grads.shape)
                )
                rep_grad_mask_nonb = (
                    np.repeat(
                        np.logical_and(grads_mask, labels_edge == 0),
                        grads_shape[-1],
                        axis=-1,
                    )
                    .astype(bool)
                    .reshape(grads.shape)
                )
                for var in track_vars:
                    var_str = self.global_config.track_inputs[var]
                    self.logger.info(
                        f"plotting gradients of variable {var_str} for model"
                        f" {model_file_number}, jet type {jet_type}"
                    )
                    dist_wozero = [
                            np.array(
                                grads[:, :8, var][rep_grad_mask_wozero[:, :8, var]]
                            ).flatten(),
                            np.array(
                                grads[:, :8, var][rep_grad_mask_b_wozero[:, :8, var]]
                            ).flatten(),
                            np.array(
                                grads[:, :8, var][rep_grad_mask_nonb_wozero[:, :8, var]]
                            ).flatten(),
                        ]
                    # minimum_dist = min([min(d) for d in dist_wozero])
                    # maximum_dist = max([max(d) for d in dist_wozero])
                    [minimum_dist, maximum_dist] = np.percentile(dist_wozero[0], q=[10, 90])
                    for dist in dist_wozero:
                        dist[dist < minimum_dist] = minimum_dist
                        dist[dist > maximum_dist] = maximum_dist
                    nbins = 25
                    self.plot_hist(
                        ylabel="normalised number of tracks",
                        xlabel="gradient",
                        plot_name=f"gradient_wzeros_per_var_{var_str}_{jet_type}",
                        vals=dist_wozero,
                        labels=[
                            "gradients, all tracks",
                            f"gradients, {jet_type} tracks",
                            f"gradients, non-{jet_type} tracks",
                        ],
                        nbins=nbins,
                        binrange=(minimum_dist, maximum_dist),
                        colours=["red", "blue", "green"],
                        logy=False,
                    )

                    dist_wzero =  [
                            np.array(
                                grads[:, :8, var][rep_grad_mask[:, :8, var]]
                            ).flatten(),
                            np.array(
                                grads[:, :8, var][rep_grad_mask_b[:, :8, var]]
                            ).flatten(),
                            np.array(
                                grads[:, :8, var][rep_grad_mask_nonb[:, :8, var]]
                            ).flatten(),
                        ]
                    minimum_dist = min([min(d) for d in dist_wzero])
                    maximum_dist = max([max(d) for d in dist_wzero])
                    self.plot_hist(
                        ylabel="normalised number of tracks",
                        xlabel="gradient",
                        plot_name=f"gradient_per_var_{var_str}_{jet_type}",
                        vals=dist_wzero,
                        labels=[
                            "gradients, all tracks",
                            f"gradients, {jet_type} tracks",
                            f"gradients, non-{jet_type} tracks",
                        ],
                        nbins=25,
                        binrange=(-0.1, 0.1),  # (minimum_dist,maximum_dist),
                        colours=["red", "blue", "green"],
                        logy=False,
                    )

    def plotting_model_weights(self, model_file_numbers):
        with File(f"{self.model_pred_folder}/epoch_pred_001.h5", "r") as f:
            keys = list(f.keys())
        keys_bias = [k for k in keys if "bias" in k]
        keys_layers = [k for k in keys if "layer" in k]
        for model_file_number in model_file_numbers:
            self.logger.info(f"plotting weights for model {model_file_number}")
            model_bias_weights = []
            model_layer_weights = []
            minimum_layer = 1000
            maximum_layer = -1000
            minimum_bias = 1000
            maximum_bias = -1000
            with File(
                f"{self.model_pred_folder}/epoch_pred_{model_file_number:03d}.h5", "r"
            ) as f:
                for name in keys_bias:
                    shape = f[name][:self.njet_test].shape
                    model_bias_weights.append(f[name][:self.njet_test])
                    minimum_bias_tmp = min(model_bias_weights[-1])
                    minimum_bias = min(minimum_bias, minimum_bias_tmp)
                    maximum_bias_tmp = max(model_bias_weights[-1])
                    maximum_bias = max(maximum_bias, maximum_bias_tmp)
                for name in keys_layers:
                    shape = f[name][:self.njet_test].shape
                    model_layer_weights.append(f[name][:self.njet_test].reshape(shape[0] * shape[1]))
                    minimum_layer_tmp = min(model_layer_weights[-1])
                    minimum_layer = min(minimum_layer, minimum_layer_tmp)
                    maximum_layer_tmp = max(model_layer_weights[-1])
                    maximum_layer = max(maximum_layer, maximum_layer_tmp)
            weight_bins = np.linspace(minimum_layer, maximum_layer, num=50)
            weight_scatter = [
                hist
                for hist, _ in [
                    np.histogram(weight, bins=weight_bins)
                    for weight in model_layer_weights
                ]
            ]
            weight_scatter = [hist / sum(hist) for hist in weight_scatter]
            self.plotting_scatter_vals(
                ylabel="weight",
                xlabel="layer",
                plot_name=f"weights_per_layer_{model_file_number}",
                yvals=weight_bins[1:] - (weight_bins[1:] - weight_bins[0:-1]) / 2,
                xvals=keys_layers,
                zvals=weight_scatter,
                title="weight per layer",
                y_ticklabels=None,
                swap_inputs=True,
            )
            bias_bins = np.linspace(minimum_bias, maximum_bias, num=50)
            bias_scatter = [
                hist
                for hist, _ in [
                    np.histogram(bias, bins=bias_bins) for bias in model_bias_weights
                ]
            ]
            bias_scatter = [hist / sum(hist) for hist in bias_scatter]
            self.plotting_scatter_vals(
                ylabel="bias",
                xlabel="layer",
                plot_name=f"biases_per_layer_{model_file_number}",
                yvals=bias_bins[1:] - (bias_bins[1:] - bias_bins[0:-1]) / 2,
                xvals=keys_bias,
                zvals=bias_scatter,
                title="bias per layer",
                y_ticklabels=None,
                swap_inputs=True,
            )

    def plotting_target_input_correlation(self):
        hist_dict = {}
        self.logger.info(f"plotting correlation between targets and input")
        with File(self.test_file, "r") as f:
            data_target = f["Y_vertex_features"][:self.njet_test]
            data_input = f["X_train_tracks"][:self.njet_test]
        mask = ~np.all(data_input[..., :3] == 0, axis=-1)
        mask_target = np.any(mask, axis=-1)
        ind_False = np.where(mask_target==False)

        input_shape = data_input.shape
        rep_mask = (
                np.array(np.repeat(mask, input_shape[-1], axis=-1))
                .astype(bool)
                .reshape(input_shape)
            )
        data_input = ma.array(data_input,mask=~rep_mask)
        data_target = ma.array(data_target,mask=~mask_target)

        for i, target_name in enumerate(self.global_config.vertex_features):
            hist_dict[target_name] = {}
            for var_num, var in enumerate(self.global_config.track_inputs):
                vertex_feat = data_target[:, i]
                d_input = data_input[:, :, var_num]
                bins_target = np.linspace(min(vertex_feat), max(vertex_feat), 30)
                bins_input = np.linspace(
                    min(d_input.flatten()), max(d_input.flatten()), 30
                )
                data_track0 = d_input[:, 0]
                vertex_track0 = ma.array(vertex_feat.data, mask=data_track0.mask)
                hist_dict[target_name][f"{var}_track0"] = np.histogram2d(
                    ma.compressed(data_track0), ma.compressed(vertex_track0), bins=[bins_input, bins_target]
                )[0]
                data_track1 = d_input[:, 1]
                vertex_track1 = ma.array(vertex_feat.data, mask=data_track1.mask)
                hist_dict[target_name][f"{var}_track1"] = np.histogram2d(
                    ma.compressed(data_track1), ma.compressed(vertex_track1), bins=[bins_input, bins_target]
                )[0]
                hist_dict[target_name][f"{var}_mean"] = np.histogram2d(
                    ma.mean(d_input, axis=-1), vertex_feat, bins=[bins_input, bins_target]
                )[0]
                hist_dict[target_name][f"{var}_sum"] = np.histogram2d(
                    ma.sum(d_input, axis=-1), vertex_feat, bins=[bins_input, bins_target]
                )[0]
                self.logger.info(
                    f"plotting correlation between {var} and {target_name} for track 1"
                )
                self.plotting_scatter_vals(
                    xlabel=var,
                    ylabel=target_name,
                    plot_name=f"{target_name}_{var}_corr_track1",
                    yvals=bins_target,
                    xvals=bins_input,
                    zvals=hist_dict[target_name][f"{var}_track0"],
                    title=f"correlation between input {var} and {target_name}, track 1",
                )
                self.logger.info(
                    f"plotting correlation between {var} and {target_name} for track 2"
                )
                self.plotting_scatter_vals(
                    xlabel=var,
                    ylabel=target_name,
                    plot_name=f"{target_name}_{var}_corr_track2",
                    yvals=bins_target,
                    xvals=bins_input,
                    zvals=hist_dict[target_name][f"{var}_track1"],
                    title=f"correlation between input {var} and {target_name}, track 2",
                )
                self.logger.info(
                    f"plotting correlation between {var} and {target_name}, taking the mean"
                )
                self.plotting_scatter_vals(
                    xlabel=var,
                    ylabel=target_name,
                    plot_name=f"{target_name}_{var}_corr_mean",
                    yvals=bins_target,
                    xvals=bins_input,
                    zvals=hist_dict[target_name][f"{var}_mean"],
                    title=f"correlation between input {var} and {target_name}, mean",
                )
                self.logger.info(
                    f"plotting correlation between {var} and {target_name}, taking the sum."
                )
                self.plotting_scatter_vals(
                    xlabel=var,
                    ylabel=target_name,
                    plot_name=f"{target_name}_{var}_corr_sum",
                    yvals=bins_target,
                    xvals=bins_input,
                    zvals=hist_dict[target_name][f"{var}_sum"],
                    title=f"correlation between input {var} and {target_name}, sum",
                )
    def plotting_scatter_vals(
        self,
        ylabel,
        xlabel,
        plot_name,
        xvals,
        yvals,
        zvals,
        title=None,
        y_ticklabels=None,
        swap_inputs=True,
    ):
        width = 5.0
        height = 3.5
        zvals = np.array(zvals)
        if swap_inputs:
            zvals_ref = np.array([zvals[:, i] for i in range(len(zvals[0]))])
        else:
            zvals_ref = zvals
        z_min, z_max = (zvals_ref).min(), np.abs(zvals_ref).max()
        figsize = (width, height)
        fig = plt.Figure(figsize=figsize, layout="constrained")
        fig, axis = plt.subplots(1)
        axis.set_ylabel(ylabel)
        axis.set_xlabel(xlabel)
        axis.set_title(title)
        c = axis.pcolor(xvals, yvals, zvals_ref, cmap="RdBu", vmin=z_min, vmax=z_max)
        axis.set_title("pcolor")
        if y_ticklabels is not None:
            axis.set_yticks(list(range(len(y_ticklabels))))
            axis.set_yticklabels(y_ticklabels)
        fig.colorbar(c, ax=axis)
        fig.tight_layout()
        for file_format in self.file_formats:
            plt.savefig(f"{self.plot_dir}/{plot_name}.{file_format}")
        fig.clear()

    def plot_vals(
        self,
        ylabel,
        xlabel,
        plot_name,
        vals,
        labels,
        point_styles,
        title=None,
        y_values_given=False,
        y_ticklabels=None,
        return_plot=False,
        legend_loc="best"
    ):
        if y_values_given:
            ymax = max(vals[0][1])
            ymin = min(vals[0][1])
            if len(vals) > 1:
                for val in vals[1:][1]:
                    ymax_tmp = max(val)
                    ymin_tmp = min(val)
                    ymax = ymax_tmp if ymax < ymax_tmp else ymax
                    ymin = ymin_tmp if ymin > ymin_tmp else ymin
        else:
            ymax = max(vals[0])
            ymin = min(vals[0])
            if len(vals) > 1:
                for val in vals[1:]:
                    ymax_tmp = max(val)
                    ymin_tmp = min(val)
                    ymax = ymax_tmp if ymax < ymax_tmp else ymax
                    ymin = ymin_tmp if ymin > ymin_tmp else ymin

        band = (ymax - ymin) / 30
        plot = PlotBase(
            ylabel=ylabel,
            xlabel=xlabel,
            n_ratio_panels=0,
            logy=False,
            title=title,
            ymax=ymax + band,
            ymin=ymin - band,
        )
        plot.initialise_figure()
        plot.initialise_plot()
        for val, label, point_style in zip(vals, labels, point_styles):
            if y_values_given:
                plot.axis_top.plot(val[0], val[1], point_style, label=label)
            else:
                plot.axis_top.plot(val, point_style, label=label)
            if y_ticklabels is not None:
                plot.axis_top.set_yticks(list(range(len(y_ticklabels))))
                plot.axis_top.set_yticklabels(y_ticklabels)
        plot.axis_top.legend(loc=legend_loc)
        plot = create_figure(plot=plot)
        if return_plot:
            return plot, (f"{self.plot_dir}/{plot_name}.{file_format}" for file_format in self.file_formats)
        else:
            for file_format in self.file_formats:
                plot.savefig(f"{self.plot_dir}/{plot_name}.{file_format}")

    def plot_hist(
        self,
        ylabel,
        xlabel,
        plot_name,
        vals,
        labels,
        colours,
        nbins=50,
        title=None,
        logy=True,
        norm=True,
        y_ticklabels=None,
        x_ticklabels=None,
        binrange=None,
    ):
        print("plot_hisy")
        if binrange is None:
            minimum_glob = min(vals[0])
            maximum_glob = max(vals[0])
            if len(vals) > 1:
                for val in vals[1:]:
                    min_tmp = min(val)
                    max_tmp = max(val)
                    minimum_glob = min_tmp if min_tmp < minimum_glob else minimum_glob
                    maximum_glob = max_tmp if max_tmp > maximum_glob else maximum_glob
            binrange = (minimum_glob, maximum_glob)
        print(binrange)

        plot_histo = HistogramPlot(
            n_ratio_panels=0,
            ylabel=ylabel,
            xlabel=xlabel,
            logy=logy,
            leg_ncol=1,
            figsize=(5.5, 4.5),
            bins=np.linspace(*binrange, nbins + 1, endpoint=True),
            y_scale=1.5,
            norm=norm,
        )

        if y_ticklabels is not None:
            plot_histo.axis_top.set_yticks(list(range(len(y_ticklabels))))
            plot_histo.axis_top.set_yticklabels(y_ticklabels)

        if x_ticklabels is not None:
            if isinstance(x_ticklabels[0], str):
                # plot_histo.axis_top.set_xticks([])
                plot_histo.axis_top.set_xticks(list(range(len(x_ticklabels))))
            else:
                # plot_histo.axis_top.set_xticks(x_ticklabels, x_ticklabels)
                plot_histo.axis_top.set_xticklabels(x_ticklabels)

        for val, label, col in zip(vals, labels, colours):
            plot_histo.add(Histogram(val, label=label, colour=col))
        plot_histo.draw()
        for file_format in self.file_formats:
            plot_histo.savefig(f"{self.plot_dir}/{plot_name}.{file_format}")

    def save_vals(self, dataset_name, data):
        with File(self.plot_file, "a") as f:
            if dataset_name in f.keys():
                del f[dataset_name]
            f.create_dataset(dataset_name, data=data)


class GetEpochPrediction:
    def __init__(self, config, epoch, vars=None):
        self.config = config
        self.jet_types = config.jet_types
        self.small_net = config.small_net
        self.epoch = epoch
        self.test_file = (
            f"{self.config.output}/{self.config.testing_file_name}".replace("//", "/")
        )
        self.used_vertex_properties = getattr(
            self.config, "used_vertex_properties", None
        )
        njets_test = getattr(config, "njets_test", -1)
        njets_test = -1 if njets_test is None else njets_test
        self.dataset = Topographs_dataset(
            filename=self.test_file,
            batch_size=min(njets_test, 1024),
            n_samples=njets_test,
            jet_types=self.jet_types,
        )
        self.dataset_loader = DataLoader(
            self.dataset,
            batch_size=None,
            # drop_last=False,
            # shuffle=False,
            # num_workers=0,
        )

        str_vars = ""
        if vars is not None:
            for var in vars:
                str_vars += f"_{var}"

        self.training_output_folder = config.output_training
        self.training_output_folder = (
            self.training_output_folder[:-1]
            if self.training_output_folder[-1] == "/"
            else self.training_output_folder
        )
        self.training_output_folder += str_vars

        edge_feat_nodes = self.config.edge_feature_network["nodes"]
        edge_weight_nodes = self.config.edge_weight_network["nodes"]
        if vars is not None:
            edge_feat_nodes[0] = len(vars)
            edge_weight_nodes[0] = len(vars)

        vertex_network_nodes = self.config.vertex_network["nodes"]
        if self.used_vertex_properties is not None:
            vertex_network_nodes[-1] = (
                1
                if isinstance(self.used_vertex_properties, int)
                else len(self.used_vertex_properties)
            )
        # layer, model_sub, model
        modelfiles = {jet_type: f"{modelfile}/checkpoints/checkpoint_train_epoch={self.epoch}.ckpt".replace(
                "//", "/"
            ) for jet_type, modelfile in self.config.model_files.items()}
        topomodels = load_topomodel(
            modelfiles,
            nodes_feat=edge_feat_nodes,
            nodes_weight=edge_weight_nodes,
            nodes_vertex=vertex_network_nodes,
            activation_name=self.config.edge_weight_network.get("add_activation", None),
            small_net=self.config.small_net,
        )

        # self.metadata_dict = {}
        # with File(self.test_file, "r") as f:
        #     (
        #         self.metadata_dict["n_jets"],
        #         self.metadata_dict["n_trks"],
        #         self.metadata_dict["n_trk_features"],
        #     ) = f[f"{self.config.tracks_name}"].shape
        #     _, self.metadata_dict["n_vertex_feat"] = f[
        #         f"{self.config.vertex_feat_name}"
        #     ].shape

        preds, labels, masks, grads = get_predictions_and_labels(
            models=topomodels, dataset=self.dataset_loader, jet_types=self.jet_types, small_net=self.small_net
        )
        # model_weights = np.array([par.detach().numpy() for par in topomodel.vertex_network.layers.parameters()])
        self.output_folder = f"{self.training_output_folder}/model_predictions".replace(
            "//", "/"
        )

        model_weight_dict = {}
        bias_weight_counter = 0
        layer_weight_counter = 0
        # for i, m in enumerate(model_weights):
        #     if len(m.shape) == 1:
        #         model_weight_dict[i] = f"bias_{bias_weight_counter}"
        #         bias_weight_counter += 1
        #     else:
        #         model_weight_dict[i] = f"layer_{layer_weight_counter}"
        #         layer_weight_counter += 1

        makedirs(self.output_folder, exist_ok=True)
        with File(f"{self.output_folder}/epoch_pred_{self.epoch:03d}.h5", "w") as f:
            for jet_type in self.jet_types:
                f.create_dataset(name=f"pred_edge_{jet_type}", data=preds[f"preds_e_{jet_type}"])
                f.create_dataset(name=f"labels_edge_{jet_type}", data=labels[f"labels_e_{jet_type}"])
                if not self.small_net[jet_type]:
                    f.create_dataset(name=f"labels_vertex_features_{jet_type}", data=labels[f"labels_v_{jet_type}"])
                    f.create_dataset(name=f"pred_vertex_features_{jet_type}", data=preds[f"preds_v_{jet_type}"])
                f.create_dataset(name=f"gradients_{jet_type}", data=grads[f"grads_{jet_type}"])
            f.create_dataset(name="mask", data=masks)
            # f.create_dataset(name="gradients_pertrack", data=grads_pertrack.data)
            # f.create_dataset(name="gradients_pertrack_mask", data=grads_pertrack.mask)
            # for i in range(len(model_weights)):
            #     f.create_dataset(name=model_weight_dict[i], data=model_weights[i])
