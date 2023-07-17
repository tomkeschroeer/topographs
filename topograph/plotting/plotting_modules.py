from glob import glob
from os import makedirs, remove, rmdir
import time

import matplotlib.pyplot as plt
import numpy as np
import numpy.ma as ma
from h5py import File
from mlxtend.evaluate import confusion_matrix
from mlxtend.plotting import plot_confusion_matrix
from puma import PlotBase, Histogram, HistogramPlot, Roc, RocPlot
from torch.nn import Module

import torch.optim as optim
from torch import load, device, tensor, Tensor, autograd, ones_like, gradient
from pytorch_lightning.callbacks import ModelSummary
from torch.utils.data import DataLoader
from torch.nn.functional import binary_cross_entropy_with_logits

from topograph.modules import (
    TopographModel,
    get_logger,
    IterableFlavourTaggingDataset,
    GlobalConfig,
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

def get_track_origin(origin):
    track_origin = {
        0: "pile-up",
        1: "Fake",
        2: "Primary",
        3: "FromB",
        4: "FromBC",
        5: "FromC",
        6: "FromTau",
        7: "Other Secondary"
    }
    return track_origin[origin]

def get_point_styles(N):
    point_styles = [
        "r-",
        "b.",
        "g.",
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
        "#9ed670"
    ]
    return colours[:N]

def load_topomodel(
        modelfile=None, 
        nodes_feat=[20, 70, 70, 70, 30],
        nodes_weight=[20, 70, 70, 70, 1],
        nodes_vertex=[30, 50, 50, 50, 1],
        activation_name=None,
    ):
    if modelfile is None:
        raise KeyError("Please provide vaild modelfile.")

    topomodel = TopographModel.load_from_checkpoint(
        checkpoint_path=modelfile, 
        save_dir="./",
        name="name",
        nodes_feat=nodes_feat,
        nodes_weight=nodes_weight,
        nodes_vertex=nodes_vertex,
        activation_name=activation_name
    )

    return topomodel


def load_loss(
        modelfile=None, 
        nodes_feat=[20, 70, 70, 70, 30],
        nodes_weight=[20, 70, 70, 70, 1],
        nodes_vertex=[30, 50, 50, 50, 1],
        activation_name=None,
    ):
    # print(modelfile)
    # topomodel = TopographModel(
    #     nodes_feat=nodes_feat,
    #     nodes_weight=nodes_weight,
    #     nodes_vertex=nodes_vertex,
    #     activation_name=activation_name,
    #     save_dir="./",
    #     name="name"
    # )
    checkpoint = load(modelfile, map_location=device('cpu'))
    loss = checkpoint["train/total"]
    return loss

def get_predictions_and_labels(model, dataset):
    preds_v, preds_e = ([],[])
    labels_e, labels_v = ([], [])
    masks = []
    # grads_perjet, grads_pertrack = ([],[])
    grads = []
    model.eval()
    for sample in dataset:
        inputs, labels_edge, labels_vertex, sample_weights, mask, mask_vertex = sample
        labels_shape = labels_edge.size()
        # labels_edge = labels_edge.reshape(labels_shape[1], labels_shape[2], labels_shape[0])
        sample_weight_shape = sample_weights.size()
        # sample_weights = sample_weights.reshape(sample_weight_shape[1], sample_weight_shape[2], sample_weight_shape[0])
        labels_vertex_shape = labels_vertex.size()
        # labels_vertex = labels_vertex.reshape(labels_vertex_shape[1], labels_vertex_shape[2])
        inputs_shape = inputs.size()
        # inputs = inputs.reshape(inputs_shape[1], inputs_shape[2], inputs_shape[3])
        # mask_shape = mask.size()
        # mask = mask.reshape(mask_shape[1], mask_shape[2])
        # mask_vertex_shape = mask_vertex.size()
        # mask_vertex = mask_vertex.reshape(mask_vertex_shape[1])
        inputs.requires_grad_()
        output_v, output_e = model.forward(inputs, mask)
        output_e.backward(gradient=ones_like(output_e))
        preds_v.append(output_v.detach().numpy())
        preds_e.append(output_e.detach().numpy())
        labels_e.append(labels_edge.detach().numpy())
        labels_v.append(labels_vertex.detach().numpy())
        masks.append(mask.detach().numpy())
        grad = inputs.grad.data
        grads.append(grad.detach().numpy())
        # repmask = np.array(np.repeat(mask, inputs_shape[-1], axis = -1)).astype(bool).reshape(inputs_shape)
        # grad = ma.array(grad, mask=~repmask)
        # grad_only_b = grad[labels_edge == 1]
        # gard_only_non_b = grad[np.logical_and(labels_edge == 0, mask)]
        # grad_perjet = grad.mean(axis=-2)
        # grads_perjet.append(grad_perjet)
        # grads_pertrack.append(grad)
    shape_labels_e = np.array(labels_e).shape
    shape_labels_v = np.array(labels_v).shape
    shape_preds_e = np.array(preds_e).shape
    shape_preds_v = np.array(preds_v).shape
    shape_masks = np.array(masks).shape
    shape_grads = np.array(grads).shape
    # shape_grads_perjet = ma.array(grads_perjet).shape
    # shape_grads_pertrack = ma.array(grads_pertrack).shape
    preds_e = np.array(preds_e).reshape(shape_preds_e[0]*shape_preds_e[1],shape_preds_e[2])#,*shape_preds_e[3:])
    preds_v = np.array(preds_v).reshape(shape_preds_v[0]*shape_preds_v[1],*shape_preds_v[2:])#,*shape_preds_v[3:])
    labels_e = np.array(labels_e).reshape(shape_labels_e[0]*shape_labels_e[1],shape_labels_e[2])
    labels_v = np.array(labels_v).reshape(shape_labels_v[0]*shape_labels_v[1],*shape_labels_v[2:])
    grads = np.array(grads).reshape(shape_grads[0]*shape_grads[1],*shape_grads[2:])
    masks = np.array(masks).reshape(shape_masks[0]*shape_masks[1],*shape_masks[2:])
    # grads_perjet = ma.array(grads_perjet)
    # grads_perjet.reshape(shape_grads_perjet[0]*shape_grads_perjet[1],*shape_grads_perjet[2:])
    # # grads_pertrack = ma.array(grads_pertrack).reshape(shape_grads_pertrack[0]*shape_grads_pertrack[1],*shape_grads_pertrack[2:]).mean(axis=0)
    return preds_e, preds_v, labels_e, labels_v, masks, grads

class Plotter:
    def __init__(self, config, cut_val=None, vars=None):
        self.config = config
        self.cut_val = cut_val
        self.global_config = GlobalConfig(alternative_conf=None)
        if cut_val is not None:
            self.cut_val = cut_val if cut_val <= 1 else cut_val/100
        self.logger = get_logger()
        self.test_file = (
            f"{self.config.output}/{self.config.testing_file_name}".replace("//", "/")
        )
        self.njet_test = self.config.njets_test
        datafilename = "plotting_data_tr.h5" if (self.cut_val is None) else f"plotting_data_tr_cutval={self.cut_val}.h5"
        self.used_vertex_properties = getattr(self.config,"used_vertex_properties",None)

        str_vars = ""
        if vars is not None:
            for var in vars:
                str_vars += f"_{var}"

        self.training_output_folder = config.output_training
        self.training_output_folder = self.training_output_folder [:-1] if self.training_output_folder[-1] == "/" else self.training_output_folder
        self.training_output_folder += str_vars

        self.plot_file = f"{self.training_output_folder}/{datafilename}"
        self.add_activation = self.config.edge_weight_network.get(
            "add_activation", None
        )
        self.model_file_numbers = self.config.evaluation.get("model_file_numbers", [1])

        self.plot_effs = self.config.evaluation.get("plot_efficiency", {}).get("plot", False)
        self.plot_effs_zeros = self.config.evaluation.get("plot_efficiency_zeros_only", {}).get(
            "plot", False
        )
        self.plot_effs_ones = self.config.evaluation.get("plot_efficiency_ones_only", {}).get(
            "plot", False
        )
        self.plot_parameters = self.config.evaluation.get("plot_parameters", {}).get(
            "plot", False
        )
        self.plot_parameters_split = self.config.evaluation.get("plot_parameters", {}).get(
            "split", True
        )
        self.plot_parameters_one = self.config.evaluation.get("plot_parameters", {}).get(
            "in_one", False
        )
        self.plot_pt = self.config.evaluation.get("plot_pt", {}).get("plot", False)
        self.plot_eta = self.config.evaluation.get("plot_eta", {}).get("plot", False)
        self.plot_loss = self.config.evaluation.get("plot_loss", {}).get("plot", False)

        self.plot_conf_matrix = self.config.evaluation.get("plot_conf_matrix", {}).get(
            "plot", False
        )
        self.plot_preds_per_epoch = self.config.evaluation.get("plot_preds_per_epoch", {}).get(
            "plot", False
        )
        self.plot_preds_scatter = self.config.evaluation.get("plot_preds_scatter", {}).get(
            "plot", False
        )
        self.plot_saliency = self.config.evaluation.get("plot_saliency", {}).get(
            "plot", False
        )
        self.plot_saliency_pertrack = self.config.evaluation.get("plot_saliency_pertrack", {}).get(
            "plot", False
        )
        self.plot_saliency_pervar = self.config.evaluation.get("plot_saliency_pervar", {}).get(
            "plot", False
        )
        self.plot_n_tracks_per_jet = self.config.evaluation.get("plot_n_tracks_per_jet", {}).get(
            "plot", False
        )
        self.plot_vertex_labels = self.config.evaluation.get("vertex_labels", {}).get(
            "plot", False
        )
        self.plot_weights = self.config.evaluation.get("weights", {}).get(
            "plot", False
        )
        self.plot_target_input_corr = self.config.evaluation.get("plot_target_input_corr", {}).get(
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
        self.recalculate_parameters = self.config.evaluation.get("plot_parameters", {}).get(
            "recalculate", False
        )
        self.recalculate_pt = self.config.evaluation.get("plot_pt", {}).get("recalculate", True)
        self.recalculate_eta = self.config.evaluation.get("plot_eta", {}).get("recalculate", True)
        self.recalculate_loss = self.config.evaluation.get("plot_loss", {}).get(
            "recalculate", False
        )
        self.recalculate_preds_scatter = self.config.evaluation.get("plot_preds_scatter", {}).get(
            "recalculate", False
        )
        self.recalculate_saliency = self.config.evaluation.get("plot_saliency", {}).get(
            "recalculate", False
        )
        self.recalculate_target_input_corr = self.config.evaluation.get("plot_target_input_corr", {}).get(
            "recalculate", False
        )

        self.plot_dir = f"{self.training_output_folder}/plots"
        self.model_pred_folder = f"{self.training_output_folder}/model_predictions"
        makedirs(self.plot_dir, exist_ok=True)

        self.metadata_dict = {}
        with File(self.test_file, "r") as f:
            (
                self.metadata_dict["n_jets"],
                self.metadata_dict["n_trks"],
                self.metadata_dict["n_trk_features"],
            ) = f[f"{self.config.tracks_name}"].shape
            _, self.metadata_dict["n_vertex_feat"] = f[
                f"{self.config.vertex_feat_name}"
            ].shape

        self.n_modelfiles = 0
        # self.effs, self.effs_zeros, self.effs_ones = [],[],[]

    def Run(self):
        nbins_scatter = 100
        self.n_modelfiles = len(glob(f"{self.model_pred_folder}/epoch_pred_*"))
        self.get_all_values()
        
        if self.check_if_recalculate():
            for i in range(self.n_modelfiles):
                with File(
                    f"{self.model_pred_folder}/epoch_pred_{i:03d}.h5", "r"
                ) as model_data:
                    preds = model_data["pred_edge"][:]
                    labels = model_data["labels_edge"][:]
                    # grads = model_data["grads"][:]
                    slope, shift, c1, c2 = None, None, None, None
                    if self.add_activation == "shifted_relu":
                        slope = model_data["slope"][()]
                        shift = model_data["shift"][()]
                    elif self.add_activation == "sigmoid":
                        c1 = model_data["c1"][()]
                        c2 = model_data["c2"][()]

                if self.plot_preds_scatter and self.recalculate_preds_scatter:
                    self.logger.info(f"getting predictions for a scatter plot for model model_epoch{i:03d}")
                    hist, _ =  np.histogram(preds, bins = nbins_scatter, range=(self.startpoint_scatter, self.endpoint_scatter))
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
                        pos=i
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
                if self.plot_parameters and self.recalculate_parameters:
                    self.logger.info(f"getting parameters for model model_epoch{i:03d}")
                    params.append([slope, shift])

                if self.plot_loss and self.recalculate_loss:
                    self.logger.info(f"getting loss for model model_epoch{i:03d}")
                    # loss = self.load_loss()
            
        if self.plot_effs:
            self.logger.info(f"plotting efficiencies...")
            if self.effs is not None:
                self.plot_vals(
                    ylabel="efficiency",
                    xlabel="epoch",
                    plot_name="eff_per_epoch" if self.cut_val is None else f"eff_per_epoch_cutval={self.cut_val}",
                    vals=[self.effs],
                    labels=[""],
                    point_styles=get_point_styles(1),
                )

        if self.plot_effs_ones:
            self.logger.info(f"plotting efficiencies, ones only...")
            if self.effs_ones is not None:
                self.plot_vals(
                    ylabel="efficiency",
                    xlabel="epoch",
                    plot_name="eff_per_epoch_ones" if self.cut_val is None else f"eff_per_epoch_ones_cutval={self.cut_val}",
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
                    plot_name="eff_per_epoch_zeros" if self.cut_val is None else f"eff_per_epoch_zeros_cutval={self.cut_val}",
                    vals=[self.effs_zeros],
                    labels=[""],
                    point_styles=get_point_styles(1),
                )

        if self.plot_parameters:
            self.logger.info(f"plotting parameters...")
            if self.config.edge_weight_network["add_activation"] == "shifted_relu":
                labels = ["slope", "shift"]
            elif self.config.edge_weight_network["add_activation"] == "sigmoid":
                labels = ["c1", "c2"]
            elif self.config.edge_weight_network["add_activation"] == "softplus":
                labels = None
            if labels is not None:
                self.params = np.array(self.params).T
                if self.plot_parameters_one:
                    self.plot_vals(
                        ylabel="parameter value",
                        xlabel="epoch",
                        plot_name="parameters",
                        vals=self.params,
                        labels=labels,
                        point_styles=get_point_styles(len(labels)),
                    )
                if self.plot_parameters_split:
                    self.logger.info(f"plotting parameters in split plots...")
                    for param, label, point_style in zip(
                        self.params, labels, get_point_styles(len(labels))
                    ):
                        self.plot_vals(
                            ylabel="parameter value",
                            xlabel="epoch",
                            plot_name=label,
                            vals=[param],
                            labels=[""],
                            point_styles=[point_style],
                            title=label,
                        )
                if self.recalculate_parameters:
                    self.save_vals(dataset_name="parameters", data=self.params)

        if self.plot_pt:
            self.logger.info(f"plotting pT...")
            self.plotting_regression_scatter(
                model_file_numbers=self.model_file_numbers,
                var="pT"
            )

        if self.plot_eta:
            self.logger.info(f"plotting eta...")
            self.plotting_regression_scatter(
                model_file_numbers=self.model_file_numbers,
                var="eta"
            )
        
        if self.plot_conf_matrix:
            self.logger.info("plotting confusion matrix...")
            self.plotting_confusion_matrix(
                model_file_numbers=self.model_file_numbers,
            )

        if self.plot_preds_per_epoch:
            self.logger.info("plotting predictions per epoch...")
            self.plotting_preds_per_epoch(
                model_file_numbers=self.model_file_numbers
            )

        if self.plot_preds_scatter:
            self.logger.info(f"plotting predictions in scatter plot...")
            self.plot_scatter_vals(
                ylabel="predicition", 
                xlabel="epoch",
                plot_name="predictions",
                xvals=list(range(0,len(self.preds_scatter))),
                yvals=np.linspace(self.startpoint_scatter, self.endpoint_scatter, num=len(self.preds_scatter[0]), endpoint=True),
                zvals=self.preds_scatter,
                title="predictions"
            )
        
        if self.recalculate_preds_scatter:
            self.save_vals(dataset_name="preds_scatter", data=self.preds_scatter)
            self.save_vals(dataset_name="endpoint_scatter", data=self.endpoint_scatter)
            self.save_vals(dataset_name="startpoint_scatter", data=self.startpoint_scatter)
        
        if self.plot_saliency:
            self.plot_saliency_map(model_file_numbers=self.model_file_numbers)

        if self.plot_saliency_pertrack:
            self.plot_saliency_map_pertrack(model_file_numbers=self.model_file_numbers)
        
        if self.plot_saliency_pervar:
            self.plot_saliency_per_var(model_file_numbers=self.model_file_numbers)
            
        if self.plot_vertex_labels:
            self.plotting_vertex_labels_per_epoch(model_file_numbers=self.model_file_numbers)
        
        if self.plot_weights:
            self.plot_model_weights(model_file_numbers=self.model_file_numbers)
        
        if self.plot_target_input_corr:
            self.plot_target_input_correlation()

        if self.plot_n_tracks_per_jet:
            self.plotting_n_tracks(model_file_numbers=self.model_file_numbers)
        
        # self.plotting_track_origin()
        
        self.plotting_roc_curves(model_file_numbers=self.model_file_numbers)
                
    def get_all_values(self):
        if self.recalculate_effs is False and self.plot_effs:
            try:
                with File(self.plot_file, "r+") as f:
                    self.effs = f["efficiency"][:]
            except (KeyError, FileNotFoundError) as er:
                self.logger.warn("No efficiencies found in file or file not found. Recalculate instead")
                self.recalculate_effs = True
                self.effs = np.full(shape=(self.n_modelfiles),fill_value=-1.0, dtype=float)
        else:
            self.effs = np.full(shape=(self.n_modelfiles),fill_value=-1.0, dtype=float)
        if self.recalculate_effs_ones is False and self.plot_effs_ones:
            try:
                with File(self.plot_file, "r+") as f:
                    self.effs_ones = f["efficiency_ones_only"][:]
            except (KeyError, FileNotFoundError) as er:
                self.logger.warn("No efficiencies (ones only) found in file or file not found. Recalculate instead")
                self.recalculate_effs_ones = True
                self.effs_ones = np.full(shape=(self.n_modelfiles),fill_value=-1.0, dtype=float)
        else:
            self.effs_ones = np.full(shape=(self.n_modelfiles),fill_value=-1.0, dtype=float)
        if self.recalculate_effs_zeros is False and self.plot_effs_zeros:
            try:
                with File(self.plot_file, "r+") as f:
                    self.effs_zeros = f["efficiency_zeros_only"][:]
            except (KeyError, FileNotFoundError) as er:
                self.logger.warn("No efficiencies (zeros only) found in file or file not found. Recalculate instead")
                self.recalculate_effs_zeros = True
                self.effs_zeros = np.full(shape=(self.n_modelfiles),fill_value=-1.0, dtype=float)
        else:
            self.effs_zeros = np.full(shape=(self.n_modelfiles),fill_value=-1.0, dtype=float)
        if self.recalculate_parameters is False and self.plot_parameters:
            try:
                with File(self.plot_file, "r+") as f:
                    self.params = f["parameters"][:]
            except (KeyError, FileNotFoundError) as er:
                self.logger.warn("No parameters found in file or file not found. Recalculate instead")
                self.recalculate_parameters = True
        else:
            self.params = []
        if self.recalculate_loss is False and self.plot_loss:
            try:
                with File(self.plot_file, "r+") as f:
                    self.loss = f["loss"][:]
            except (KeyError, FileNotFoundError) as er:
                self.logger.warn("No loss found in file or file not found. Recalculate instead")
                self.recalculate_loss = True
        else:
            self.loss = []
        if self.recalculate_preds_scatter is False and self.plot_preds_scatter:
            try:
                with File(self.plot_file, "r+") as f:
                    self.preds_scatter = f["preds_scatter"][:]
                    self.endpoint_scatter = f["endpoint_scatter"][()]
                    self.startpoint_scatter = f["startpoint_scatter"][()]
            except (KeyError, FileNotFoundError) as er:
                self.logger.warn("No loss found in file or file not found. Recalculate instead")
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
            or (self.recalculate_parameters)
            or (self.recalculate_loss)
            or (self.recalculate_preds_scatter)
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
        ones_only=False
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
            cut_val=self.cut_val
        )()
        effs[pos] = eff
        return effs

    def plotting_regression_scatter(self, model_file_numbers, var):
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
                except ValueError:
                    preds = f["pred_vertex_features"][:]
                try:
                    labels = f["labels_vertex_features"][:, var_numb]
                except ValueError:
                    labels = f["labels_vertex_features"][:]
            
            var_min = np.min(labels[~np.isnan(labels)])
            var_max = np.max(labels[~np.isnan(labels)])
            var_min_pred = np.min(preds[~np.isnan(preds)])
            var_max_pred = np.max(preds[~np.isnan(preds)])
            bins = np.linspace(min(var_min, var_min_pred), max(var_max, var_max_pred), 20)
            bins = np.linspace(-2,2,60)
            hist = np.histogram2d(preds, labels, bins=[bins, bins])[0]
            self.plot_scatter_vals(
                    ylabel=f"true {var_str}",
                    xlabel=f"predicted {var_str}", 
                    plot_name=f"{var}_regression_model_{model_file_number}", 
                    xvals=bins, 
                    yvals=bins, 
                    zvals=hist, 
                    title=None, 
                    y_ticklabels=None, 
                    swap_inputs=True
                )

            regs = calculate_pT_diff(pred=preds, label=labels)()
            var_min = np.min(regs) #[~np.isnan(regs)])
            var_max = np.max(regs) #[~np.isnan(regs)])
            var_min_pred = np.min(preds[~np.isnan(preds)])
            var_max_pred = np.max(preds[~np.isnan(preds)])
            bins_x = np.linspace(var_min_pred, var_max_pred, 30)
            bins_x = np.linspace(-2, 2, 30)
            bins_y = np.linspace(var_min, var_max, 30)
            bins_y = np.linspace(-1.5, 1.5, 30)
            # bins = np.linspace(-2,2,30)
            hist_Delta = np.histogram2d(preds, regs, bins=[bins_x, bins_y])[0]
            self.plot_scatter_vals(
                    ylabel=f"Delta {var_str}",
                    xlabel=f"predicted {var_str}", 
                    plot_name=f"Delta_{var}_model_{model_file_number}", 
                    xvals=bins_x, 
                    yvals=bins_y, 
                    zvals=hist_Delta, 
                    title=None, 
                    y_ticklabels=None, 
                    swap_inputs=True
            )

    def plotting_roc_curves(self, model_file_numbers):
        self.logger.info("plotting roc curves...")
        for model_file_number in model_file_numbers:
            self.logger.info(f"plotting predictions per epoch for model {model_file_number}")
            with File(
                f"{self.model_pred_folder}/epoch_pred_{model_file_number:03d}.h5", "r"
            ) as f:
                labels = f["labels_edge"][:]
                preds = f["pred_edge"][:]
            Pos = preds[labels==1]
            Neg = preds[labels==0]
            percentages = np.linspace(start=0, end=100, num=50, endpoint=True)
            percentages = 100-percentages
            cut_vals_tpr = np.percentile(Pos, percentages)
            fpr = [sum(Neg > cut_val_tpr)/len(Neg) for cut_val_tpr in cut_vals_tpr]
            tpr = [sum(Pos > cut_val_tpr)/len(Pos) for cut_val_tpr in cut_vals_tpr]
            self.plot_vals(
                ylabel="TPR",
                xlabel="FPR",
                plot_name="ROC_curve",
                vals=[tpr, fpr],
                labels=[''],
                title="ROC curve",
                y_values_given=True
            )


    def plotting_track_origin(self):
        test_file = (
            f"{self.config.output}/{self.config.testing_file_name}".replace("//", "/")
        )
        with File(test_file, "r") as test:
            edge_origin = test["edge_origin"][:self.config.njets_test]
            edge_label = test["Y_edge"][:self.config.njets_test]
        track_origin = [
            "pile-up",
            "Fake",
            "Primary",
            "FromB",
            "FromBC",
            "FromC",
            "FromTau",
            "Other Secondary"
        ]
        self.plot_hist(
            ylabel="number of tracks",
            xlabel="track origin",
            plot_name="origin_labels",
            vals=[edge_origin, edge_origin[edge_label==1], edge_origin[edge_label==0]],
            labels=["all tracks", "b-tracks", "non-b tracks"],
            title="track origins",
            logy=False,
            norm=False,
            nbins=8,
            binrange=(-0.5,7.5),
            y_ticklabels=track_origin,
            colours=get_colours(3)
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
                    preds = f["pred_vertex_features"][:]
                    labels = f["labels_vertex_features"][:]
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
                ymax=var_max_pred
            )
            plot_var.initialise_figure()
            plot_var.axis_top.plot(labels, preds, "b.")
            plot_var.axis_top.plot([var_min, var_max], [var_min, var_max], "r-")
            plot_var.set_y_lim()
            plot_var = create_figure(plot=plot_var)
            plot_var.savefig(
                f"{self.plot_dir}/{var}_regression_model_{model_file_number}.pdf"
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
            plot_var_diff.savefig(
                f"{self.plot_dir}/Delta_{var}_model_{model_file_number}.pdf"
            )

    def plotting_n_tracks(self, model_file_numbers):
        self.logger.info("plotting number of tracks...")
        with File(
            f"{self.model_pred_folder}/epoch_pred_{model_file_numbers[0]:03d}.h5", "r"
        ) as f:
            labels_e = f["labels_edge"][:]
            mask = f["mask"][:]
        n_b_tracks = np.sum(labels_e, axis=1)
        n_tracks = np.sum(mask, axis=1)
        n_non_b_tracks = n_tracks - n_b_tracks
        dists = [n_tracks, n_b_tracks, n_non_b_tracks]
        legend_labels = ["all tracks", "b-tracks", "non-b tracks"]
        self.plot_hist(
            ylabel="number of jets",
            xlabel="number of tracks",
            vals=dists,
            labels=legend_labels,
            nbins=int(max(n_tracks))+1,
            binrange=(-0.5,max(n_tracks)+0.5),
            plot_name="number_of_tracks",
            colours=get_colours(len(legend_labels)),
            norm=False,
            logy=False
        )


    def plotting_confusion_matrix(self, model_file_numbers):
        for model_file_number in model_file_numbers:
            self.logger.info(f"plotting confusion matrix for model {model_file_number}")
            with File(
                f"{self.model_pred_folder}/epoch_pred_{model_file_number:03d}.h5", "r"
            ) as f:
                preds = f["pred_edge"][:].flatten()
                labels = f["labels_edge"][:].flatten()
                try:
                    slope = f["slope"][()]
                    shift = f["shift"][()]
                except KeyError:
                    slope = None
                    shift = None
                try:
                    c1 = f["c1"][()]
                    c2 = f["c2"][()]
                except KeyError:
                    c1 = None
                    c2 = None

            preds = calculate_binary_preds(preds=preds, slope=slope, shift=shift, c1=c1, c2=c2)()
            conf_mat = confusion_matrix(labels, preds, binary=True)
            plot_confusion_matrix(
                conf_mat=conf_mat,
                colorbar=True,
                show_normed=True,
                show_absolute=True,
                class_names=[0, 1],
            )

            plt.tight_layout()
            plt.savefig(f"{self.plot_dir}/conf_matrix_model_{model_file_number}.pdf")
            plt.close()
            
    def plotting_preds_per_epoch(self, model_file_numbers):
        for model_file_number in model_file_numbers:
            self.logger.info(f"plotting predictions per epoch for model {model_file_number}")
            with File(
                f"{self.model_pred_folder}/epoch_pred_{model_file_number:03d}.h5", "r"
            ) as f:
                preds = f["pred_edge"][:].flatten()
                labels = f["labels_edge"][:].flatten()
            nbins = 50
            binrange = (min(labels),max(labels))
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
                plot_name=f"predicitions_split_epoch_{model_file_number}"
            )
    
    def plotting_vertex_labels_per_epoch(self, model_file_numbers):
        for model_file_number in model_file_numbers:
            self.logger.info(f"plotting predictions per epoch for model {model_file_number}")
            with File(
                f"{self.model_pred_folder}/epoch_pred_{model_file_number:03d}.h5", "r"
            ) as f:
                labels = f["labels_vertex_features"][:].flatten()
            nbins = 50
            binrange = (min(labels),max(labels))            
            self.plot_hist(
                ylabel="normalised number of tracks",
                xlabel="prediction", 
                vals=[labels],
                labels=[''], 
                colours=get_colours(1),
                title=f"labels for epoch {model_file_number}",
                nbins=nbins,
                binrange=binrange,
                plot_name=f"labels_split_epoch_{model_file_number}"
            )

    def plot_saliency_map(self, model_file_numbers):
        for model_file_number in model_file_numbers:
            self.logger.info(f"plotting saliency map for model {model_file_number}")
            with File(
                f"{self.model_pred_folder}/epoch_pred_{model_file_number:03d}.h5", "r"
            ) as f:
                grads = f["gradients"][:]
                grads_mask = f["mask"][:]
                grads_shape = grads.shape
                labels_edge = f["labels_edge"][:]
            rep_grad_mask = np.array(np.repeat(grads_mask, grads_shape[-1], axis = -1)).astype(bool).reshape(grads_shape)
            rep_grad_mask_b = np.repeat(np.logical_and(grads_mask, labels_edge == 1), grads_shape[-1], axis = -1).astype(bool).reshape(grads.shape)
            rep_grad_mask_nonb = np.repeat(np.logical_and(grads_mask, labels_edge == 0), grads_shape[-1], axis = -1).astype(bool).reshape(grads.shape)
            grads_all = ma.array(grads, mask=~rep_grad_mask).mean(axis=1)
            grads_b = ma.array(grads, mask=~rep_grad_mask_b).mean(axis=1)
            grads_nonb = ma.array(grads, mask=~rep_grad_mask_nonb).mean(axis=1)
            track_vars = list(range(len(self.global_config.track_inputs)))

            if len(track_vars) != grads_shape[-1]:
                self.logger.warning("Number of track variables is not the same as the one indicated by the saved gradients. Only use the numbers of variables as y-axis")
                track_vars = list(range(grads_shape[-1]))

            sal_bins_all = np.linspace(min(grads_all.flatten()), max(grads_all.flatten()), num=25)
            hists_all = [np.histogram(grads_all[:,i][grads_all[:,i] != 0.0], bins=sal_bins_all)[0]/len(grads_all[:,i]) for i in track_vars]
            minimal_perc = np.concatenate(np.array([np.argwhere(hist>0.1).flatten() for hist in hists_all]))
            minimum = min(minimal_perc)
            maximum = max(minimal_perc)
            # sal_bins_all = np.linspace(minimum, maximum, num=25)
            grads_all[grads_all<sal_bins_all[minimum]] = sal_bins_all[minimum]
            grads_all[grads_all>sal_bins_all[maximum]] = sal_bins_all[maximum]
            sal_bins_all = np.linspace(min(grads_all.flatten()), max(grads_all.flatten()), num=25)
            # sal_bins_all = np.linspace(-0.1,0.01,50)
            hists_all = [np.histogram(grads_all[:,i], bins=sal_bins_all)[0]/len(grads_all[:,i]) for i in track_vars]
            self.plot_scatter_vals(
                ylabel="input variable",
                xlabel="gradient",
                xvals=sal_bins_all[1:]-(sal_bins_all[1:]-sal_bins_all[0:-1])/2,
                yvals=track_vars,
                zvals=np.stack((hists_all)),
                plot_name=f"saliency_map_alltracks_model_{model_file_number:03d}",
                y_ticklabels=self.global_config.track_inputs,
                swap_inputs=False
            )

            sal_bins_b = np.linspace(min(grads_b.flatten()), max(grads_b.flatten()), num=25)
            hists_b = [np.histogram(grads_b[:,i][grads_b[:,i] != 0.0], bins=sal_bins_b)[0]/len(grads_b[:,i]) for i in track_vars]
            minimal_perc = np.concatenate(np.array([np.argwhere(hist>0.1).flatten() for hist in hists_b]))
            minimum = min(minimal_perc)
            maximum = max(minimal_perc)
            # sal_bins_b = np.linspace(minimum, maximum, num=25)
            grads_b[grads_b<sal_bins_b[minimum]] = sal_bins_b[minimum]
            grads_b[grads_b>sal_bins_b[maximum]] = sal_bins_b[maximum]
            sal_bins_b = np.linspace(min(grads_b.flatten()), max(grads_b.flatten()), num=25)
            self.plot_scatter_vals(
                ylabel="input variable",
                xlabel="gradient",
                xvals=sal_bins_b[1:]-(sal_bins_b[1:]-sal_bins_b[0:-1])/2,
                yvals=track_vars,
                zvals=np.stack((hists_b)),
                plot_name=f"saliency_map_btracks_model_{model_file_number:03d}",
                y_ticklabels=self.global_config.track_inputs,
                swap_inputs=False
            )

            sal_bins_nonb = np.linspace(min(grads_nonb.flatten()), max(grads_nonb.flatten()), num=25)
            hists_nonb = [np.histogram(grads_nonb[:,i][grads_nonb[:,i] != 0.0], bins=sal_bins_nonb)[0]/len(grads_nonb[:,i]) for i in track_vars]
            minimal_perc = np.concatenate(np.array([np.argwhere(hist>0.1).flatten() for hist in hists_nonb]))
            minimum = min(minimal_perc)
            maximum = max(minimal_perc)
            # sal_bins_nonb = np.linspace(minimum, maximum, num=25)
            grads_nonb[grads_nonb<sal_bins_nonb[minimum]] = sal_bins_nonb[minimum]
            grads_nonb[grads_nonb>sal_bins_nonb[maximum]] = sal_bins_nonb[maximum]
            sal_bins_nonb = np.linspace(min(grads_nonb.flatten()), max(grads_nonb.flatten()), num=25)
            self.plot_scatter_vals(
                ylabel="input variable",
                xlabel="gradient",
                xvals=sal_bins_nonb[1:]-(sal_bins_nonb[1:]-sal_bins_nonb[0:-1])/2,
                yvals=track_vars,
                zvals=np.stack((hists_nonb)),
                plot_name=f"saliency_map_nonbtracks_model_{model_file_number:03d}",
                y_ticklabels=self.global_config.track_inputs,
                swap_inputs=False
            )


    def plot_saliency_map_pertrack(self, model_file_numbers):
        for model_file_number in model_file_numbers:
            self.logger.info(f"plotting saliency map per track for model {model_file_number}")
            with File(
                f"{self.model_pred_folder}/epoch_pred_{model_file_number:03d}.h5", "r"
            ) as f:
                grads = f["gradients"][:]
                grads_mask = f["mask"][:]
                grads_shape = grads.shape
                labels_edge = f["labels_edge"][:]
            rep_grad_mask = np.array(np.repeat(grads_mask, grads_shape[-1], axis = -1)).astype(bool).reshape(grads_shape)
            rep_grad_mask_b = np.repeat(np.logical_and(grads_mask, labels_edge == 1), grads_shape[-1], axis = -1).astype(bool).reshape(grads.shape)
            rep_grad_mask_nonb = np.repeat(np.logical_and(grads_mask, labels_edge == 0), grads_shape[-1], axis = -1).astype(bool).reshape(grads.shape)
            grads_all = ma.array(grads, mask=~rep_grad_mask).mean(axis=0)
            grads_b = ma.array(grads, mask=~rep_grad_mask_b).mean(axis=0)
            grads_nonb = ma.array(grads, mask=~rep_grad_mask_nonb).mean(axis=0)
            track_vars = list(range(len(self.global_config.track_inputs)))
            if len(track_vars) != grads_shape[-1]:
                self.logger.warning("Number of track variables is not the same as the one indicated by the saved gradients. Only use the numbers of variables as y-axis")
                track_vars = list(range(grads_shape[-1]))
            ntracks = 8
            sal_bins = np.linspace(0,ntracks,ntracks+1)
            self.plot_scatter_vals(
                ylabel="input variable",
                xlabel="tracks",
                xvals=sal_bins[1:]-(sal_bins[1:]-sal_bins[0:-1])/2,
                yvals=track_vars,
                zvals=grads_all[:ntracks], #np.stack((hists)),
                plot_name=f"saliency_map_alltracks_pertrack_model_{model_file_number:03d}",
                y_ticklabels=self.global_config.track_inputs,
                swap_inputs=True
            )
            self.plot_scatter_vals(
                ylabel="input variable",
                xlabel="tracks",
                xvals=sal_bins[1:]-(sal_bins[1:]-sal_bins[0:-1])/2,
                yvals=track_vars,
                zvals=grads_b[:ntracks], #np.stack((hists)),
                plot_name=f"saliency_map_btracks_pertrack_model_{model_file_number:03d}",
                y_ticklabels=self.global_config.track_inputs,
                swap_inputs=True
            )
            self.plot_scatter_vals(
                ylabel="input variable",
                xlabel="tracks",
                xvals=sal_bins[1:]-(sal_bins[1:]-sal_bins[0:-1])/2,
                yvals=track_vars,
                zvals=grads_nonb[:ntracks], #np.stack((hists)),
                plot_name=f"saliency_map_nonbtracks_pertrack_model_{model_file_number:03d}",
                y_ticklabels=self.global_config.track_inputs,
                swap_inputs=True
            )

    def plot_jet_pt_from_tracks(self, model_file_numbers):
        for model_file_number in model_file_numbers:
            with File(
                f"{self.model_pred_folder}/epoch_pred_{model_file_number:03d}.h5", "r"
            ) as f:
                grads = f["gradients"][:]

    def plot_saliency_per_var(self, model_file_numbers):
        track_vars = list(range(len(self.global_config.track_inputs)))
        for model_file_number in model_file_numbers:
            with File(
                f"{self.model_pred_folder}/epoch_pred_{model_file_number:03d}.h5", "r"
            ) as f:
                grads = f["gradients"][:]
                grads_mask = f["mask"][:]
                grads_shape = grads.shape
                labels_edge = f["labels_edge"][:]
            rep_grad_mask_wozero = np.logical_and((np.repeat(grads_mask, grads_shape[-1], axis = -1)).astype(bool).reshape(grads_shape), grads!=0)
            rep_grad_mask_b_wozero = np.logical_and(np.repeat(np.logical_and(grads_mask, labels_edge == 1), grads_shape[-1], axis = -1).astype(bool).reshape(grads.shape), grads!=0)
            rep_grad_mask_nonb_wozero = np.logical_and(np.repeat(np.logical_and(grads_mask, labels_edge == 0), grads_shape[-1], axis = -1).astype(bool).reshape(grads.shape), grads!=0)
            rep_grad_mask = np.repeat(grads_mask, grads_shape[-1], axis = -1).astype(bool).reshape(grads_shape)
            rep_grad_mask_b = np.repeat(np.logical_and(grads_mask, labels_edge == 1), grads_shape[-1], axis = -1).astype(bool).reshape(grads.shape)
            rep_grad_mask_nonb = np.repeat(np.logical_and(grads_mask, labels_edge == 0), grads_shape[-1], axis = -1).astype(bool).reshape(grads.shape)
            for var in track_vars:
                var_str = self.global_config.track_inputs[var]
                self.logger.info(f"plotting gradients of variable {var_str} for model {model_file_number}")
                dist_wozero = np.array(
                    [
                        np.array(grads[:,:,var][rep_grad_mask_wozero[:,:,var]]).flatten(), 
                        np.array(grads[:,:,var][rep_grad_mask_b_wozero[:,:,var]]).flatten(), 
                        np.array(grads[:,:,var][rep_grad_mask_nonb_wozero[:,:,var]]).flatten()
                    ]
                )
                minimum_dist = min([min(d) for d in dist_wozero])
                maximum_dist = max([max(d) for d in dist_wozero])
                self.plot_hist(
                    ylabel="normalised number of tracks",
                    xlabel="gradient",
                    plot_name=f"gradient_wzeros_per_var_{var_str}",
                    vals=dist_wozero,
                    labels=["gradients, all tracks", "gradients, b tracks", "gradients, non-b tracks"],
                    nbins=25,
                    binrange=(-0.1,0.1), #(minimum_dist,maximum_dist),
                    colours=["red", "blue", "green"],
                    logy=False
                )
        
                dist_wzero = np.array(
                    [
                        np.array(grads[:,:,var][rep_grad_mask[:,:,var]]).flatten(), 
                        np.array(grads[:,:,var][rep_grad_mask_b[:,:,var]]).flatten(), 
                        np.array(grads[:,:,var][rep_grad_mask_nonb[:,:,var]]).flatten()
                    ]
                )
                minimum_dist = min([min(d) for d in dist_wzero])
                maximum_dist = max([max(d) for d in dist_wzero])
                self.plot_hist(
                    ylabel="normalised number of tracks",
                    xlabel="gradient",
                    plot_name=f"gradient_per_var_{var_str}",
                    vals=dist_wzero,
                    labels=["gradients, all tracks", "gradients, b tracks", "gradients, non-b tracks"],
                    nbins=25,
                    binrange=(-0.1,0.1), #(minimum_dist,maximum_dist),
                    colours=["red", "blue", "green"],
                    logy=False
                )
            


    def plot_model_weights(self, model_file_numbers):
        with File(
                f"{self.model_pred_folder}/epoch_pred_001.h5", "r"
            ) as f:
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
                    shape = f[name][:].shape
                    model_bias_weights.append(f[name][:])
                    minimum_bias_tmp = min(model_bias_weights[-1])
                    minimum_bias = min(minimum_bias,minimum_bias_tmp)
                    maximum_bias_tmp = max(model_bias_weights[-1])
                    maximum_bias = max(maximum_bias,maximum_bias_tmp)
                for name in keys_layers:
                    shape = f[name][:].shape
                    model_layer_weights.append(f[name][:].reshape(shape[0]*shape[1]))
                    minimum_layer_tmp = min(model_layer_weights[-1])
                    minimum_layer = min(minimum_layer,minimum_layer_tmp)
                    maximum_layer_tmp = max(model_layer_weights[-1])
                    maximum_layer = max(maximum_layer,maximum_layer_tmp)
            weight_bins = np.linspace(minimum_layer, maximum_layer, num=50)
            weight_scatter = [hist for hist, _ in [np.histogram(weight, bins=weight_bins) for weight in model_layer_weights]]
            weight_scatter = [hist/sum(hist) for hist in weight_scatter]
            self.plot_scatter_vals(
                ylabel="weight",
                xlabel="layer",
                plot_name=f"weights_per_layer_{model_file_number}", 
                yvals=weight_bins[1:]-(weight_bins[1:]-weight_bins[0:-1])/2, 
                xvals=keys_layers,
                zvals=weight_scatter,
                title="weight per layer",
                y_ticklabels=None,
                swap_inputs=True
            )
            bias_bins = np.linspace(minimum_bias, maximum_bias, num=50)
            bias_scatter = [hist for hist, _ in [np.histogram(bias, bins=bias_bins) for bias in model_bias_weights]]
            bias_scatter = [hist/sum(hist) for hist in bias_scatter]
            self.plot_scatter_vals(
                ylabel="bias",
                xlabel="layer", 
                plot_name=f"biases_per_layer_{model_file_number}", 
                yvals=bias_bins[1:]-(bias_bins[1:]-bias_bins[0:-1])/2, 
                xvals=keys_bias,
                zvals=bias_scatter,
                title="bias per layer",
                y_ticklabels=None,
                swap_inputs=True
            )


    def plot_target_input_correlation(self):
        hist_dict = {}
        self.logger.info(f"plotting correlation between targets and input")
        with File(self.test_file, "r") as f:
            data_target = f["Y_vertex_features"][:self.njet_test]
            data_input = f["X_train_tracks"][:self.njet_test]
        for i, target_name in enumerate(self.global_config.vertex_features):
            hist_dict[target_name] = {}
            for var_num, var in enumerate(self.global_config.track_inputs):
                vertex_feat = data_target[:,i]
                d_input = data_input[:,:,var_num]
                bins_target = np.linspace(min(vertex_feat), max(vertex_feat), 30)
                bins_input = np.linspace(min(d_input.flatten()), max(d_input.flatten()), 30)
                hist_dict[target_name][f"{var}_track0"] = np.histogram2d(d_input[:,0], vertex_feat, bins=[bins_input, bins_target])[0]
                hist_dict[target_name][f"{var}_track1"] = np.histogram2d(d_input[:,1], vertex_feat, bins=[bins_input, bins_target])[0]
                self.logger.info(f"plotting correlation between {var} and {target_name} for track 1")
                self.plot_scatter_vals(
                    xlabel=var,
                    ylabel=target_name,
                    plot_name=f"{target_name}_{var}_corr_track1",
                    yvals=bins_target,
                    xvals=bins_input,
                    zvals=hist_dict[target_name][f"{var}_track0"],
                    title=f"correlation between input {var} and {target_name}, track 1"
                )
                self.logger.info(f"plotting correlation between {var} and {target_name} for track 2")
                self.plot_scatter_vals(
                    xlabel=var,
                    ylabel=target_name,
                    plot_name=f"{target_name}_{var}_corr_track2",
                    yvals=bins_target,
                    xvals=bins_input,
                    zvals=hist_dict[target_name][f"{var}_track1"],
                    title=f"correlation between input {var} and {target_name}, track 2"
                )

    def plot_scatter_vals(
        self, ylabel, xlabel, plot_name, xvals, yvals, zvals, title=None, y_ticklabels=None, swap_inputs=True
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
        c = axis.pcolor(xvals, yvals, zvals_ref, cmap='RdBu',vmin=z_min, vmax=z_max)
        axis.set_title('pcolor')
        if y_ticklabels is not None:
            axis.set_yticks(list(range(len(y_ticklabels))))
            axis.set_yticklabels(y_ticklabels)
        fig.colorbar(c, ax=axis)
        fig.tight_layout()
        plt.savefig(f"{self.plot_dir}/{plot_name}.pdf")
        fig.clear()

    def plot_vals(
        self, ylabel, xlabel, plot_name, vals, labels, point_styles, title=None, y_values_given=False, y_ticklabels=None
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

        band = (ymax-ymin)/30
        plot = PlotBase(
            ylabel=ylabel, xlabel=xlabel, n_ratio_panels=0, logy=False, title=title, ymax = ymax + band, ymin = ymin - band
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
        plot.axis_top.legend()
        plot = create_figure(plot=plot)
        plot.savefig(f"{self.plot_dir}/{plot_name}.pdf")

    def plot_hist(
        self, ylabel, xlabel, plot_name, vals, labels, nbins, binrange, colours, title=None, logy=True, norm=True, y_ticklabels=None
    ):
        plot_histo = HistogramPlot(
            n_ratio_panels=0,
            ylabel=ylabel,
            xlabel=xlabel,
            logy=logy,
            leg_ncol=1,
            figsize=(5.5, 4.5),
            bins=np.linspace(*binrange, nbins, endpoint=True),
            y_scale=1.5,
            norm=norm
        )

        if y_ticklabels is not None:
            plot_histo.axis_top.set_yticks(list(range(len(y_ticklabels))))
            plot_histo.axis_top.set_yticklabels(y_ticklabels)

        for val, label, col in zip(vals, labels, colours):
            plot_histo.add(
                Histogram(
                    val,
                    label=label,
                    colour=col
                )
            )
        plot_histo.draw()
        plot_histo.savefig(f"{self.plot_dir}/{plot_name}.pdf")

    def save_vals(self, dataset_name, data):
        with File(self.plot_file, "a") as f:
            if dataset_name in f.keys():
                del f[dataset_name]
            f.create_dataset(dataset_name, data=data)

class GetEpochPrediction:
    def __init__(self, config, epoch, vars=None):
        self.config = config
        self.epoch = epoch
        self.test_file = (
            f"{self.config.output}/{self.config.testing_file_name}".replace("//", "/")
        )
        self.used_vertex_properties = getattr(self.config,"used_vertex_properties",None)
        njets_test = getattr(config, "njets_test", -1)
        njets_test = -1 if njets_test is None else njets_test
        self.dataset = IterableFlavourTaggingDataset(
            dset="test",
            buffer_shuffle=False,
            file_name = self.test_file,
            batch_size = 100, # min(njets_test, 1024),
            drop_last = True,
            buffer_size = 10_000,
            njets = njets_test,
            vars=vars,
            used_vertex_properties=getattr(self.config,"used_vertex_properties",None),
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
        self.training_output_folder = self.training_output_folder [:-1] if self.training_output_folder[-1] == "/" else self.training_output_folder
        self.training_output_folder += str_vars
        
        edge_feat_nodes = self.config.edge_feature_network["nodes"]
        edge_weight_nodes = self.config.edge_weight_network["nodes"]
        if vars is not None:
            edge_feat_nodes[0] = len(vars)
            edge_weight_nodes[0] = len(vars)

        vertex_network_nodes = self.config.vertex_network["nodes"]
        if self.used_vertex_properties is not None:
            vertex_network_nodes[-1] = 1 if isinstance(self.used_vertex_properties, int) else len(self.used_vertex_properties)
        # layer, model_sub, model 
        topomodel = load_topomodel(
            modelfile=f"{self.training_output_folder}/checkpoints/checkpoint_train_epoch={self.epoch}.ckpt".replace("//","/"),
            nodes_feat=edge_feat_nodes,
            nodes_weight=edge_weight_nodes,
            nodes_vertex=vertex_network_nodes,
            activation_name=self.config.edge_weight_network.get("add_activation", None)
        )

        self.metadata_dict = {}
        with File(self.test_file, "r") as f:
            (
                self.metadata_dict["n_jets"],
                self.metadata_dict["n_trks"],
                self.metadata_dict["n_trk_features"],
            ) = f[f"{self.config.tracks_name}"].shape
            _, self.metadata_dict["n_vertex_feat"] = f[
                f"{self.config.vertex_feat_name}"
            ].shape

        pars = topomodel.state_dict()
        activation = config.edge_weight_network["add_activation"]
        if activation == "shifted_relu":
            slope = pars[f"add_activation.slope"]
            shift = pars[f"add_activation.shift"]
        elif activation == "sigmoid":
            c1 = pars[f"add_activation.c1"]
            c2 = pars[f"add_activation.c1"]
        preds_e, preds_v, labels_e, labels_v, masks, grads  = get_predictions_and_labels(model=topomodel, dataset=self.dataset_loader)
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
            f.create_dataset(name="pred_edge", data=preds_e)
            f.create_dataset(name="pred_vertex_features", data=preds_v)
            f.create_dataset(name="labels_edge", data=labels_e)
            f.create_dataset(name="labels_vertex_features", data=labels_v)
            f.create_dataset(name="mask", data=masks)
            f.create_dataset(name="gradients", data=grads)
            # f.create_dataset(name="gradients_pertrack", data=grads_pertrack.data)
            # f.create_dataset(name="gradients_pertrack_mask", data=grads_pertrack.mask)
            # for i in range(len(model_weights)):
            #     f.create_dataset(name=model_weight_dict[i], data=model_weights[i])
            if activation == "shifted_relu":
                f.create_dataset(name="slope", data=slope)
                f.create_dataset(name="shift", data=shift)
            elif activation == "sigmoid":
                f.create_dataset(name="c1", data=c1)
                f.create_dataset(name="c2", data=c2)
