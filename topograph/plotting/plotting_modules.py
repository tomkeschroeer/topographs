from glob import glob
from os import makedirs, remove, rmdir
import time

import matplotlib.pyplot as plt
import numpy as np
from h5py import File
from mlxtend.evaluate import confusion_matrix
from mlxtend.plotting import plot_confusion_matrix
from puma import PlotBase, Histogram, HistogramPlot

import torch.optim as optim
from torch import load, device, tensor
from pytorch_lightning.callbacks import ModelSummary
from torch.utils.data import DataLoader

from topograph.modules import (
    TopographModel,
    get_logger,
    IterableFlavourTaggingDataset,
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

def get_var_names(var):
    vardict = {
        "pT": "log($p_T$)",
        "eta": "$\eta$"
    }
    
    return vardict[var], list(vardict.keys()).index(var)

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
    for sample in dataset:
        inputs, labels_edge, labels_vertex, _, mask, mask_vertex = sample
        output = model(inputs, mask)
        output_v = output[0].detach().numpy()
        output_e = output[1].detach().numpy()
        preds_v.append(output_v)
        preds_e.append(output_e)
        labels_e.append(labels_edge.detach().numpy())
        labels_v.append(labels_vertex.detach().numpy())
        masks.append(mask.detach().numpy())
    shape_labels_e = np.array(labels_e).shape
    shape_labels_v = np.array(labels_v).shape
    shape_preds_e = np.array(preds_e).shape
    shape_preds_v = np.array(preds_v).shape
    shape_masks = np.array(masks).shape
    preds_e = np.array(preds_e).reshape(shape_preds_e[0]*shape_preds_e[1]*shape_preds_e[2],*shape_preds_e[3:])
    preds_v = np.array(preds_v).reshape(shape_preds_v[0]*shape_preds_v[1]*shape_preds_v[2],*shape_preds_v[3:])
    labels_e = np.array(labels_e).reshape(shape_labels_e[0]*shape_labels_e[1]*shape_labels_e[2],*shape_labels_e[3:])
    labels_v = np.array(labels_v).reshape(shape_labels_v[0]*shape_labels_v[1]*shape_labels_v[2],*shape_labels_v[3:])
    masks = np.array(masks).reshape(shape_masks[0]*shape_masks[1],*shape_masks[2:])
    return preds_e, preds_v, labels_e, labels_v, masks

class Plotter:
    def __init__(self, config, cut_val=None, vars=None):
        self.config = config
        self.cut_val = cut_val
        if cut_val is not None:
            self.cut_val = cut_val if cut_val <= 1 else cut_val/100
        self.logger = get_logger()
        self.test_file = (
            f"{self.config.output}/{self.config.testing_file_name}".replace("//", "/")
        )
        datafilename = "plotting_data_tr.h5" if (self.cut_val is None) else f"plotting_data_tr_cutval={self.cut_val}.h5"

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
                    slope, shift, c1, c2 = None, None, None, None
                    if self.add_activation == "shifted_relu":
                        slope = model_data["slope"][()]
                        shift = model_data["shift"][()]
                    elif self.add_activation == "sigmoid":
                        c1 = model_data["c1"][()]
                        c2 = model_data["c2"][()]
                if self.plot_preds_scatter and self.recalculate_preds_scatter:
                    self.logger.info(f"getting predictions for a scatter plot for model model_epoch{i:03d}")
                    preds_s = preds.shape
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
            self.plotting_regression(
                model_file_numbers=self.model_file_numbers,
                var="pT"
            )

        if self.plot_eta:
            self.logger.info(f"plotting eta...")
            self.plotting_regression(
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
            or ( self.recalculate_preds_scatter)
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

    def plotting_regression(self, model_file_numbers, var):
        for model_file_number in model_file_numbers:
            self.logger.info(f"plotting pT regression for model {model_file_number}")
            var_str, var_numb = get_var_names(var)
            with File(
                f"{self.model_pred_folder}/epoch_pred_{model_file_number:03d}.h5", "r"
            ) as f:
                preds = f["pred_vertex_features"][:, var_numb]
                labels = f["labels_vertex_features"][:, var_numb]
            var_min = np.min(labels[~np.isnan(labels)])
            var_max = np.max(labels[~np.isnan(labels)])
            plot_var = PlotBase(
                ylabel=f"predicted {var_str}",
                xlabel=f"true {var_str}",
                n_ratio_panels=0,
                logy=False,
            )
            plot_var.initialise_figure()
            plot_var.axis_top.plot(labels, preds, "b.")
            plot_var.axis_top.plot([var_min, var_max], [var_min, var_max], "r-")
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
            binrange = (0,1)
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

    def plot_scatter_vals(
        self, ylabel, xlabel, plot_name, xvals, yvals, zvals, title=None
    ):
        width = 5.0
        height = 3.5
        zvals = np.array(zvals)
        zvals_ref = np.array([zvals[:, i] for i in range(len(zvals[0]))])
        z_min, z_max = (zvals_ref).min(), np.abs(zvals_ref).max()
        figsize = (width, height)
        fig = plt.Figure(figsize=figsize, layout="constrained")
        fig, axis = plt.subplots(1)
        axis.set_ylabel(ylabel)
        axis.set_xlabel(xlabel)
        axis.set_title(title)
        c = axis.pcolor(xvals, yvals, zvals_ref, cmap='RdBu',vmin=z_min, vmax=z_max)
        axis.set_title('pcolor')
        fig.colorbar(c, ax=axis)
        fig.tight_layout()
        plt.savefig(f"{self.plot_dir}/{plot_name}.pdf")
        fig.clear()

    def plot_vals(
        self, ylabel, xlabel, plot_name, vals, labels, point_styles, title=None
    ):  
        ymax = max(vals[0])
        ymin = min(vals[0])
        if len(vals) > 1:
            for val in vals[0:]:
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
            plot.axis_top.plot(val, point_style, label=label)
        plot.axis_top.legend()
        plot = create_figure(plot=plot)
        plot.savefig(f"{self.plot_dir}/{plot_name}.pdf")

    def plot_hist(
        self, ylabel, xlabel, plot_name, vals, labels, nbins, binrange, colours, title=None
    ):
        plot_histo = HistogramPlot(
            n_ratio_panels=0,
            ylabel=ylabel,
            xlabel=xlabel,
            logy=True,
            leg_ncol=1,
            figsize=(5.5, 4.5),
            bins=np.linspace(*binrange, nbins, endpoint=True),
            y_scale=1.5,
            norm=True
        )

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
        njets_test = getattr(config, "njets_test", -1)
        njets_test = -1 if njets_test is None else njets_test
        self.dataset = IterableFlavourTaggingDataset(
            dset="test",
            buffer_shuffle=False,
            file_name = self.test_file,
            batch_size = 1024,
            drop_last = True,
            buffer_size = 10_000,
            njets = njets_test,
            vars=vars
        )
        self.dataset_loader = DataLoader(
            self.dataset,
            # batch_size=None,
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
        
        # layer, model_sub, model 
        topomodel = load_topomodel(
            modelfile=f"{self.training_output_folder}/checkpoints/checkpoint_train_epoch={self.epoch}.ckpt".replace("//","/"),
            nodes_feat=edge_feat_nodes,
            nodes_weight=edge_weight_nodes,
            nodes_vertex=self.config.vertex_network["nodes"],
            activation_name=self.config.edge_weight_network.get("add_activation", None)
        )

        # loss = load_loss(
        #     modelfile=f"{self.training_output_folder}/checkpoints/checkpoint_train_epoch={self.epoch}.ckpt".replace("//","/"),
        #     nodes_feat=self.config.edge_feature_network["nodes"],
        #     nodes_weight=self.config.edge_weight_network["nodes"],
        #     nodes_vertex=self.config.vertex_network["nodes"],
        #     activation_name=self.config.edge_weight_network.get("add_activation", None)
        # )

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
        preds_e, preds_v, labels_e, labels_v, mask = get_predictions_and_labels(model=topomodel, dataset=self.dataset_loader)

        self.output_folder = f"{self.training_output_folder}/model_predictions".replace(
            "//", "/"
        )
        makedirs(self.output_folder, exist_ok=True)
        with File(f"{self.output_folder}/epoch_pred_{self.epoch:03d}.h5", "w") as f:
            f.create_dataset(name="pred_edge", data=preds_e)
            f.create_dataset(name="pred_vertex_features", data=preds_v)
            f.create_dataset(name="labels_edge", data=labels_e)
            f.create_dataset(name="labels_vertex_features", data=labels_v)
            f.create_dataset(name="mask", data=mask)
            # f.create_dataset(name="loss", data=loss)
            if activation == "shifted_relu":
                f.create_dataset(name="slope", data=slope)
                f.create_dataset(name="shift", data=shift)
            elif activation == "sigmoid":
                f.create_dataset(name="c1", data=c1)
                f.create_dataset(name="c2", data=c2)
