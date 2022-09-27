from glob import glob
from os import makedirs

import matplotlib.pyplot as plt
import numpy as np
from h5py import File
from mlxtend.evaluate import confusion_matrix
from mlxtend.plotting import plot_confusion_matrix
from puma import PlotBase

import torch.optim as optim
from torch import load
from torch import tensor

from topograph.modules import (
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


def load_topomodel(
        modelfile=None, 
        add_activation=None, 
        lr=0.01, 
        nodes_feat=[20, 70, 70, 70, 30],
        nodes_weight=[20, 70, 70, 70, 1],
        nodes_vertex=[30, 50, 50, 50, 1]
    ):
    if modelfile is None:
        raise KeyError("Please provide vaild modelfile.")
    topomodel = TopographModel(
        nodes_feat=nodes_feat,
        nodes_weight=nodes_weight,
        nodes_vertex=nodes_vertex
    )
    optimiser = optim.Adam(topomodel.parameters(), lr=lr)
    checkpoint = load(modelfile)
    topomodel.load_state_dict(checkpoint["model_state_dict"])
    optimiser.load_state_dict(checkpoint["optimiser_state_dict"])
    loss = checkpoint["loss"]

    layers = [layer for layer in topomodel.modules]

    layer_names = [layer.name for layer in model.layers]
    if add_activation is None:
        activation_name = "edge_weight"
    elif add_activation == "shifted_relu":
        activation_name = [layer for layer in layer_names if "relu" in layer][0]
    elif add_activation == "sigmoid":
        activation_name = [layer for layer in layer_names if "sigmoid" in layer][0]
    else:
        raise KeyError(
            f"Undefined additional actrivation: {add_activation}. Please"
            ' select one of the following: ["shifted_relu", "sigmoid"] or leave'
            " empty/remove option."
        )

    return topomodel


def get_predictions(model, dataset, full_model=False):
    if full_model:
        _, preds = model.predict(dataset)
    else:
        preds = model.predict(dataset)
    return preds


def get_labels(label_name, file_name, n_samples):
    with File(file_name, "r") as f:
        return f[label_name][:n_samples]


class Plotter:
    def __init__(self, config):
        self.config = config
        logger = get_logger()
        self.test_file = (
            f"{self.config.output}/{self.config.testing_file_name}".replace("//", "/")
        )
        self.plot_file = f"{self.config.output_training}/plotting_data_tr.h5"
        self.add_activation = self.config.edge_weight_network.get(
            "add_activation", None
        )
        self.model_file_numbers = self.config.evaluation.get("model_file_numbers", [1])

        self.plot_effs = self.config.evaluation["plot_efficiency"].get("plot", False)
        self.plot_effs_zeros = self.config.evaluation["plot_efficiency_zeros_only"].get(
            "plot", False
        )
        self.plot_effs_ones = self.config.evaluation["plot_efficiency_ones_only"].get(
            "plot", False
        )
        self.plot_parameters = self.config.evaluation["plot_parameters"].get(
            "plot", False
        )
        self.plot_parameters_split = self.config.evaluation["plot_parameters"].get(
            "split", True
        )
        self.plot_parameters_one = self.config.evaluation["plot_parameters"].get(
            "in_one", False
        )
        self.plot_pt = self.config.evaluation["plot_pt"].get("plot", False)
        self.plot_conf_matrix = self.config.evaluation["plot_conf_matrix"].get(
            "plot", False
        )

        self.recalculate_effs = self.config.evaluation["plot_efficiency"].get(
            "recalculate", True
        )
        self.recalculate_effs_zeros = self.config.evaluation[
            "plot_efficiency_zeros_only"
        ].get("recalculate", True)
        self.recalculate_effs_ones = self.config.evaluation[
            "plot_efficiency_ones_only"
        ].get("recalculate", True)
        self.recalculate_parameters = self.config.evaluation["plot_parameters"].get(
            "recalculate", True
        )
        self.recalculate_pt = self.config.evaluation["plot_pt"].get("recalculate", True)

        self.plot_dir = f"{self.config.output_training}/plots"
        makedirs(self.plot_dir, exist_ok=True)

        self.model_pred_folder = f"{self.config.output_training}/model_predictions"

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
            x=tensor(f[f"{self.config.X_train_tracks}"])
            y_edge=tensor(f[f"{self.config.Y_edge}"])
            y=tensor(f[f"{self.config.Y_vertex_features}"])

        effs = []
        effs_ones = []
        effs_zeros = []
        params = []

        if self.recalculate_effs is False and self.plot_effs:
            with File(self.plot_file, "r+") as f:
                effs = f["efficiency"][:]
        if self.recalculate_effs_ones is False and self.plot_effs_ones:
            with File(self.plot_file, "r+") as f:
                effs_ones = f["efficiency_ones_only"][:]
        if self.recalculate_effs_zeros is False and self.plot_effs_zeros:
            with File(self.plot_file, "r+") as f:
                effs_zeros = f["efficiency_zeros_only"][:]
        if self.recalculate_parameters is False and self.plot_parameters:
            with File(self.plot_file, "r+") as f:
                params = f["parameters"][:]

        if (
            (self.plot_effs and self.recalculate_effs)
            or (self.plot_effs_ones and self.recalculate_effs_zeros)
            or (self.plot_effs_zeros and self.recalculate_effs_ones)
            or (self.plot_parameters and self.recalculate_pt)
        ):
            n_modelfiles = len(glob(f"{self.model_pred_folder}/epoch_pred_*"))
            for i in range(1, n_modelfiles + 1):
                with File(
                    f"{self.model_pred_folder}/epoch_pred_{i:03d}.h5", "r"
                ) as model_data:
                    preds = model_data["pred_edge"][:]
                    labels = model_data["labels_edge"][:]
                    slope = model_data["slope"][()]
                    shift = model_data["shift"][()]

                if self.plot_effs and self.recalculate_effs:
                    logger.info(f"plotting efficiency for model model_epoch{i:03d}")
                    effs = self.get_efficiency(
                        preds=preds,
                        labels=labels,
                        slope=slope,
                        shift=shift,
                        effs=effs,
                    )
                if self.plot_effs_ones and self.recalculate_effs_ones:
                    logger.info(
                        f"plotting efficiency, ones only, for model model_epoch{i:03d}"
                    )
                    effs_ones = self.get_efficiency(
                        preds=preds,
                        labels=labels,
                        slope=slope,
                        shift=shift,
                        effs=effs_ones,
                        ones_only=True,
                    )
                if self.plot_effs_zeros and self.recalculate_effs_zeros:
                    logger.info(
                        f"plotting efficiency, zeros only, for model model_epoch{i:03d}"
                    )
                    effs_zeros = self.get_efficiency(
                        preds=preds,
                        labels=labels,
                        slope=slope,
                        shift=shift,
                        effs=effs_zeros,
                        zeros_only=True,
                    )
                if self.plot_parameters and self.recalculate_parameters:
                    logger.info(f"plotting parameters for model model_epoch{i:03d}")
                    params.append([slope, shift])

        if self.plot_effs:
            self.plot_vals(
                ylabel="efficiency",
                xlabel="epoch",
                plot_name="eff_per_epoch",
                vals=[effs],
                labels=[""],
                point_styles=get_point_styles(1),
            )
            if self.recalculate_effs:
                self.save_vals(dataset_name="efficiency", data=effs)

        if self.plot_effs_ones:
            self.plot_vals(
                ylabel="efficiency",
                xlabel="epoch",
                plot_name="eff_per_epoch_ones",
                vals=[effs_ones],
                labels=[""],
                point_styles=get_point_styles(1),
            )
            if self.recalculate_effs_ones:
                self.save_vals(dataset_name="efficiency_ones_only", data=effs_ones)

        if self.plot_effs_zeros:
            self.plot_vals(
                ylabel="efficiency",
                xlabel="epoch",
                plot_name="eff_per_epoch_zeros",
                vals=[effs_zeros],
                labels=[""],
                point_styles=get_point_styles(1),
            )
            if self.recalculate_effs_zeros:
                self.save_vals(dataset_name="efficiency_zeros_only", data=effs_zeros)

        if self.plot_parameters:
            labels = ["slope", "shift"]
            params = np.array(params).T
            if self.plot_parameters_one:
                self.plot_vals(
                    ylabel="parameter value",
                    xlabel="epoch",
                    plot_name="parameters",
                    vals=params,
                    labels=labels,
                    point_styles=get_point_styles(len(labels)),
                )
            if self.plot_parameters_split:
                for param, label, point_style in zip(
                    params, labels, get_point_styles(len(labels))
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
                self.save_vals(dataset_name="parameters", data=params)

        if self.plot_pt:
            logger.info("Plotting pT...")
            self.plotting_pT_regression(
                logger=logger,
                model_file_numbers=self.model_file_numbers,
            )

        if self.plot_conf_matrix:
            logger.info("Plotting confusion matrix...")
            self.plotting_confusion_matrix(
                logger=logger,
                model_file_numbers=self.model_file_numbers,
            )

    def get_efficiency(
        self, preds, labels, slope, shift, effs, zeros_only=False, ones_only=False
    ):
        Ntotal = self.metadata_dict["n_trks"] * len(preds)
        eff = calculate_efficiency(
            preds,
            labels,
            Ntotal,
            slope,
            shift,
            zeros_only=zeros_only,
            ones_only=ones_only,
        )()
        effs.append(eff)
        return effs

    def plotting_pT_regression(self, logger, model_file_numbers):
        for model_file_number in model_file_numbers:
            logger.info(f"plotting pT regression for model {model_file_number}")
            with File(
                f"{self.model_pred_folder}/epoch_pred_{model_file_number:03d}.h5", "r"
            ) as f:
                preds = f["pred_vertex_features"][:]
                labels = f["labels_vertex_features"][:]
            min = np.min(labels)
            max = np.max(labels)
            plot_pT = PlotBase(
                ylabel="predicted log($p_T$)",
                xlabel="true log($p_T$)",
                n_ratio_panels=0,
                logy=False,
            )
            plot_pT.initialise_figure()
            plot_pT.axis_top.plot(labels, preds, "b.")
            plot_pT.axis_top.plot([min, max], [min, max], "r-")
            plot_pT = create_figure(plot=plot_pT)
            plot_pT.savefig(
                f"{self.plot_dir}/pT_regression_model_{model_file_number}.pdf"
            )

            plot_pT_diff = PlotBase(
                ylabel="Delta log($p_T$)",
                xlabel="predicted log($p_T$)",
                n_ratio_panels=0,
                logy=False,
            )
            regs = calculate_pT_diff(pred=preds, label=labels)()
            plot_pT_diff.initialise_figure()
            plot_pT_diff.axis_top.plot(labels, regs, "b.")
            plot_pT_diff.axis_top.plot([min, max], [0, 0], "r-")
            plot_pT_diff = create_figure(plot=plot_pT_diff)
            plot_pT_diff.savefig(
                f"{self.plot_dir}/Delta_pT_model_{model_file_number}.pdf"
            )

    def plotting_confusion_matrix(self, logger, model_file_numbers):
        for model_file_number in model_file_numbers:
            logger.info(f"plotting confusion matrix for model {model_file_number}")
            with File(
                f"{self.model_pred_folder}/epoch_pred_{model_file_number:03d}.h5", "r"
            ) as f:
                preds = f["pred_edge"][:].flatten()
                labels = f["labels_edge"][:].flatten()
                slope = f["slope"][()]
                shift = f["shift"][()]
            preds = calculate_binary_preds(preds=preds, slope=slope, shift=shift)()
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

    def plot_vals(
        self, ylabel, xlabel, plot_name, vals, labels, point_styles, title=None
    ):
        plot = PlotBase(
            ylabel=ylabel, xlabel=xlabel, n_ratio_panels=0, logy=False, title=title
        )
        plot.initialise_figure()
        for val, label, point_style in zip(vals, labels, point_styles):
            plot.axis_top.plot(val, point_style, label=label)
        plot.axis_top.legend()
        plot = create_figure(plot=plot)
        plot.savefig(f"{self.plot_dir}/{plot_name}.pdf")

    def save_vals(self, dataset_name, data):
        with File(self.plot_file, "a") as f:
            if dataset_name in f.keys():
                del f[dataset_name]
            f.create_dataset(dataset_name, data=data)


class GetEpochPrediction:
    def __init__(self, config, epoch):
        self.config = config
        self.epoch = epoch
        self.test_file = (
            f"{self.config.output}/{self.config.testing_file_name}".replace("//", "/")
        )

        layer, model_sub, model = load_topomodel(
            modelfile=f"{self.config.output_training}/modelfiles/model_epoch{self.epoch:03d}.h5",
            add_activation=self.config.edge_weight_network["add_activation"],
            lr=self.config.lr,
            nodes_feat=self.config.edge_feature_network["nodes"],
            nodes_weight=self.config.edge_weight_network["nodes"],
            nodes_vertex=self.config.vertex_network["nodes"]
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
            

        weights = layer.trainable_weights
        slope = [weight for weight in weights if "relu_slope" in weight.name][0]
        shift = [weight for weight in weights if "relu_shift" in weight.name][0]
        pred_sub = get_predictions(model=model_sub, dataset=self.dataset)
        pred = get_predictions(model=model, dataset=self.dataset, full_model=True)
        label_sub = get_labels(
            label_name=self.config.edge_name,
            file_name=self.test_file,
            n_samples=len(pred_sub),
        )
        label = get_labels(
            label_name=self.config.vertex_feat_name,
            file_name=self.test_file,
            n_samples=len(pred),
        )
        self.output_folder = f"{self.config.output_training}/model_predictions".replace(
            "//", "/"
        )
        makedirs(self.output_folder, exist_ok=True)
        with File(f"{self.output_folder}/epoch_pred_{self.epoch:03d}.h5", "w") as f:
            f.create_dataset(name="pred_edge", data=pred_sub)
            f.create_dataset(name="pred_vertex_features", data=pred)
            f.create_dataset(name="labels_edge", data=label_sub)
            f.create_dataset(name="labels_vertex_features", data=label)
            f.create_dataset(name="slope", data=slope)
            f.create_dataset(name="shift", data=shift)
