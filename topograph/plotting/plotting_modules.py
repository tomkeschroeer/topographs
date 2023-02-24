from glob import glob
from os import makedirs

import matplotlib.pyplot as plt
import numpy as np
from h5py import File
from mlxtend.evaluate import confusion_matrix
from mlxtend.plotting import plot_confusion_matrix
from puma import PlotBase

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
    
def get_predictions_and_labels(model, dataset, full_model=True):
    preds_v, preds_e = ([],[])
    labels_e, labels_v = ([], [])
    masks = []
    for sample in dataset:
        inputs, labels_edge, labels_vertex, _, mask = sample
        output = model(inputs, mask)
        output_v = output[0].detach().numpy()
        output_e = output[1].detach().numpy()
        preds_v.append(*output_v)
        preds_e.append(output_e)
        labels_e.append(labels_edge.detach().numpy())
        labels_v.append(labels_vertex.detach().numpy())
        masks.append(mask.detach().numpy())
    shape_labels_e = np.array(labels_e).shape
    shape_labels_v = np.array(labels_v).shape
    shape_preds_e = np.array(preds_e).shape
    shape_preds_v = np.array(preds_v).shape
    shape_masks = np.array(masks).shape
    preds_e = np.array(preds_e).reshape(shape_preds_e[0]*shape_preds_e[1],*shape_preds_e[2:])
    preds_v = np.array(preds_v).reshape(shape_preds_v[0]*shape_preds_v[1],*shape_preds_v[2:])
    labels_e = np.array(labels_e).reshape(shape_labels_e[0]*shape_labels_e[1],*shape_labels_e[2:])
    labels_v = np.array(labels_v).reshape(shape_labels_v[0]*shape_labels_v[1],*shape_labels_v[2:])
    masks = np.array(masks).reshape(shape_masks[0]*shape_masks[1],*shape_masks[2:])
    return preds_e, preds_v, labels_e, labels_v, masks

class Plotter:
    def __init__(self, config, cut_val=None):
        self.config = config
        self.cut_val = cut_val
        if cut_val is not None:
            self.cut_val = cut_val if cut_val <= 1 else cut_val/100
        logger = get_logger()
        self.test_file = (
            f"{self.config.output}/{self.config.testing_file_name}".replace("//", "/")
        )
        datafilename = "plotting_data_tr.h5" if (self.cut_val is None) else f"plotting_data_tr_cutval={self.cut_val}.h5"
        self.plot_file = f"{self.config.output_training}/{datafilename}"
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
        self.plot_loss = self.config.evaluation["plot_loss"].get("plot", False)
        
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
        self.recalculate_loss = self.config.evaluation["plot_loss"].get(
            "recalculate", True
        )

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
        
        effs = []
        effs_ones = []
        effs_zeros = []
        params = []

        if self.recalculate_effs is False and self.plot_effs:
            try:
                with File(self.plot_file, "r+") as f:
                    effs = f["efficiency"][:]
            except KeyError:
                logger.warn("No efficiencies found in file. Recalculate instead")
                self.recalculate_effs = True
        if self.recalculate_effs_ones is False and self.plot_effs_ones:
            try:
                with File(self.plot_file, "r+") as f:
                    effs_ones = f["efficiency_ones_only"][:]
            except KeyError:
                logger.warn("No efficiencies (ones only) found in file. Recalculate instead")
                self.recalculate_effs_ones = True
        if self.recalculate_effs_zeros is False and self.plot_effs_zeros:
            try:
                with File(self.plot_file, "r+") as f:
                    effs_zeros = f["efficiency_zeros_only"][:]
            except KeyError:
                logger.warn("No efficiencies (zeros only) found in file. Recalculate instead")
                self.recalculate_effs_zeros = True
        if self.recalculate_parameters is False and self.plot_parameters:
            try:
                with File(self.plot_file, "r+") as f:
                    params_old = f["parameters"][:]
            except KeyError:
                logger.warn("No parameters found in file. Recalculate instead")
                self.recalculate_parameters = True
        if self.recalculate_loss is False and self.plot_loss:
            try:
                with File(self.plot_file, "r+") as f:
                    loss = f["loss"][:]
            except KeyError:
                logger.warn("No loss found in file. Recalculate instead")
                self.recalculate_loss = True
        if (
            (self.plot_effs and self.recalculate_effs)
            or (self.plot_effs_ones and self.recalculate_effs_zeros)
            or (self.plot_effs_zeros and self.recalculate_effs_ones)
            or (self.plot_parameters and self.recalculate_parameters)
            or (self.plot_loss and self.recalculate_loss)
        ):
            n_modelfiles = len(glob(f"{self.model_pred_folder}/epoch_pred_*"))
            for i in range(0,n_modelfiles):
                with File(
                    f"{self.model_pred_folder}/epoch_pred_{i:03d}.h5", "r"
                ) as model_data:
                    # mask = model_data["mask"][:]
                    # mask = mask.reshape(*mask.shape, 1)
                    preds = model_data["pred_edge"][:]
                    labels = model_data["labels_edge"][:]
                    slope, shift, c1, c2 = None, None, None, None
                    if self.add_activation == "shifted_relu":
                        slope = model_data["slope"][()]
                        shift = model_data["shift"][()]
                    elif self.add_activation == "sigmoid":
                        c1 = model_data["c1"][()]
                        c2 = model_data["c2"][()]
                if self.plot_effs and self.recalculate_effs:
                    logger.info(f"plotting efficiency for model model_epoch{i:03d}")
                    effs = self.get_efficiency(
                        preds=preds,
                        labels=labels,
                        effs=effs,
                        slope=slope,
                        shift=shift,
                        c1=c1,
                        c2=c2,
                    )
                if self.plot_effs_ones and self.recalculate_effs_ones:
                    logger.info(
                        f"plotting efficiency, ones only, for model model_epoch{i:03d}"
                    )
                    effs_ones = self.get_efficiency(
                        preds=preds,
                        labels=labels,
                        effs=effs_ones,
                        slope=slope,
                        shift=shift,
                        c1=c1,
                        c2=c2,
                        ones_only=True,
                    )
                if self.plot_effs_zeros and self.recalculate_effs_zeros:
                    logger.info(
                        f"plotting efficiency, zeros only, for model model_epoch{i:03d}"
                    )
                    effs_zeros = self.get_efficiency(
                        preds=preds,
                        labels=labels,
                        effs=effs_zeros,
                        slope=slope,
                        shift=shift,
                        c1=c1,
                        c2=c2,
                        zeros_only=True,
                    )
                if self.plot_parameters and self.recalculate_parameters:
                    logger.info(f"plotting parameters for model model_epoch{i:03d}")
                    params.append([slope, shift])
                if self.plot_loss and self.recalculate_loss:
                    logger.info(f"plotting loss for model model_epoch{i:03d}")
                    # loss = self.load_loss()

        if self.plot_effs:
            self.plot_vals(
                ylabel="efficiency",
                xlabel="epoch",
                plot_name="eff_per_epoch" if self.cut_val is None else f"eff_per_epoch_cutval={self.cut_val}",
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
                plot_name="eff_per_epoch_ones" if self.cut_val is None else f"eff_per_epoch_ones_cutval={self.cut_val}",
                vals=[effs_ones],
                labels=[""],
                point_styles=get_point_styles(1),
            )
            if self.recalculate_effs_ones:
                self.save_vals(dataset_name="efficiency_ones_only", data=effs_ones)

        if self.plot_effs_zeros:
            # print(effs_zeros)
            self.plot_vals(
                ylabel="efficiency",
                xlabel="epoch",
                plot_name="eff_per_epoch_zeros" if self.cut_val is None else f"eff_per_epoch_zeros_cutval={self.cut_val}",
                vals=[effs_zeros],
                labels=[""],
                point_styles=get_point_styles(1),
            )
            if self.recalculate_effs_zeros:
                self.save_vals(dataset_name="efficiency_zeros_only", data=effs_zeros)

        if self.plot_parameters:
            if config.edge_weight_network["add_activation"] == "shifted_relu":
                labels = ["slope", "shift"]
            elif config.edge_weight_network["add_activation"] == "sigmoid":
                labels = ["c1", "c2"]
            elif config.edge_weight_network["add_activation"] == "softplus":
                labels = None
            if labels is not None:
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
        self,
        preds,
        labels,
        effs,
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

    def plot_vals(
        self, ylabel, xlabel, plot_name, vals, labels, point_styles, title=None
    ):  
        print(f"vals = {vals}")
        ymax = max(vals[0])
        ymin = min(vals[0])
        band = (ymax-ymin)/30
        # print(ymax + band)
        # print(ymin - band)
        plot = PlotBase(
            ylabel=ylabel, xlabel=xlabel, n_ratio_panels=0, logy=False, title=title, ymax = ymax + band, ymin = ymin - band
        )
        print("plot vals")
        plot.initialise_figure()
        plot.initialise_plot()
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
        njets_test = getattr(config, "njets_test", -1)
        njets_test = -1 if njets_test is None else njets_test
        self.dataset = IterableFlavourTaggingDataset(
            dset="test",
            buffer_shuffle=False,
            file_name = self.test_file,
            batch_size = 1024,
            drop_last = True,
            buffer_size = 10_000,
            njets = njets_test
        )
        self.dataset_loader = DataLoader(
            self.dataset,
            batch_size=None,
            drop_last=False,
            shuffle=False,
            num_workers=0,
        )
        # layer, model_sub, model 
        topomodel = load_topomodel(
            modelfile=f"{self.config.output_training}/checkpoints/checkpoint_train_epoch={self.epoch}.ckpt".replace("//","/"),
            nodes_feat=self.config.edge_feature_network["nodes"],
            nodes_weight=self.config.edge_weight_network["nodes"],
            nodes_vertex=self.config.vertex_network["nodes"],
            activation_name=self.config.edge_weight_network.get("add_activation", None)
        )

        # loss = load_loss(
        #     modelfile=f"{self.config.output_training}/checkpoints/checkpoint_train_epoch={self.epoch}.ckpt".replace("//","/"),
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

        self.output_folder = f"{self.config.output_training}/model_predictions".replace(
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
