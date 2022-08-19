from glob import glob
from os import makedirs

import numpy as np
from h5py import File
from puma import PlotBase
from tensorflow.data import Dataset
from tensorflow.keras import Model
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import CustomObjectScope

from topograph.modules import (
    DataLoader,
    DenseNetwork,
    DotProduct,
    EdgeLayers,
    FeatLayers,
    ShiftRelu,
    Sigmoid,
    get_logger,
)
from topograph.plotting.plotting_tools import calculate_efficiency


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


class Plotter:
    def __init__(self, config):
        self.config = config
        logger = get_logger()
        self.metadata_dict = {}
        self.test_file = f"{self.config.output}/{self.config.testing_file_name}"
        self.add_activation = self.config.edge_weight_network.get(
            "add_activation", None
        )

        with File(self.test_file, "r") as f:
            (
                self.metadata_dict["n_jets"],
                self.metadata_dict["n_trks"],
                self.metadata_dict["n_trk_features"],
            ) = f[f"{self.config.tracks_name}"].shape
            _, self.metadata_dict["n_vertex_feat"] = f[
                f"{self.config.vertex_feat_name}"
            ].shape

        self.plot_dir = f"{self.config.output}/plots"
        makedirs(self.plot_dir, exist_ok=True)

        if config.evaluation["plot_efficiency"] or config.evaluation["plot_parameters"]:
            n_modelfiles = len(glob(f"{self.config.output}/modelfiles/model_epoch*"))
            effs = []
            params = []
            for i in range(1, n_modelfiles + 1):
                layer, model_sub, model = self.load_topomodel(
                    f"{self.config.output}/modelfiles/model_epoch{i:03d}.h5"
                )
                if config.evaluation["plot_efficiency"]:
                    logger.info(f"plotting efficiency for model model_epoch{i:03d}")
                    effs = self.plotting_efficiency(
                        model_sub,
                        Ntotal=self.metadata_dict["n_jets"]
                        * self.metadata_dict["n_trks"],
                        logger=logger,
                        effs=effs,
                    )
                if config.evaluation["plot_parameters"]:
                    logger.info(f"plotting parameters for model model_epoch{i:03d}")
                    params = self.plotting_parameters(layer, params=params)

            if config.evaluation["plot_efficiency"]:
                self.plot_vals(
                    ylabel="efficiency",
                    xlabel="epoch",
                    plot_name="eff_per_epoch",
                    vals=[effs],
                    labels=[""],
                    point_styles=get_point_styles(1),
                )

            if config.evaluation["plot_parameters"]:
                params = np.array(params).T.squeeze(0)
                labels = [var.name.replace(":0", "") for var in layer.trainable_weights]
                if self.config.evaluation.get("plot_params_one", False):
                    self.plot_vals(
                        ylabel="parameter value",
                        xlabel="epoch",
                        plot_name="parameters",
                        vals=params,
                        labels=labels,
                        point_styles=get_point_styles(len(labels)),
                    )
                if self.config.evaluation.get("plot_params_split", True):
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

        if config.evaluation["plot_pt"]:
            logger.info("Plotting pT...")
            self.plotting_pT_regression(logger=logger)

    def load_topomodel(self, modelfile=None):
        if modelfile is None:
            modelfile_name = self.config.evaluation["model"]
            modelfile = f"{self.config.output}/modelfiles/{modelfile_name}".replace(
                "//", "/"
            )
        with CustomObjectScope(
            {
                "Sigmoid": Sigmoid,
                "ShiftRelu": ShiftRelu,
                "EdgeLayers": EdgeLayers,
                "FeatLayers": FeatLayers,
                "DenseNetwork": DenseNetwork,
                "DotProduct": DotProduct,
            }
        ):
            model = load_model(filepath=modelfile)

        input = model.input
        layer_names = [layer.name for layer in model.layers]
        if self.add_activation is None:
            activation_name = "edge_weight"
        elif self.add_activation == "shifted_relu":
            activation_name = [layer for layer in layer_names if "relu" in layer][0]
        elif self.add_activation == "sigmoid":
            activation_name = [layer for layer in layer_names if "sigmoid" in layer][0]
        else:
            raise KeyError(
                f"Undefined additional actrivation: {self.add_activation}. Please"
                ' select one of the following: ["shifted_relu", "sigmoid"] or leave'
                " empty/remove option."
            )
        layer = model.get_layer(name=activation_name)
        model_sub = Model(inputs=[input], outputs=[layer.output])
        return layer, model_sub, model

    def get_predictions(self, model, full_model=False):
        DatasetGenerator = DataLoader(
            input=self.test_file,
            metadata_dict=self.metadata_dict,
            get_inputs=True,
            get_labels=False,
            get_weight_labels=False,
            stepsize=5000,
            savetracks=True,
            track_name=self.config.tracks_name,
            edge_name=self.config.edge_name,
            edge_feat_name=self.config.edge_feat_name,
            vertex_feat_name=self.config.vertex_feat_name,
            n_samples=self.config.evaluation.get("n_samples", None),
        )

        types, shapes = DatasetGenerator.get_types_shapes()

        dataset = Dataset.from_generator(DatasetGenerator, types, shapes)
        if full_model:
            _, preds = model.predict(dataset)
        else:
            preds = model.predict(dataset)
        return preds

    def get_labels(self, get_weight_labels=False, get_labels=False):
        with File(self.test_file, "r") as f:
            if get_weight_labels:
                return f[self.config.edge_name][:]
            if get_labels:
                return f[self.config.vertex_feat_name][:]

    def plotting_efficiency(self, model, Ntotal, logger, effs):
        preds = self.get_predictions(model)
        labels = self.get_labels(get_weight_labels=True).astype(int)
        eff = calculate_efficiency(preds, labels, Ntotal)
        effs.append(eff)
        with File(f"{self.config.output}/plotting_data.h5", "a") as f:
            if "efficiency" in f.keys():
                del f["efficiency"]
            f.create_dataset("efficiency", data=effs)
        return effs

    def plotting_pT_regression(self, logger, modelfile=None):
        logger.info("plotting pT regression for model model_epoch")
        model, _ = self.load_topomodel(modelfile)
        preds = self.get_predictions(model, full_model=True)
        labels = self.get_labels(get_labels=True).flatten()
        plot_pT = PlotBase(
            ylabel="predicted pT",
            xlabel="true pT",
            n_ratio_panels=0,
            logy=False,
        )
        plot_pT.initialise_figure()
        plot_pT.axis_top.plot(labels, preds, "bo")
        plot_pT = create_figure(plot=plot_pT)
        plot_pT.savefig(f"{self.plot_dir}/pT_regression.pdf")

    def plotting_parameters(self, layer, params):
        weights = layer.trainable_weights
        params.append(np.array(weights))
        return params

    def plot_vals(
        self, ylabel, xlabel, plot_name, vals, labels, point_styles, title=None
    ):
        plot_eff = PlotBase(
            ylabel=ylabel, xlabel=xlabel, n_ratio_panels=0, logy=False, title=title
        )
        plot_eff.initialise_figure()
        for val, label, point_style in zip(vals, labels, point_styles):
            print(point_style)
            plot_eff.axis_top.plot(val, point_style, label=label)
        plot_eff.axis_top.legend()
        plot_eff = create_figure(plot=plot_eff)
        plot_eff.savefig(f"{self.plot_dir}/{plot_name}.pdf")
