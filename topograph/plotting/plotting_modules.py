from glob import glob
from os import makedirs

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


class Plotter:
    def __init__(self, config):
        self.config = config
        logger = get_logger()
        self.metadata_dict = {}
        self.test_file = f"{self.config.output}/{self.config.testing_file_name}"

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

        if config.evaluation["plot_efficiency"]:
            logger.info("Plotting efficiency...")
            n_modelfiles = len(glob(f"{self.config.output}/modelfiles/model_epoch*"))
            self.plotting_efficiency(
                n_modelfiles,
                self.metadata_dict["n_jets"] * self.metadata_dict["n_trks"],
                logger,
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
        print(modelfile)
        with CustomObjectScope(
            {
                "ShiftRelu": ShiftRelu,
                "EdgeLayers": EdgeLayers,
                "FeatLayers": FeatLayers,
                "DenseNetwork": DenseNetwork,
                "DotProduct": DotProduct,
            }
        ):
            model = load_model(filepath=modelfile)
        #     custom_objects={
        #         "ShiftRelu": ShiftRelu,
        #         "EdgeLayers": EdgeLayers,
        #         "FeatLayers": FeatLayers,
        #         "DenseNetwork": DenseNetwork,
        #         "DotProduct": DotProduct
        #     }
        # )
        input = model.input
        layer = model.get_layer(name="shift_relu")
        print(layer.trainable_weights)
        model_sub = Model(inputs=[input], outputs=[layer.output])
        return model, model_sub

    def get_predictions(self, model, full_model=False):
        DatasetGenerator = DataLoader(
            input=self.test_file,
            metadata_dict=self.metadata_dict,
            get_inputs=True,
            get_labels=False,
            get_weight_labels=False,
            # stepsize=600,
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

    def plotting_efficiency(self, n_modelfiles, Ntotal, logger):
        effs = []
        for i in range(1, n_modelfiles + 1):
            logger.info(f"plotting efficiency for model model_epoch{i:03d}")
            _, model = self.load_topomodel(
                f"{self.config.output}/modelfiles/model_epoch{i:03d}.h5"
            )
            preds = self.get_predictions(model)
            labels = self.get_labels(get_weight_labels=True).astype(int)
            eff = calculate_efficiency(preds, labels, Ntotal)
            effs.append(eff)
        with File(f"{self.config.output}/plotting_data.h5", "a") as f:
            if "efficiency" in f.keys():
                del f["efficiency"]
            f.create_dataset("efficiency", data=effs)
        plot_eff = PlotBase(
            ylabel="efficiency",
            xlabel="epoch",
            n_ratio_panels=0,
            logy=False,
        )
        plot_eff.initialise_figure()
        plot_eff.axis_top.plot(effs, "bo")
        plot_eff = create_figure(plot=plot_eff)
        plot_eff.savefig(f"{self.plot_dir}/eff_per_epoch.pdf")

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
