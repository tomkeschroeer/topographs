from puma import Histogram, HistogramPlot, PlotBase, PlotObject
from h5py import File
from glob import glob
from os import makedirs
import numpy as np
from tensorflow.keras import Model
from tensorflow.keras.models import load_model
from tensorflow.data import Dataset
from tensorflow import (
    TensorShape,
    float32,
    int32
)
from topograph.modules import (
    step_activation,
    shifted_relu_activation,
    DataLoader,
    get_logger
)
from topograph.plotting.plotting_tools import (
    calculate_efficiency,
    calculate_jetwise_efficiency
)

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
        self.train_file = f"{self.config.output}/{self.config.training_file_name}"
        
        with File(self.train_file, "r") as f:
            self.metadata_dict["n_jets"], self.metadata_dict["n_trks"], self.metadata_dict["n_trk_features"] = f[f"{self.config.tracks_name}"].shape
            _, self.metadata_dict["n_vertex_feat"] = f[f"{self.config.vertex_feat_name}"].shape

        self.plot_dir = f"{self.config.output}/plots"
        makedirs(self.plot_dir, exist_ok=True)
        
        if config.evaluation["plot_efficiency"]:
            n_modelfiles = len(glob(f"{self.config.output}/modelfiles/model_epoch*"))
            self.plotting_efficiency(n_modelfiles, self.metadata_dict["n_jets"]*self.metadata_dict["n_trks"], logger)
        
        # if config.plot_pt:
        #     pT_predictions = self.get_predictions(model)
        #     pT_labels = self.get_labels(get_labels=False)
        

    def load_model(self, modelfile=None):
        if modelfile is None:
            modelfile_name = self.config.evaluation["model"]
            modelfile = f"{self.config.output}/modelfiles/{modelfile_name}".replace("//","/")
        model = load_model(
            modelfile,
            custom_objects={"step_activation":step_activation, "shifted_relu_activation":shifted_relu_activation}
        )
        input = model.input 
        layer = model.get_layer(name="edge_weight").output
        model_sub = Model(inputs = [input], outputs = [layer])
        return model, model_sub

    def get_predictions(self, model):
        DatasetGenerator = DataLoader(
            input=f"{self.config.output}/{self.config.training_file_name}",
            metadata_dict=self.metadata_dict,
            get_inputs=True,
            get_labels=False,
            get_weight_labels=False,
            savetracks=True,
            track_name=self.config.tracks_name,
            edge_name=self.config.edge_name,
            edge_feat_name=self.config.edge_feat_name,
            vertex_feat_name=self.config.vertex_feat_name
        )

        types, shapes = DatasetGenerator.get_types_shapes()

        dataset = Dataset.from_generator(
            DatasetGenerator,
            types,
            shapes
        )
        preds = model.predict(dataset) 
        return preds
    
    def get_labels(self, get_weight_labels=False, get_labels=False):
        with File(self.train_file, "r") as f:
            if get_weight_labels:
                return f[self.config.edge_name][:]
            if get_labels:
                return f[self.config.vertex_feat_name][:]

    def plotting_efficiency(self, n_modelfiles, Ntotal, logger):
        effs = []
        #for i in range(1,n_modelfiles+1):
        for i in [1,10,20]:
            logger.info(f"plotting efficiency for model model_epoch{i:03d}")
            _, model = self.load_model(f"{self.config.output}/modelfiles/model_epoch{i:03d}.h5")
            preds = self.get_predictions(model)
            labels = self.get_labels(get_weight_labels=True).astype(int)
            eff = calculate_efficiency(preds, labels, Ntotal)
            effs.append(eff)
        plot_eff = PlotBase(
            ylabel="efficiency",
            xlabel="epoch",
            n_ratio_panels=0,
            logy=False,
            )
        plot_eff.initialise_figure()
        plot_eff.axis_top.plot(effs, 'bo')
        plot_eff = create_figure(plot=plot_eff)
        plot_eff.savefig(f"{self.plot_dir}/eff_per_epoch.pdf")

    def plotting_pT_regression(self, modelfile, logger):
        logger.info(f"plotting pT regression for model model_epoch")
        model, _ = self.load_model(f"{self.config.output}/modelfiles/{modelfile}.h5")
        preds = self.get_predictions(model).flatten()
        labels = self.get_labels(get_labels=True).flatten()
        plot_pT = PlotBase(
            ylabel="predicted pT",
            xlabel="true pT",
            n_ratio_panels=0,
            logy=False,
            )
        plot_pT.initialise_figure()
        plot_pT.axis_top.plot(labels, preds, 'bo')
        plot_pT = create_figure(plot=plot_pT)
        plot_pT.savefig(f"{self.plot_dir}/pT_regression.pdf")
        
            




