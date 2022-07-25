from puma import Histogram, HistogramPlot
from h5py import File
from tensorflow.keras.models import load_model
from tensorflow.data import Dataset
from tensorflow import (
    TensorShape,
    float32,
    int32
)

from topograph.modules import (
    step_activation,
    DataLoader
)

class Plotter:
    def __init__(self, config):
        self.config = config
        self.model = self.load_model()
        self.get_predictions()

    def load_model(self):
        modelfile = self.config.evaluation["model"]
        modelpath = f"{self.config.output}/modelfiles/{modelfile}".replace("//","/")
        model = load_model(
            modelpath,
            custom_objects={"step_activation":step_activation}
        )
        return model

    def get_predictions(self):
        metadata_dict = {}
        with File(f"{self.config.output}/{self.config.training_file_name}", "r") as f:
            metadata_dict["n_jets"], metadata_dict["n_trks"], metadata_dict["n_trk_features"] = f[f"{self.config.tracks_name}"].shape
            _, metadata_dict["n_vertex_feat"] = f[f"{self.config.vertex_feat_name}"].shape
        
        types = ({
            "input_1": float32,
            "input_2": float32
        })
        shapes = ({
            "input_1": TensorShape((None, metadata_dict["n_trks"], metadata_dict["n_trk_features"])),
            "input_2": TensorShape((None, metadata_dict["n_trks"], metadata_dict["n_trk_features"]))
        })

        dataset = Dataset.from_generator(
            DataLoader(
                input=f"{self.config.output}/{self.config.training_file_name}",
                train_dataset=False,
                metadata_dict=metadata_dict,
                savetracks=True,
                track_name=self.config.tracks_name,
                edge_name=self.config.edge_name,
                edge_feat_name=self.config.edge_feat_name,
                vertex_feat_name=self.config.vertex_feat_name
            ),
            types,
            shapes
        )
        pred = self.model.predict(dataset)
        print(pred)

    def plotting_efficiency(self):
        plot = "h"


