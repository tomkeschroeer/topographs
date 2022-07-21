from puma import Histogram, HistogramPlot
from tensorflow.keras.models import load_model

class Plotter:
    def __init__(self, config):
        self.config = config

    def plotting_efficiency(self):
        plot = "h"


