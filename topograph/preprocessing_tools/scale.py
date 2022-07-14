import numpy as np

from topograph.modules.tools import (
    get_logger
)

class Scaler:
    def __init__(self, config):
        self.config = config

    def Run(self):
        logger = get_logger()
        logger.info("scale inputs...")
