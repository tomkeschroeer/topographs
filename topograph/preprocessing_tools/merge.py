from h5py import File
from glob import glob

from topograph.modules.tools import (
    get_logger
)

class Merge:
    def __init__(self, config):
        self.config = config
         
    def Run(self):
        logger = get_logger()
        with File(f"{self.config.output}/training_files/training_dataset_merged.h5","w") as h5fw:
            row1 = {}
            logger.info("Merging datasets...")
            for i, h5name in enumerate(glob(f"{self.config.output}/training_files/*.h5")):
                if "merged" in h5name: continue
                logger.info(f"Reading from file {h5name}...")
                with File(h5name,'r') as h5fr:
                    for key in list(h5fr.keys()):
                        arr_data = h5fr[key][:]
                        shape = list(arr_data.shape)
                        dslen = shape[0]
                        maxshape = shape.copy()
                        maxshape[0]=None
                        if i == 0:
                            h5fw.create_dataset(key, dtype="f", shape=(shape), maxshape=(maxshape))
                            row1[key] = dslen
                        else:
                            if (row1[key]+dslen <= len(h5fw[key])):
                                h5fw[key][row1[key]:row1[key]+dslen,:] = arr_data[:]
                            else:
                                shape[0] = row1[key]+dslen
                                h5fw[key].resize((shape))
                                h5fw[key][row1[key]:row1[key]+dslen,:] = arr_data[:]
                            row1[key] += dslen
                    logger.info(f"File {h5name} appended.")
            logger.info("Merging sucessfully done.")