from glob import glob
import numpy as np
from h5py import File

from topograph.modules.tools import get_logger, scary_shuffle


class Merge:
    def __init__(self, config, dataset_types):
        self.config = config
        self.dataset_types = dataset_types

    def Run(self):
        logger = get_logger()
        output_file = (
            f"{self.config.output}/{self.config.preprocessing_file_name}".replace(
                "//", "/"
            ).replace(".h5","")
        )
        jet_types = self.config.jet_types
        if len(jet_types) == 1:
            logger.info("no need to merge, only one jet type is used.")
        njets = int((
            self.dataset_types[""]["njets"]
            + self.dataset_types["_val"]["njets"]
            + self.dataset_types["_test"]["njets"]
        )/len(jet_types))
        stepsize = min(500_000, int(njets / 2))
        n_steps = njets // stepsize + 1
        array_full={}
        array_dict_labels_full={}
        array_dict_comb_labels_full = {}
        non_stack_keys = ["Y_edge", "Y_vertex_features"]
        keys = [key for key in File(f"{output_file}_{jet_types[0]}.h5").keys() if key not in non_stack_keys]
        with File(
            f"{output_file}.h5", "w"
        ) as h5fw:
            for step in range(n_steps):
                for jet_type in jet_types:
                    with File(f"{output_file}_{jet_type}.h5", "r") as h5fr:
                        array_dict = {f"{key}_{jet_type}": h5fr[f"/{key}"][step*stepsize:(step+1)*stepsize] for key in keys}
                        array_full.update(array_dict)
                        for jet_type_2 in jet_types:
                            if jet_type_2 == jet_type:
                                array_dict_labels = {f"{key}_{jet_type}_{jet_type_2}": h5fr[f"/{key}"][step*stepsize:(step+1)*stepsize] for key in non_stack_keys}
                                array_dict_labels_full.update(array_dict_labels)
                            else:
                                with File(f"{output_file}_{jet_type_2}.h5", "r") as h5fr_2:
                                    y_edge = h5fr_2[f"/Y_edge"][step*stepsize:(step+1)*stepsize]
                                    y_vertex = h5fr_2[f"/Y_vertex_features"][step*stepsize:(step+1)*stepsize]
                                array_dict_labels = {
                                    f"Y_edge_{jet_type}_{jet_type_2}": np.full(shape=y_edge.shape, dtype=y_edge.dtype, fill_value=0),
                                    f"Y_vertex_features_{jet_type}_{jet_type_2}": np.full(shape=y_vertex.shape, dtype=y_vertex.dtype, fill_value=-999.),
                                }
                                array_dict_labels_full.update(array_dict_labels)
                    # array_dict_comb_labels = {key: np.concatenate([array_dict_labels_full[f"{key}_{jet_type}_{jet_type_2}"] for jet_type_2 in jet_types]) for key in non_stack_keys}
                    array_dict_comb_labels = {}
                    for key in non_stack_keys:
                        array_dict_comb_labels[f"{key}_{jet_type}"] = np.concatenate([array_dict_labels_full[f"{key}_{jet_type}_{jet_type_2}"] for jet_type_2 in jet_types])
                    array_dict_comb_labels_full.update(array_dict_comb_labels)
                array_dict_comb = {key: np.concatenate([array_full[f"{key}_{jet_type}"] for jet_type in jet_types]) for key in list(keys)}
                array_dict_comb.update(array_dict_comb_labels_full)
                nentries = len(next(iter(array_dict_comb.values())))
                indices = np.linspace(0, nentries-1, nentries).astype(int)
                scary_shuffle(indices)
                for key in array_dict_comb.keys():
                    array_dict_comb[key] = array_dict_comb[key][indices]
                    shape = array_dict_comb[key].shape
                    maxshape = (None,) if len(shape) == 1 else (None, *shape[1:])
                    if step == 0:
                        h5fw.create_dataset(key, data=array_dict_comb[key],chunks=True,  maxshape=maxshape)
                    else:
                        data = array_dict_comb[key]
                        h5fw[key].resize((h5fw[key].shape[0] + shape[0]), axis=0)
                        h5fw[key][-shape[0]:] = data