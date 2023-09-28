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
            with File(
                f"{output_file}_{jet_types[0]}.h5", "a"
            ) as f:
                if "jet_type" not in f.keys():
                    f.create_dataset("jet_type", data=np.full(shape=(len(f[f"{self.config.edge_name}_{jet_types[0]}"])), fill_value=0))
            exit()
        njets = int((
            self.dataset_types[""]["njets"]
            + self.dataset_types["_val"]["njets"]
            + self.dataset_types["_test"]["njets"]
        )/len(jet_types))
        stepsize = min(50_000, int(njets / 2))
        print(stepsize)
        n_steps = njets // stepsize + 1
        array_full={}
        array_dict_labels_full={}
        array_dict_comb_labels_full = {}
        truthOrigin_dict = {}
        truthOrigin_dict_full = {}
        njets_dict = {jet_type: len(File(f"{output_file}_{jet_type}.h5", "r")[f"{self.config.edge_name}_{jet_type}"]) for jet_type in jet_types}
        non_stack_keys = [self.config.edge_name, self.config.vertex_feat_name]
        non_stack_keys_for_check = [f"{self.config.edge_name}_{jet_types[0]}", f"{self.config.vertex_feat_name}_{jet_types[0]}"]
        keys = [key for key in File(f"{output_file}_{jet_types[0]}.h5").keys()]# if key not in non_stack_keys_for_check]
        with File(
            f"{output_file}.h5", "w"
        ) as h5fw:
            for step in range(n_steps):
                logger.info(f"merging in process, step {step} from {n_steps}")
                for i, jet_type in enumerate(jet_types):
                    logger.info(f"getting data for {jet_type}-jets...")
                    with File(f"{output_file}_{jet_type}.h5", "r") as h5fr:
                        truthorigin = h5fr["/track_extra_truth"][step*stepsize:(step+1)*stepsize]
                        inputs = h5fr["/X_train_tracks"][step*stepsize:(step+1)*stepsize]
                        array_dict = {f"{key}_{jet_type}": h5fr[f"/{key}"][step*stepsize:(step+1)*stepsize] for key in keys}
                        array_dict[f"jet_type_{jet_type}"] = np.full(shape=(len(array_dict[f"{keys[0]}_{jet_type}"])), fill_value=i)
                        array_dict[f"truthorigin_{jet_type}"] = truthorigin
                        array_full.update(array_dict)
                        # for jet_type_2 in jet_types:
                            #  if jet_type_2 == jet_type:
                            # array_dict_labels = {f"{key}_{jet_type}_{jet_type_2}": h5fr[f"/{key}_{jet_type}"][step*stepsize:(step+1)*stepsize] for key in non_stack_keys}
                            # array_dict_labels_full.update(array_dict_labels)
                            # else:
                            #     with File(f"{output_file}_{jet_type_2}.h5", "r") as h5fr_2:
                            #         y_edge = h5fr_2[f"/{self.config.edge_name}_{jet_type_2}"][step*stepsize:(step+1)*stepsize]
                            #         y_vertex = h5fr_2[f"/{self.config.vertex_feat_name}_{jet_type_2}"][step*stepsize:(step+1)*stepsize]
                            #     array_dict_labels = {
                            #         f"{self.config.edge_name}_{jet_type}_{jet_type_2}": np.full(shape=y_edge.shape, dtype=y_edge.dtype, fill_value=0),
                            #         f"{self.config.vertex_feat_name}_{jet_type}_{jet_type_2}": np.full(shape=y_vertex.shape, dtype=y_vertex.dtype, fill_value=-999.),
                            #     }
                            #     array_dict_labels_full.update(array_dict_labels)
                    # array_dict_comb_labels = {key: np.concatenate([array_dict_labels_full[f"{key}_{jet_type}_{jet_type_2}"] for jet_type_2 in jet_types]) for key in non_stack_keys}
                    array_dict_comb_labels = {}
                    # for key in non_stack_keys:
                    #     array_dict_comb_labels[f"{key}_{jet_type}"] = np.concatenate([array_dict_labels_full[f"{key}_{jet_type}_{jet_type_2}"] for jet_type_2 in jet_types])
                    array_dict_comb_labels_full.update(array_dict_comb_labels)
                # for jet_type in jet_types:
                #     array_full[f"jet_type_{jet_types}"] = np.full(shape=(array_full[f"{key}_{jet_type}"]))
                array_dict_comb = {key: np.concatenate([array_full[f"{key}_{jet_type}"] for jet_type in jet_types]) for key in list(keys)+ ["jet_type", "truthorigin"]}
                array_dict_comb.update(array_dict_comb_labels_full)
                # for jet_type in self.jet_types:
                #     mask_b = array_dict_comb["jet_type"] == jet_types.index("b")
                #     mask_c = array_dict_comb["jet_type"] == jet_types.index("c")
                #     _, comm_b, comm_c  = np.intersect1d(
                #         array_dict_comb["jet_pt"]["eventNumber"][mask_b],
                #         array_dict_comb["jet_pt"]["eventNumber"][mask_c],
                #         return_indices=True
                #     )
                #     index_add_c = 0
                #     index_add_b = 0
                    
                #     for jet_type in jet_types:
                #         if jet_types.index(jet_type) < jet_types.index("c"):
                #             index_add_c += sum(array_dict_comb["jet_type"] == jet_types.index(jet_type))
                #         if jet_types.index(jet_type) < jet_types.index("b"):
                #             index_add_b += sum(array_dict_comb["jet_type"] == jet_types.index(jet_type))
                #     comm_b += index_add_b
                #     comm_c += index_add_c

                #     inputs_b = array_dict_comb[self.config.tracks_name][comm_b]
                #     inputs_c = array_dict_comb[self.config.tracks_name][comm_c]

                #     eq = [np.array_equal(i_b[:1], i_c[:1]) for i_b, i_c in zip(inputs_b, inputs_c)]
                #     comm_b = comm_b[eq]
                #     comm_c = comm_c[eq]

                #     Y_edge_c = array_dict_comb["Y_edge_c"]
                #     Y_vertex_feat_c = array_dict_comb["Y_vertex_features_c"]

                #     array_dict_comb["Y_edge_c"][comm_b] = Y_edge_c[comm_c]
                #     array_dict_comb["Y_vertex_features_c"][comm_b] = Y_vertex_feat_c[comm_c]
                #     array_dict_comb["jet_type"][comm_b] = 12
 
                #     for key in array_dict_comb.keys():
                #         array_dict_comb[key] = np.delete(array_dict_comb[key],comm_c, axis=0)
                nentries = len(next(iter(array_dict_comb.values())))
                # print(nentries)
                # print(array_dict_comb["Y_vertex_features_c"])
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