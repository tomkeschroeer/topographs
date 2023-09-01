from glob import glob
import numpy as np
from h5py import File

from topograph.modules.tools import get_logger, scary_shuffle


class MergeTestFile:
    def __init__(self, config, dataset_types):
        self.config = config
        self.dataset_types = dataset_types
        self.test_files = self.config.test_files

    def Run(self):
        logger = get_logger()
        output_file = (
            f"{self.config.output}/{self.config.testing_file_name}".replace(
                "//", "/"
            ).replace(".h5","")
        )
        jet_types = self.config.jet_types
        if len(jet_types) == 1:
            logger.info("no need to merge, only one jet type is used.")
        njets = self.config.njets_test
        stepsize = min(500_000, int(njets / 2))
        n_steps = njets//stepsize
        if njets%stepsize != 0:
            n_steps +=1
        with File(self.test_files[jet_types[0]],"r") as f:
            test_file_keys = list(f.keys())
        jet_type_keys = [key[:-(len(jet_types[0])+1)] for key in test_file_keys if f"_{jet_types[0]}" == key[-(len(jet_types[0])+1):]]
        for jet_type_key in jet_type_keys:
            test_file_keys.remove(f"{jet_type_key}_{jet_types[0]}")
        n_start = 0
        with File(f"{self.config.output}/{self.config.testing_file_name}".replace(".h5","") + ".h5", "w") as h5fo:
            create_file = True
            for i, jet_type in enumerate(jet_types):
                with File(
                    self.test_files[jet_type], "r"
                ) as h5fr:
                    n_jets_file = len(h5fr[test_file_keys[0]])
                    for step in range(n_steps):
                        if create_file:
                            for key in test_file_keys:
                                h5fo.create_dataset(data = h5fr[key][step*stepsize:(step+1)*stepsize], name=key, maxshape=(None,*h5fr[key].shape[1:]))
                            shape_edge = h5fr[f"{self.config.edge_name}_{jet_type}"][step*stepsize:(step+1)*stepsize].shape
                            h5fo.create_dataset(data = h5fr[f"{self.config.edge_name}_{jet_type}"][step*stepsize:(step+1)*stepsize], name=f"{self.config.edge_name}_{jet_type}", maxshape=(None,*shape_edge[1:]))
                            shape_vertex = h5fr[f"{self.config.vertex_feat_name}_{jet_type}"][step*stepsize:(step+1)*stepsize].shape
                            h5fo.create_dataset(data = h5fr[f"{self.config.vertex_feat_name}_{jet_type}"][step*stepsize:(step+1)*stepsize], name=f"{self.config.vertex_feat_name}_{jet_type}", maxshape=(None,*shape_vertex[1:]))
                            jet_types_remain = jet_types.copy()
                            jet_types_remain.remove(jet_type)
                            for jet_type_inter in jet_types_remain:
                                data_edge = np.full(shape_edge, fill_value=0)
                                h5fo.create_dataset(data = data_edge, name=f"{self.config.edge_name}_{jet_type_inter}", maxshape=(None,*shape_edge[1:]))
                                data_vertex = np.full(shape_vertex, fill_value=-999.0)
                                h5fo.create_dataset(data = data_vertex, name=f"{self.config.vertex_feat_name}_{jet_type_inter}", maxshape=(None,*shape_vertex[1:]))
                            create_file = False
                        else:
                            for key in test_file_keys:
                                h5fo[key].resize((h5fo[key].shape[0]+ stepsize),axis=0)
                                h5fo[key][-stepsize:] = h5fr[key][step*stepsize:(step+1)*stepsize]
                            h5fo[f"/{self.config.edge_name}_{jet_type}"].resize((h5fo[f"{self.config.edge_name}_{jet_type}"].shape[0]+stepsize), axis=0)
                            h5fo[f"/{self.config.edge_name}_{jet_type}"][-stepsize:] = h5fr[f"{self.config.edge_name}_{jet_type}"][step*stepsize:(step+1)*stepsize]
                            h5fo[f"/{self.config.vertex_feat_name}_{jet_type}"].resize((h5fo[f"{self.config.vertex_feat_name}_{jet_type}"].shape[0]+stepsize), axis=0)
                            h5fo[f"/{self.config.vertex_feat_name}_{jet_type}"][-stepsize:] = h5fr[f"{self.config.vertex_feat_name}_{jet_type}"][step*stepsize:(step+1)*stepsize]
                            jet_types_remain = jet_types.copy()
                            jet_types_remain.remove(jet_type)
                            for jet_type_inter in jet_types_remain:
                                first_dim = (h5fr[test_file_keys[0]][step*stepsize:(step+1)*stepsize].shape)[0]
                                data_edge = np.full((first_dim, *shape_edge[1:]), fill_value=0)
                                h5fo[f"{self.config.edge_name}_{jet_type_inter}"].resize((h5fo[f"{self.config.edge_name}_{jet_type_inter}"].shape[0]+stepsize), axis=0)
                                h5fo[f"{self.config.edge_name}_{jet_type_inter}"][-stepsize:] = data_edge
                                data_vertex = np.full((first_dim, *shape_vertex[1:]),  fill_value=-999.0)
                                h5fo[f"{self.config.vertex_feat_name}_{jet_type_inter}"].resize((h5fo[f"{self.config.vertex_feat_name}_{jet_type_inter}"].shape[0]+stepsize), axis=0)
                                h5fo[f"{self.config.vertex_feat_name}_{jet_type_inter}"][-stepsize:] = data_vertex