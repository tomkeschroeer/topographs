import argparse as pars
from h5py import File
import os
from glob import glob

from matplotlib.pyplot import get 

from topograph.modules import (
    GetConfiguration,
    DatasetCreater,
    GlobalConfig,
    get_logger
)

def get_parser():
    """
    Argument parser for the train script

    Returns
    -------
    args: parse_args
    """
    parser = pars.ArgumentParser()
    parser.add_argument(
        '--config', 
        '-c', 
        type=str,
        required=True, 
        help='config file giving the network parameters'
    )
    parser.add_argument(
        '--prepare',
        '-p',
        action="store_true",
        help='prepares samples'
    )
    parser.add_argument(
        '--merge',
        '-m',
        action="store_true",
        help='merge samples'
    )

    args = parser.parse_args()
    return args

if __name__ == "__main__":

    args = get_parser()
    config = GetConfiguration(args.config)
    stepsize = 300_000
    logger = get_logger()
    if args.prepare:
        global_conf = GlobalConfig()
        input_files = config.get_all_input_files()
        training_file_dir = f"{config.output}/training_files"
        os.makedirs(training_file_dir, exist_ok=True)
        for input_file_ind in range(len(input_files)):
            with File(input_files[input_file_ind], "r") as f:
                njets = len(f["/jets"][:])
            n_steps = njets//stepsize
            for step in range(n_steps):
                logger.info(f"Process file number {input_file_ind+1} from {len(input_files)}, step {step+1}/{n_steps}")
                datasets = DatasetCreater(input_file=input_files[input_file_ind], step=step, stepsize=stepsize)
                if step == 0 and input_file_ind == 0:
                    with File(f"{training_file_dir}/{config.training_file_name}", "w") as train_file:
                        train_file.create_dataset("Y_vertex_features", data = datasets.get_vertex_feat_y(), chunks=True, maxshape=(None,len(global_conf.vertex_features)))
                        train_file.create_dataset("Y_edge_features", data = datasets.get_edge_feat_y(), chunks=True, maxshape=(None,40,len(global_conf.edge_features)))
                        train_file.create_dataset("Y_edge", data = datasets.get_edge_y(), chunks=True, maxshape=(None,40,1))
                        train_file.create_dataset("X_train_tracks", data = datasets.get_track_input(), chunks=True, maxshape=(None,40,len(global_conf.track_inputs)))
                else:
                    njets_step = datasets.get_n_valid_jets()
                    with File(f"{training_file_dir}/{config.training_file_name}", "a") as train_file:
                        train_file["Y_vertex_features"].resize((train_file["Y_vertex_features"].shape[0] + njets_step), axis=0)
                        train_file["Y_vertex_features"][-njets_step:] = datasets.get_vertex_feat_y()
                        train_file["Y_edge_features"].resize((train_file["Y_edge_features"].shape[0] + njets_step), axis=0)
                        train_file["Y_edge_features"][-njets_step:] = datasets.get_edge_feat_y()
                        train_file["Y_edge"].resize((train_file["Y_edge"].shape[0] + njets_step), axis=0)
                        train_file["Y_edge"][-njets_step:] = datasets.get_edge_y()
                        train_file["X_train_tracks"].resize((train_file["X_train_tracks"].shape[0] + njets_step), axis=0)
                        train_file["X_train_tracks"][-njets_step:] = datasets.get_track_input()
    if args.merge:
        with File(f"{config.output}/training_files/training_dataset_merged.h5","w") as h5fw:
            row1 = {}
            logger.info("Merging datasets...")
            for i, h5name in enumerate(glob(f"{config.output}/training_files/*.h5")):
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
