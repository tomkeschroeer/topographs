import argparse as pars

from topograph.modules import GetConfiguration
from topograph.preprocessing_tools import (  # H5toTfrecordsConverter,
    Apply_Scaler,
    Merge,
    Prepare,
    Scaler,
    MergeTestFile,
    Salt_preprocess,
)
from topograph.modules import (
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
        "--config",
        "-c",
        type=str,
        required=True,
        help="config file giving the network parameters",
    )
    parser.add_argument(
        "--onefile", "-o", action="store_true", help="make one samples files"
    )
    parser.add_argument("--scale", "-s", action="store_true", help="scale samples")
    parser.add_argument(
        "--apply_scales", "-a", action="store_true", help="apply scales"
    )
    parser.add_argument("--prepare", "-p", action="store_true", help="prepares samples")
    parser.add_argument("--merge", "-m", action="store_true", help="merge samples")
    parser.add_argument("--merge_test_files", "-mtf", action="store_true", help="merge test files")
    parser.add_argument("--saltpreprocess", "-sp", action="store_true", help="salt preprocess")
    parser.add_argument("--to_records", "-r", action="store_true", help="merge samples")

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    # logger = get_logger()
    args = get_parser()
    config = GetConfiguration(args.config)
    output_dir = f"{config.output}"
    dataset_types = {
        "": {
            "njets": config.njets,
            "final_filename": f"{output_dir}/{config.training_file_name}",
        },
        "_val": {
            "njets": config.njets_val,
            "final_filename": f"{output_dir}/{config.validation_file_name}",
        },
        "_test": {
            "njets": config.njets_test,
            "final_filename": f"{output_dir}/{config.testing_file_name}",
        },
    }
    if args.prepare:
        prepare = Prepare(config, dataset_types)
        prepare.Run()
    if args.scale:
        scale = Scaler(config)
        scale.Run()
    if args.merge:
        merge = Merge(config, dataset_types)
        merge.Run()
    if args.apply_scales:
        apply_scales = Apply_Scaler(config, dataset_types)
        apply_scales.Run()
    if args.merge_test_files:
        merge_test_file = MergeTestFile(config, dataset_types)
        merge_test_file.Run()
    if args.saltpreprocess:
        saltpreprocess = Salt_preprocess(config, dataset_types)
        saltpreprocess.Run()
