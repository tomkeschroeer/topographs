import argparse as pars
from glob import glob

from matplotlib.pyplot import get 

from topograph.modules import (
    GetConfiguration,
)

from topograph.preprocessing_tools import (
    Prepare,
    Merge,
    Scale
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
        '--scale',
        '-s',
        action="store_true",
        help='scale samples'
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
    if args.scale:
        scale = Scale(config)
        scale.Run()
    if args.prepare:
        prepare = Prepare(config)
        prepare.Run()
    if args.merge:
        merge = Merge(config)
        merge.Run()
        