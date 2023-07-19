import argparse as pars

import numpy as np

from topograph.modules import GetConfiguration
from topograph.plotting import GetEpochPrediction, Plotter


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
        "--epoch", "-e", type=int, default=None, help="epoch number to evaluate"
    )
    parser.add_argument(
        "--cutval", "-v", type=float, default=None, help="cut value used for efficiency"
    )
    parser.add_argument(
        "--vars", "-o", type=str, default=None, help="numbers of variable"
    )

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = get_parser()
    config = GetConfiguration(args.config)
    vars = args.vars
    if vars is not None:
        vars = np.array(vars.split(",")).astype(int)
    if args.epoch is not None:
        GetEpochs = GetEpochPrediction(config, args.epoch, vars=vars)
    else:
        Plotting = Plotter(config, args.cutval, vars)
        Plotting.Run()
