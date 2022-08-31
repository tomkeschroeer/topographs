import argparse as pars

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
    parser.add_argument("--epoch", "-e", type=int, help="epoch number to evaluate")

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = get_parser()
    config = GetConfiguration(args.config)
    if args.epoch:
        GetEpochs = GetEpochPrediction(config, args.epoch)
    else:
        Plotting = Plotter(config)
