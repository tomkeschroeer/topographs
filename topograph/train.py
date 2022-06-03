import argparse as pars
import numpy as np

from modules import GetConfiguration
from modules import get_model

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

    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = get_parser()
    config = GetConfiguration(args.config)
    x = np.arange(60000000).reshape(100000,40,15)
    y = np.arange(1800000).reshape(100000,18)

    model = get_model(input_feat=(40,15), input_weight=(40,15), config=config)
    model.fit([x,x],y)