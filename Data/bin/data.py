from typing import Sequence, Union, Optional
import argparse
import sys, time
import numpy as np
import pandas as pd
sys.path.append('.')
from src import Ifgs
from IPython.display import clear_output

class Action(argparse.Action):
    def __call__(
        self,
        parser: argparse.ArgumentParser,
        namespace: argparse.Namespace,
        values: Optional[Union[str, Sequence[str]]],
        option_string: Optional[str] = None,
    ) -> None:
        args = {}
        for idx in range(len(values) // 2):
            key = values[idx * 2]
            try:
                value = float(values[idx * 2 + 1])
            except:
                value = values[idx + 1]
            args.update({ key: value})
        setattr(namespace, self.dest, args)



parser = argparse.ArgumentParser(
                    prog='Create Data',
                    description='this program creates Time series of interferograms',
                    epilog='Text at the bottom of help')
parser.add_argument('-d',  '--dir',             dest='dir',       default='./Datasets', type=str,   help='The directory to store interferograms, e.g., DEFAULT = ./Dataset')
parser.add_argument('-ph', '--phase',           dest='phase',     default='train',      type=str,   help='The type of the generated interferograms, e.g., DEFAULT = train')
parser.add_argument('-nd', '--ndata',           dest='ndata',     default=100000,       type=int,   help='The number of the generated timeseries, e.g., DEFAULT = 10')
 
parser.add_argument('-n',  '--nifgs',           dest='nifgs',     default=10,           type=int,   help='The number of interferograms, e.g., DEFAULT = 10')
parser.add_argument('-r',  '--resolution',      dest='shape',     default=(48, 48),     type=int,   help='The spatial resolution of each interferogram , e.g., (<width>, <length>) DEFAULT = (48, 48)', 
                                                                                                         nargs=2, metavar=('<length>', '<width>'))
parser.add_argument('-l',  '--location',        dest='location',  default=None,                     help='The spatial location of the interferogram , e.g., (west <float>, east <float>, south <float>, north <float>).', 
                                                                                                         nargs=8, metavar=('west', '<float>', 'east', '<float>', 'south', '<float>', 'north', '<float>'), 
                                                                                                                                                              action=Action)
parser.add_argument('-lf', '--lfile',           dest='lfile',     default=None,         type=str,   help='The file contains the spatial location of the interferograms , e.g., (<dir>).', 
                                                                                                                 metavar='<dir>')
parser.add_argument('-p',  '--platform',        dest='platform',  default='SENTINEL',   type=str,   help='The satellite sensor , e.g., ("ASAR", "ERS", "ALOS1", "ALOS2", "RADARSAT", "SENTINEL", "CSK", "TSK") DEFAULT = SENTINEL.')
parser.add_argument('-od', '--orbitdeg',        dest='polydeg',   default=None,         type=str,   help='The dgree of orbital ramps , e.g., ("second", "third", and "fifth") DEFAULT = None.')
parser.add_argument('-m',  '--turbmethod',      dest='method',    default=None,         type=str,   help='The method to create turblent APS , e.g., ("fft", "cov", "eig", "fractal", and "trend") DEFAULT = None.')
parser.add_argument('-s',  '--source',          dest='source',    default=None,         type=str,   help='the source of deformation e.g., ("mogi", "quake", "sill", "dyke", "complex", "cone" and "peak") DEFAULT = None.')      # option that takes a value
parser.add_argument('-tr', '--trend', '--disp', dest='disp',      default=None,         type=str,   help='the trend of deformation displacment e.g., ("stable", "linear", "sinusoidal", "cosinusoidal", "periodic", "onset", "pulse", logarithmic", "exponential", "power", "coseismic", "postseismic", "longwave", "accumilationl", "timerelated", "complex", '
                                                                                                        f' "stable+sinusoidal", "stable+cosinusoidal", "stable+periodic", "linear+sinusoidal", "linear+cosinusoidal", "linear+periodic", "linear+logarithmic", "linear+exponential", "linear+power", and "linear+longwave") DEFAULT = None.')      # option that takes a value
parser.add_argument('-t',  '--track',           dest='track',     default=None,         type=str,   help='the track of satellite e.g., ("asc", "desc", and "random") DEFAULT = None.')      # option that takes a value
parser.add_argument('-sd', '--startday',        dest='startday',  default=None,         type=int,   help='The first day of deformation, e.g., DEFAULT = None')
parser.add_argument('-dl', '--defolimits',      dest='limits',    default=None,                     help='The deformation limits the interferogram , e.g., (min <float>, max <float>) DEFAULT = None',
                                                                                                         nargs=4, metavar=('min', '<float>', 'max', '<float>'), action=Action)
parser.add_argument('-ct', '--cohthreshold',    dest='threshold', default=0.2,          type=float, help='The threshold for coherance mask, e.g., DEFAULT = 0.20')
parser.add_argument('-sn', '--snr',             dest='snr',       default=2.0,          type=float, help='The signal to noise ration , e.g., DEFAULT = 2.0')
parser.add_argument('-v',  '--verbose',         dest='verbose',                                    help='Boolean for showing the informations',                action='store_true',)  # on/off flag
parser.add_argument('-w',  '--warning',         dest='warning',                                    help='Boolean for showing the warnings',                    action='store_true',)  # on/off flag

args = parser.parse_args()
# print(args)

def main(inps:dict) -> None:
    start = time.time()
    print(f'startTime: {time.strftime("%Y_%m_%d_%H_%M_%S")}')
    if inps['lfile']:
        csvdata = pd.read_csv(inps['lfile'], index_col=False)

    idx = 0
    while idx < inps['ndata']:
        now = time.time()
        clear_output()
        sys.stdout.write(f'\r file # {idx:06d}, {(now - start) / 60 : 0.3f} mins, {int(idx / ((now - start) / 60)) : 03d} [files/min]')
        sys.stdout.flush()
        if inps['lfile']:
            inps['location'] = dict(csvdata.iloc[np.random.randint(0, len(csvdata))])
        try:
            ifgs = Ifgs.create(**inps)
            ifgs.write(f'{inps["dir"]}/{inps["source"]}/{inps["phase"]}/stack_{inps["source"]}_{idx:06d}.pkl', 'obj')
            idx += 1
        except:
            pass
    print(f'finishTime: {time.strftime("%Y_%m_%d_%H_%M_%S")}')

if __name__ == '__main__':
    # pass
    main(vars(args))