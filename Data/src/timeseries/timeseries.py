from typing import Tuple, Literal, Optional, Union, Mapping, Sequence
from ..base import Stack
from ..ifg import Ifgs
from ..plot import Ploter
    
class TimeSeries(Stack):
    mode = 'timeseries'
    @classmethod
    def create(cls, 
               ndates:int, 
               shape:Tuple[int, int],
               location:Mapping[str, Union[str, float, Tuple[float, float]]],
               resolution:Optional[float]=None,
               reshape:Literal['crop', 'resize']='resize',
               platform:Literal['ASAR', 'ERS', 'ALOS1', 'ALOS2', 'RADARSAT', 'SENTINEL', 'CSK', 'TSK']='SENTINEL', 
               polydeg:Optional[Literal['second', 'third', 'fifth']]=None,
               method:Optional[Literal['fft', 'cov', 'eig', 'fractal', 'trend']]=None,
               order:Optional[Literal['first', 'second']]=None,
               source:Optional[Literal['cone', 'peak', 'complex', 'mogi', 'sill', 'dyke', 'quake']]=None,
               disp:Optional[Literal['stable', 'linear', 'sinusoidal', 'cosinusoidal', 'periodic', 'onset', 'pulse', 
                                     'logarithmic', 'exponential', 'power', 'coseismic', 'postseismic', 'longwave', 
                                     'stable+sinusoidal', 'stable+cosinusoidal', 'stable+periodic', 
                                     'linear+sinusoidal', 'linear+cosinusoidal', 'linear+periodic', 
                                     'linear+logarithmic', 'linear+exponential', 'linear+power', 'linear+longwave', 
                                     'accumilationl', 'timerelated', 'complex']]=None,
               track:Optional[Literal['asc', 'desc', 'random']]=None, 
               startday:Optional[int]=None, 
               limits:Optional[Mapping[str, float]]=None,
               duration:Optional[int]=None,
               threshold:float=0.2,
               snr:float=2.0,
               verbose:bool=False,
               warning:bool=False, **kwargs) -> 'TimeSeries':
        return super().create(ndates, shape, location, resolution, reshape, platform, polydeg, method, order, source, disp, track, 
                              startday, limits, duration, threshold, snr, verbose, warning)
    
    def getIfgs(self) -> Ifgs:
        return Ifgs(self.trend.torad().diff(),
                    self.topo.torad().diff(),
                    self.turb.torad().diff(),
                    self.tropo.torad().diff(),
                    self.orbit.torad().diff(),
                    self.thermo.torad().diff(),
                    self.decorr.torad().diff(),
                    self.cohmask.diff(),
                    self.watermask,
                    self.demheights)
    
    def plot(self, 
             labels:Sequence[Literal['topo', 'turb', 'tropo', 'orbit', 'trend', 'noisy']]=['turb', 'tropo', 'orbit', 'trend', 'noisy'], 
             unit:Literal['rad', 'm', 'cm']='cm', 
             save:bool=False):
        return Ploter(self).run(labels, unit, -9.0, 9.0, save)
    