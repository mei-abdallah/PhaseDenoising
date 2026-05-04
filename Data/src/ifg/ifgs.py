from typing import Tuple, Literal, Optional, Union, Mapping, Sequence
import numpy as np
from ..base import Stack
from ..dtype import Data
from ..slc import SlcData
from ..plot import Ploter

class Ifgs(Stack):
    mode = 'ifgs' 

    @classmethod
    def create(cls, 
               nifgs:int, 
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
               warning:bool=False, **kwargs) -> 'Ifgs':
        
        return super().create(nifgs + 1, shape, location, resolution, reshape, platform, polydeg, method, order, source, disp, track, 
                              startday, limits, duration, threshold, snr, verbose, warning)   
    
    def getSlcs(self, scale:float=1.0) -> SlcData:
        """get the SAR images data"""
        data = self.getNoisy()
        if data.ndim == 2:
            data = data.unsqueeze()

        shape = data.shape[:2]
        noise = Data(2 * np.pi * np.random.rand(*shape, 1))
        amp = np.random.rayleigh(scale, (*shape, 1))
        data = Data(np.concatenate((np.zeros((*shape, 1)), np.cumsum(data, axis=-1)), axis=-1) + noise) # np.cumsum(np.concatenate((noise, data), axis=-1), axis=-1)
        data = np.exp(1j * data.wrap())
        slcs = amp * data
        return SlcData(slcs, complex)
    
    def plot(self, 
             labels:Sequence[Literal['topo', 'turb', 'tropo', 'orbit', 'trend', 'noisy']]=['turb', 'tropo', 'orbit', 'trend', 'noisy'], 
             unit:Literal['rad', 'm', 'cm']='cm', 
             save:bool=False):
        return Ploter(self).run(labels, unit, -9.0, 9.0, save)