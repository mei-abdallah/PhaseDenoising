from typing import Tuple, Literal, Union, Callable, Mapping, Optional
import numpy as np
from .source import Mogi, Okada
from ..dtype import Data
from ..objects import Sensor

class WrapperData(Data):
    def plot(self, save:bool=False) -> None:
        return super().plot('Deformation Map', 'RdBu_r', 'rad', save)

class Wrapper:
    def __init__(self, shape:Tuple[int, int], nu:float=0.25) -> None:
        self.shape = shape
        self.nu = nu

    def create(self, resolution:Mapping[str, float], sensor:Union[Sensor, str], 
                source:Literal['mogi', 'okada', 'sill', 'dyke', 'quake', 'normal', 'thrust', 'strike-slip', 'left-lateral', 'right-lateral'], 
                center:Tuple[float, float], source_kwargs:Mapping[str, Union[int, float]], proj:Callable[[np.ndarray], np.ndarray], 
                defolimits:Optional[Mapping[str, float]]=None) -> Tuple[WrapperData, Mapping[str, Union[int, float]]]:
        
        defo = self.getMap(resolution, source, center, source_kwargs, proj)
        
        if defolimits is not None:
            accepted = True if defolimits['min'] <= np.max(np.abs(defo)) <= defolimits['max'] else False
            scale, ratio = 1.0, 0.1 # the ratio to change the major source kwargs 

            while not accepted:
                defo = self.getMap(resolution, source, center, source_kwargs, proj)

                if defolimits['min'] <= np.max(np.abs(defo)) <= defolimits['max']:
                    accepted = True
                
                else:
                    if defolimits['min'] > np.max(np.abs(defo)):
                        scale += ratio
                    else:
                        scale -= ratio

                    for kwarg in ['slip', 'opening', 'volume_change']:
                        if kwarg in source_kwargs:
                            source_kwargs[kwarg] *= scale

        defo *= ((4 * np.pi) / sensor.getWaveLength()) 

        return WrapperData(defo), source_kwargs
    
    def getMap(self, resolution:Mapping[str, float], source:Literal['mogi', 'okada', 'sill', 'dyke', 'quake', 'normal', 'thrust', 'strike-slip', 'left-lateral', 'right-lateral'], 
                 center:Tuple[float, float], source_kwargs:Mapping[str, Union[int, float]], proj:Callable[[np.ndarray], np.ndarray]) -> np.ndarray:
        """ Computes the displacement map in the east-north-up directions 
        
        Parameters:
            - resolution | dict | the spatial resolution in x, y directions in meters.
            - source | str | the event source that caused the deformation
            - center | (float, float) | the (x, y) coordinates of the defo source
            - source_kwargs | dict | keyword arguments to define the source
            - scale | float | the scale of the deformation
            - track | str | the tracking of the satellite mission 
        
        Returns:
        - disps | 2D Array | the diplacment in the east-noth-updirection, shape >> (3, npixs)
        """

        coord = self.getCoord(resolution)
        model = Mogi(self.nu) if source == 'mogi' else Okada(self.nu)
        disp = model(center, coord, **source_kwargs)
        defo = proj(disp).reshape(self.shape)
        return defo
    
    def getCoord(self, resolution:Mapping[str, float]) -> np.ndarray:
        """ Computes the coordinate at given resolution the refrence point is the upper left point """
        length, width = self.shape
        yy, xx = np.mgrid[0:length:length*1j, 0:width:width*1j]
        yy, xx =  np.asarray(yy) * resolution['y'], np.asarray(xx) * resolution['x']
        return np.vstack((xx.reshape(1, -1), yy.reshape(1, -1)))