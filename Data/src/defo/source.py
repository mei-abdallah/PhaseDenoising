from typing import Tuple, Literal, Mapping, Union, Optional
import numpy as np
from .model import Mogi, Okada
from .kwargs import SrcKwargs
from ..dtype import Data
from ..utils import Unit
from ..objects import Sensor, Projection

class SourceData(Data):
    def plot(self, save:bool=False) -> None:
        return super().plot('Deformation Map', 'coolwarm', 'm', save)
     
class Source:
    def __init__(self, shape:Tuple[int, int]) -> None:
        self.shape = shape
        self.kwargs = SrcKwargs(shape)
 
    def create(self, source:Literal['mogi', 'sill', 'dyke', 'quake', 'normal', 'thrust', 'strike-slip', 'left-lateral', 'right-lateral'], 
               sensor:Union[Sensor, str], track:Optional[Literal['asc', 'desc', 'random']]=None, ) -> SourceData:
        sensor = Sensor(sensor)
        if source == 'mogi':
            model = Mogi() 
            
        elif source in ['sill', 'dyke', 'quake', 'normal', 'thrust', 'strike-slip', 'left-lateral', 'right-lateral']:
            model = Okada()

        else:
            raise ValueError(f'Invalid pattern of {source} type')
        
        resolution = {'x': 1e5 / self.shape[1],
                      'y': 1e5 / self.shape[0]}
        
        if source in ['mogi', 'dyke', 'sill']:
            resolution = {key : value * 0.1 for key, value in resolution.items()}
        
        coord = self.getCoord(resolution)

        center = self.kwargs.getCentre(resolution, edge=0.35)
        
        srckwargs = self.kwargs.getKwargs(source, unit=True)

        proj = Projection().create(sensor, track)
        
        disp = model(center, coord, **srckwargs)

        defo = proj(disp).reshape(self.shape)

        # defo = MinMax.apply(defo, -1, 1) 
        # defo = Unit.apply(defo) # Unit pattern

        return SourceData(defo)
    
    def getCoord(self, resolution:Mapping[str, float]) -> np.ndarray:
        """ get a random coordinate for the source"""
        length, width = self.shape
        yy, xx = np.mgrid[0:length:length*1j, 0:width:width*1j]

        coord =  np.vstack((xx.reshape(1, -1) * resolution['x'], 
                            yy.reshape(1, -1) * resolution['y']))
        return coord
    