from typing import Tuple, Union, Optional, Literal
import numpy as np
from ..dtype import Data, MaskedData
from ..utils import MeanVar
class TroposphericData(Data):
    def plot(self, save:bool=False) -> None:
        return super().plot('Tropospheric', 'RdBu_r', 'rad', save)
    
class TroposphericMaskedData(MaskedData):
    def plot(self, save:bool=False) -> None:
        return super().plot('Tropospheric', 'RdBu_r', 'rad', save)


class Tropospheric:
    def __init__(self, shape:Tuple[int, int]):
        self.shape = shape

    def create(self, nslcs:int, heights:np.ndarray, order:Literal['first', 'second']='first', tropograde:float=4.5, tropovar:float=1.0, zerodelay:Optional[float]=None) -> Union[TroposphericData, TroposphericMaskedData]:
        """ Create troposheric delay for stack of slc(s)
        >>> 0.0003 m delay / m height -> 0.3 m / km -> 68 rad / km (-ve for large zerodelay)
        >>> 0.00005 m delay / m hieght -> 0.05 m / km -> 11 rad/km

        """
        degree = {'first' : 1, 'second' : 2}.get(order)
        heights = heights[:self.shape[0], :self.shape[1]] * 1e-3

        if zerodelay is not None:
            heights -= zerodelay

        grads = tropograde + tropovar * np.random.randn(degree, nslcs)
        
        tropodelay = np.zeros((*self.shape, nslcs))

        if order == 'first':
            tropodelay = np.matmul(heights[..., np.newaxis] , grads[0:1,])  # Topographic delay in an interferogram is the difference between two acquisition, still in rad

        elif  order == 'second':
            tropodelay = np.matmul(heights[..., np.newaxis] , grads[0:1,]) + \
                         np.matmul(heights[..., np.newaxis]**2, grads[1:2])
        else:
            raise NotImplementedError(f'number of order coefficents can only be either 1 or 2. But {degree} was found')
        
        if zerodelay is not None:
            tropodelay -= tropodelay[np.unravel_index(np.argmin(heights), heights.shape)]

        tropodelay = MeanVar.apply(tropodelay, tropograde, tropovar, axis=-1)

        tropodelay = np.squeeze(tropodelay)

        if isinstance(heights, (MaskedData, np.ma.MaskedArray)):
            return TroposphericMaskedData(tropodelay)
        
        return TroposphericData(tropodelay)