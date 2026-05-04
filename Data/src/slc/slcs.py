from typing import Tuple
import numpy as np
from ..dtype import Data

class SlcsData(Data):
    def plot(self, save:bool=False):
        return super().plot('Single Look Complex', 'RdBu_r', 'rad', save)
    
class Slcs:
    def __init__(self, shape:Tuple[int, int]) -> None:
        self.shape = shape

    def create(self, defo:Data, scale:float=1.0) -> SlcsData:
        if defo.ndim == 2:
            defo = defo.unsqueeze()
            
        noise = Data(2 * np.pi * np.random.rand(*self.shape, 1))
        amp = np.random.rayleigh(scale, (*self.shape, 1))
        defo = Data(np.concatenate((np.zeros((*self.shape, 1)), np.cumsum(defo, axis=-1)), axis=-1) + noise) # np.cumsum(np.concatenate((noise, defo), axis=-1), axis=-1)
        defo = np.exp(1j * defo.wrap())
        slcs = amp * defo
        return SlcsData(slcs, complex)