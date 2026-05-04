from typing import Tuple
import numpy as np
from ..dtype import Data

class SlcData(Data):
    def plot(self, save:bool=False):
        return super().plot('Single Look Complex', 'RdBu_r', 'rad', save)

class Slc:
    def __init__(self, shape:Tuple[int, int]) -> None:
        self.shape = shape
    
    def create(self, scale:float=1.0) -> SlcData:
        phase = np.exp(1j * 2 * np.pi * np.random.rand(*self.shape))
        amp = np.random.rayleigh(scale, self.shape)
        slc = amp * phase
        return SlcData(slc, complex)
    

