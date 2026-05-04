from typing import Tuple
import numpy as np
import os
import warnings 
from ..utils import MinMax
from ..dtype import Data

try:
    import pyfftw
    pyfftw.config.NUM_THREADS = min(os.cpu_count(), 4)
    fft = pyfftw.interfaces.numpy_fft
except:
    fft = np.fft

warnings.filterwarnings('ignore')

class CoherenceData(Data):
    def plot(self, save:bool=False) -> None:
        return super().plot('Cohernece Mask', 'gray', None, save)

class Coherence:
    def __init__(self, shape:Tuple[int, int]) -> None:
        self.shape = shape

    
    def create(self, nslcs:int, threshold:float=0.2, beta:float=2.0) -> CoherenceData:
        masks = np.zeros((*self.shape, nslcs), dtype=np.float32)

        for nslc in range(nslcs):
            masks[..., nslc] = self.getMask(threshold, beta)

        return CoherenceData(masks, bool)

    
    def getMask(self, threshold:float=0.2, beta:float=2.0) -> np.ndarray:
        """simulate the random decorelation using FFT approach. """
        x, y = np.meshgrid(0.25 + np.arange(-self.shape[1]/2, self.shape[1]/2), 
                           0.25 + np.arange(-self.shape[0]/2, self.shape[0]/2))
        
        dist = np.sqrt(np.square(x) + np.square(y))

        noemer = dist ** (beta/2)
        noemer[noemer == 0] = 1

        # Simulate a uniform random signal
        noise = np.random.rand(*self.shape)

        noise = fft.fftshift(fft.fft2(noise))

        cohmap = np.abs(fft.ifft2(noise / noemer))

        cohmap = MinMax.apply(cohmap, 0, 1) # rescale to range [0, 1]  
        mask = np.where(cohmap <= threshold, np.ones_like(cohmap), np.zeros_like(cohmap)).astype(np.float32)                       # anything above the threshold is masked, creating blothcy areas of incoherence.
        return mask