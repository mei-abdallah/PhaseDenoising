from typing import Tuple, Union
import os
import numpy as np
from ..dtype import Data
from ..objects import Sensor
from ..utils import MinMax
import warnings

try:
    import pyfftw
    pyfftw.config.NUM_THREADS = min(os.cpu_count(), 4)
    fft = pyfftw.interfaces.numpy_fft
except:
    fft = np.fft

warnings.filterwarnings('ignore')

class TopographicData(Data):
    def plot(self, save:bool=False) -> None:
        return super().plot('Topographic error', 'RdBu_r', 'rad', save)
    
class Topographic:
    def __init__(self, shape:Tuple[int, int]) -> None:
        self.shape = shape

    def create(self, heights:np.ndarray, spatial:np.ndarray, sensor:Union[Sensor, str], beta:float=2.2) -> TopographicData:
        """Create topography data for a given set of heights and slope angle (beta)."""

        noise = np.random.randn(*self.shape)

        noise = fft.fftshift(fft.fft2(noise))

        # scale the spectrum with the power law
        # avoid zero distance
        x = 0.25 + np.arange(-self.shape[1]/2, self.shape[1]/2)
        y = 0.25 + np.arange(-self.shape[0]/2, self.shape[0]/2)

        xx, yy = np.meshgrid(x, y)
        k = np.sqrt(np.square(xx) + np.square(yy))

        # primary DEM error
        pbeta = 8 - 2 * beta
        pnoemer = k ** (pbeta/2)
        pnoemer[pnoemer == 0] = 1

        # small DEM error
        sbeta = 8 - 3 * beta
        snoemer = k ** (sbeta / 2)
        snoemer[snoemer == 0] = 1

        #  combine
        perror = abs(fft.ifft2(noise / pnoemer))
        serror = abs(fft.ifft2(noise / snoemer))

        perror = MinMax.apply(perror, -0.1, 0.1)
        serror = MinMax.apply(serror, -0.1, 0.1)

        error = heights * np.where(heights > heights.mean(), perror, serror)

        sensor = Sensor(sensor)

        coeff = spatial / (sensor.getSlantRange() * np.sin(np.deg2rad(sensor.getIncidentAngle())))

        topographic = np.matmul(error[..., np.newaxis], coeff[np.newaxis,])

        topographic *= ((4 * np.pi) / sensor.getWaveLength())

        return TopographicData(topographic)