from typing import Tuple, Union
import numpy as np
from ..dtype import Data
from ..objects import Sensor
from ..utils import MeanStd


class SpeckleData(Data):
    def plot(self, save:bool=False) -> None:
        return super().plot('White noise', 'RdBu_r', 'rad', save)

class Speckle:
    def __init__(self, shape:Tuple[int, int]) -> None:
        self.shape = shape

    def create(self, nslcs:int) -> SpeckleData:
        # simulate thermal nosie
        thermnoise = np.random.randn(*self.shape, nslcs)
        thermnoise = MeanStd.apply(thermnoise, 0, 0.2)
        return SpeckleData(thermnoise)


class ThermoData(Data):
    def plot(self, save:bool=False) -> None:
        return super().plot('Thermal noise', 'RdBu_r', 'rad', save)

class Thermal:
    def __init__(self, shape:Tuple[int, int]) -> None:
        self.shape = shape

    def create(self, nslcs:int, ratio:float=0.01) -> ThermoData:
        # simulate thermal nosie
        thermnoise = np.random.randn(*self.shape, nslcs)
        thermnoise = MeanStd.apply(thermnoise)
        # thermnoise = (thermnoise - thermnoise.mean()) / thermnoise.std()

        thermnoise *= np.sqrt(2) * ratio
        thermnoise[:, :, 0] = np.zeros(self.shape)
        
        return ThermoData(thermnoise)


class DecorrData(Data):
    def plot(self, save:bool=False) -> None:
        return super().plot('Decorrelation noise', 'RdBu_r', 'rad', save)

class Decorelation:
    def __init__(self, shape:Tuple[int, int]) -> None:
        self.shape = shape

    def create(self, nslcs:int, temporal:np.ndarray, spatial:np.ndarray, sensor:Union[Sensor, str]) -> DecorrData:
        sensor = Sensor(sensor)
        # simulate decorrelation noise
        decorrnoise = np.random.randn(*self.shape, nslcs)
        decorrnoise -= decorrnoise.mean()
        decorrnoise = MeanStd.apply(decorrnoise, 0.0, 0.1, axis=0)

        alpha = np.where(1 - (np.abs(spatial) / sensor.getCriticalBaseLine()) > 0, 
                         1 - (np.abs(spatial) / sensor.getCriticalBaseLine()), 
                         np.zeros_like(spatial))
        beta = np.exp(-temporal / sensor.getCriticalBaseTime())
        
        decorrnoise *= np.arccos(alpha * beta * 0.96 * 0.92)

        return DecorrData(decorrnoise)
