from typing import Union
import numpy as np
from ..objects import Sensor
from ..dtype import Vector


class SpatioVector(Vector):
    def plot(self, save:bool=False) -> None:
        return super().plot('Spatial BaseLine', 'Prependicular Baseline [m]', save)

class Spatial:
    def create(self, nslcs:int, sensor:Union[Sensor, str]) -> SpatioVector:
        sensor = Sensor(sensor)
        spatio = sensor.getBaseLineSTD() * np.random.randn(nslcs -1)
        return SpatioVector(np.concatenate(([0.], spatio)))