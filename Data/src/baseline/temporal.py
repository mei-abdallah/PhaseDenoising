from typing import Union
import numpy as np
from ..objects import Sensor
from ..dtype import Vector

class TemporalVector(Vector):
    def plot(self, save:bool=False) -> None:
        return super().plot('Temporal BaseLine', 'Time [days]', save)


class Temporal:
    def create(self, nslcs:int, sensor:Union[Sensor, str]) -> TemporalVector:
        sensor = Sensor(sensor)
        # temporal = ((np.arange(nslcs - 1) + 1.0) * sensor.getCycle()).astype(np.float32) # np.concatenate(([0.], temporal))
        temporal = np.arange(nslcs) * sensor.getCycle()
        return TemporalVector(temporal)
    