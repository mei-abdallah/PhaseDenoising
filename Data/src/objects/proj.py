from typing import Callable, Tuple, Union, Optional, Literal
from ..objects import Sensor
import numpy as np


class Projection:
    def create(self, sensor:Union[Sensor, str], track:Optional[Literal['asc', 'desc', 'random']]=None) -> Callable[[np.ndarray], np.ndarray]:
        """ create the projection function to project the enu displacment to the sattelite line of site """
        sensor = Sensor(sensor)

        matrix = self.getMatrix(sensor.getIncidentAngle(), 
                                sensor.getTrackAngle(track) if track else sensor.getHeadingAngle())
        
        return Transform(matrix) 
    
    def getMatrix(self, incident:Union[float, Tuple[float, float]], heading:float) -> np.ndarray:
        """Return a transformation matrix based on incident and heading angles 
        
        Paramters:
            - incident | tuple | inclinde angle representing the position where the sensor was pointing when it took this measurement
            - heading | float | azimuthal angle of the satellite relative to north, in radians

        Returns:
            - matrix | ND array | representing the transform matrix that can be used.
        """

        return np.array([np.sin(np.deg2rad(incident)) * np.sin(np.deg2rad(heading)),
                        -np.sin(np.deg2rad(incident)) * np.cos(np.deg2rad(heading)),
                         np.cos(np.deg2rad(incident))]).reshape(-1, 1)
    

class Transform():
    def __init__(self, matrix:np.ndarray) -> None:
        self.matrix = matrix

    def __call__(self, disp:np.ndarray) -> np.ndarray:
        return np.sum(np.multiply(self.matrix, disp), axis=0, keepdims=True) if isinstance(disp, np.ndarray) else type(disp)(np.sum(np.multiply(self.matrix, disp), axis=0, keepdims=True))
    
    def __repr__(self) -> str:
        return f"<Transform Matrix: \n{self.matrix.__repr__()}>"
    
    def __str__(self) -> str:
        return f"<Transform Matrix: \n{self.matrix}>"
    
    