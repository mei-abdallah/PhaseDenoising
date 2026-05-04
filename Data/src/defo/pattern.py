from typing import Tuple, Literal
from .model import Cone, Peak
from .complex import Complex
from ..dtype import Data
from ..utils import Unit, MinMax
import numpy as np

class PatternData(Data):
    def plot(self, save:bool=False) -> None:
        return super().plot('Deformation Map', 'RdBu_r', 'rad', save)

class Pattern:
    def __init__(self, shape:Tuple[int, int]) -> None:
        self.shape = shape

    def create(self, pattern:Literal['cone', 'peak', 'complex'] = 'cone') -> PatternData:
        if pattern == 'cone':
            defo = Cone(self.shape).create()

        elif pattern == 'peak':
            defo = Peak(self.shape).create()

        elif pattern == 'complex':
            defo = Complex(self.shape).create()

        else:
            raise ValueError(f'Invalid pattern of {pattern} type')
        
        defo = MinMax.apply(defo, np.random.uniform(-0.25, 0), 
                                  np.random.uniform(0, 0.25)) 
        # defo = Unit.apply(defo) # Unit pattern
        
        return PatternData(defo)



        





    
