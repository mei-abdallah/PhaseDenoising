from typing import Union, Tuple, Optional
import numpy as np

class MinMax():
    def __init__(self, ymin:float=0.0, ymax:float=1.0) -> None:
        self.ymin = ymin
        self.ymax = ymax

    def __call__(self, x:np.ndarray) -> np.ndarray:
        return ((x - x.min()) * (self.ymax - self.ymin) / (x.max() - x.min())) + self.ymin
    
    @classmethod
    def apply(cls, x:np.ndarray, ymin:float=0.0, ymax:float=1.0) -> np.ndarray:
        return cls(ymin, ymax)(x)

class MeanStd():
    def __init__(self, ymean:float=0.0, ystd:float=1.0) -> None:
        self.ymean = ymean 
        self.ystd = ystd
    
    def __call__(self, x:np.ndarray, axis:Optional[Union[int, Tuple[int, ...]]]=None) -> np.ndarray:
        return ((x - x.mean(axis)) * self.ystd / x.std()) + self.ymean
    
    @classmethod
    def apply(cls, x:np.ndarray, ymean:float=0.0, ystd:float=1.0, 
              axis:Optional[Union[int, Tuple[int, ...]]]=None) -> np.ndarray:
        return cls(ymean, ystd)(x, axis)


class MeanVar():
    def __init__(self, ymean:float=0.0, yvar:float=1.0) -> None:
        self.ymean = ymean 
        self.yvar = yvar
    
    def __call__(self, x:np.ndarray, axis:Optional[Union[int, Tuple[int, ...]]]=None) -> np.ndarray:
        # print(x.mean(axis, keepdims=True))
        x -= x.mean(axis, keepdims=True)
        
        power = self.ymean + self.yvar * np.random.randn()
        return x * (power / np.max(np.abs(x)))
    
    @classmethod
    def apply(cls, x:np.ndarray, ymean:float=0.0, ystd:float=1.0, 
              axis:Optional[Union[int, Tuple[int, ...]]]=None) -> np.ndarray:
        return cls(ymean, ystd)(x, axis)


class Unit():
    def __init__(self, unit:1.0) -> None:
        self.unit = unit 
    
    def __call__(self, x:np.ndarray) -> np.ndarray:
        return x * (self.unit / np.max(np.abs(x)))
    
    @classmethod
    def apply(cls, x:np.ndarray, unit:float=1.0) -> np.ndarray:
        return cls(unit)(x)