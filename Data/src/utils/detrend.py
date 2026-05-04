from typing import Optional
import numpy as np

class DeTrend():
    def __init__(self, slope:Optional[float]=None, intercept:Optional[float]=None) -> None:
        self.slope = slope
        self.intercept = intercept

    def fit(self, x:np.ndarray, ymin:float=-1.0, ymax:float=1.0) -> 'DeTrend':
        xmin, xmax = x.min(), x.max()
        if xmax == xmin:
            raise ValueError("there is no variation")
        
        self.slope = (ymax - ymin) / (xmax - xmin)
        self.intercept = (-xmin*(ymax-ymin)/(xmax-xmin)) + ymin
        return self


    def __call__(self, x:np.ndarray) -> np.ndarray:
        return x * self.slope + self.intercept
    
    def reverse(self, y:np.ndarray) -> np.ndarray:
        return (y - self.intercept) / self.slope
    
    def __repr__(self) -> str:
        return f'<slope: {self.slope}, intercept: {self.intercept}>'
    
    @classmethod
    def feed(cls, x:np.ndarray, ymin:float=-1.0, ymax:float=1.0) -> 'DeTrend':
        xmin, xmax = x.min(), x.max()
        if xmax == xmin:
            raise ValueError("there is no variation")
        
        slope = (ymax - ymin) / (xmax - xmin)
        intercept = (-xmin*(ymax-ymin)/(xmax-xmin)) + ymin
        return cls(slope, intercept)
    

    @classmethod
    def apply(cls, x:np.ndarray, ymin:float=-1.0, ymax:float=1.0) -> np.ndarray:
        return cls.feed(x, ymin, ymax)(x)
    