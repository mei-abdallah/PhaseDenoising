import numpy as np

class Wrap:
    def __init__(self, value:int) -> None:
        self.value = value

    def __call__(self, x:np.ndarray) -> np.ndarray:
        return np.mod(x + np.pi, self.value * np.pi) - (0.5 * self.value * np.pi)
    
    @classmethod
    def apply(cls, x:np.ndarray, value:int) -> np.ndarray:
        return cls(value)(x)