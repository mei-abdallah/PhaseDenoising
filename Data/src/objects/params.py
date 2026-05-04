from typing import Union
import numpy as np

class Params:
    __data__ = {}
    def __setattr__(self, name:str, value:Union[str, float, int, bool, tuple, list, np.ndarray]) -> None:
        self.__data__[name] = value

    def __getattr__(self, name: str) -> Union[str, float, int, bool, tuple, list, np.ndarray]:
        return self.__data__[name]