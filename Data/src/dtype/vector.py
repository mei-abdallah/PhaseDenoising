from typing import Optional
import numpy as np
import matplotlib.pyplot as plt
import os

class Vector(np.ndarray):
    def __new__(cls, data:np.ndarray) -> 'Vector':
        return np.asarray(data, dtype=np.float32).view(cls)
    
    def diff(self, axis:int=-1) -> 'Vector':
        """Calculate the difference between two successive elements along an axis"""
        return type(self)(np.ma.diff(self, n=1, axis=axis))
    
    def plot(self, title:Optional[str]=None, ylabel:Optional[str]=None, save:bool=False) -> None:
        fig = plt.figure(figsize=(5, 5))
        axe = fig.add_subplot()

        axe.plot(self)
        axe.set_ylabel(f'{ylabel}')
        axe.set_xlabel('# of Interferograms')
        axe.set_title(title if title else self.__class__.__name__)
        
        if save:
            if not os.path.exists('./figures/'):
                os.makedirs('./figures/', exist_ok=True)
            fig.savefig(f'./figures/{self.__class__.__name__}.png', dpi=750, bbox_inches='tight')
        plt.show()

# class Vector:
#     def __init__(self, data:np.ndarray) -> None:
#         self.data = data.astype(np.float32)

#     def __repr__(self) -> str:
#         return f'<{self.__class__.__name__}: \n{self.data.__repr__()}>'
    
#     def __str__(self) -> str:
#         return f'<{self.__class__.__name__}: \n{self.data.__str__()}>'

#     def __array__(self) -> np.ndarray:
#         return self.data
    

#     def plot(self, title:Optional[str]=None, save:bool=False) -> None:
#         fig = plt.figure(figsize=(5, 5))
#         axe = fig.add_subplot()

#         axe.plot(self.data)
#         axe.set_ylabel('Baseline')
#         axe.set_xlabel('interferoframs')
#         axe.set_title(title if title else self.__class__.__name__)
        
#         if save:
#             if not os.path.exists('./figures/'):
#                 os.makedirs('./figures/', exist_ok=True)
#             fig.savefig(f'./figures/{self.__class__.__name__}.png', dpi=750, bbox_inches='tight')
#         plt.show()