
from typing import Tuple, Optional, Literal
import numpy as np
from ..dtype import Data

class OrbitData(Data):
    def plot(self, save:bool=False) -> None:
        return super().plot('Orbital Ramps', 'RdBu_r', 'rad', save)


class Orbit:
    def __init__(self, shape:Tuple[int, int]) -> None:
        self.shape = shape

    def create(self, nslcs:int, polydeg:Literal['second', 'third', 'fifth']='third', orbitsig:float=1.0, naslcs:Optional[int]=None) -> OrbitData:
        naslcs = min(naslcs or nslcs - 1, nslcs - 1)
        degree = {'second' : 2, 'third'  : 3, 'fifth'  : 5}.get(polydeg, 2)
        x, y = np.meshgrid(np.linspace(0, 1, self.shape[1]), np.linspace(0, 1, self.shape[0]))

        idx = np.random.choice(np.arange(1, nslcs), naslcs, replace=False)
        logic = np.zeros((nslcs,), dtype=np.int8)
        logic[idx] = 1
        const = 5
        
        polymatrix = orbitsig * np.random.randn(degree, nslcs)
        polyx = np.random.randint(-orbitsig*const, orbitsig*const, size=(nslcs,))
        polymatrix[0, :] += polyx
        polyy = np.random.randint(-orbitsig*const, orbitsig*const, size=(nslcs,))
        polymatrix[1, :] += polyy

        # polymatrix = np.hstack((np.zeros((degree, 1)), polymatrix))
        polymatrix[:, np.logical_not(logic)] = polymatrix[:, np.logical_not(logic), ] / np.max(np.abs(polymatrix)) 

        orbit = np.zeros((*self.shape, nslcs))
        if polydeg == 'second':
            orbit = np.matmul(x[..., np.newaxis], polymatrix[0:1,]) + \
                    np.matmul(y[..., np.newaxis], polymatrix[1:2,])

        elif polydeg == 'third':
            orbit = np.matmul(x[..., np.newaxis], polymatrix[0:1,]) + \
                    np.matmul(y[..., np.newaxis], polymatrix[1:2,]) + \
                    np.matmul(x[..., np.newaxis] * y[..., np.newaxis], polymatrix[2:3,])

        elif polydeg == 'fifth':
            orbit = np.matmul(x[..., np.newaxis], polymatrix[0:1,]) + \
                    np.matmul(y[..., np.newaxis], polymatrix[1:2,]) + \
                    np.matmul(x[..., np.newaxis] * y[..., np.newaxis], polymatrix[2:3,]) + \
                    np.matmul(x[..., np.newaxis] * x[..., np.newaxis], polymatrix[3:4,]) + \
                    np.matmul(y[..., np.newaxis] * y[..., np.newaxis], polymatrix[4:5,])
        else:
            raise NotImplementedError(f'number of polynomial coefficents can only be either 2, 3 or 5. But {degree} was found')
        
        # for nslc in range(nslcs):
        #     if polydeg == 'second':
        #         orbit[:, :, nslc] = polymatrix[0, nslc] * x  + polymatrix[1, nslc] * y

        #     elif polydeg == 'third':
        #         orbit[:, :, nslc] = polymatrix[0, nslc] * x  + polymatrix[1, nslc] * y + polymatrix[2, nslc] * x * y

        #     elif polydeg == 'fifth':
        #         orbit[:, :, nslc] = polymatrix[0, nslc] * x  + polymatrix[1, nslc] * y + polymatrix[2, nslc] * x * y + polymatrix[3, nslc] * x**2 + polymatrix[4, nslc] * y**2
            
        #     else:
        #         raise NotImplementedError('number of polynomial coefficents can only be either 2, 3 or 5')
            
        return OrbitData(orbit)