from typing import Tuple, Callable, Mapping, Union
import numpy as np
import os
from .model import Mogi, Okada
from ..utils import MinMax

try:
    import pyfftw
    pyfftw.config.NUM_THREADS = min(os.cpu_count(), 4)
    fft = pyfftw.interfaces.numpy_fft
except:
    fft = np.fft

class Complex:
    def __init__(self, shape:Tuple[int, int]) -> None:
        self.shape = shape

    def create(self) -> np.ndarray:
        defo = np.zeros(self.shape)
        # ['mogi', 'sill', 'dyke', 'quake']
        idx, sources = 0, np.random.choice(['mogi', 'sill', 'dyke', 'quake'], size=5)
        itr, maxitr = 0, 10
       
        while idx < len(sources) and itr < maxitr:
            source = sources[idx]
            resolution = self.getResolution(source)
            coord = self.getCoord(resolution)
            kwargs = self.getKwargs(source)
            centre = self.getCentre(resolution, edge=0.0)
            disp = np.sum(np.multiply(self.getProj(), 
                                      self.getModel(source)(centre, coord, **kwargs)), 
                          axis=0, keepdims=True).reshape(self.shape)
            if np.max(np.abs(disp)) <= 2.0:
                defo += disp
                idx += 1
            itr += 1
        return defo
    
    def getResolution(self, source:str) -> Mapping[str, float]:
        resolution = {'x': 1e5 / self.shape[1],
                      'y': 1e5 / self.shape[0]}
        if source in ['mogi', 'dyke', 'sill']:
            resolution = {key : value * 0.1 for key, value in resolution.items()}
        return resolution

    def getModel(self, source:str) -> Callable[[Tuple[float, float], 
                                                np.ndarray, 
                                                Mapping[str, Union[int, float]]], 
                                                np.ndarray]:
        if source == 'mogi':
            model = Mogi() 
            
        elif source in ['sill', 'dyke', 'quake']:
            model = Okada()

        else:
            raise ValueError(f'Invalid pattern of {source} type')
        
        return model


    def getKwargs(self, source:str) -> Mapping[str, Union[int, float]]:
        if source.lower() == 'mogi':
            src_kwargs = self.getMogi()

        elif source.lower() == 'dyke':
            src_kwargs = self.getDyke()

        elif source.lower() == 'sill':
            src_kwargs = self.getSill()

        elif source.lower() == 'quake':
            src_kwargs = self.getQuake()

        else:
            raise ValueError('Invalid source type!')
        return src_kwargs
            
    def getProj(self) -> np.ndarray:
        incident =  np.random.randint(30, 50)
        heading = np.random.randint(-360, 360)
        return np.array([np.sin(np.deg2rad(incident)) * np.sin(np.deg2rad(heading)),
                        -np.sin(np.deg2rad(incident)) * np.cos(np.deg2rad(heading)),
                         np.cos(np.deg2rad(incident))]).reshape(-1, 1)
    
    def getCentre(self, resolution:Mapping[str, float], edge:float=0.2) -> Tuple[float, float]:
        x = (edge + np.random.rand() * (1 - 2 * edge)) * self.shape[1] * resolution['x']
        y = (edge + np.random.rand() * (1 - 2 * edge)) * self.shape[0] * resolution['y']
        return x, y
    
    def getCoord(self, resolution:Mapping[str, Union[int, float]]) -> np.ndarray:
        """ get a random coordinate for the source"""
        yy, xx = np.mgrid[0:self.shape[0]:self.shape[0]*1j, 0:self.shape[1]:self.shape[1]*1j]
        coord =  np.vstack((xx.reshape(1, -1) * resolution['x'], 
                            yy.reshape(1, -1) * resolution['y']))
        return coord
    
    def getMogi(self) -> Mapping[str, Union[int, float]]:
        src_kwargs = {'volume_change' :  int(2e6 + 1e6 * np.random.rand()),                             # in metres, always positive
                      'depth'         :  1000 + 3000 * np.random.rand()}                                  # in metres                         
        return src_kwargs
    
    def getSill(self) -> Mapping[str, Union[int, float]]:
        width = 2000 + 4000 * np.random.rand()
        dip = np.random.randint(0, 5)
        return {'strike'   : np.random.randint(0, 359),                                            # in degrees
                'dip'      : dip,                                                                  # in degrees
                'rake'     : 0,                                                                    # in degrees
                'length'   : 2000 + 4000 * np.random.rand(),                                       # in meters
                'width'    : width,                                                                # in metres
                'depth'    : 0.5 * width * np.sin(np.deg2rad(dip)) + 4000 * np.random.rand(),      # in metres
                'slip'     : 0.0,                                                                  # in meters
                'opening'  : 0.2 + 0.8 * np.random.rand()}                                         # in metres
    
    def getDyke(self) -> Mapping[str, Union[int, float]]:
        width = 6000 * np.random.rand()
        dip = np.random.randint(75, 90)
        return {'strike'   : np.random.randint(0, 359),                                            # in degrees
                'dip'      : dip,                                                                  # in degrees
                'rake'     : 0,                                                                    # in degrees
                'length'   : 10000 * np.random.rand(),                                             # in metres
                'width'    : width,                                                                # in meters  
                'depth'    : 0.5 * width * np.sin(np.deg2rad(dip)) + 2000 * np.random.rand(),      # bottom depth is top depth plus a certain random amount
                'slip'     : 0.0,                                                                  # in meters
                'opening'  : 0.1 + 0.6 * np.random.rand()}                                         # in metres  

    
    def getQuake(self) -> Mapping[str, Union[int, float]]:
        width = np.random.choice(np.arange(5, 30, 1))*1000
        dip = np.random.randint(10, 80)
        return {'strike'    : np.random.randint(0, 359),                                       # in degrees [0 >> 360]
                'dip'       : dip,
                'rake'      : np.random.randint(-180, 180),
                'length'    : np.random.choice(np.arange(10, 50, 1))*1000,                     # in metres [10 >> 50 km]
                'width'     : width,                      # in meters [5 >> 30 km]
                'depth'     : 0.5 * width * np.sin(np.deg2rad(dip)) + np.random.choice(np.arange(0, 20, 1))*1000,              # in metres [0 >> 20 km]
                'slip'      : np.random.choice(np.arange(0.5, 12, 0.5)),                       # in meters [0.5 >> 12 m]
                'opening'   : 0.,}