from typing import  Union, Literal, Mapping, Tuple
import bz2, pickle, os
import numpy as np
from ..dtype import Data
from ..defo import DefoData
from ..delays import DelayData
from ..noise import NoiseData
from ..mask import MaskData
from ..slc import SlcData

class SingleData(Data):
    def plot(self, save:bool=False) -> None:
        return super().plot('Single', 'RdBu_r', 'rad', save)

class Single:
    def __init__(self, defo:DefoData, 
                       turb:DelayData, 
                       tropo:DelayData,  
                       orbit:DelayData, 
                       noise:NoiseData,
                       watermask:MaskData,
                       cohmask:MaskData,
                       srckwargs:Mapping[str, Union[int, float]],
                       center:Tuple[float, float],
                       bbox:Tuple[float, float, float, float]) -> None:
        self.defo = defo
        self.turb = turb
        self.tropo = tropo
        self.orbit = orbit
        self.noise = noise
        self.watermask = watermask
        self.cohmask = cohmask
        self.srckwargs = srckwargs
        self.center = center
        self.bbox = bbox

    def getDefo(self) -> DefoData:
        """ get the deformation map for either for IFG """
        return self.defo.copy()
    
    def getDisp(self) -> DefoData:
        """ get the displament map for either for IFG """
        return self.defo.copy()
    
    def getTropoDelay(self) -> DelayData:
        """ get the Tropospheric delay for either for IFG """
        return self.tropo.copy()
    
    def getTurbDelay(self) -> DelayData:
        """ get the Turblent delay for either for IFG """
        return self.turb.copy()

    def getOrbitalRamps(self) -> DelayData:
        """ get the Orbital ramps for either for IFG """
        return self.orbit.copy()
    
    def getNoise(self) -> NoiseData:
        """ get the Noise for either for IFG """
        return self.noise.copy()

    def getWaterMask(self) -> MaskData:
        """ get the Water mask for either for IFG """
        return self.watermask.copy()
    
    def getCohMask(self) -> MaskData:
        """ get the Coherence mask for either for IFG """
        return self.cohmask.copy()
        
    def getClear(self) -> SingleData:
        """ get the clear map for either for IFG """
        return SingleData(self.defo.copy() + self.turb.copy() + self.tropo.copy() + self.orbit.copy())
    
    def getNoisy(self) -> SingleData:
        """ get the noisy map for either for IFG """
        return SingleData(self.defo.copy() + self.turb.copy() + self.tropo.copy() + self.orbit.copy() + self.noise.copy())
    
    def getDelay(self) -> DelayData:
        """ get the delay maps for either for IFG """
        return DelayData(self.turb.copy() + self.tropo.copy() + self.orbit.copy())
    
    def getMask(self) -> MaskData:
        """ get the Mask  for either for IFG """
        return MaskData(np.logical_or(self.watermask.copy(), self.cohmask.copy()))
    
    def getSrcKwargs(self) -> Mapping[str, Union[int, float]]:
        """ get the Source paramters for either for IFG """
        return self.srckwargs
    
    def getCenter(self) -> Tuple[float, float]:
        """ get the Source center for either for IFG """
        return self.center
    
    def getBbox(self) -> Tuple[float, float, float, float]:
        """ get the Source bounding box for either for IFG """
        return self.bbox
    
    def getSlcs(self, scale:float=1.0) -> SlcData:
        """get the SAR images data"""
        data = self.getNoisy()
        if data.ndim == 2:
            data = data.unsqueeze()

        shape = data.shape[:2]
        noise = Data(2 * np.pi * np.random.rand(*shape, 1))
        amp = np.random.rayleigh(scale, (*shape, 1))
        data = Data(np.cumsum(np.concatenate((noise, data), axis=-1), axis=-1)) # np.concatenate((np.zeros((*self.shape, 1)), np.cumsum(defo, axis=-1)), axis=-1) + noise
        data = np.exp(1j * data.wrap())
        slcs = amp * data
        return SlcData(slcs, complex)

    
    def write(self, filename:str, out:Literal['obj', 'dict']) -> None:
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        with bz2.BZ2File(filename, 'wb') as file:
            pickle.dump(self if out == 'obj' else self.__dict__, file)
        file.close()
    
    @classmethod
    def read(cls, filename:str) -> 'Single':
        with bz2.BZ2File(filename, 'rb') as file:
            data = pickle.load(file)
        
        if isinstance(data, Single):
            return data
        
        return cls(**data)
    
    def plot(self, save:bool=False):
        methods = filter(lambda x : x[:3] == 'get', dir(self))
        for method in methods:
            method = getattr(self, method)
            if method() is not None and isinstance(method(), np.ndarray):
                method().plot(save)

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__},\n\t(defo: {self.defo},\n\tturb:{self.turb},\n\ttropo:{self.tropo},\n\t{self.orbit},\n\tnoise: {self.noise},\n\tkwargs: {self.srckwargs},\n\tkwargs: {self.bbox})>"
