from typing import Tuple, Literal, Optional, Union, Mapping
import pickle, bz2, os
import numpy as np
from ..dtype import Data
from ..defo import Pattern, Source, Trend, DefoData
from ..dem import Dem, DemData
from ..delays import Tropospheric, Turblent, Topographic, Orbit, DelayData, DelayMaskedData
from ..noise import Thermal, Decorelation, NoiseData
from ..mask import Coherence, MaskData
from ..baseline import Spatial, Temporal
from ..objects import Sensor

class StackData(Data):
    def plot(self, save:bool=False) -> None:
        return super().plot('Stack', 'RdBu_r', 'rad', save)
    
class Stack:
    mode:Literal['stack', 'timeseries', 'ifgs'] = 'stack'
    def __init__(self, trend:DefoData, 
                       topo:DelayData,
                       turb:DelayData, 
                       tropo:Union[DelayData, DelayMaskedData],  
                       orbit:DelayData, 
                       thermo:NoiseData,
                       decorr:NoiseData, 
                       cohmask:MaskData,
                       watermask:MaskData,
                       demheights:DemData) -> None:
        self.trend = trend
        self.topo = topo
        self.turb = turb
        self.tropo = tropo
        self.orbit = orbit
        self.thermo = thermo
        self.decorr = decorr
        self.cohmask = cohmask
        self.watermask = watermask
        self.demheights = demheights
        
    @classmethod
    def create(cls, 
               nslcs:int, 
               shape:Tuple[int, int],
               location:Mapping[str, Union[str, float, Tuple[float, float]]],
               resolution:Optional[float]=None,
               reshape:Literal['crop', 'resize']='resize',
               platform:Literal['ASAR', 'ERS', 'ALOS1', 'ALOS2', 'RADARSAT', 'SENTINEL', 'CSK', 'TSK']='SENTINEL', 
               polydeg:Optional[Literal['second', 'third', 'fifth']]=None,
               method:Optional[Literal['fft', 'cov', 'eig', 'fractal', 'trend']]=None,
               order:Optional[Literal['first', 'second']]=None,
               source:Optional[Literal['cone', 'peak', 'complex', 'mogi', 'sill', 'dyke', 'quake']]=None,
               disp:Optional[Literal['stable', 'linear', 'sinusoidal', 'cosinusoidal', 'periodic', 'onset', 'pulse', 
                                     'logarithmic', 'exponential', 'power', 'coseismic', 'postseismic', 'longwave', 
                                     'stable+sinusoidal', 'stable+cosinusoidal', 'stable+periodic', 
                                     'linear+sinusoidal', 'linear+cosinusoidal', 'linear+periodic', 
                                     'linear+logarithmic', 'linear+exponential', 'linear+power', 'linear+longwave', 
                                     'accumilationl', 'timerelated', 'complex']]=None,
               track:Optional[Literal['asc', 'desc', 'random']]=None, 
               startday:Optional[int]=None, 
               limits:Optional[Mapping[str, float]]=None,
               duration:Optional[int]=None,
               threshold:float=0.2,
               snr:float=2.0,
               verbose:bool=False,
               warning:bool=False, **kwargs) -> 'Stack':
        
        """Create a stack of interferometric synthetic data for testing purposes."""
        sensor = Sensor(platform)

        # **************************************************************************
        #  step 1 --- generate DEM
        # **************************************************************************

        if verbose:
            print(' ')
            print('******************************************************************')
            print('                          1. generate DEM                         ')
            print('******************************************************************')

        dem = Dem(shape).create(location, resolution, reshape)

        # **************************************************************************
        #  step 2 --- generate heights and Watermask 
        # **************************************************************************

        if verbose:
            print(' ')
            print('******************************************************************')
            print('                 2. generate heights and Watermask                ')
            print('******************************************************************')

        heights = dem.getData()
        watermask = dem.getMask()

        # **************************************************************************
        #  step 3 --- generate timebase & shortbaseline
        # **************************************************************************

        if verbose:
            print(' ')
            print('******************************************************************')
            print('               3. generate timebase & shortbaseline               ')
            print('******************************************************************')

        spatial = Spatial().create(nslcs, sensor)
        temporal = Temporal().create(nslcs, sensor)

        # **************************************************************************
        #  step 4 --- simulate Coherence
        # **************************************************************************

        if verbose:
            print(' ')
            print('******************************************************************')
            print('                       4. simulate coherence                      ')
            print('******************************************************************')

        cohmask = Coherence(shape).create(nslcs, threshold)

        # **************************************************************************
        #  step 5 --- simulate DEM error
        # **************************************************************************

        if verbose:
            print(' ')
            print('******************************************************************')
            print('                       5. simulate DEM error                      ')
            print('******************************************************************')

        topo = Topographic(shape).create(heights, spatial, sensor, beta=2.2)

        # **************************************************************************
        #  step 6 --- simulate orbital error
        # **************************************************************************

        if verbose:
            print(' ')
            print('******************************************************************')
            print('                     6. simulate orbital error                    ')
            print('******************************************************************')

        polydeg = polydeg or np.random.choice(['second', 'third', 'fifth'])

        orbit = Orbit(shape).create(nslcs, polydeg, orbitsig=0.4) # , naslcs=int(np.random.rand() * nslcs)

        # **************************************************************************
        #  step 7 --- simulate turblent atmospheric error
        # **************************************************************************

        if verbose:
            print(' ')
            print('******************************************************************')
            print('                   7. simulate atmospheric error                  ')
            print('******************************************************************')
        
        method  = method or np.random.choice(['fft', 'fractal', 'trend'])

        turb = Turblent(shape).create(nslcs, dem.getResolution(), method) # , naslcs=int(np.random.uniform(0.7, 1.0) * nslcs)

        # **************************************************************************
        #  step 8 --- simulate tropospheric error
        # **************************************************************************

        if verbose:
            print(' ')
            print('******************************************************************')
            print('                  8. simulate tropospheric error                  ')
            print('******************************************************************')

        order = order or np.random.choice(['first', 'second'])

        tropo = Tropospheric(shape).create(nslcs, heights, order)

        
        # **************************************************************************
        #  step 9 --- simulate noise
        # **************************************************************************

        if verbose:
            print(' ')
            print('******************************************************************')
            print('                         9. simulate noise                        ')
            print('******************************************************************')

        thermal = Thermal(shape).create(nslcs)

        decorr = Decorelation(shape).create(nslcs, temporal, spatial, sensor)

        # **************************************************************************
        #  step 10 --- simulate deformation
        # **************************************************************************

        if verbose:
            print(' ')
            print('******************************************************************')
            print('                     10. simulate pattern                         ')
            print('******************************************************************')
            
        source = source or np.random.choice(['cone', 'peak', 'complex', 'mogi', 'sill', 'dyke', 'quake'])

        pattern = Pattern(shape).create(source) if source in ['cone', 'peak', 'complex'] else Source(shape).create(source, sensor, track)

        viablesnr, itr, maxitrs = False, 0, 10
        while (not viablesnr) and (itr <= maxitrs):
            # **************************************************************************
            #  step 11 --- simulate deformation
            # **************************************************************************

            if verbose:
                print(' ')
                print('******************************************************************')
                print('                   11. simulate deformation                       ')
                print('******************************************************************')
            
            startday = startday if startday is not None else np.random.choice(np.arange(-int(0.3 * sensor.getCycle() * nslcs), 
                                                                                         int(0.3 * sensor.getCycle() * nslcs)))

            disp = disp or np.random.choice(['linear', 'sinusoidal', 'cosinusoidal', 'periodic', 'onset', 'pulse', 'exponential', 
                                             'coseismic', 'postseismic', 'accumilationl', 'timerelated', 'complex'
                                             'stable+sinusoidal', 'stable+cosinusoidal', 'stable+periodic', 
                                             'linear+sinusoidal', 'linear+cosinusoidal', 'linear+periodic', 
                                             'linear+exponential'])

            trend = Trend(shape).create(pattern, temporal, sensor, disp, startday, limits, duration)

            viablesnr = Stack.checkVisibleSignal(trend, tropo, turb, orbit, thermal, decorr, snr)

            if not viablesnr:
                itr += 1

        if viablesnr:
            if verbose:
                print(' ')
                print('******************************************************************')
                print('                          | Viable SNR |                          ')
                print('******************************************************************')
        else:
            if verbose:
                print(' ')
                print('******************************************************************')
                print('                        | Non-Viable SNR |                        ')
                print('******************************************************************')

            if warning:
                raise Exception('SNR is too low !.')

        if verbose:
            print(' ')
            print('******************************************************************')
            print('                             | Done |                             ')
            print('******************************************************************')
        
        if cls.mode == 'ifgs':
            return cls(trend.diff(), topo.diff(), turb.diff(), tropo.diff(), orbit.diff(), thermal.diff(), decorr.diff(), cohmask.diff(), watermask, heights)
        
        return cls(trend, topo, turb, tropo, orbit, thermal, decorr, cohmask, watermask, heights)
    
    def checkVisibleSignal(defo:Data, tropodelay:Data, turbdelay:Data, rampdelay:Data, thermnoise:Data, decorrnoise:Data, threshold:float=2.0, mask:Optional[Data]=None) -> bool:
        viable = True

        defo = np.ma.array(defo, mask=mask)
        residues =  np.ma.array((tropodelay + turbdelay + rampdelay + thermnoise + decorrnoise), mask=mask)
        snr = np.var(np.ma.compressed(defo)) / np.var(np.ma.compressed(residues))

        if snr < threshold:
            viable = False
        

        return viable
    
    def getTrend(self) -> DefoData:
        """ get the displacment Trend for either for SLC or IFG """
        return self.trend.copy()
    
    def getDefo(self) -> DefoData:
        """ get the deformation map for either for SLC or IFG """
        return self.trend.copy()
    
    def getDisp(self, indcies:Optional[Tuple[int, int]]=(-1, 0)) -> DefoData:
        """ get the displacment map for either for SLC or IFG """
        trend = self.trend.torad().copy()
        
        if self.mode == 'ifgs':
            trend = np.concatenate((np.zeros((*trend.shape[:-1], 1)), trend.cumsum(-1)), axis=-1)

        disp = trend[..., indcies[1]] - trend[..., indcies[0]]
        
        return DefoData(disp)
    
    def getTropo(self) -> Union[DelayData, DelayMaskedData]:
        """ get the Tropospheric delay for either for SLC or IFG """
        return self.tropo.copy()
    
    def getTurb(self) -> DelayData:
        """ get the Turblent delay for either for SLC or IFG """
        return self.turb.copy()
    
    def getDelay(self) -> DelayData:
        """ get the delay maps for either for SLC or IFG """
        return DelayData(self.tropo.copy() + self.turb.copy())
    
    def getTopo(self) -> DelayData:
        """ get the Topographic error for either for SLC or IFG """
        return self.topo.copy()
    
    def getError(self) -> DelayData:
        """ get the error maps for either for SLC or IFG """
        return self.topo.copy()

    def getOrbit(self) -> DelayData:
        """ get the Orbital ramps for either for SLC or IFG """
        return self.orbit.copy()
    
    def getRamps(self) -> DelayData:
        """ get the ramp stack containing all the orbital ramps"""
        return self.orbit.copy()
    
    def getThermo(self) -> NoiseData:
        """ get the Thermal noise for either for SLC or IFG """
        return self.thermo.copy()
    
    def getDecorr(self) -> NoiseData:
        """ get the Decorrelation noise for either for SLC or IFG """
        return self.decorr.copy()
    
    def getNoise(self)-> NoiseData:
        """ get the noise maps for either for SLC or IFG """
        return NoiseData(self.thermo.copy() + self.decorr.copy())
    
    def getWaterMask(self) -> MaskData:
        """ get the Water mask for either for SLC or IFG """
        return self.watermask.copy()
    
    def getCohMask(self) -> MaskData:
        """ get the Coherence mask for either for SLC or IFG """
        return self.cohmask.copy()
    
    def getMask(self) -> MaskData:
        """ get the Mask mask for either for SLC or IFG """
        return MaskData(np.logical_or(self.watermask.copy().unsqueeze(), self.cohmask.copy()))
    
    def getDemHeights(self) -> DemData:
        """ get the dem heights for either for SLC or IFG """
        return self.demheights.copy()
        
    def getClear(self) -> StackData:
        """ get the clear map for either for SLC or IFG """
        return StackData(self.trend.copy() + self.tropo.copy() + self.turb.copy() + self.orbit.copy())
    
    def getNoisy(self) -> StackData:
        """ get the noisy map for either for SLC or IFG """
        return StackData(self.trend.copy() + self.tropo.copy() + self.turb.copy() + self.orbit.copy() + self.thermo.copy() + self.decorr.copy())
    
    def getResidue(self) -> DelayData:
        """ get the residual maps for either for SLC or IFG """
        return DelayData(self.tropo.copy() + self.turb.copy() + self.orbit.copy() + self.thermo.copy() + self.decorr.copy())
    
    def getScale(self) -> Mapping[str, float]:
        return {'min' : self.getNoisy().min(),
                'max' : self.getNoisy().max()}
    
    def getSNR(self) -> float:
        """ get the Signal to Noise Ratio (SNR) of the data """
        snr = self.getDefo().var() / self.getResidue().var()
        return float(snr)
    
    def write(self, filename:str, out:Literal['obj', 'dict']) -> None:
        """ write the data to a file """
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        with bz2.BZ2File(filename, 'wb') as file:
            pickle.dump(self if out == 'obj' else self.__dict__, file)
        file.close()
    
    @classmethod
    def read(cls, filename:str) -> 'Stack':
        with bz2.BZ2File(filename, 'rb') as file:
            data = pickle.load(file)
        
        try:
            return cls(**data)
        except:
            return data
        
    
    def plot(self, save:bool=False):
        methods = filter(lambda x : x[:3] == 'get', dir(self))
        for method in methods:
            method = getattr(self, method)
            if method() is not None and isinstance(method(), np.ndarray):
                method().plot(save)
        
    
    def __repr__(self) -> str:
        return f"<{self.__class__.__name__},\n\t(trend: {self.trend},\n\ttopo: {self.topo},\n\tturb:{self.turb},\n\ttropo:{self.tropo},\n\t{self.orbit},\n\tthermo: {self.thermo},\n\tdecorr: {self.decorr})>"
