from typing import Tuple, Literal, Union, Optional, Mapping
import numpy as np
import scipy.special as sp
from ..dtype import Data, Vector
from ..objects import Sensor, Params
from ..utils import MinMax



class TrendData(Data):
    def plot(self, save:bool=False) -> None:
        return super().plot('Displacement Map', 'RdBu_r', 'rad', save)

class TrendVector(Vector):
    def plot(self, save:bool=False) -> None:
        return super().plot('Displacement', 'Displacement [mm]', save)

class Trend:
    params = Params()

    def __init__(self, shape:Tuple[int, int], ) -> None:
        self.shape = shape

    def create(self, 
               pattern:np.ndarray, 
               temporal:np.ndarray, 
               sensor:Union[Sensor, str],
               trend:Literal['stable', 'linear', 'sinusoidal', 'cosinusoidal', 'periodic', 'onset', 'pulse', 
                             'logarithmic', 'exponential', 'power', 'coseismic', 'postseismic', 'longwave', 
                             'stable+sinusoidal', 'stable+cosinusoidal', 'stable+periodic', 
                             'linear+sinusoidal', 'linear+cosinusoidal', 'linear+periodic', 
                             'linear+logarithmic', 'linear+exponential', 'linear+power', 'linear+longwave', 
                             'accumilationl', 'timerelated', 'complex',], 
               startday:int=0, 
               defolimits:Optional[Mapping[str, float]]=None, 
               duration:Optional[int]=None, 
               relative:bool=False,
               assgin:bool=True) -> TrendData:
        
        sensor = Sensor(sensor)

        endday = duration + startday if duration is not None else temporal.diff().mean() * len(temporal)
        
        active = np.where(temporal >= startday, np.ones_like(temporal), np.zeros_like(temporal)) 

        inactive = np.where(temporal > endday, np.ones_like(temporal), np.zeros_like(temporal)) 

        temporal = (temporal - startday)/ 365.25
        
        stack = np.zeros((*self.shape, len(temporal)), dtype= np.float32)

        if relative:
            if assgin:
                self.params.indx = np.random.randint(low=0, high=len(temporal))

        successful = False

        while not successful:

            if trend == 'stable':
                if assgin:
                    self.params.rate = 0 * np.random.uniform(low=-1, high=1) # max trend
                    self.params.const = 20 * np.random.uniform(-1, 1)
                
                trendcoef = TrendVector(self.params.rate * temporal + self.params.const)

            elif trend  == 'linear':
                if assgin:
                    self.params.rate = 25 * np.random.uniform(low=-1, high=1) # max trend
                    self.params.const = 20 * np.random.uniform(-1, 1)
                
                trendcoef = TrendVector(self.params.rate * temporal + self.params.const)

            elif trend == 'sinusoidal':
                if assgin:
                    self.params.sinamp =  10 * np.random.uniform(low=-1, high=1)
                    self.params.sinfreq = 10 * np.random.uniform(low=0.5, high=1)
                    self.params.sininit = 1.0
                
                trendcoef = TrendVector(self.params.sinamp * np.sin((self.params.sininit * np.pi) + (2 * np.pi * self.params.sinfreq * temporal)))

            elif trend == 'cosinusoidal':
                if assgin:
                    self.params.cosamp =  10 * np.random.uniform(low=-1, high=1)
                    self.params.cosfreq = 10 * np.random.uniform(low=0.5, high=1)
                    self.params.cosinit = 0.5
                
                trendcoef = TrendVector(self.params.cosamp * np.cos((self.params.cosinit * np.pi) + (2 * np.pi * self.params.cosfreq * temporal)))
            
            elif trend == 'periodic':
                if assgin:
                    self.params.sinamp =  5 * np.random.uniform(low=-1, high=1)
                    self.params.sinfreq = 10 * np.random.uniform(low=0.5, high=1)
                    self.params.sininit = 1.0
                    self.params.cosamp =  5 * np.random.uniform(low=-1, high=1)
                    self.params.cosfreq = 10 * np.random.uniform(low=0.5, high=1)
                    self.params.cosinit = 0.5
                
                trendcoef = TrendVector(self.params.sinamp * np.sin((self.params.sininit * np.pi) + (2 * np.pi * self.params.sinfreq * temporal)) + 
                                        self.params.cosamp * np.cos((self.params.cosinit * np.pi) + (2 * np.pi * self.params.cosfreq * temporal)))
            
            elif trend == 'onset':
                if assgin:
                    self.params.rate = 20 * np.random.uniform(low=-1, high=1)
                    self.params.seq = np.random.choice([-1, 0, 1], size=len(temporal))
                
                trendcoef = TrendVector(self.params.rate * self.params.seq)

            elif trend == 'pulse':
                if assgin:
                    self.params.rate = 25 * np.random.uniform(low=-1, high=1)
                    self.params.pos = np.random.choice(np.arange(len(temporal)))

                trendcoef = TrendVector(self.params.rate * np.where(temporal > temporal[self.params.pos], np.ones_like(temporal), np.zeros_like(temporal)))

            elif trend == 'logarithmic':
                if assgin:
                    self.params.logamp = 10 * np.random.uniform(low=-1, high=1)
                    self.params.logcoef = 10 * np.random.uniform(low=0.5, high=1)

                trendcoef = TrendVector(self.params.logamp * np.log(self.params.logcoef * temporal))         

            elif trend == 'exponential':
                if assgin:
                    self.params.expamp = 10 * np.random.uniform(low=-1, high=1)
                    self.params.expcoef = 5 * np.random.uniform(low=0.5, high=1)
                
                trendcoef = TrendVector(self.params.expamp * np.exp(self.params.expcoef * temporal))

            elif trend == 'power':
                if assgin:
                    self.params.powamp = 10 * np.random.uniform(low=-1, high=1)
                    self.params.powcoef = 5 * np.random.uniform(low=0.5, high=1)
                
                trendcoef = TrendVector(self.params.powamp * np.power(self.params.powcoef, temporal))

            elif trend == 'coseismic':
                if assgin:
                    self.params.seisamp = 5 * np.random.uniform(low=-1, high=1)
                    self.params.seisoff = 0.01 * np.random.uniform(low=0, high=1)
                    self.params.seiscoff = 0.01 * np.random.uniform(low=0, high=1)
                
                trendcoef = TrendVector(self.params.seisamp * (temporal - self.params.seisoff) / np.sqrt(self.params.seiscoff + np.square(temporal - self.params.seisoff)))

            elif trend == 'postseismic':
                if assgin:
                    self.params.beslamp = 25  * np.random.uniform(low=0.5, high=1)
                    self.params.beslcoef = 40  * np.random.uniform(low=0.5, high=1)
                
                trendcoef = TrendVector(self.params.beslamp * sp.jv(1, self.params.beslcoef * temporal))

            elif trend == 'longwave':
                if assgin:
                    self.params.lwrate = 5 * np.random.uniform(low=-1, high=1)
                wavepattern = self.params.lwrate * MinMax.apply(np.ones((self.shape[0] , 1)) * np.linspace(0, 1, self.shape[1]), -1, 1)
                
                trendcoef = TrendVector(np.ones_like(temporal))

            elif trend in ['accumilationl', 'timerelated'] :
                if assgin:
                    self.params.alpha = 25 * np.random.uniform(low=-1, high=1)
                    self.params.beta = 25 * np.random.uniform(low=-1, high=1)
                    self.params.gamma = 25 * np.random.uniform(low=-1, high=1)
                
                trendcoef = TrendVector(self.params.alpha * temporal + 
                                        self.params.beta * np.power(temporal, 2) + 
                                        self.params.gamma * np.power(temporal, 3))
                
            elif trend == 'stable+sinusoidal':
                if assgin:
                    self.params.rate = 0 * np.random.uniform(low=-1, high=1)
                    self.params.const = 20 * np.random.uniform(-1, 1)

                    self.params.sinamp =  10 * np.random.uniform(low=-1, high=1)
                    self.params.sinfreq = 10 * np.random.uniform(low=0.5, high=1)
                    self.params.sininit = 1.0
                
                trendcoef = TrendVector(self.params.rate * temporal + self.params.const +
                                        self.params.sinamp * np.sin((self.params.sininit * np.pi) + (2 * np.pi * self.params.sinfreq * temporal)))

            elif trend == 'stable+cosinusoidal':
                if assgin:
                    self.params.rate = 0 * np.random.uniform(low=-1, high=1)
                    self.params.const = 20 * np.random.uniform(-1, 1)

                    self.params.cosamp =  10 * np.random.uniform(low=-1, high=1)
                    self.params.cosfreq = 10 * np.random.uniform(low=0.5, high=1)
                    self.params.cosinit = 0.5
                
                trendcoef = TrendVector(self.params.rate * temporal + self.params.const +
                                        self.params.cosamp * np.cos((self.params.cosinit * np.pi) + (2 * np.pi * self.params.cosfreq * temporal)))
            
            elif trend  == 'stable+periodic':
                if assgin:
                    self.params.rate = 0 * np.random.uniform(low=-1, high=1)
                    self.params.const = 20 * np.random.uniform(-1, 1)

                    self.params.sinamp =  5 * np.random.uniform(low=-1, high=1)
                    self.params.sinfreq = 10 * np.random.uniform(low=0.5, high=1)
                    self.params.sininit = 1.0

                    self.params.cosamp =  5 * np.random.uniform(low=-1, high=1)
                    self.params.cosfreq = 10 * np.random.uniform(low=0.5, high=1)
                    self.params.cosinit = 0.5
                
                trendcoef = TrendVector(self.params.rate * temporal + self.params.const +
                                        self.params.sinamp * np.sin((self.params.sininit * np.pi) + (2 * np.pi * self.params.sinfreq * temporal)) + 
                                        self.params.cosamp * np.cos((self.params.cosinit * np.pi) + (2 * np.pi * self.params.cosfreq * temporal)))
            
            elif trend == 'linear+sinusoidal':
                if assgin:
                    self.params.rate = 25 * np.random.uniform(low=-1, high=1)
                    self.params.const = 20 * np.random.uniform(-1, 1)

                    self.params.sinamp =  10 * np.random.uniform(low=-1, high=1)
                    self.params.sinfreq = 10 * np.random.uniform(low=0.5, high=1)
                    self.params.sininit = 1.0
                
                trendcoef = TrendVector(self.params.rate * temporal + self.params.const +
                                        self.params.sinamp * np.sin((self.params.sininit * np.pi) + (2 * np.pi * self.params.sinfreq * temporal)))

            elif trend == 'linear+cosinusoidal':
                if assgin:
                    self.params.rate = 25 * np.random.uniform(low=-1, high=1)
                    self.params.const = 20 * np.random.uniform(-1, 1)

                    self.params.cosamp =  10 * np.random.uniform(low=-1, high=1)
                    self.params.cosfreq = 10 * np.random.uniform(low=0.5, high=1)
                    self.params.cosinit = 0.5
                
                trendcoef = TrendVector(self.params.rate * temporal + self.params.const +
                                        self.params.cosamp * np.cos((self.params.cosinit * np.pi) + (2 * np.pi * self.params.cosfreq * temporal)))
            
            elif trend  == 'linear+periodic':
                if assgin:
                    self.params.rate = 25 * np.random.uniform(low=-1, high=1)
                    self.params.const = 20 * np.random.uniform(-1, 1)

                    self.params.sinamp =  5 * np.random.uniform(low=-1, high=1)
                    self.params.sinfreq = 10 * np.random.uniform(low=0.5, high=1)
                    self.params.sininit = 1.0

                    self.params.cosamp =  5 * np.random.uniform(low=-1, high=1)
                    self.params.cosfreq = 10 * np.random.uniform(low=0.5, high=1)
                    self.params.cosinit = 0.5
                
                trendcoef = TrendVector(self.params.rate * temporal + self.params.const +
                                        self.params.sinamp * np.sin((self.params.sininit * np.pi) + (2 * np.pi * self.params.sinfreq * temporal)) + 
                                        self.params.cosamp * np.cos((self.params.cosinit * np.pi) + (2 * np.pi * self.params.cosfreq * temporal)))

            elif trend == 'linear+logarithmic':
                if assgin:
                    self.params.rate = 25 * np.random.uniform(low=-1, high=1)
                    self.params.const = 20 * np.random.uniform(-1, 1)

                    self.params.logamp = 5 * np.random.uniform(low=-1, high=1)
                    self.params.logcoef = 5 * np.random.uniform(low=0.5, high=1)
                
                trendcoef = TrendVector(self.params.rate * temporal + self.params.const +
                                        self.params.logamp * np.log(self.params.logcoef * temporal))
                
            elif trend == 'linear+exponential':
                if assgin:
                    self.params.rate = 25 * np.random.uniform(low=-1, high=1)
                    self.params.const = 20 * np.random.uniform(-1, 1)

                    self.params.expamp = 10 * np.random.uniform(low=-1, high=1)
                    self.params.expcoef = 5 * np.random.uniform(low=0.5, high=1)
                
                trendcoef = TrendVector(self.params.rate * temporal + self.params.const +
                                        self.params.expamp * np.exp(self.params.expcoef * temporal))
                
            elif trend == 'linear+power':
                if assgin:
                    self.params.rate = 25 * np.random.uniform(low=-1, high=1)
                    self.params.const = 20 * np.random.uniform(-1, 1)

                    self.params.powamp = 10 * np.random.uniform(low=-1, high=1)
                    self.params.powcoef = 5 * np.random.uniform(low=0.5, high=1)

                trendcoef = TrendVector(self.params.rate * temporal + self.params.const +
                                        self.params.powamp * np.power(self.params.powcoef, temporal))

            elif trend == 'linear+longwave':
                if assgin:
                    self.params.rate = 25 * np.random.uniform(low=-1, high=1)
                    self.params.const = 20 * np.random.uniform(-1, 1)

                    self.params.lwrate = 5 * np.random.uniform(low=-1, high=1)
                wavepattern = self.params.lwrate * MinMax.apply(np.ones((self.shape[0] , 1)) * np.linspace(0, 1, self.shape[1]), -1, 1)
                
                trendcoef = TrendVector(self.params.rate * temporal + self.params.const)

            elif trend == 'complex':
                if assgin:
                    self.params.rate = 25 * np.random.uniform(low=-1, high=1)
                    self.params.const = 20 * np.random.uniform(-1, 1)

                    self.params.logamp = 5 * np.random.uniform(low=-1, high=1)
                    self.params.logcoef = 1.0
                    self.params.logidx = np.random.choice(np.arange(0, int(0.3 * len(temporal))))

                    self.params.expamp = 5 * np.random.uniform(low=-1, high=1)
                    self.params.expcoef = 1.0
                    self.params.expidx = np.random.choice(np.arange(int(0.4 * len(temporal)), 
                                                                    int(0.7 * len(temporal))))

                    self.params.releaseidx = np.random.choice(np.arange(int(0.8 * len(temporal)), 
                                                                        len(temporal)))

                trendcoef = TrendVector(self.params.rate * temporal + self.params.const +
                                        self.params.logamp * np.log(self.params.logcoef * (temporal - temporal[self.params.logidx - 1])) * 
                                                                            np.where(np.logical_and(temporal >= temporal[self.params.logidx], temporal <= temporal[self.params.expidx]), np.ones_like(temporal), np.zeros_like(temporal)) +
                                        self.params.expamp * np.exp(self.params.expcoef * (temporal - temporal[self.params.expidx - 1])) * 
                                                                            np.where(np.logical_and(temporal >= temporal[self.params.expidx], temporal <= temporal[self.params.releaseidx]), np.ones_like(temporal), np.zeros_like(temporal)))
        
            else:
                raise NotImplementedError(f'{trend} was not implemented yet!')
            
            trendcoef = np.nan_to_num(trendcoef, nan=0.0, posinf=0.0, neginf=0.0)

            successful = bool(trendcoef.sum()) # If the sum of coefficients is zero, then we need to re-generate a new set of parameters
        
        trendcoef = np.where(active, trendcoef, np.zeros_like(trendcoef)) # trendcoef *= active
        if relative:
            trendcoef -= trendcoef[self.params.indx] # to preserve the same attiude
        trendcoef = np.where(inactive, trendcoef[np.argwhere((inactive * active) == 0)[-1]], trendcoef)
        
        tomtr = 0.01

        if defolimits is not None:
            accepted = True if defolimits['min'] <= np.max(np.abs(trendcoef)) * tomtr <= defolimits['max'] else False
            scale, ratio = 1.0, 0.2
            while not accepted:
                if defolimits['min'] <= np.max(np.abs(trendcoef)) * tomtr <= defolimits['max']:
                    accepted = True
                else:
                    if defolimits['min'] > np.max(np.abs(trendcoef)) * tomtr:
                        scale += ratio
                    else:
                        scale -= ratio
                
                    trendcoef *= scale
        # print(trendcoef.max(), trendcoef.min())

        stack += np.matmul(pattern[..., np.newaxis], trendcoef[np.newaxis,])
        if 'longwave' in trend:
            temporal -= temporal[self.params.indx]
            stack += np.matmul(wavepattern[..., np.newaxis], temporal[np.newaxis,])
        
        stack *= ((4 * np.pi * tomtr) / sensor.getWaveLength()) # Units in [m]

        return TrendData(stack)
    
