from typing import Literal, Union
import numpy as np
class Sensor():
    data = {'asar' : {'wavelength' : 0.0562,
                      'slantrange' : 845331.9499,
                      'incangle'   : 22.8076,
                      'cycle'      : 35,
                      'std'        : 200,
                      'firstacq'   : '20030927',
                      'critical'   : {'baseline' : 5100,
                                      'time'     : 1200}},

            'sentinel' : {'wavelength' : 0.0562,
                          'slantrange' : 851628.6300,
                          'incangle'   : (30, 50), #33.8299
                          'headangle'  : (-360, 360),
                          'trackangle'  : {'asc'  : 195,
                                          'desc' : 385},
                          'cycle'      : 12,
                          'firstacq'   : '20140403',
                          'std'        : 100,
                          'critical'   : {'baseline' : 5188,
                                          'time'     : 1200}},

            'csk' : {'wavelength' : 0.0311,
                     'slantrange' : 714121.1074,
                     'incangle'   : 29.5061,
                     'std'        : 150,
                     'critical'   : {'baseline' : 1300,
                                     'time'     : 400}},

            'alos1' : {'wavelength' : 0.2361,
                       'slantrange' : 867919.5698,
                       'incangle'   : 38.7491,
                       'cycle'      : 46,
                       'std'        : 100,
                       'firstacq'   : '20060927',
                       'critical'   : {'baseline' : 1000,
                                       'time' : 200}},

            'alos2' : {'wavelength' : 0.2361,
                       'slantrange' : 838154.6476,
                       'incangle'   : 43.3408,
                       'cycle'      : 11,
                       'std'        : 100,
                       'firstacq'   : '20150927',
                       'critical'   : {'baseline' : 1000,
                                       'time'     : 200}},

            'tsk' : {'wavelength' : 0.0311,
                     'slantrange' : 625804.5425,
                     'incangle'   : 37.3477,
                     'cycle'      : 11,
                     'std'        : 50,
                     'firstacq'   : '20070927',
                     'critical'   : {'baseline' : 800,
                                     'time'     : 365}},

            'ers' : {'wavelength' : 0.0562,
                     'slantrange' : 845283.4859,
                     'incangle'   : 22.8359,
                     'std'        : 100,
                     'critical'   : {'baseline' : 1000,
                                     'time'     : 200}}, 

            'radarsat' : {'wavelength' : 0.0555,
                          'slantrange' : 952019.5478,
                          'incangle'   : 35.5103,
                          'cycle'      : 12,
                          'std'        : 100,
                          'firstacq'   : '20080927',
                          'critical'   : {'baseline' : 1000,
                                          'time'     : 200}}}
    def __new__(cls, platform:Union['Sensor', Literal['asar', 'sentinel', 'csk', 'alos1', 'alos2', 'tsk', 'ers', 'radarsat']]='sentinel') -> 'Sensor':
        if isinstance(platform, str):
            cls.platform = platform.lower()
            cls.__data__ = {key : np.random.uniform(*value) if isinstance(value, (list, tuple)) else value 
                                    for key, value in cls.data[platform.lower()].items()} 
            return super().__new__(cls)
        return platform
    
    # def __init__(self, platform:Union['Sensor', Literal['asar', 'sentinel', 'csk', 'alos1', 'alos2', 'tsk', 'ers', 'radarsat']]='asar') -> None:
    #     self.platform = platform.lower()
    #     self.__data__ = {key : np.random.uniform(*value) if isinstance(value, (list, tuple)) else value 
    #                             for key, value in self.data[platform.lower()].items()} 

    def getName(self) -> str:
        return self.platform.upper()
    
    def getBaseLineSTD(self) -> float:
        return self.__data__['std']
    
    def getCycle(self) -> float:
        return self.__data__['cycle']
    
    def getWaveLength(self) -> float:
        return self.__data__["wavelength"]
    
    def getSlantRange(self) -> float:
        return self.__data__["slantrange"]
    
    def getIncidentAngle(self) -> float:
        return self.__data__["incangle"]
    
    def getHeadingAngle(self) -> float:
        return self.__data__["headangle"]
    
    def getTrackAngle(self, track:Literal['asc', 'desc', 'random']='random') -> float:
        if track == 'random':
            track = 'asc' if np.random.rand() > 0.5 else 'desc'
        return self.__data__["trackangle"][track]
    
    def getCriticalBaseTime(self) -> float:
        return self.__data__["critical"]["time"]
    
    def getCriticalBaseLine(self) -> float:
        return self.__data__["critical"]["baseline"]
    
    def getFirstAcquisition(self) -> str:
        return self.__data__['firstacq']

    def __repr__(self) -> str:
        return (f'<name : {self.platform}, '
                f'wavelength: {self.getWaveLength()} m, '
                f'slantrange: {self.getSlantRange()} m, '
                f'incangle : {self.getIncidentAngle()} dgrees, '
                f'cirtical basleline: {self.getCriticalBaseLine()} m, '
                f'critical time: {self.getCriticalBaseTime()} days>')

    def __str__(self) -> str:
        return f'<{self.getName()} Satellite>'

