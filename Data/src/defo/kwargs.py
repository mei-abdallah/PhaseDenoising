from typing import Literal, Dict, Tuple, Union
import numpy as np

class SrcKwargs:
    def __init__(self, shape:Tuple[int, int]) -> None:
        self.shape = shape

    def getKwargs(self, name:Literal['mogi', 'quake', 'dyke', 'sill'], unit:bool=False) -> Dict[str, Union[int, float]]:
        if name.lower() == 'mogi':
            src_kwargs = self.getMogi()
            
            if unit:
                src_kwargs['volume_change'] = 1e7

        elif name.lower() == 'dyke':
            src_kwargs = self.getDyke()
            
            if unit:
                src_kwargs['opening'] = 1.0

        elif name.lower() == 'sill':
            src_kwargs = self.getSill()

            if unit:
                src_kwargs['opening'] = 1.0

        elif name.lower() in ['quake', 'normal', 'thrust', 'strike-slip', 'left-lateral', 'right-lateral']:
            src_kwargs = self.getQuake(name.lower())

            if unit:
                src_kwargs['slip'] = 1.0

        else:
            raise ValueError('Invalid source type!')
        return src_kwargs
    

    def getMogi(self, deflation:bool=False) -> Dict[str, Union[int, float]]:
        src_kwargs = {'volume_change' :  int(2e6 + 1e6 * np.random.rand()),                             # in metres, always positive
                      'depth'         :  1000 + 3000 * np.random.rand()}                                  # in metres
        if deflation:
            src_kwargs['volume_change'] *=  np.random.choice([-1, 1])                          
        return src_kwargs
    
    def getSill(self) -> Dict[str, Union[int, float]]:
        # maximum change due to depth flaculation will be 261 m so it will be ignored
        return {'strike'   : np.random.randint(0, 359),                                            # in degrees
                'dip'      : np.random.randint(0, 5),                                              # in degrees
                'rake'     : 0,                                                                  # in degrees
                'length'   : 2000 + 4000 * np.random.rand(),                                       # in meters
                'width'    : 2000 + 4000 * np.random.rand(),                                       # in metres
                'depth'    : 1500 + 2000 * np.random.rand(),                                       # in metres
                'slip'     : 0.0,                                                                  # in meters
                'opening'  : 0.2 + 0.8 * np.random.rand()}                                         # in metres
            

    def getDyke(self) -> Dict[str, Union[int, float]]:
        src_kwargs = {'strike'   : np.random.randint(0, 359),                                            # in degrees
                      'dip'      : np.random.randint(75, 90),                                            # in degrees
                      'rake'     : 0,                                                                    # in degrees
                      'length'   : 10000 * np.random.rand(),                                             # in metres
                      'width'    : 6000 * np.random.rand(),                                              # in meters  
                      'slip'     : 0.0,                                                                  # in meters
                      'opening'  : 0.1 + 0.6 * np.random.rand()}                                         # in metres  
        # bottom depth is top depth plus a certain random amount
        src_kwargs['depth'] = (2000 * np.random.rand()) +  0.5 * src_kwargs['width'] * np.sin(np.deg2rad(src_kwargs['dip']))                        # centroid_depth = top_depth + width/2 * sin(dip) 
        return src_kwargs

    def getQuake(self, source:str) -> Dict[str, Union[int, float]]:
        top_depth = np.random.choice(np.arange(0, 20, 1))*1000                                            # in metres [0 >> 20 km]

        if source in ['quake', 'normal', 'thrust', 'strike-slip', 'left-lateral', 'right-lateral']:
            src_kwargs = {'strike'      : np.random.randint(0, 359),                                             # in degrees [0 >> 360]
                            'length'    : np.random.choice(np.arange(10, 50, 1))*1000,                              # in metres [10 >> 50 km]
                            'width'     : np.random.choice(np.arange(5, 30, 1))*1000,                               # in meters [5 >> 30 km]
                            'slip'      : np.random.choice(np.arange(0.5, 12, 0.5)),                                 # in meters [0.5 >> 12 m]
                            'opening'   : 0.0,}
        #src_kwargs['bottom_depth'] = 5000 + src_kwargs['top_depth'] + 2500 * np.random.rand()        # bottom depth is top depth plus a certain random amount
        if source == 'quake':
            src_kwargs['rake'] = np.random.choice([0, 180, 90, -90])                                             # in degrees 0 for left lateral ss, 180 or -180 for right lateral ss, -90 for normal, and 90 for thrust 
            
            if src_kwargs['rake'] == 0 or src_kwargs['rake'] == 180:
                src_kwargs['dip'] = np.random.randint(50, 90)                                                    # for strike-slip >> dip range [50 : 90]
            
            elif src_kwargs['rake'] == -90 or src_kwargs['rake'] == 90:
                src_kwargs['dip'] = np.random.randint(10, 60)                                                    # for strike-slip >> dip range [50 : 90]
            
            else:
                raise Exception("rake must be 0 or 180 or 90 or -90")
            
            if top_depth < 5000:
                src_kwargs['slip'] = np.random.choice(np.arange(0.5,5,0.5))                                         # if top_depth < 5 km rearange the slip to be [0.5 >> 12 m] >>> SarNet
        
        elif source == 'normal':
            src_kwargs['rake'] = -90
            src_kwargs['dip'] = np.random.randint(10, 60)                                                # in degrees
        
        elif source == 'thrust':
            src_kwargs['rake'] = 90
            src_kwargs['dip'] = np.random.randint(10, 90)
        
        elif source == 'strike-slip':
            src_kwargs['rake'] = np.random.choice([0, 180])
            src_kwargs['dip'] = np.random.randint(50, 90)

        elif source == 'right-lateral':
            src_kwargs['rake'] = 0
            src_kwargs['dip'] = np.random.randint(50, 90)

        elif source == 'left-lateral':
            src_kwargs['rake'] = 180
            src_kwargs['dip'] = np.random.randint(50, 90)

        src_kwargs['depth'] = top_depth +  0.5 * src_kwargs['width'] * np.sin(np.deg2rad(src_kwargs['dip']))                        # centroid_depth = top_depth + width/2 * sin(dip)

        return src_kwargs
    
    def getCentre(self, resolution:Dict[str, float], edge:float=0.2) -> Tuple[float, float]:
        """  get the center of the source the corrdinate is claculated according to the upperleft point """
        # shematic design of the prpoposed center
        # |* refpoint     |
        # |    ______     |
        # |edge|     |edge|
        # |    |  *  |    |
        # |    |_____|    |
        # |               |
        # |               |

        length, width = self.shape

        x = (edge + np.random.rand() * (1 - 2 * edge)) * width * resolution['x']
        y = (edge + np.random.rand() * (1 - 2 * edge)) * length * resolution['y']

        return x, y
    
    