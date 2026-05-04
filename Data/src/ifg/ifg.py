from typing import Tuple, Literal, Optional, Union, Mapping
import numpy as np
from ..dem import Dem
from ..delays import Tropospheric, Turblent, Orbit
from ..noise import Speckle
from ..mask import Coherence
from ..defo import Wrapper, SrcKwargs
from ..objects import Sensor, Projection, Converter
from ..base import Single
from ..dtype import Data

class Ifg(Single):
    @classmethod
    def create(cls, 
               shape:Tuple[int, int],
               location:Mapping[str, Union[str, float, Tuple[float, float]]],
               resolution:Optional[float]=None,
               reshape:Literal['crop', 'resize']='resize',
               platform:Literal['ASAR', 'ERS', 'ALOS1', 'ALOS2', 'RADARSAT', 'SENTINEL', 'CSK', 'TSK']='SENTINEL', 
               polydeg:Optional[Literal['second', 'third', 'fifth']]=None,
               method:Optional[Literal['fft', 'cov', 'eig', 'fractal', 'trend']]=None,
               order:Optional[Literal['first', 'second']]=None,
               source:Optional[Literal['nodefo','mogi', 'sill', 'dyke', 'quake', 'normal', 'thrust', 'left-lateral', 'right-lateral']]=None, 
               track:Optional[Literal['asc', 'desc', 'random']]=None,
               limits:Optional[Mapping[str, float]]=None,
               threshold:float=0.2,
               snr:float=2.0,
               verbose:bool=False,
               warning:bool=False) -> 'Ifg':
        
        polydeg = polydeg or np.random.choice(['second', 'third', 'fifth'])
        source = source or np.random.choice(['nodefo', 'mogi', 'sill', 'dyke', 'quake', 'normal', 'thrust', 'left-lateral', 'right-lateral'])
        method = method or np.random.choice(['fft', 'eig', 'fractal', 'trend'])
        order = order or np.random.choice(['first', 'second'])

        if verbose:
            print('Initializing Interferogram... | ', end='')

        if verbose:
            print('Processing the DEM... | ', end='')
        
        dem = Dem(shape).create(location, resolution, reshape)

        if verbose:
            print('Processing the heights and WaterMask... | ', end='')

        heights = dem.getData()
        watermask = dem.getMask()

        if verbose:
            print('Getting Sensor Values...| ', end='')
        
        sensor = Sensor(platform)

        if verbose:
            print('Creating the proj function... | ', end='')

        proj = Projection().create(sensor, track)

        if verbose:
            print('Setting the grid converter ... | ', end='')

        converter = Converter().setGridParams(*dem.getGeoTransform())

        if verbose:
            print('Creating the Decorrlation mask... | ', end='')

        decorrmask = Coherence(shape).create(2, threshold).diff().squeeze()

        succesful, count, maxcount = False, 0, 8

        while not succesful and (count < maxcount):
            if source == 'nodefo':
                visibledefo = visiblesignal = True

                if verbose:
                    print('Getting the source parameters... | ', end='')

                source_kwargs = {}

                if verbose:
                    print('Getting the source epicenter... | ', end='')

                center = None

                if verbose:
                    print('Creating the surface deformation... | ', end='')
                
                defo = Data(np.zeros(shape))

                if verbose:
                    print('Creating the topographic delay... | ', end='')

                tropodelay = Tropospheric(shape).create(2, heights, order, tropovar=10).diff().squeeze()
                
                if verbose:
                    print('Creating the turbulent delay... | ', end='')

                turbdelay = Turblent(shape).create(2, dem.getResolution(), method).diff().squeeze()

                if verbose:
                    print('Creating the orbital ramp delay... | ', end='')

                rampdelay = Orbit(shape).create(2, polydeg).diff().squeeze()

                if verbose:
                    print('Creating the noise... | ', end='')

                noise = Speckle(shape).create(2).diff().squeeze()


            else:
                visibledefo, itr, maxitrs = False, 0, 8

                if verbose:
                    print('Getting the source parameters... | ', end='')

                source_kwargs = SrcKwargs(shape).getKwargs(source)
                
                while (not visibledefo) and (itr <= maxitrs):
                    
                    if verbose:
                        print('Getting the source epicenter... | ', end='')
                    
                    center = SrcKwargs(shape).getCentre(dem.getResolution())

                    if verbose:
                        print('Creating the surface deformation... | ', end='')
                    
                    defo, source_kwargs = Wrapper(shape).create(dem.getResolution(), sensor, source, center, source_kwargs, proj, limits)

                    if verbose:
                        print('Check the visible defo... | ', end='')
                    
                    visibledefo, defomask = Ifg.checkVisibleDefo(defo, watermask, decorrmask)

                    if not visibledefo:
                        itr += 1
                

                if visibledefo:
                    
                    if verbose:
                        print(f'Viable location... | ', end = '')

                    visiblesignal, itr, maxitrs = False, 0, 8

                    while (not visiblesignal) and (itr <= maxitrs):
                        
                        if verbose:
                            print('Creating the topographic delay... | ', end='')

                        tropodelay = Tropospheric(shape).create(2, heights).diff().squeeze()
                        
                        if verbose:
                            print('Creating the turbulent delay... | ', end='')

                        turbdelay = Turblent(shape).create(2, dem.getResolution(), method).diff().squeeze()

                        if verbose:
                            print('Creating the orbital ramp delay... | ', end='')

                        rampdelay = Orbit(shape).create(2, polydeg).diff().squeeze()

                        if verbose:
                            print('Creating the noise... | ', end='')
                            
                        noise = Speckle(shape).create(2).diff().squeeze()

                        if verbose:
                            print('Check the visible signal... | ', end='')
                        
                        visiblesignal = Ifg.checkVisibleSignal(defo, defomask, tropodelay, turbdelay, rampdelay, noise, snr)

                        if not visiblesignal:
                            itr += 1
                    
                    if visiblesignal:
                        if verbose:
                            print('Viable SNR ... | ', end = '')
                    else:
                        if verbose:
                            print('SNR is too low.')
                
                else:
                    if verbose:
                        print('No viable location found.')

            if visibledefo and visiblesignal:
                succesful = True

            else:
                count += 1

        if succesful:
            if verbose:
                print('Locating signals... | ', end='')

            bbox = Ifg.locateSignal(defo, converter, center)

        else:
            if verbose:
                print('Failed to locate a suitable visibility signal in this Interferogram !.')
            
            if warning:
                raise Exception('Failed to locate a suitable visibility signal in this Interferogram !.')
            
        if verbose:
            print('Done.')

        return cls(defo, turbdelay, tropodelay, rampdelay, noise, watermask, decorrmask, source_kwargs, center, bbox)
    

    @staticmethod
    def checkVisibleDefo(defo:np.ndarray, watermask:np.ndarray, decorrmask:Optional[np.ndarray], threshold:float=0.3, fraction:float=0.8) -> Tuple[bool, np.ndarray]:
        """Check if the deformation is visible. 
        
        Parameters:
            - defo | 2D Array | the created deformation
            - watermask | 2D Array | mask due to water areas. i.e., zero is the value of the masked pixels.
            - decorrmask | 2D Array | mask due to in coherent reigons. i.e., zero is the value of the masked pixels.
            - threshold | float | is the limiting ration of the maximum deformation to considere as a deformation 
            - fraction | float | if this fraction of deformation is not in masked area, defo, coh mask, and water mask are deemed compatible

        Returns:
            - viable | bool | if the deformation is visible or not
            - defomask | 2D array | mask of the pixels which has less than % of the maximim displacment
        """

        viable = True

        defomask = np.where(np.abs(defo) > (threshold * np.max(np.abs(defo))), np.zeros_like(defo), np.ones_like(defo))
        combinedmask = np.invert(np.maximum(watermask.astype(bool), decorrmask.astype(bool)))

        ratio = np.ma.mean(np.ma.array(combinedmask, mask=defomask))

        if ratio < fraction:
            viable = False

        return viable, defomask
    
    @staticmethod
    def checkVisibleSignal(defo:np.ndarray, defomask:np.ndarray, tropodelay:np.ndarray, turbdelay:np.ndarray, rampdelay:np.ndarray, noise:np.ndarray, threshold:float=2.0) -> bool:
        """"""

        viable = True
        defo = np.ma.array(defo, mask=defomask)

        delays =  np.ma.array((tropodelay + turbdelay + rampdelay + noise), mask=defomask)
        snr = np.var(np.ma.compressed(defo)) / np.var(np.ma.compressed(delays))
        
        if snr < threshold:
            viable = False

        return viable
    
    
    @staticmethod
    def locateSignal(defo:np.ndarray, converter:Converter, center:Optional[Tuple[float, float]]=None, threshold:Optional[float]=None, dtype:Literal['ratio', 'pixs']='pixs') -> Tuple[float, float, float, float]:
        """Return a region that contains deformation above the threshold 
        (both positive and negative are considered)
        
        Parameters:
            - defo | 2D Array | the surface deformation. i.e., equals zeros if source defo is `nodefo`
            - center | tuple | center of deformation signal, appears to be in matrix notation (ie 0,0 is top left)
            - threshold | float | value above which deformation is selected. if None, default, set to 20% of maximum absolute deformation
            - dtype | str | if 'ratio', return ratios, if 'pixs', return pixel indices

        Returns:
            - bbox | 1D Array | returns the center, and half the dimensions in pixles or ratios
        """

        if np.any(defo):
            if center is None:
                ycenter, xcenter = np.unravel_index(np.argmax(defo), defo.shape)
            else:
                xcenter, ycenter = converter.getXpixYpix(*center, dtype='mtr')

            if not threshold:
                threshold = 0.2
            
            limits =  threshold * np.max(np.abs(defo))
            defoargs = np.argwhere(np.abs(defo) > limits)        # a matrix of the pixels that have a magnitude above the threshold (ie both positive and negative deformation)
            xstart, xstop = np.min(defoargs[:, 1]), np.max(defoargs[:, 1])                 # column 1 is for xs
            ystart, ystop = np.min(defoargs[:, 0]), np.max(defoargs[:, 0])                 # column 0 is for ys
            xhalfwidth, yhalfwidth = int(np.ceil(sum([xcenter - xstart, xstop - xcenter])/2)), int(np.ceil(sum([ycenter - ystart, ystop - ycenter])/2))           # the size of the pattern in x direction is the mean of the distance from the centre to each edge

        else:
            xcenter, ycenter = 0, 0
            xhalfwidth, yhalfwidth = 0, 0


        if dtype == 'pixs':
            bbox = (xcenter, 
                    ycenter, 
                    xhalfwidth, 
                    yhalfwidth)

        elif dtype == 'ratio':
            bbox = (xcenter / defo.shape[1], 
                    ycenter / defo.shape[1],
                    xhalfwidth / defo.shape[1],
                    yhalfwidth / defo.shape[1])

        else:
            raise ValueError(f"dtype should be 'ratio' or 'pixs'. Got {dtype}")

        return bbox