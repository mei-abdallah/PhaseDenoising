from typing import Mapping, Literal, Tuple, Optional, Union
from .height import Heights
from .water import Water
from .raster import Raster

class DemRaster(Raster):
    def plot(self, save:bool=False) -> None:
        return super().getArray().plot(save)

class Dem:
    """ A class to create a degital elevation model and the associated water mask
    
    Parameters:
        - shape | Tuple | the shape of the output dem and watermask
        - srtm | str | either use SRTM3 (~90m pixels) or SRTM1 (~30m pixels)
        - resolution | str | the resolution of the water lines e.g., 
                                    'f' -> full resolution: Original (full) data resolution.
                                    'h' -> high resolution: About 80 % reduction in size and quality.
                                    'i' -> intermediate resolution: Another ~80 % reduction.
                                    'l' -> low resolution: Another ~80 % reduction.
                                    'c' -> crude resolution: Another ~80 % reduction.
        - workdir | str | the processing directory 
    """
    def __init__(self, shape:Tuple[int, int], srtm:Literal['srtm1', 'srtm3']='srtm3', resolution:Literal['c', 'l', 'i', 'h', 'f']='f',
                 workdir:str='./', username:Optional[str]=None, password:Optional[str]=None, gshhsurl='http://www.soest.hawaii.edu/pwessel/gshhg/', 
                 verbose:bool=False) -> None:
        self.shape = shape
        self.heights = Heights(None, srtm, workdir, username, password, verbose)
        self.water = Water(None, resolution, srtm, workdir, gshhsurl, verbose)
        

    def create(self, location:Mapping[str, Union[str, float, Tuple[float, float]]], pixelsize:Optional[float]=None, reshape:Literal['crop', 'resize']='resize') -> DemRaster:
        """ Create a digital elevation model for a given location """
        heights = self.heights.create(location, pixelsize).getReference()
        water = self.water.create(location, pixelsize).getReference()

        return DemRaster(heights, water).reshape(reshape, self.shape)

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} (\n srtm: {self.heights.srtm.type},\n water: {self.water.waterlines.gshhs.resolution}" + (f",\n bounds: {self.heights.location}" if self.heights.location else "") + (f",\n shape: {self.heights.shape}" if self.heights.shape else "") + ")>"
    
    def __str__(self) -> str:
        return f"<{self.__class__.__name__} (\n srtm: {self.heights.srtm.type},\n water: {self.water.waterlines.gshhs.resolution}" + (f",\n bounds: {self.heights.location}" if self.heights.location else "") + (f",\n shape: {self.heights.shape}" if self.heights.shape else "") + ")>"