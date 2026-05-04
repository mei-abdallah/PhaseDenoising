from typing import Mapping, Literal, Tuple, Optional, Union, List

from osgeo import gdal
import numpy as np
import os
from .waterlines import WaterLines
from .raster import Raster
from .geolocation import GeoLocation


class WaterRaster(Raster):
    def plot(self, save:bool=False) -> None:
        return super().getArray().plot(save)

class Water(GeoLocation):
    def __init__(self, shape:Optional[Tuple[int, int]]=None, resolution:Literal['c', 'l', 'i', 'h', 'f']='f', srtm:Literal['srtm1', 'srtm3']='srtm3',
                 workdir:str='./', gshhsurl='http://www.soest.hawaii.edu/pwessel/gshhg/', 
                 verbose:bool=False) -> None:
        super().__init__(shape, os.path.join(workdir, 'Water/downloads', f'uncropped.vrt'))
        
        self.waterlines = WaterLines(f'{workdir}/Water/downloads', resolution, srtm, gshhsurl, verbose)
        self.verbose = verbose

    def create(self, location:Mapping[str, Union[str, float, Tuple[float, float]]], pixelsize:Optional[float]=None) -> WaterRaster:

        self.setLocation(**location)

        self.setSize(self.waterlines.pixs2deg)

        linesfiles = self.getWaterFiles()

        gdal.BuildVRT(self.vrtfiledir, linesfiles)

        dataset = gdal.Translate('', self.vrtfiledir, 
                options=gdal.TranslateOptions(format='MEM', 
                                              outputType=gdal.GDT_Int16,
                                              projWin=self.getWindow(),
                                              **(self.getResolution(pixelsize)
                                                 if pixelsize is not None else 
                                                 {'height' : self.shape[0], 'width' : self.shape[1], 'outputBounds' : self.getWindow()})
                                                ))

        return WaterRaster(mask=dataset)

    def getWaterFiles(self) -> List[str]:
        waterfiles = []
        if self.location is None:
            raise Exception('`location` of the ditial elvation model is not known pleas identify it using `setLocation` method')

        west, east, south, north = self.getWaterEdges()

        # 1: Initiliase the big DEM:
        lats = np.arange(south, north, 1)
        lons = np.arange(west, east, 1)

        # 2: Work through each water area
        for lon in lons:                                                                                  # one column first, make the name for the water to try and download
            for lat in lats:                                                                              # and then rows for that column
            # 2.0 read or download the water file
                if self.verbose:
                    print(f'{self.waterlines.getName(lon, lat)} : Trying to open locally... | ', end = '')
                existflag = self.waterlines.getExists(lon, lat)
                if existflag:
                    waterfiles.append(self.waterlines.getPath(lon, lat))
                    if self.verbose:
                        print('Done!')

                else:
                    if self.verbose:
                        print('Failed.  ')
                        print(f'{self.waterlines.getName(lon, lat)} : Trying to create it... | ', end = '')
                    self.waterlines.create(lon, lat)   
                    waterfiles.append(self.waterlines.getPath(lon, lat))
                    if self.verbose:
                        print('Done!')
        return waterfiles


    def getWaterEdges(self) -> Tuple[int, int, int, int]:
        """ Given DEM limits as floats (ie not integers), determine the limits of the DEM in integers.  
            e.g. if western limit is 3.5, this will return 3 as the western extent of the westernmost tile required.  
 
        Returns:
            w/e/s/n | int | orthern, eastern, western, and southern edges of tile required to span the DEM
            as above for the other edges.  
        History:
            2020/09/21 | MEG | Written 
        """
        if self.location is None:
            raise Exception('`location` of the ditial elvation model is not known pleas identify it using `setLocation` method')
        
        west, south, east, north = self.location
        
        west = int(np.floor(west))
        east = int(np.ceil(east))
        south = int(np.floor(south))
        north = int(np.ceil(north))

        return west, east, south, north
    
    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} (\n resolution: {self.waterlines.gshhs.resolution}" + (f",\n bounds: {self.location}" if self.location else "") + (f",\n shape: {self.shape}" if self.shape else "") + ")>"
    
    def __str__(self) -> str:
        return f"<{self.__class__.__name__} (\n resolution: {self.waterlines.gshhs.resolution}" + (f",\n bounds: {self.location}" if self.location else "") + (f",\n shape: {self.shape}" if self.shape else "") + ")>"