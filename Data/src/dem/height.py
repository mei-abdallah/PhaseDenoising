from typing import Tuple, Optional, Literal, Union, Mapping, List
from osgeo import gdal
import numpy as np
import os
from .srtm import Srtm
from .raster import Raster
from .geolocation import GeoLocation


class HeightsRaster(Raster):    
    def plot(self, save:bool=False) -> None:
        return super().getArray().plot(save)

class Heights(GeoLocation):
    """ A class to create and open the digtal elvetion model. Given lons and lats (integers), make a multi tile DEM from either SRTM1 or 3 data.
    
    Parameters:
        - shape | (int, int) | the length and width of the preocessed dem
        - srtm | str | either use SRTM3 (~90m pixels) or SRTM1 (~30m pixels)
        - workdir | str | the working dir
        - username | str | Earthdata username, to apply: https://earthdata.nasa.gov/eosdis/science-system-description/eosdis-components/earthdata-login
        - password | str | Earthdata password
        - fillvoid | boolean | If true, will try to linearly interpolate across voids in data.  
        - verbose | boolean | print processing steps if true
    """
    def __init__(self, shape:Optional[Tuple[int, int]]=None, srtm:Literal['srtm1', 'srtm3']='srtm3', 
                 workdir:str='./', username:Optional[str]=None, password:Optional[str]=None,
                 verbose:bool=False) -> None:
        super().__init__(shape, os.path.join(workdir, 'Dem/downloads', f'uncropped.vrt'))
        
        self.srtm = Srtm(srtm, f'{workdir}/Dem/downloads', username, password, verbose)
        self.verbose = verbose

        
    def create(self, location:Mapping[str, Union[str, float, Tuple[float, float]]], pixelsize:Optional[float]=None) -> HeightsRaster:
        
        self.setLocation(**location)

        self.setSize(self.srtm.pixs2deg)

        tilefiles = self.getTileFiles()

        gdal.BuildVRT(self.vrtfiledir, tilefiles)

        dataset = gdal.Translate('', self.vrtfiledir, 
                            options=gdal.TranslateOptions(format='MEM', 
                                                          outputType=gdal.GDT_Float32,
                                                          projWin=self.getWindow(),
                                                          **(self.getResolution(pixelsize)
                                                             if pixelsize is not None else 
                                                             {'height' : self.shape[0], 'width' : self.shape[1], 'outputBounds' : self.getWindow()})
                                                            ))

        return HeightsRaster(data=dataset)
    

    def getTileFiles(self) -> List[str]:
        
        if self.location is None:
            raise Exception('`location` of the digital elvation model is not known pleas identify it using `setLocation` method')
        
        tilefiles = []

        west, east, south, north = self.getTileEdges()

        # 1: Initiliase the big DEM:
        lats = np.arange(south, north, 1)
        lons = np.arange(west, east, 1)

        # 2: Work through each tile
        for lon in lons:                                                                                  # one column first, make the name for the tile to try and download
            for lat in lats:                                                                              # and then rows for that column
            # 2.0 read or download the tile file
                if self.verbose:
                    print(f'{self.srtm.getName(lon, lat)} : Trying to open locally... | ', end = '')
                existflag = self.srtm.getExists(lon, lat)
                if existflag:
                    tilefiles.append(self.srtm.getPath(lon, lat))
                    if self.verbose:
                        print('Done!')

                else:
                    if self.verbose:
                        print('Failed.  ')
                        print(f'{self.srtm.getName(lon, lat)} : Trying to download it... | ', end = '')
                    successful = self.srtm.getDownload(lon, lat)
                    if successful:
                        tilefiles.append(self.srtm.getPath(lon, lat))
        return tilefiles
    

    def getTileEdges(self) -> Tuple[int, int, int, int]:
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
        return f"<{self.__class__.__name__} (\n srtm: {self.srtm.type}" + (f",\n bounds: {self.location}" if self.location else "") + (f",\n shape: {self.shape}" if self.shape else "") + ")>"
    
    def __str__(self) -> str:
        return f"<{self.__class__.__name__} (\n srtm: {self.srtm.type}" + (f",\n bounds: {self.location}" if self.location else "") + (f",\n shape: {self.shape}" if self.shape else "") + ")>"