from typing import Tuple, Mapping, Optional, Literal, Union
import numpy as np
from osgeo import gdal
from geopy.distance import geodesic
from ..dtype import Data, MaskedData

class RasterData(Data):
    def plot(self, save:bool=False) -> None:
        """Plot the raster data."""
        return super().plot('Heights Data', 'terrain', 'm', save)

class RasterMask(Data):
    def plot(self, save:bool=False) -> None:
        """Plot the mask raster data."""
        return super().plot('Water Mask', 'gray', None, save)
    
class RasterMaskedData(MaskedData):
    def plot(self, save:bool=False) -> None:
        """Plot the masked raster data."""
        return super().plot('Masked Heights Data', 'terrain', 'm', save)

class Raster():
    def __init__(self, data:Optional[gdal.Dataset]=None, mask:Optional[gdal.Dataset]=None) -> None:
        self.data = data
        self.mask = mask
    
    def getReference(self) -> gdal.Dataset:
        """Get the Raster reference."""
        return self.data or self.mask
    
    def getBounds(self) -> Tuple[float, float, float, float]:
        """get the bounds of the Raster e.g., (minlon, minlat, maxlon, maxlat)"""
        geoTrans, xPixs, yPixs  = self.getReference().GetGeoTransform(), self.getReference().RasterXSize, self.getReference().RasterYSize
        minlon = round(geoTrans[0], 1)
        minlat = round(geoTrans[3] + geoTrans[5] * yPixs, 1) 
        maxlon = round(geoTrans[0] + geoTrans[1] * xPixs, 1)
        maxlat = round(geoTrans[3], 1)
        return minlon, minlat, maxlon, maxlat
    
    def getShape(self) -> Tuple[int, int]:
        """get the shape of the Raster e.g., (height, width)"""
        # print(self.shape)
        return self.getReference().RasterYSize, self.getReference().RasterXSize
    
    def getResolution(self) -> Mapping[str, float]:
        """Get the Raster resolution."""
        geoTrans = self.getReference().GetGeoTransform()
        lon, lonres, lat, latres = geoTrans[0], geoTrans[1], geoTrans[3], geoTrans[5]
        return {'x' : np.abs(geodesic((lat, lon), (lat, lon + lonres)).meters),
                'y' : np.abs(geodesic((lat, lon), (lat + latres, lon)).meters)}
    
    def getGeoTransform(self) -> Tuple[Tuple[float, float], Tuple[float, float]]:
        """get the geotransform of the Raster in (lon, lat), (lonres, latres) in degrees"""
        geoTrans = self.getReference().GetGeoTransform()
        # lon, lat, lonres, latres = geoTrans[0], geoTrans[3], geoTrans[1], geoTrans[5]
        return (geoTrans[0], geoTrans[3]), (geoTrans[1], geoTrans[5])
    
    def getLonsLats(self) -> Tuple[np.ndarray, np.ndarray]:
        """get the longitudes and Latitudes from a the processed Raster"""
        geoTrans, xPixs, yPixs  = self.getReference().GetGeoTransform(), self.getReference().RasterXSize, self.getReference().RasterYSize
        lats = np.linspace(geoTrans[3], geoTrans[3]+(geoTrans[5]*yPixs), yPixs)
        lats = np.repeat(lats[:, np.newaxis], xPixs, axis=1)
        lons = np.linspace(geoTrans[0], geoTrans[0]+(geoTrans[1]*xPixs), xPixs)
        lons = np.repeat(lons[np.newaxis, :], yPixs, axis=0)
        return  lons, lats
    
    def getLatitudes(self) -> np.ndarray:
        """get the Latitudes from a the processed Raster"""
        geoTrans, xPixs, yPixs  = self.getReference().GetGeoTransform(), self.getReference().RasterXSize, self.getReference().RasterYSize
        lats = np.linspace(geoTrans[3], geoTrans[3]+(geoTrans[5]*yPixs), yPixs)
        lats = np.repeat(lats[:, np.newaxis], xPixs, axis=1)
        return lats
    
    def getLongitudes(self) -> np.ndarray:
        """get the Longitudes from a the processed Raster"""
        geoTrans, xPixs, yPixs  = self.getReference().GetGeoTransform(), self.getReference().RasterXSize, self.getReference().RasterYSize
        lons = np.linspace(geoTrans[0], geoTrans[0]+(geoTrans[1]*xPixs), xPixs)
        lons = np.repeat(lons[np.newaxis, :], yPixs, axis=0)
        return lons
    
    def reshape(self, method:Literal['crop', 'resize']='resize', shape:Optional[Tuple[int, int]]=None) -> 'Raster':
        if shape is not None:
            if shape != self.getShape():
                if method == 'resize':
                    if self.data is not None:
                        self.data = gdal.Translate('', self.data, 
                                                  options=gdal.TranslateOptions(format='MEM',
                                                                                height=shape[0],
                                                                                width=shape[1]))
                    if self.mask is not None:
                        self.mask = gdal.Translate('', self.mask, 
                                                  options=gdal.TranslateOptions(format='MEM',
                                                                                height=shape[0],
                                                                                width=shape[1]))
                    
                elif method == 'crop':
                    try:
                        xstart = np.random.randint(low=0, high=abs(shape[1] - self.getShape()[1]))
                    except:
                        xstart = 0

                    try:
                        ystart = np.random.randint(low=0, high=abs(shape[0] - self.getShape()[0]))
                    except:
                        ystart = 0

                    if self.data is not None:
                        self.data = gdal.Translate('', self.data, 
                                                  options=gdal.TranslateOptions(format='MEM',
                                                                                srcWin=[xstart, ystart, shape[1], shape[0]]))
                    if self.mask is not None:
                        self.mask = gdal.Translate('', self.mask, 
                                                  options=gdal.TranslateOptions(format='MEM',
                                                                                srcWin=[xstart, ystart, shape[1], shape[0]]))
                    
        return type(self)(self.data, self.mask)
    

    def getArray(self) -> Union[RasterData, RasterMask, RasterMaskedData]:
        """Return the data as a numpy array"""

        if self.data is not None and self.mask is not  None:
            array = RasterMaskedData(self.data.ReadAsArray(), self.mask.ReadAsArray())
        
        elif self.data is not None:
            array = RasterData(self.data.ReadAsArray())
            
        elif self.mask is not None:
            array = RasterMask(self.mask.ReadAsArray())

        return array
    
    def getData(self) -> Optional[RasterData]:
        if self.data is not None:
            return RasterData(self.data.ReadAsArray())
    
    def getMask(self) -> Optional[MaskedData]:
        if self.mask is not None:
            return RasterMask(self.mask.ReadAsArray())

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} (\n{self.getArray().__repr__()}, \n resolution: {self.getResolution()},\n bounds: {self.getBounds()},\n shape: {self.getShape()})>"

    def __str__(self) -> str:
        return f"<{self.__class__.__name__} (\n array: {self.getArray()}, \n resolution: {self.getResolution()},\n bounds: {self.getBounds()},\n shape: {self.getShape()})>"
    