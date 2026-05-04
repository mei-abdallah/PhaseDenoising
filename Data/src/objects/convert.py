from typing import Tuple, Optional, Union, Literal
from geopy.distance import geodesic


class Converter():
    def __init__(self, lon:Optional[float]=None,       lat:Optional[float]=None, 
                 lonresdeg:Optional[float]=None, latresdeg:Optional[float]=None, 
                 lonresmtr:Optional[float]=None, latresmtr:Optional[float]=None):
        
        self.lon = lon # west
        self.lat = lat # north

        self.lonresdeg = lonresdeg # longitude resolution step in degrees (+ve)
        self.latresdeg = latresdeg # latitude resolution step in degrees (-ve)

        self.lonresmtr = lonresmtr # x-direction resolution step in meter
        self.latresmtr = latresmtr # x-direction resolution step in meter

    def setGridParams(self, lonlat:Tuple[float, float], lonresdeglatresdeg:Tuple[float, float]) -> 'Converter':
        """ set the grid conversion prameters"""
        self.lon, self.lat = lonlat
        self.lonresdeg, self.latresdeg = lonresdeglatresdeg
        self.lonresmtr = geodesic((lonlat[1], lonlat[0]), (lonlat[1], lonlat[0] + lonresdeglatresdeg[0])).meters # Point((lat, lon))
        self.latresmtr = geodesic((lonlat[1], lonlat[0]), (lonlat[1] + lonresdeglatresdeg[1], lonlat[0])).meters # Point((lat, lon))

        return self
    
    def getXpixYpix(self, x:float, y:float, dtype:Literal['deg', 'mtr']='deg') -> Tuple[int, int]:
        """ Calculate the location of the point in pixels (int) according to its x and y (deg or mtr)"""
        
        if dtype == 'deg':
            londiff = x - self.lon
            latdiff = y - self.lat
            xpix = int(londiff / self.lonresdeg)
            ypix = int(latdiff / self.latresdeg)

        elif dtype == 'mtr':
            xpix = int(x / self.lonresmtr)
            ypix = int(y / self.latresmtr)
        
        else:
            raise ValueError(f"dtype should be 'deg' or 'mtr'. Got {dtype}")

        return (xpix, ypix)

    def getXmtrYmtr(self, x:Union[int, float], y:Union[int, float], dtype:Literal['deg', 'pix']='deg') -> Tuple[float, float]:
        """ Calculate the location of the point in meters (float) according to its lon and lat """

        if dtype == 'deg':
            londiff = x - self.lon
            latdiff = y - self.lat

            xmtr = (londiff * self.lonresmtr) / self.lonresdeg
            ymtr = (latdiff * self.latresmtr) / self.latresdeg

        elif dtype == 'pix':
            xmtr = x * self.latresmtr
            ymtr = y * self.lonresmtr

        else:
            raise ValueError(f"dtype should be 'deg' or 'pix'. Got {dtype}")

        return xmtr, ymtr


    def getXdegYdeg(self, x:Union[int, float], y:Union[int, float], dtype:Literal['pix', 'mtr']='pix') -> Tuple[float, float]:
        """ Calculate the lon and lat of the point in degrees (float) according to its x and y (pixs or mtr)"""
        
        if dtype == 'pix':
            lon = self.lon + x * self.lonresdeg
            lat = self.lat + y * self.latresdeg

        elif dtype == 'mtr':
            lon = self.lon + ((x * self.lonresdeg) / self.lonresmtr)
            lat = self.lat + ((y * self.latresdeg) / self.latresmtr)
        
        else:
            raise ValueError(f"dtype should be 'pix' or 'mtr'. Got {dtype}")
        
        return lon, lat