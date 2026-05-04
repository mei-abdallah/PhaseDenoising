from typing import Tuple, Optional, Mapping, Union, Sequence, overload
from osgeo import gdal
from geopy import Point
from geopy.distance import geodesic
from ..objects import GeoFile

class GeoLocation:
    def __init__(self, shape:Optional[Tuple[int, int]], vrtfiledir:str) -> None:
        self.shape = shape
        self.vrtfiledir = vrtfiledir
        self.location = None

    def process(self, location:Mapping[str, Union[str, float, Tuple[float, float]]], files:Sequence[str], dtype:gdal.ExtendedDataType) -> gdal.Dataset:
        self.setLocation(**location)

        self.setSize()

        gdal.BuildVRT(self.vrtfiledir, files)

        dataset = gdal.Translate('', self.vrtfiledir, 
                            options=gdal.TranslateOptions(format='MEM', 
                                                    outputBounds=self.getWindow(), 
                                                    outputType=dtype,
                                                    projWin=self.getWindow(),
                                                    height=self.shape[0],
                                                    width=self.shape[1]))
        return dataset

    def getWindow(self) -> Tuple[float, float, float, float]:
        return self.location[0], self.location[3], self.location[2], self.location[1]
    
    def getResolution(self, pixelsize:float) -> Mapping[str, float]:
        """Get the resolution of the data in x and y directions"""
        west, south, east, north = self.location
        return {'xRes' : 2 * pixelsize / (abs(geodesic((south, west), (south, west + 1)).meters) + 
                                          abs(geodesic((north, west), (north,  west + 1)).meters)), 
                'yRes' : 2 * pixelsize / (abs(geodesic((south, west), (south + 1, west)).meters) + 
                                          abs(geodesic((south, east), (south + 1, east)).meters))}

    @overload
    def setLocation(self, geofile:str) -> None: 
        """Set the location of the given dem using geofile
        e.g., SierraNegra.json
        """
        ...

    @overload
    def setLocation(self, name:str) -> None: 
        """Set the location of the given dem using name 
            e.g., N00E099, S05W095    
        """
        ...

    @overload
    def setLocation(self, bounds:Tuple[float, float, float, float]) -> None: 
        """Set the location of the given dem using bounds in formate of (ulx, uly, lrx, lry)
            e.g., (5, 12, 7, 10)   
        """
        ...
    @overload
    def setLocation(self, center:Tuple[float, float], length:Tuple[float, float]) -> None: 
        """Set the location of the given dem using the center (lon, lat) and side length (x, y)
            e.g., (50, 85) and (20.4, 20.4)
        """
        ...

    @overload
    def setLocation(self, west:float, south:float, east:float, north:float) -> None: 
        """Set the location of the given dem using the w/s/e/n
        e.g., 5, 10, 7, 12
        """
        ...

    def setLocation(self, *args, **kwargs) -> None:
        """ Set the location of the DEM.  
        If a geofile is given, will crop the DEM to the give coordinate of the polyggon.  
        If a center and length are given, will crop the DEM to the shape of the rectangle.  
        If a news are given, will crop the DEM to the shape of the rectangle. """

        if not len(args) and not len(kwargs):
            raise Exception("No arguments provided")

        if len(args):
            if len(args) == 1:
                if isinstance(args[0], str):
                    if args[0].endswith('.json'):
                        west, south, east, north = GeoFile(args[0]).open(0, 0).bounds
                    else:
                        south = float(args[0][1:3])
                        west = float(args[0][4:])
                        if args[0][0].upper() == 'S':
                            south = -south
                        if args[0][3].upper() == 'W':
                            west = -west
                        east = west + 1
                        north = south + 1
                
                elif isinstance(args[0], (tuple, list)):
                    west, south, east, north = args

            elif len(args) == 2:
                origin = Point(args[0][1], args[0][0])                            # geopy uses lat then lon notation, here for the centre of the DEM
                west = geodesic(meters=(args[1][0])/2).destination(origin, 270)[1] 
                east = geodesic(meters=(args[1][0])/2).destination(origin, 90)[1]  
                south = geodesic(meters=(args[1][1])/2).destination(origin, 180)[0]
                north = geodesic(meters=(args[1][1])/2).destination(origin, 000)[0]

            elif len(args) == 4:
                west, south, east, north = args
        
        if len(kwargs):
            if 'geofile' in kwargs.keys():
                    west, south, east, north = GeoFile(kwargs['geofile']).open(0, 0).bounds 

            elif 'name' in kwargs.keys():
                
                south = float(kwargs['name'][1:3])
                west = float(kwargs['name'][4:])
                if kwargs['name'][0].upper() == 'S':
                    south = -south
                if kwargs['name'][3].upper() == 'W':
                    west = -west
                east = west + 1
                north = south + 1

            elif 'bounds' in kwargs.keys():
                west, south, east, north = kwargs['bounds']

            elif ('centre' in kwargs.keys()) and ('length' in kwargs.keys()):
                origin = Point(kwargs['centre'][1], kwargs['centre'][0])                            # geopy uses lat then lon notation, here for the centre of the DEM
                west = geodesic(meters=(kwargs['length'][0])/2).destination(origin, 270)[1] 
                east = geodesic(meters=(kwargs['length'][0])/2).destination(origin, 90)[1]  
                south = geodesic(meters=(kwargs['length'][1])/2).destination(origin, 180)[0]
                north = geodesic(meters=(kwargs['length'][1])/2).destination(origin, 000)[0]

            elif ('west' in kwargs.keys()) and ('east' in kwargs.keys()) and ('south' in kwargs.keys()) and ('north' in kwargs.keys()):                 # or in terms of bounds (edges)
                west = kwargs['west']                                                                             # get edges of DEM
                east = kwargs['east']
                south = kwargs['south']
                north = kwargs['north']

        self.location = west, south, east, north

    def setSize(self, pixs2deg:int) -> None:
        """ Set the size of the output array. This will be used when reading multiple tiles into one array."""

        if self.shape is None:
            self.shape = (int((self.location[3] - self.location[1]) * pixs2deg), 
                          int((self.location[2] - self.location[0]) * pixs2deg))