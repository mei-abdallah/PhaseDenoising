import os
from matplotlib.path import Path
from osgeo import gdal, osr
from .gshhs import GSHHS
import numpy as np
from typing import Literal, overload


class WaterLines():
    """ 
        Load the GSHHS (global self-consistent heirarchical high-resolution shorelines), available from: http://www.soest.hawaii.edu/pwessel/gshhg/
        Return as list of numpy arrays for shorelines.   Note, as this only works with shorelines, these are the L1 products 
        in the GSHHS docs, so it won't include lakes and rivers.  
        
    Parameters:
        - workdir | str | the working directory
        - resoltuion | string | From GSHHS docs:
                                The geography data come in five resolutions:
                                    'f' -> full resolution: Original (full) data resolution.
                                    'h' -> high resolution: About 80 % reduction in size and quality.
                                    'i' -> intermediate resolution: Another ~80 % reduction.
                                    'l' -> low resolution: Another ~80 % reduction.
                                    'c' -> crude resolution: Another ~80 % reduction.
        - gshhsurl | url | default: http://www.soest.hawaii.edu/pwessel/gshhg/'
        - verbose | boolean | print processing steps if true
        
    """
    def __init__(self, workdir:str='./', resolution:Literal['c', 'l', 'i', 'h', 'f']='i', srtm:Literal['srtm1', 'srtm3']='srtm3',
                 gshhsurl='http://www.soest.hawaii.edu/pwessel/gshhg/', verbose:bool=False) -> None:
        self.gshhs = GSHHS(workdir, resolution, gshhsurl, verbose)
        self.pixs2deg = 1201 if srtm.lower() == 'srtm3' else 3601
        self.linesdir = f'{workdir}/lines'
        self.verbose = verbose

    def create(self, lon:int, lat:int) -> None:
        edgefraction = 0.1

        if self.verbose:
            print('\nCreating a mask of pixels that lie in water (sea and lakes)... | ', end = '')
        # 1: deal with sizes of various things and make a grid of points
        # length, width = self.shape                                                                     # number of pixels vetically and horizontally            
        # west, south, east, north = self.location
        
        lons, lats = np.meshgrid(np.linspace(lon, lon+1, self.pixs2deg), 
                                    np.linspace(lat, lat+1, self.pixs2deg))
        
        locations = np.hstack((np.ravel(lons)[:,np.newaxis], np.ravel(lats)[:,np.newaxis]))            # x and y (lon and lat) for all pixels in the dem, size (n_pixels x 2)
        
        
        bbox = {"west" : lon - edgefraction, "east" : lon + 1 + edgefraction,
                "south": lat - edgefraction, "north" : lat + 1 + edgefraction}

        
        l1_mask = np.zeros(len(locations), dtype=np.bool_)                                                     # initialise as false, will be True (1) where land
        l2_mask = np.zeros(len(locations), dtype=np.bool_)                                                     # initialise as false, will be True (1) where land
        l3_mask = np.zeros(len(locations), dtype=np.bool_)                                                     # initialise as false, will be True (1) where land
        l4_mask = np.zeros(len(locations), dtype=np.bool_)                                                     # initialise as false, will be True (1) where land
        
        for level in ['L1','L2','L3','L4']:                                                              # the GSHHS products
            print(f"Completed product {level} ", end='')
            coastlines = self.gshhs.loadCostline(bbox, level)[-1]                               # open the GSHHS product (list of numpy arrays of lon lats of water/land boundary)
            
            for indx, coastline in enumerate(coastlines):
                landmass = Path(coastline)  
                if level == 'L1':
                    l1_mask += np.array(landmass.contains_points(locations))                               # True if pixel is in land
                elif level == 'L2':
                    l2_mask += np.array(landmass.contains_points(locations))                               # True if pixel is in lake
                elif level == 'L3':
                    l3_mask += np.array(landmass.contains_points(locations))                               # True if pixel is in island
                elif level == 'L4':
                    l4_mask += np.array(landmass.contains_points(locations))                               # True if pixel is in pond (on island)
                
                if self.verbose:
                    print(f"coastline {indx}", end='')
            
            if self.verbose:
                print('Done!')                                                        
        
        # l12_mask = np.logical_and(l1_mask, np.invert(l2_mask))                                             # combine L1 with L2
        # l13_mask = np.logical_or(l12_mask, (l3_mask))                                                      # combine L12 with 3
        # l14_mask = np.logical_and(l13_mask, np.invert(l4_mask))                                            # combine L123 with 4
        # mask = np.invert(l14_mask)                                                                   # water is where not land, so invert (now True for land)
        
        mask = np.invert(np.logical_and(np.logical_or(np.logical_and(l1_mask, np.invert(l2_mask)), (l3_mask)), np.invert(l4_mask)))
        mask = np.reshape(mask, (self.pixs2deg, self.pixs2deg))                                            # reshape, not sure why flipud as geographic but numpy is referring to top left pixel
        mask = np.flipud(mask)
        self.rasterize(mask, lon, lat)
        
    def rasterize(self, array:np.ndarray, lon:int, lat:int) -> None:
        # xmin, ymin, xmax, ymax = location['west'], location['south'], location['east'], location['north']

        # # length, width = np.shape(array)
        # xres = (location['east'] - location['west'])/float(array.shape[1])
        # yres = (location['north'] - location['south'])/float(array.shape[0])
        geotransform=(lon, 
                      (1.0)/float(array.shape[1]), 
                      0, 
                      lat+1, 
                      0, 
                      -(1.0)/float(array.shape[0]))
          
        # That's (top left x, w-e pixel resolution, rotation (0 if North is up), 
        #         top left y, rotation (0 if North is up), n-s pixel resolution)
        # I don't know why rotation is in twice???

        raster = gdal.GetDriverByName('ENVI').Create(f'{self.linesdir}/{self.getName(lon, lat)}.wtr', array.shape[1], array.shape[0], 1, gdal.GDT_Byte)  # Open the file
        raster.SetGeoTransform(geotransform)  # Specify its coordinates
        srs = osr.SpatialReference()                 # Establish its coordinate encoding
        srs.ImportFromEPSG(4326)                     # This one specifies WGS84 lat long.
                                                    # Anyone know how to specify the 
                                                    # IAU2000:49900 Mars encoding?
        raster.SetProjection( srs.ExportToWkt() )   # Exports the coordinate system 
                                                        # to the file
        raster.GetRasterBand(1).WriteArray(array)   # Writes my array to the raster

        raster.FlushCache()
    
        raster = None

        # os.remove(f'{self.linesdir}/{self.getName(lon, lat)}.hdr')

    def getName(self, lon:int, lat:int) -> str:
        """ Given longitude and latitude in the form of - for west south, conver to 
            awlays positive format prefixed by NESW format used by USGS.  
        """
        name = ''
        if lat >= 0 and lon >= 0:                       # north east quadrant
            name = 'N' + str(lat).zfill(2) + 'E' + str(lon).zfill(3)                                    # zfill pads to the left with zeros so always 2 or 3 digits long. 
        if lat >= 0 and lon < 0:                        # north west quadant
            name = 'N' + str(lat).zfill(2) + 'W' + str(-lon).zfill(3)
        if lat < 0 and lon >= 0:                        # south east quadrantexistsname
            name = 'S' + str(-lat).zfill(2) + 'E' + str(lon).zfill(3)
        if lat < 0 and lon < 0:                         # south east quadrant
            name = 'S' + str(-lat).zfill(2) + 'W' + str(-lon).zfill(3)
        
        return name + self.gshhs.resolution
    

    @overload
    def getExists(self, filename:str) -> bool: 
        """Check the extance of a tile file usning tile name"""
        ...

    @overload
    def getExists(Self, lon:int, lat:int) -> bool: 
        """Check the extance of a tile file usning tile lon/lat"""
        ...

    def getExists(self, *args, **kwargs) -> bool:
        """Check the extance of a tile file"""
        if len(args) == 2:
            filename = self.getName(*args)

        elif len(args) == 1:
            filename = args[0]

        elif 'filename' in list(kwargs.keys()):
            filename = kwargs['filename']

        elif 'lon' in list(kwargs.keys()) and 'lat' in list(kwargs.keys()):
            filename = self.getName(**kwargs)
        return self.exists(filename)
    

    def exists(self, filename:str) -> bool:
        """Check the extance of a tile file"""
        if not os.path.exists(self.linesdir):
            os.makedirs(self.linesdir, exist_ok=True)
            
        existflag = os.path.exists(f'{self.linesdir}/{filename}.wtr')
        if existflag:
            dataset = gdal.Open(f'{self.linesdir}/{filename}.wtr', gdal.GA_ReadOnly)
            if dataset.ReadAsArray().shape != (self.pixs2deg, self.pixs2deg):
                existflag = False
            dataset = None
        return existflag
    
    @overload
    def getPath(self, filename:str) -> str: 
        """Get the path to a tile file usning tile name"""
        ...

    @overload
    def getPath(Self, lon:int, lat:int) -> str: 
        """Get the path to a tile file usning tile lon/lat"""
        ...

    def getPath(self, *args, **kwargs) -> str:
        """Get the path to a tile file"""
        if len(args) == 2:
            filename = self.getName(*args)

        elif len(args) == 1:
            filename = args[0]

        elif 'filename' in list(kwargs.keys()):
            filename = kwargs['filename']

        elif 'lon' in list(kwargs.keys()) and 'lat' in list(kwargs.keys()):
            filename = self.getName(**kwargs)
        return self.path(filename)

    def path(self, filename:str) -> str:
        """Get the path to a tile file"""
        return os.path.abspath(f'{self.linesdir}/{filename}.wtr')
          



    