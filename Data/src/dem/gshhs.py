import os
from shapefile import Reader
from shapely.geometry import Polygon, MultiPolygon
from matplotlib.path import Path
from osgeo import gdal, osr
import numpy as np
from bs4 import BeautifulSoup
from urllib.request import urlopen
import requests
import zipfile
from typing import Literal, Dict, Tuple, List


class GSHHS():
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
    def __init__(self, workdir:str='./', resolution:Literal['c', 'l', 'i', 'h', 'f']='i', 
                 gshhsurl='http://www.soest.hawaii.edu/pwessel/gshhg/', verbose:bool=False) -> None:
        self.gshhsdir = f'{workdir}/gshhs'
        self.resolution = resolution
        self.gshhsurl = gshhsurl
        self.verbose = verbose

        if not os.path.exists(self.gshhsdir):
            self.downloadGshhsGiles()
    
    def downloadGshhsGiles(self) -> None:
        """ Download the GSHHS Gile shoreline product """
        # Connect to the website and return the html to the variable ‘page’
        try:
            page = urlopen(self.gshhsurl)
        except:
            print(f"Error opening the gshhs webpage with {self.gshhsurl}")
        
        # parse the html using beautiful soup and store in variable `soup`
        soup = BeautifulSoup(page, 'html.parser')
        for link in soup.findAll('a'):
            if 'http' in link.get('href') and 'shp' in link.get('href'):
                url = link.get('href')
                if self.verbose:
                    print(f'gshhs files was found at {url}')
        
        if self.verbose:
            print('Downloading started... | ', end='')
        # Downloading the file by sending the request to the URL
        req = requests.get(url)

        # Split URL to get the file name
        zippedfile = url.split('/')[-1]


        # Writing the file to the local file system
        with open(zippedfile,'wb') as file:
            file.write(req.content)
        print('Done!')
        
        # unzip to get a gshhs, and then delete zip file
        if self.verbose:
            print(f"{zippedfile}: Unzipping... | ", end='')                                               # unzip the file
        with zipfile.ZipFile(zippedfile ,"r") as zipref:                                
            zipref.extractall(self.gshhsdir)
        os.remove(zippedfile)                                                                     # remove the redundant zip file
        
        if self.verbose:
            print("Done!")

    def loadCostline(self, bbox:Dict[str, float], level:Literal['L1', 'L2', 'L3', 'L4']='L1') -> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray]]:
        """ load the cost lines from the GSHHS files.
        
        Parameters:
            - bbox | dict |  contains west south east north in degress, and ensures only an area of interest is returned.  
            - level | int | From GSHHS docs:
                                L1: boundary between land and ocean, except Antarctica.
                                L2: boundary between lake and land.
                                L3: boundary between island-in-lake and lake.
                                L4: boundary between pond-in-island and island.
                                L5: boundary between Antarctica ice and ocean.
                                L6: boundary between Antarctica grounding-line and ocean.
        
                                
        Returns:
        allcostlinePolygons | [arrays] | all coastlines in the file, as a list of numpy arrays
        intersectcostlinePolygons | [arrays] | all coastlines that cross the bounding box, as list of arrays
        croppedcostlinePolygons | [arrays] |  all coastlines inside the bounding box (and use the edge of the bounding box for bits that go outside it), as list of arrays
        
        History:
            - 2022_01_26 | MEG | Written
        """
        shapefile = f'{self.gshhsdir}/GSHHS_shp/{self.resolution}/GSHHS_{self.resolution}_{level}.shp'
        shapefile = Reader(shapefile)
        landmasses = len(shapefile.shapes())


        
        allcostlinePolygons = []                                     # all coastlines in the file, as a list of numpy arrays
        intersectcostlinePolygons = []                           # all coastlines that cross the bounding box, as list of arrays
        croppedcostlinePolygons = []                             # all coastlines inside the bounding box (and use the edge of the bounding box for bits that go outside it), as list of arrays
        
        bboxPolygon = Polygon([(bbox['west'], bbox['south']),    # build the bounding box (bbox), if required.
                                   (bbox['east'], bbox['south']),
                                   (bbox['east'], bbox['north']),
                                   (bbox['west'], bbox['north'])])

        for landmass in range(landmasses):                                  # loop through each landmass (could be an island, could be all of eurasia)
            landmassPoints = shapefile.shape(landmass).points        # get the lon and lats of the points defining its coastline
            landmassPolygon = Polygon(landmassPoints)                # convert to shapely polygon
            allcostlinePolygons.append(np.array(landmassPoints))     # convert to numpy array and add to list
            if landmassPolygon.intersects(bboxPolygon):                                                 # if any part of the coastline goes into the bounding box 
                intersectcostlinePolygons.append(np.array(landmassPoints))                              # convert it to a numpy array and store

                intersectshapelyPolygon = landmassPolygon.intersection(bboxPolygon)                     # find which bits are in the bbox, this can be a Polygon or a MultiPolygon 
                                                                                                        # (if the coastline has now broken into two parts 
                                                                                                        # (e.g. a horshoe shaped island could become two islands if the bbox only capture the bottom part of it))
                
                if isinstance(intersectshapelyPolygon, Polygon):
                    croppedcostlinePolygons.append(self.toArray(intersectshapelyPolygon))       # conver the Polygon to a numpy array and store

                elif isinstance(intersectshapelyPolygon, MultiPolygon):
                    for polygon in intersectshapelyPolygon.geoms:
                        croppedcostlinePolygons.append(self.toArray(polygon))
                

        return allcostlinePolygons, intersectcostlinePolygons, croppedcostlinePolygons
    

    def toArray(self, polygon:Polygon) -> np.ndarray:
        """ Convert a shapely Polygon to a numpy array (lons in column 0, lats in column 1)
        """
        x, y = polygon.exterior.coords.xy                                                   # get the coords of that
        x = np.array(x)                                                                     # make into a numpy array, rank 1
        y = np.array(y)                                                                     # make into a numpy array, rank 1
        xy = np.concatenate((x[:, np.newaxis], y[:, np.newaxis]), axis = 1)

        return xy
            



    