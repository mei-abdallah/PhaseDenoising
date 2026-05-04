from typing import Literal, Optional, overload
import os, netrc, requests, zipfile
from osgeo import gdal
class Srtm:
    """ A class to handel the download of strm file from the online hub 
    
    Parameters:
        - type | str | either use SRTM3 (~90m pixels) or SRTM1 (~30m pixels)
        - workdir | str | the working dir
        - username | str | Earthdata username, to apply: https://earthdata.nasa.gov/eosdis/science-system-description/eosdis-components/earthdata-login
        - password | str | Earthdata password
        - verbose | boolean | print processing steps if true
    """
    def __init__(self, type:Literal['srtm1', 'srtm3']='srtm3', workdir:str='./', username:Optional[str]=None, password:Optional[str]=None, verbose:bool=False):
        if not type.lower() in ['srtm1', 'srtm3']:
            raise ValueError('Invalid SRTM type. Must be either srtm1 or strm3. Exiting...')
        
        self.type = type.lower()
        self.url = 'https://e4ftl01.cr.usgs.gov/MEASURES/SRTMGL3.003/2000.02.11/' if type.lower() == 'srtm3' else 'http://e4ftl01.cr.usgs.gov/MEASURES/SRTMGL1.003/2000.02.11/' # Where USGS keeps SRTM1 or SRTM3 tiles
        self.pixs2deg = 1201 if type.lower() == 'srtm3' else 3601
        self.srtmres = 3 if type.lower() == 'srtm3' else 1
        self.tilesdir = f'{workdir}/{type}'
        self.username = self.getUsername(username)
        self.password = self.getPassword(password)
        self.verbose = verbose

    def getUsername(self, username:Optional[str]=None) -> str:
        if not username:
            if os.path.exists(os.path.expanduser('~/.netrc')):
                username = netrc.netrc().authenticators('urs.earthdata.nasa.gov')[0]
            else:
                raise Exception ('No netric file found. Exiting...')
        return username

    def getPassword(self, password:Optional[str]=None) -> str:
        if not password:
            if os.path.exists(os.path.expanduser('~/.netrc')):
                password = netrc.netrc().authenticators('urs.earthdata.nasa.gov')[-1]
            else:
                raise Exception ('No netric file found. Exiting...')
        return password
    
    @overload
    def getDownload(self, tilename:str) -> bool: 
        """ Download SRTM tiles from Earthdata using tile name """
        ...

    @overload
    def getDownload(Self, lon:int, lat:int) -> bool: 
        """ Download SRTM tiles from Earthdata using tile lon/lat """
        ...

    def getDownload(self, *args, **kwargs) -> bool:
        """ Download SRTM tiles from Earthdata """
        if len(args) == 2:
            tilename = self.getName(*args)

        elif len(args) == 1:
            tilename = args[0]

        elif 'tilename' in list(kwargs.keys()):
            tilename = kwargs['tilename']

        elif 'lon' in list(kwargs.keys()) and 'lat' in list(kwargs.keys()):
            tilename = self.getName(**kwargs)
        return self.download(tilename)

        
    def download(self, tilename:str) -> bool:
        """ A function to download and unzip tile files. Given lons and lats (integers), make a multi tile DEM from either SRTM1 or 3 data.  
            An Earthdata account is needed to succesfully download tiles.
            To apply: https://earthdata.nasa.gov/eosdis/science-system-description/eosdis-components/earthdata-login 
        
        Parameters:
            - tilename | str | e.g. N51W004
        
        Returns:
            - file | hgt file | the downlaoded strm file
        
        History:
            2020/05/10 | MEG | Written
            2021/02/24 | MEG | Update to handle both SRTM1 and 3 tiles. 
        """
        ext = f'SRTMGL{self.srtmres}.hgt.zip'
        # if self.verbose:
        #     print(f"{tilename}: Downloading the zip file... | ", end = '')

        zippedfile = f"{self.tilesdir}/{tilename}.hgt.zip"                                                  # name to save downloaded file to
        
        with requests.Session() as session:
            session.auth = (self.username, self.password)                                               # login steps
            request = session.request('get', f"{self.url}{tilename}.{ext}")  
            request = session.get(request.url, auth=(self.username, self.password))                           # download file
            if request.ok:
                with open(zippedfile, 'wb') as file:                                            # write the .hgt.zip file
                    for chunk in request.iter_content(chunk_size=1024*1024):
                        file.write(chunk)
                successful = True
                if self.verbose:
                    print('Done!')
            else:
                successful = False
                if self.verbose:
                    print('Failed!')
                # raise Exception('Download failed.  ')

        if successful:            
            # 2: unzip to get a hgt, and then delete zip file
            if self.verbose:
                print(f"{tilename}: Unzipping... | ", end = '')                                               # unzip the file
            try:
                with zipfile.ZipFile(zippedfile ,"r") as zip_ref:                                
                    zip_ref.extractall(self.tilesdir)
                os.remove(zippedfile)                                                                     # remove the redundant zip file
                if self.verbose:
                    print("Done!")
            except:
                successful = False
                if self.verbose:
                    print("Failed!")
            
        return successful
    
    def getName(self, lon:int, lat:int) -> str:
        """ Given longitude and latitude in the form of - for west south, conver to 
            awlays positive format prefixed by NESW format used by USGS.  
        """
        tilename = ''
        if lat >= 0 and lon >= 0:                       # north east quadrant
            tilename = 'N' + str(lat).zfill(2) + 'E' + str(lon).zfill(3)                                    # zfill pads to the left with zeros so always 2 or 3 digits long. 
        if lat >= 0 and lon < 0:                        # north west quadant
            tilename = 'N' + str(lat).zfill(2) + 'W' + str(-lon).zfill(3)
        if lat < 0 and lon >= 0:                        # south east quadrant
            tilename = 'S' + str(-lat).zfill(2) + 'E' + str(lon).zfill(3)
        if lat < 0 and lon < 0:                         # south east quadrant
            tilename = 'S' + str(-lat).zfill(2) + 'W' + str(-lon).zfill(3)
        return tilename
    
    @overload
    def getExists(self, tilename:str) -> bool: 
        """Check the existence of a tile file usning tile name"""
        ...

    @overload
    def getExists(Self, lon:int, lat:int) -> bool: 
        """Check the existence of a tile file usning tile lon/lat"""
        ...

    def getExists(self, *args, **kwargs) -> bool:
        """Check the existence of a tile file"""
        if len(args) == 2:
            tilename = self.getName(*args)

        elif len(args) == 1:
            tilename = args[0]

        elif 'tilename' in list(kwargs.keys()):
            tilename = kwargs['tilename']

        elif 'lon' in list(kwargs.keys()) and 'lat' in list(kwargs.keys()):
            tilename = self.getName(**kwargs)
        return self.exists(tilename)
    

    def exists(self, tilename:str) -> bool:
        """Check the existence of a tile file"""
        if not os.path.exists(self.tilesdir):
            os.makedirs(self.tilesdir, exist_ok=True)

        existflag = os.path.exists(f'{self.tilesdir}/{tilename}.hgt')
        if existflag:
            dataset = gdal.Open(f'{self.tilesdir}/{tilename}.hgt', gdal.GA_ReadOnly)
            if dataset.ReadAsArray().shape != (self.pixs2deg, self.pixs2deg):
                existflag = False
            dataset = None
        return existflag
    
    @overload
    def getPath(self, tilename:str) -> str: 
        """Get the path to a tile file usning tile name"""
        ...

    @overload
    def getPath(Self, lon:int, lat:int) -> str: 
        """Get the path to a tile file usning tile lon/lat"""
        ...

    def getPath(self, *args, **kwargs) -> str:
        """Get the path to a tile file"""
        if len(args) == 2:
            tilename = self.getName(*args)

        elif len(args) == 1:
            tilename = args[0]

        elif 'tilename' in list(kwargs.keys()):
            tilename = kwargs['tilename']

        elif 'lon' in list(kwargs.keys()) and 'lat' in list(kwargs.keys()):
            tilename = self.getName(**kwargs)
        return self.path(tilename)

    def path(self, tilename:str) -> str:
        """Get the path to a tile file"""
        return os.path.abspath(f'{self.tilesdir}/{tilename}.hgt')
    

        
    
       