from typing import Union, Literal, Sequence, Tuple
from osgeo import gdal, ogr
from shapely.wkt import loads
from shapely.geometry import Polygon
import matplotlib.path as mpath
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import os
import numpy as np

gdal.UseExceptions()
gdal.PushErrorHandler('CPLQuietErrorHandler')

class GeoFile():
    def __init__(self, name:str) -> None:
        self.name = name
        # print(os.path.exists(self.name))


    def open(self, lyrIndx:Union[str, int], ftrIndx:int) -> Polygon:
        """Open a existing shapefile and pass the coordinates back."""
        # opening the file
        file_bbox = ogr.Open(self.name)

        #If layer name provided
        if isinstance(lyrIndx, str):
            file_bbox = file_bbox.GetLayerByName(lyrIndx).GetFeature(ftrIndx)

        #If layer index provided
        else:
            file_bbox = file_bbox.GetLayerByIndex(lyrIndx).GetFeature(ftrIndx)

        return loads(file_bbox.GetGeometryRef().ExportToWkt())
    

    def save(self, polygon:Union[Sequence[Tuple[float, float]],Polygon], drivername:Literal['GeoJSON']='GeoJSON') -> None:
        if not isinstance(polygon, Polygon):
            polygon = Polygon(polygon)

        """Save a polygon shapefile."""
        # open file # create layer
        ds = ogr.GetDriverByName(drivername).CreateDataSource(self.name)
        lyr = ds.CreateLayer('', None, ogr.wkbPolygon)

        lyr.CreateField(ogr.FieldDefn('id', ogr.OFTInteger)) #Add 1 attribute
        # Create a new feature (attribute and geometry)
        feat = ogr.Feature(lyr.GetLayerDefn())
        feat.SetField('id', 0)

        # Make a geometry, from input Shapely object
        geom = ogr.CreateGeometryFromWkb(polygon.wkb)
        feat.SetGeometry(geom)
        lyr.CreateFeature(feat)

        ds = None


    def plot(self) -> None:
        # Extract first layer of features from shapefile using OGR
        ds = ogr.Open(self.name, gdal.GA_ReadOnly)
        lyr = ds.GetLayer(0)

        ds = None
        # Get extent and calculate buffer size
        ext = lyr.GetExtent()
        xoff = (ext[1]-ext[0])/50
        yoff = (ext[3]-ext[2])/50

        paths = []
        lyr.ResetReading()

        # Read all features in layer and store as paths
        for i, feat in enumerate(lyr):
            geom = feat.geometry()
            geom_name = geom.GetGeometryName()
            codes = []
            all_x = []
            all_y = []
            for i in range(geom.GetGeometryCount()):
                # Read ring geometry and create path
                r = geom.GetGeometryRef(i)
                if geom_name == 'MULTIPOLYGON':
                    r = geom.GetGeometryRef(i)
                    for j in range(r.GetGeometryCount()):
                        p = r.GetGeometryRef(j)
                        r = p
                x = [r.GetX(j) for j in range(r.GetPointCount())]
                y = [r.GetY(j) for j in range(r.GetPointCount())]

                # skip boundary between individual rings
                codes += [mpath.Path.MOVETO] + (len(x)-1)*[mpath.Path.LINETO]
                all_x += x
                all_y += y

            path = mpath.Path(np.column_stack((all_x,all_y)), codes)
            paths.append(path)

        with plt.style.context(('seaborn')):
            fig = plt.figure(figsize=(12, 9))
            ax = fig.add_subplot(111)
            ax.set_xlim(ext[0]-xoff,ext[1]+xoff)
            ax.set_ylim(ext[2]-yoff,ext[3]+yoff)

            # Add paths as patches to axes
            for path in paths:
                patch = mpatches.PathPatch(path, fill=False, facecolor='blue', \
                    edgecolor='black', linewidth=1)

                ax.add_patch(patch)

            ax.set_xlabel('longitude', labelpad=15, fontsize=15)
            ax.set_ylabel('latitude', labelpad=15, fontsize=15)
            ax.set_title(os.path.basename(os.path.splitext(self.name)[0]), \
                fontsize=15)
            ax.set_aspect(1.0)
            ax.grid(False)
        plt.show()