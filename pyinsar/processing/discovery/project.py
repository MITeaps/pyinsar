# The MIT License (MIT)
# Copyright (c) 2018 Massachusetts Institute of Technology
#
# Authors: Cody Rude
# This software is part of the NSF DIBBS Project "An Infrastructure for
# Computer Aided Discovery in Geoscience" (PI: V. Pankratius) and
# NASA AIST Project "Computer-Aided Discovery of Earth Surface
# Deformation Phenomena" (PI: V. Pankratius)
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

# Pyinsar imports
from pyinsar.processing.utilities.generic import project_insar_data, AffineGlobalCoords

# Scikit data access imports
from skdaccess.utilities.image_util import AffineGlobalCoords

# Scikit discovery imports
from skdiscovery.data_structure.framework.base import PipelineItem

# 3rd party imports
from osgeo import gdal, osr, gdal_array



class Project(PipelineItem):
    """
    *** In Development *** Pipeline item to project and image
    """

    def __init__(self, str_description, target_projection='tm', center_coords = 'all'):
        """
        Initialize TransformImage item

        @param str_description: String describing item
        @param target_projection: Target projection (currently unused)
        @param center_coords: What to use for the central coordinates for the projection
                              'all': Use each images center coordinates for it's central projection coordinates
                              'first': Use the center of the first image

        """

        self._target_projecton = target_projection
        self.center_coords = center_coords

        super(Project, self).__init__(str_description)


    def _get_center_coords(self, wkt, geotransform, data_shape):

        wgs84 = osr.SpatialReference()
        wgs84.ImportFromEPSG(4326)

        spatial = osr.SpatialReference()
        spatial.ImportFromWkt(wkt)

        affine_transform = AffineGlobalCoords(geotransform)

        if len(data_shape) == 3:
            y_size = data_shape[1]
            x_size = data_shape[2]

        else:
            y_size = data_shape[0]
            x_size = data_shape[1]
        

        transform = osr.CreateCoordinateTransformation(spatial, wgs84)
        proj_y, proj_x = affine_transform.getProjectedYX(y_size/2, x_size/2)
        center_lon, center_lat = transform.TransformPoint(proj_x, proj_y)[:2]

        return center_lon, center_lat
        

    def process(self, obj_data):
        """
        Project data in an image wrapper

        @param obj_data: Image wrapper
        """

        for index, (label, data) in enumerate(obj_data.getIterator()):

            wkt = obj_data.info(label)['WKT']
            geotransform = obj_data.info(label)['GeoTransform']

            if (self.center_coords.lower() == 'first' and index == 0) or \
               self.center_coords.lower() == 'all':
                center_lon, center_lat = self._get_center_coords(wkt, geotransform, data.shape)

            ds = gdal_array.OpenNumPyArray(data)

            ds.SetGeoTransform(geotransform)
            ds.SetProjection(obj_data.info(label)['WKT'])            

            reprojected_ds = project_insar_data(ds, center_lon, center_lat)

            obj_data.updateData(label, reprojected_ds.ReadAsArray())
            obj_data.info(label)['WKT'] = reprojected_ds.GetProjection()
            obj_data.info(label)['GeoTransform'] = reprojected_ds.GetGeoTransform()

        
