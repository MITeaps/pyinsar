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
from pyinsar.processing.utilities.generic import get_gdal_dtype
from pyinsar.processing.geography.coordinates import georaster_vertical_datum_shift

# Scikit data access imports
from skdaccess.utilities.image_util import AffineGlobalCoords
from skdaccess.generic.file.cache import DataFetcher as FILEDF
from skdaccess.framework.param_class import *

# Scikit discovery imports
from skdiscovery.data_structure.framework.base import PipelineItem

# 3rd party imports
from osgeo import gdal, osr, gdal_array



class SRTM_Transform(PipelineItem):
    """
    *** In Development *** Pipeline item to transfrom heights from
        SRTM from EGM96 geoid to WGS84 ellipsoid
    """

    def __init__(self, str_description):
        """
        Initialize Convert

        """
        super(SRTM_Transform, self).__init__(str_description)


    def process(self, obj_data):
        """
        Project data in an image wrapper

        @param obj_data: Image wrapper
        """

        egmdf = FILEDF([AutoList(['http://download.osgeo.org/proj/vdatum/egm96_15/egm96_15.gtx'])])
        egmdw = egmdf.output()

        egmurl, egmfilename = next(egmdw.getIterator())

        for index, (label, data) in enumerate(obj_data.getIterator()):

            wkt = obj_data.info(label)['WKT']
            geotransform = obj_data.info(label)['GeoTransform']

            ds = gdal_array.OpenNumPyArray(data)

            ds.SetGeoTransform(geotransform)
            ds.SetProjection(obj_data.info(label)['WKT'])

            gdal_dtype = get_gdal_dtype(data.dtype)

            reprojected_ds = georaster_vertical_datum_shift(
                georaster = ds,
                old_datum_proj4 = '+proj=longlat +datum=WGS84 +no_defs +geoidgrids=' + egmfilename,
                new_datum_proj4 = '+proj=longlat +datum=WGS84 +no_defs'
            )


            obj_data.updateData(label, reprojected_ds.ReadAsArray())
            # obj_data.info(label)['WKT'] = reprojected_ds.GetProjection()
            # obj_data.info(label)['GeoTransform'] = reprojected_ds.GetGeoTransform()
