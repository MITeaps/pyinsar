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

# PyInSAR imports
import pyinsar.processing.utilities.generic as generic_tools
from pyinsar.processing.geography.coordinates import reproject_georaster
from pyinsar.processing.discovery.srtm_egm96_wgs84 import SRTM_Transform

# Scikit discovery imports
from skdiscovery.data_structure.framework.base import PipelineItem
from skdiscovery.data_structure.framework.discoverypipeline import DiscoveryPipeline
from skdiscovery.data_structure.framework.stagecontainers import *
from skdiscovery.data_structure.generic.accumulators import DataAccumulator


# Scikit data access imports
from skdaccess.utilities import srtm_util
from skdaccess.geo.srtm.cache import DataFetcher as SRTMDF


# 3rd part imports
from osgeo import osr, gdal_array, gdal
import numpy as np

class SRTM_Fusion(PipelineItem):
    """
    Find appropriate elevation data from SRTM

    Puts the elevation data as another layer in the image.
    Must have WKT and GeoTransform information available in metadata.
    """

    def __init__(self, str_description, username, password, convert_to_wgs84=False, **kwargs):
        """
        Initialize SRTM Fusion object

        @param username: Earth data username
        @param password: Earth data password
        @param convert_to_wgs84: Convert heights from EGM96 geoid to WGS84 ellipsoid
        @param kwargs:  additional keyword arguments are given to the SRTM data fetcher
        """

        self.convert_to_wgs84 = convert_to_wgs84
        self.kwargs = kwargs
        self.kwargs['username'] = username
        self.kwargs['password'] = password

        super(SRTM_Fusion, self).__init__(str_description)

    def process(self, obj_data):
        """
        Add SRTM layer to image data

        @param obj_data: Image Data Wrapper
        """
        for label, data in obj_data.getIterator():


            geotransform = obj_data.info(label)['GeoTransform']
            wkt = obj_data.info(label)['WKT']

            lon_lat_extents =  generic_tools.get_lonlat_bounds(data.shape, wkt, geotransform)

            if data.ndim == 3:
                shape = data.shape[1:]
            else:
                shape = data.shape

            extents = generic_tools.get_image_extents(geotransform, shape)

            min_lat = lon_lat_extents[2]
            max_lat = lon_lat_extents[3]
            min_lon = lon_lat_extents[0]
            max_lon = lon_lat_extents[1]

            srtm_lat_lon = srtm_util.getSRTMLatLon(min_lat,
                                                   max_lat,
                                                   min_lon,
                                                   max_lon)

            srtmdf = SRTMDF(*srtm_lat_lon, **self.kwargs)

            if self.convert_to_wgs84:
                fl_transform = SRTM_Transform('SRTM_Transform')
                sc_transform = StageContainer(fl_transform)

                acc_data = DataAccumulator("Data", save_wrapper=True)
                sc_data = StageContainer(acc_data)

                pipe = DiscoveryPipeline(srtmdf, [sc_transform, sc_data])
                pipe.run()

                my_dw = pipe.getResults(0)['Data']

            else:
                my_dw = srtmdf.output()


            srtm_info = list(my_dw.info().values())[0]

            srtm_data, srtm_extents, srtm_geotransform = srtm_util.getSRTMData(my_dw, min_lat, max_lat, min_lon, max_lon)

            gdal_srtm_ds = generic_tools.get_gdal_dataset(srtm_data, srtm_info['WKT'], srtm_geotransform)

            gdal_dtype = generic_tools.get_gdal_dtype(data.dtype)

            transformed_ds = reproject_georaster(gdal_srtm_ds, (geotransform[1], np.abs(geotransform[5])), new_projection_wkt=wkt,
                                                 no_data_value = np.nan, new_extent = extents, data_type=gdal_dtype)

            transformed_data = transformed_ds.ReadAsArray()


            if data.ndim == 3:
                new_data = np.concatenate((data, transformed_data.reshape(1, *transformed_data.shape)))

            else:
                new_data = np.stack((data, transformed_data))

            obj_data.updateData(label, new_data)
