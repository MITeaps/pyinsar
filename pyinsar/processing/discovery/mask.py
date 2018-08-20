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

# Standard library imports
import math

# Pyinsar imports
from pyinsar.output.export_georaster import create_georaster_from_array
from pyinsar.processing.geography.coordinates import reproject_georaster
from pyinsar.processing.utilities.generic import get_image_extents

# skdiscovery imports
from skdiscovery.data_structure.framework.base import PipelineItem

# 3rd party imports
from osgeo import gdal

class Mask(PipelineItem):
    """
    Pipeline item used for masking images
    """

    def __init__(self, str_description, mask, mask_value=math.nan, geotransform=None, wkt=None):
        """
        Initialize Mask item

        If geotransform and wkt are provided, then mask 
        will be transformed before being applied

        @param str_description: String describing item
        @param mask: Array of zeros and ones with the same shape as the input images (1 for mask, 0 for no mask)
        @param mask_value: Value to set the masked values to
        @param geotransform: Geotransform of mask
        @param wkt: String of the well known text describing projection
        """

        if geotransform is None and wkt is not None or \
           geotransform is not None and wkt is None:
            raise RuntimeError('Must supply both geotransform and wkt or neither of them')
        elif geotransform is not None and wkt is not None:
            self._apply_transform = True
        else:
            self._apply_transform = False
        
        self.mask = mask
        self.mask_value = mask_value
        self._geotransform = geotransform
        self._wkt = wkt


        super(Mask, self).__init__(str_description)


    def process(self, obj_data):
        """
        Mask images

        @param obj_data: Image data wrapper
         """
        for label, data in obj_data.getIterator():

            if self._apply_transform:
                new_ds = create_georaster_from_array(georaster_array = self.mask,
                                                     geotransform = self._geotransform,
                                                     projection = self._wkt,
                                                     data_type = gdal.GDT_Int16,
                                                     no_data_value=-1)


                new_geotransform = obj_data.info(label)['GeoTransform']
                new_wkt = obj_data.info(label)['WKT']
                data_extent = get_image_extents(new_geotransform, data.shape)


                transformed_ds = reproject_georaster(new_ds, (new_geotransform[1], abs(new_geotransform[5])), new_projection_wkt=new_wkt,
                                                     no_data_value = -1, new_extent = data_extent, interpolation_method=gdal.GRA_NearestNeighbour,
                                                     data_type=gdal.GDT_Int16)

                mask = transformed_ds.ReadAsArray().astype(bool)

            else:
                mask = self.mask

                
                
            data[mask] = self.mask_value
                
