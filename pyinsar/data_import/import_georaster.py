# The MIT License (MIT)
# Copyright (c) 2018 Massachusetts Institute of Technology
#
# Author: Guillaume Rongier
# This software has been created in projects supported by the US National
# Science Foundation and NASA (PI: Pankratius)
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

import numpy as np

from osgeo import gdal
from gdalconst import GA_ReadOnly, GA_Update

def open_georaster(georaster_path, read_only = True):
    '''
    Open a georaster with GDAL
    
    @param georaster_path: Location of the georaster
    @param read_only: Determine if the georaster can be modified
    
    @return The georaster as a GDAL data set
    '''
    read_status = GA_ReadOnly
    if read_only == False:
        read_status = GA_Update
    gdal_georaster = gdal.Open(georaster_path, read_status)
    if gdal_georaster is None:
        print('Could not open file')
        
    return gdal_georaster

def get_georaster_array(gdal_georaster, remove_ndv = True, as_float = True):
    '''
    Get a NumPy array from a georaster opened with GDAL
    
    @param gdal_georaster: A georaster opened with GDAL
    @param remove_ndv: Replace the no-data value as mentionned in the label by np.nan
    @param as_float: Transform the array to a float array
    
    @return The array
    '''
    assert gdal_georaster is not None, 'No georaster available'
    
    number_of_bands = gdal_georaster.RasterCount
    georaster_array = gdal_georaster.ReadAsArray()
    if as_float == True:
        georaster_array = georaster_array.astype(np.float)
    for i_band in range(number_of_bands):
        georaster_band = gdal_georaster.GetRasterBand(i_band + 1)
        no_data_value = georaster_band.GetNoDataValue()
        if no_data_value is not None and remove_ndv == True:
            if number_of_bands > 1:
                georaster_array[i_band, :, :][georaster_array[i_band, :, :] == no_data_value] = np.nan
            else:
                georaster_array[georaster_array == no_data_value] = np.nan
        scale = georaster_band.GetScale()
        if scale is None:
            scale = 1.
        offset = georaster_band.GetOffset()
        if offset is None:
            offset = 0.
        if number_of_bands > 1:
            georaster_array[i_band, :, :] = georaster_array[i_band, :, :]*scale + offset
        else:
            georaster_array = georaster_array*scale + offset
            
    return georaster_array

def get_georaster_extent(gdal_georaster):
    '''
    Get the extent of a georaster opened with GDAL
    
    @param gdal_georaster: A georaster opened with GDAL
    
    @return The georaster extent
    '''
    assert gdal_georaster is not None, 'No georaster available'
    
    georaster_x_size = gdal_georaster.RasterXSize
    georaster_y_size = gdal_georaster.RasterYSize
    geotransform = gdal_georaster.GetGeoTransform()
    xmin = geotransform[0]
    ymax = geotransform[3]
    xmax = xmin + geotransform[1]*georaster_x_size + geotransform[2]*georaster_y_size
    ymin = ymax + geotransform[4]*georaster_x_size + geotransform[5]*georaster_y_size
    
    return (xmin, xmax, ymin, ymax)

def print_georaster_info(gdal_georaster):
    '''
    Print some information about the GDAL georaster 
    
    @param gdal_georaster: A georaster opened with GDAL
    '''
    assert gdal_georaster is not None, 'No georaster available'

    print('Driver: ', gdal_georaster.GetDriver().ShortName, '/', gdal_georaster.GetDriver().LongName)
    print('Size of the cube is ', gdal_georaster.RasterXSize, 'x', gdal_georaster.RasterYSize, 'x', gdal_georaster.RasterCount)
    print('Projection is ', gdal_georaster.GetProjection())
    geotransform = gdal_georaster.GetGeoTransform()
    if not geotransform is None:
        print('Origin = (', geotransform[0], ',', geotransform[3], ')')
        print('Pixel Size = (', geotransform[1], ',', geotransform[5], ')')