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

from osgeo import gdal, osr

################################################################################
# Array coordinates
################################################################################

def transform_to_pixel_coordinates(x, y,
                                   x_min, x_max,
                                   y_min, y_max,
                                   array_width, array_height):
    '''
    Transform some geographic coordinates to pixel coordinates in an array
    
    @param x: Coordinate along the x axis to tranform
    @param y: Coordinate along the y axis to tranform
    @param x_min: Minimal coordinate of the array along the x axis (along the cell border)
    @param x_max: Maximal coordinate of the array along the x axis (along the cell border)
    @param y_min: Minimal coordinate of the array along the y axis (along the cell border)
    @param y_max: Maximal coordinate of the array along the y axis (along the cell border)
    @param array_width: Width of the array (i.e., along the x axis)
    @param array_height: Height of the array (i.e., along the y axis)
    
    @return The pixel coordinates
    '''
    u = (x - x_min)*(array_width/(x_max - x_min))
    v = (y_max - y)*(array_height/(y_max - y_min))
    
    return np.int_(u), np.int_(v)

def transform_to_geographic_coordinates(u, v,
                                        x_min, x_max,
                                        y_min, y_max,
                                        array_width, array_height):
    '''
    Transform some pixel coordinates in an array to geographic coordinates
    
    @param u: Pixel coordinate along the x axis to tranform
    @param v: Pixel coordinate along the y axis to tranform
    @param x_min: Minimal coordinate of the array along the x axis (along the cell border)
    @param x_max: Maximal coordinate of the array along the x axis (along the cell border)
    @param y_min: Minimal coordinate of the array along the y axis (along the cell border)
    @param y_max: Maximal coordinate of the array along the y axis (along the cell border)
    @param array_width: Width of the array (i.e., along the x axis)
    @param array_height: Height of the array (i.e., along the y axis)
    
    @return The geographic coordinates at the center of the pixel
    '''
    pixel_x_size = (x_max - x_min)/array_width
    pixel_y_size = (y_max - y_min)/array_height
    x = x_min + u*pixel_x_size + 0.5*pixel_x_size
    y = y_max - v*pixel_y_size - 0.5*pixel_y_size
    
    return x, y

def compute_x_and_y_coordinates_maps(x_min, x_max, 
                                     y_min, y_max, 
                                     array_width, array_height):
    '''
    Compute an array of x and y coordinates based on an extent and array shape
    
    @param x_min: Minimal coordinate along the x axis (along the cell border)
    @param x_max: Maximal coordinate along the x axis (along the cell border)
    @param y_min: Minimal coordinate along the y axis (along the cell border)
    @param y_max: Maximal coordinate along the y axis (along the cell border)
    @param array_width: Width of the array (i.e., along the x axis)
    @param array_height: Height of the array (i.e., along the y axis)
    
    @return The coordinates' arrays
    '''
    pixel_x_size = (x_max - x_min)/array_width
    pixel_y_size = (y_max - y_min)/array_height
    x_array = np.linspace(x_min + 0.5*pixel_x_size, 
                          x_max - 0.5*pixel_x_size, 
                          array_width)
    y_array = np.linspace(y_max - 0.5*pixel_y_size,
                          y_min + 0.5*pixel_y_size,
                          array_height)
    
    return np.meshgrid(x_array, y_array)

################################################################################
# Projection
################################################################################

def reproject_point(lon, lat, old_projection_wkt, new_projection_wkt):
    '''
    Reproject a single point
    
    @param lon: Longitude of the point
    @param lat: Latitude of the point
    @param old_projection_wkt: WKT code of the current projection
    @param new_projection_wkt: WKT code of the new projection
    
    @return The coordinates' arrays
    '''
    old_spatial_reference = osr.SpatialReference()
    old_spatial_reference.ImportFromWkt(old_projection_wkt)
    
    new_spatial_reference = osr.SpatialReference()
    new_spatial_reference.ImportFromWkt(new_projection_wkt)

    transform = osr.CoordinateTransformation(old_spatial_reference,
                                             new_spatial_reference)
    
    return transform.TransformPoints([(lon, lat)])[0]

def reproject_georaster(georaster,
                        new_cell_sizes,
                        new_projection_EPSG = None,
                        new_projection_wkt = None,
                        interpolation_method = gdal.GRA_Cubic,
                        file_type = 'MEM',
                        file_path = '',
                        data_type = gdal.GDT_Float64,
                        no_data_value = -99999.,
                        scale = 1.,
                        offset = 0.):
    '''
    Change the projection of a GDAL georaster
    
    @param georaster: The GDAL georaster
    @param new_cell_sizes: Sizes (x, y) for cells of the georaster in the new projection
    @param new_projection_EPSG: EPSG code of the new projection
    @param new_projection_wkt: WKT code of the new projection (can be used instead
                               of the new_projection_EPSG)
    @param interpolation_method: Interpolation method used during the projection
    @param file_type: Type to save the file (default is memory)
    @param file_path: Where to store the new georasterEPSG_code (default is memory)
    @param data_type: Data type of the georaster
    @param no_data_value: No data value for the georaster
    @param scale: Scaling factor for the georaster
    @param offset: Offset factor for the georaster
    
    @return The GDAL georaster
    '''
    old_spatial_reference = osr.SpatialReference()
    old_spatial_reference.ImportFromWkt(georaster.GetProjectionRef())
    old_geotransform = georaster.GetGeoTransform()
    old_georaster_x_size = georaster.RasterXSize
    old_georaster_y_size = georaster.RasterYSize

    new_spatial_reference = osr.SpatialReference()
    if new_projection_EPSG is not None:
        new_spatial_reference.ImportFromEPSG(new_projection_EPSG)
    elif new_projection_wkt is not None:
        new_spatial_reference.ImportFromWkt(new_projection_wkt)
    else:
        print('No new spatial reference provided, will use the one from the georaster')
        new_spatial_reference.ImportFromWkt(georaster.GetProjectionRef())

    transform = osr.CoordinateTransformation(old_spatial_reference,
                                             new_spatial_reference) 
    (ulx, uly, ulz) = transform.TransformPoint(old_geotransform[0],
                                               old_geotransform[3])
    (lrx, lry, lrz) = transform.TransformPoint(old_geotransform[0] + old_geotransform[1]*old_georaster_x_size,
                                               old_geotransform[3] + old_geotransform[5]*old_georaster_y_size)
    new_geotransform = (ulx,
                        new_cell_sizes[0],
                        old_geotransform[2],
                        uly,
                        old_geotransform[4],
                        -new_cell_sizes[1])

    driver = gdal.GetDriverByName(file_type)
    new_georaster = driver.Create(file_path,
                               int(abs(lrx - ulx)/new_cell_sizes[0]),
                               int(abs(uly - lry)/new_cell_sizes[1]),
                               1,
                               data_type)
    new_georaster.SetGeoTransform(new_geotransform)
    new_georaster.SetProjection(new_spatial_reference.ExportToWkt())
    new_georaster_band = new_georaster.GetRasterBand(1)
    new_georaster_band.SetNoDataValue(no_data_value)
    new_georaster_band.SetScale(scale)
    new_georaster_band.SetOffset(offset)
    # Fill the georaster band, otherwise no data values are not set in the new georaster
    new_georaster_band.Fill(no_data_value)

    res = gdal.ReprojectImage(georaster,
                              new_georaster,
                              old_spatial_reference.ExportToWkt(),
                              new_spatial_reference.ExportToWkt(),
                              interpolation_method)

    return new_georaster