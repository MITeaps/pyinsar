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
from numba import jit

################################################################################
# Array coordinates
################################################################################

def transform_to_pixel_coordinates(x, y,
                                   x_min, x_max,
                                   y_min, y_max,
                                   array_width, array_height):
    '''
    Transform some geographic coordinates to pixel coordinates in an array
    
    @param x: Coordinate along the x axis to transform
    @param y: Coordinate along the y axis to transform
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
    
    @param u: Pixel coordinate along the x axis to transform
    @param v: Pixel coordinate along the y axis to transform
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

def extract_subgeoarray(georaster_array,
                        georaster_extent,
                        x_min, x_max,
                        y_min, y_max,
                        center_extent = False):
    '''
    Extract a sub-array given some geographical coordinates. The new extent's
    coordinates don't have to be along a pixel border
    
    @param georaster_array: A 2D NumPy array
    @param georaster_extent: The array current extent (along the cells' borders)
    @param x_min: New minimal coordinate along the x axis (along the cell border)
    @param x_max: New maximal coordinate along the x axis (along the cell border)
    @param y_min: New minimal coordinate along the y axis (along the cell border)
    @param y_max: New maximal coordinate along the y axis (along the cell border)
    @param center_extent: Whether the new extent should be along the cells' 
                          borders or centers
    
    @return The sub-array and its extent
    '''
    assert len(georaster_array.shape) >= 2, "The array must be two-dimensional"
    assert len(georaster_extent) == 4, "The extent must have contain 4 coordinates"

    i_min, j_min = transform_to_pixel_coordinates(x_min, y_min, 
                                                  georaster_extent[0], georaster_extent[1],
                                                  georaster_extent[2], georaster_extent[3], 
                                                  georaster_array.shape[-1], georaster_array.shape[-2])
    i_max, j_max = transform_to_pixel_coordinates(x_max, y_max, 
                                                  georaster_extent[0], georaster_extent[1],
                                                  georaster_extent[2], georaster_extent[3], 
                                                  georaster_array.shape[-1], georaster_array.shape[-2])
    new_georaster_array = georaster_array[..., j_max:j_min, i_min:i_max]

    new_x_min, new_y_min = transform_to_geographic_coordinates(i_min, j_min, 
                                                               georaster_extent[0], georaster_extent[1],
                                                               georaster_extent[2], georaster_extent[3], 
                                                               georaster_array.shape[-1], georaster_array.shape[-2])
    new_x_max, new_y_max = transform_to_geographic_coordinates(i_max, j_max, 
                                                               georaster_extent[0], georaster_extent[1],
                                                               georaster_extent[2], georaster_extent[3], 
                                                               georaster_array.shape[-1], georaster_array.shape[-2])
    
    if center_extent == False:
        pixel_x_size = (georaster_extent[1] - georaster_extent[0])/georaster_array.shape[-1]
        pixel_y_size = (georaster_extent[3] - georaster_extent[2])/georaster_array.shape[-2]
        new_x_min -= 0.5*pixel_x_size
        new_y_min += 0.5*pixel_y_size
        new_x_max -= 0.5*pixel_x_size
        new_y_max += 0.5*pixel_y_size

    new_georaster_extent = (new_x_min, new_x_max, new_y_min, new_y_max)
    
    return new_georaster_array, new_georaster_extent

@jit(nopython = True)
def get_valid_subarray_indexes(array,
                               subarray_shape,
                               steps = (1, 1),
                               pads = (0, 0),
                               is_shape_centered = False):
    '''
    Get the indexes of all the possible sub-arrays that do not contain any NaN
    
    @param array: A ND NumPy array. The N - 2 first axes represent the variables,
                  the last two axes represent which the indexes are gotten
    @param subarray_shape: The 2D shape of the sub-arrays
    @param steps: The step between each sub-array for the last two axes, to avoid 
                  sampling all the possible sub-arrays
    @param pads: Extra row(s) and column(s) to get around the subarrays, without
                 influencing the subarray indexes
    @param is_shape_centered: True if the sub-arrays are defined from the
                              central cell, false if they are defined from the
                              top-left cell
    
    @return The 2D indexes of the sub-arrays in a list
    '''
    assert len(array.shape) >= 2, 'Array must be at least 2D'
    assert len(subarray_shape) == 2, 'Sub-array shape must be 2D'
    
    left_shape = (0, 0)
    right_shape = (subarray_shape[-2],
                   subarray_shape[-1])
    if is_shape_centered == True:
        left_shape = (int(subarray_shape[-2]/2),
                      int(subarray_shape[-1]/2))
        right_shape = (int(subarray_shape[-2]/2) + 1,
                       int(subarray_shape[-1]/2) + 1)

    subarray_indexes = []
    for j in range(left_shape[-2] + pads[-2],
                   array.shape[-2] - right_shape[-2] - pads[-2],
                   steps[-2]):
        for i in range(left_shape[-1] + pads[-1],
                       array.shape[-1] - right_shape[-1] - pads[-1],
                       steps[-1]):
            subarray = array[...,
                             j - left_shape[-2]:j + right_shape[-2],
                             i - left_shape[-1]:i + right_shape[-1]]
            if np.isnan(subarray).any() == False:
                subarray_indexes.append((j, i))
                
    return subarray_indexes
                
@jit(nopython = True)
def extract_subarrays(array,
                      sample_array_shape,
                      subarray_indexes,
                      pads = (0, 0),
                      is_shape_centered = False):
    '''
    Extract all the possible sub-arrays from their indexes
    
    @param array: A ND NumPy array. The N - 2 first axes represent the variables,
                  the last two axes represent which the indexes are gotten
    @param sample_array_shape: The (N + 1)D shape of the array that will contain
                               the sub-arrays
    @param subarray_indexes: A list of 2D indexes that locate the sub-arrays
    @param pads: Extra row(s) and column(s) to get around the subarrays, without
                 influencing the subarray indexes
    @param is_shape_centered: True if the sub-arrays are defined from the
                              central cell, false if they are defined from the
                              top-left cell
    
    @return The sub-arrays in a (N + 1)D NumPy array
    '''
    assert len(array.shape) >= 2, 'Array must be at least 2D'
    assert len(sample_array_shape) == len(array.shape) + 1, 'Sample array shape must be one dimension higher than array'
    
    left_shape = (pads[-2], pads[-1])
    right_shape = (sample_array_shape[-2] - pads[-2],
                   sample_array_shape[-1] - pads[-1])
    if is_shape_centered == True:
        left_shape = (int(sample_array_shape[-2]/2),
                      int(sample_array_shape[-1]/2))
        right_shape = (int(sample_array_shape[-2]/2) + 1,
                       int(sample_array_shape[-1]/2) + 1)
    
    sample_array = np.empty(sample_array_shape)
    for i in range(sample_array_shape[0]):
        sample_array[i] = array[...,
                                subarray_indexes[i][0] - left_shape[0]:subarray_indexes[i][0] + right_shape[0],
                                subarray_indexes[i][1] - left_shape[1]:subarray_indexes[i][1] + right_shape[1]]
        
    return sample_array

def sample_array(array,
                 subarray_shape,
                 steps = (1, 1),
                 pads = (0, 0),
                 is_shape_centered = False,
                 return_subarray_indexes = False):
    '''
    Sample an array of all the possible sub-arrays that do not contain any NaN
    
    @param array: A ND NumPy array. The N - 2 first axes represent the variables,
                  the last two axes represent which the indexes are gotten
    @param subarray_shape: The 2D shape of the sub-arrays
    @param steps: The step between each sub-array for the last two axes, to avoid 
                  sampling all the possible sub-arrays
    @param pads: Extra row(s) and column(s) to get around the subarrays, without
                 influencing the subarray indexes
    @param is_shape_centered: True if the sub-arrays are defined from the
                              central cell, false if they are defined from the
                              top-left cell
    
    @return The sub-arrays in a (N + 1)D NumPy array
    '''
    assert len(array.shape) >= 2, 'Array must be at least 2D'
    assert len(subarray_shape) == 2, 'Sub-array shape must be 2D'
    assert len(steps) == 2, 'Steps must be 2D'
    assert (is_shape_centered == False or 
            (is_shape_centered == True
             and (subarray_shape[0] - 1)%2 == 0
             and (subarray_shape[1] - 1)%2 == 0)), 'When centered, the subarray shape must be odd'
    
    subarray_indexes = get_valid_subarray_indexes(array,
                                                  subarray_shape,
                                                  steps = steps,
                                                  pads = pads,
                                                  is_shape_centered = is_shape_centered)
    sample_array_shape = tuple([len(subarray_indexes)]
                               + list(array.shape[:-2])
                               + [subarray_shape[-2] + 2*pads[-2], subarray_shape[-1] + 2*pads[-1]])
    samples_array = extract_subarrays(array,
                                      sample_array_shape,
                                      subarray_indexes,
                                      pads = pads,
                                      is_shape_centered = is_shape_centered)
    
    if return_subarray_indexes == False:
        return samples_array
    else:
        return samples_array, subarray_indexes

################################################################################
# Projection
################################################################################

def reproject_point(lon,
                    lat,
                    old_projection_EPSG = None,
                    old_projection_wkt = None,
                    old_projection_utm = None,
                    new_projection_EPSG = None,
                    new_projection_wkt = None,
                    new_projection_utm = None):
    '''
    Reproject a single point
    
    @param lon: Longitude of the point
    @param lat: Latitude of the point
    @param old_projection_EPSG: EPSG code of the old projection
    @param old_projection_wkt: WKT code of the old projection (can be used instead
                               of the old_projection_EPSG)
    @param old_projection_utm: Tuple with the UTM zone code and if it's northern or not
    @param new_projection_EPSG: EPSG code of the new projection
    @param new_projection_wkt: WKT code of the new projection (can be used instead
                               of the new_projection_EPSG)
    @param new_projection_utm: Tuple with the UTM zone code and if it's northern or not
    
    @return The coordinates' arrays
    '''
    assert (old_projection_EPSG is not None
            or old_projection_wkt is not None
            or old_projection_utm is not None), 'No old projection provided'
    assert (new_projection_EPSG is not None
            or new_projection_wkt is not None
            or new_projection_utm is not None), 'No new projection provided'

    old_spatial_reference = osr.SpatialReference()
    if old_projection_EPSG is not None:
        old_spatial_reference.ImportFromEPSG(old_projection_EPSG)
    elif old_projection_wkt is not None:
        old_spatial_reference.ImportFromWkt(old_projection_wkt)
    elif new_projection_utm is not None:
        assert len(new_projection_utm) == 2, 'UTM projection requires a zone code and an hemisphere code'
        old_spatial_reference.SetUTM(old_projection_utm[0], old_projection_utm[1])
    
    new_spatial_reference = osr.SpatialReference()
    if new_projection_EPSG is not None:
        new_spatial_reference.ImportFromEPSG(new_projection_EPSG)
    elif new_projection_wkt is not None:
        new_spatial_reference.ImportFromWkt(new_projection_wkt)
    elif new_projection_utm is not None:
        assert len(new_projection_utm) == 2, 'UTM projection requires a zone code and an hemisphere code'
        new_spatial_reference.SetUTM(new_projection_utm[0], new_projection_utm[1])

    transform = osr.CoordinateTransformation(old_spatial_reference,
                                             new_spatial_reference)
    
    return transform.TransformPoints([(lon, lat)])[0]

def find_utm_area(longitude, latitude):
    '''
    Find the UTM code and hemisphere from the longitude and latitude of a point
    
    @param longitude: A float for the longitude
    @param latitude: A float for the latitude
    
    @return A tuple with the code of the UTM zone and the hemisphere 
            (1: northern hemisphere; 0: southern hemisphere)
    '''
    utm_zone = int(1 + (longitude + 180.)/6.)
    is_northern = 1
    if latitude < 0.:
        is_northern = 0
    
    return (utm_zone, is_northern)

def reproject_georaster(georaster,
                        new_cell_sizes,
                        new_projection_EPSG = None,
                        new_projection_wkt = None,
                        new_projection_utm = None,
                        new_extent = None,
                        interpolation_method = gdal.GRA_Cubic,
                        file_type = 'MEM',
                        file_path = '',
                        data_type = gdal.GDT_Float64,
                        no_data_value = -99999.,
                        scale = 1.,
                        offset = 0.,
                        options = []):
    '''
    Change the projection of a GDAL georaster
    
    @param georaster: The GDAL georaster
    @param new_cell_sizes: Sizes (x, y) for cells of the georaster in the new projection
    @param new_projection_EPSG: EPSG code of the new projection
    @param new_projection_wkt: WKT code of the new projection (can be used instead
                               of the new_projection_EPSG)
    @param new_projection_utm: Tuple with the UTM zone code and if it's northern or not
    @param new_extent: Tuple with the minimal x, maximal x, minimal y, maximal y
                       for the new georaster
    @param interpolation_method: Interpolation method used during the projection
    @param file_type: Type to save the file (default is memory)
    @param file_path: Where to store the new georasterEPSG_code (default is memory)
    @param data_type: Data type of the georaster
    @param no_data_value: No data value for the georaster
    @param scale: Scaling factor for the georaster
    @param offset: Offset factor for the georaster
    @param options: List of options for compression
    
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
    elif new_projection_utm is not None:
        assert len(new_projection_utm) == 2, 'UTM projection requires a zone code and an hemisphere code'
        new_spatial_reference.SetUTM(new_projection_utm[0], new_projection_utm[1])
    else:
        if new_extent is None:
            print('No new spatial reference provided, will use the one from the georaster')
        new_spatial_reference.ImportFromWkt(georaster.GetProjectionRef())
    
    new_x_size = None
    new_y_size = None
    new_geotransform = None
    if new_extent is None:
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
        new_x_size = int(abs(lrx - ulx)/new_cell_sizes[0])
        new_y_size = int(abs(uly - lry)/new_cell_sizes[1])
    else:
        new_geotransform = (new_extent[0],
                            new_cell_sizes[0],
                            0., 
                            new_extent[3], 
                            0., 
                            -new_cell_sizes[1])
        new_x_size = int(round(abs(new_extent[1] - new_extent[0])/new_cell_sizes[0]))
        new_y_size = int(round(abs(new_extent[3] - new_extent[2])/new_cell_sizes[1]))

    number_of_bands = georaster.RasterCount
    
    driver = gdal.GetDriverByName(file_type)
    new_georaster = driver.Create(file_path,
                                  new_x_size,
                                  new_y_size,
                                  number_of_bands,
                                  data_type,
                                  options = options)
    new_georaster.SetGeoTransform(new_geotransform)
    new_georaster.SetProjection(new_spatial_reference.ExportToWkt())
    for i_band in range(1, number_of_bands + 1):
        new_georaster_band = new_georaster.GetRasterBand(i_band)
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

def georaster_vertical_datum_shift(georaster,
                                   old_datum_proj4 = '+proj=longlat +datum=WGS84 +no_defs +geoidgrids=egm96_15.gtx',
                                   new_datum_proj4 = '+proj=longlat +datum=WGS84 +no_defs',
                                   file_type = 'MEM',
                                   file_path = '',
                                   data_type = gdal.GDT_Float64,
                                   no_data_value = -99999.,
                                   scale = 1.,
                                   offset = 0.):
    '''
    Change the vertical datum of a GDAL georaster, from a geoid to an ellipsoid
    or the other way around
    
    @param georaster: The GDAL georaster
    @param old_datum_proj4: Proj4 code for the old datum, with the elevation
                            being defined relatide to an ellipsoid or a geoid.
                            Default value is for the EGM96 geoid, whose grid
                            needs to be downloaded beforehand
    @param new_datum_proj4: Proj4 code for the new datum, with the elevation
                            being defined relatide to an ellipsoid or a geoid.
                            Default value is for the WGS84 ellipsoid.
    @param file_type: Type to save the file (default is memory)
    @param file_path: Where to store the new georasterEPSG_code (default is memory)
    @param data_type: Data type of the georaster
    @param no_data_value: No data value for the georaster
    @param scale: Scaling factor for the georaster
    @param offset: Offset factor for the georaster
    
    @return The GDAL georaster
    '''
    geotransform = georaster.GetGeoTransform()
    georaster_x_size = georaster.RasterXSize
    georaster_y_size = georaster.RasterYSize

    number_of_bands = georaster.RasterCount
    
    driver = gdal.GetDriverByName(file_type)
    new_georaster = driver.Create(file_path,
                                  georaster_x_size,
                                  georaster_y_size,
                                  number_of_bands,
                                  data_type)
    new_georaster.SetGeoTransform(geotransform)
    new_georaster.SetProjection(new_datum_proj4)
    for i_band in range(1, number_of_bands + 1):
        new_georaster_band = new_georaster.GetRasterBand(i_band)
        new_georaster_band.SetNoDataValue(no_data_value)
        new_georaster_band.SetScale(scale)
        new_georaster_band.SetOffset(offset)
        # Fill the georaster band, otherwise no data values are not set in the new georaster
        new_georaster_band.Fill(no_data_value)

    res = gdal.Warp(new_georaster,
                    georaster,
                    srcSRS = old_datum_proj4,
                    dstSRS = new_datum_proj4)

    return new_georaster
