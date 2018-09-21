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

import math
import numpy as np

from numba import jit

from pyinsar.processing.geography.coordinates import transform_to_pixel_coordinates

@jit(nopython = True)
def divide_quadrant(quadrant):
    '''
    Divide a quadrant into four subquadrants
    
    @param quadrant: A 2D NumPy array
    
    @return A 2D NumPy array with the corner indexes of each subquadrant within
            the quadrant
    '''
    subquadrant_indexes = np.empty((4, 4), dtype = np.int32)
    subquadrant_indexes[0, 0] = 0
    subquadrant_indexes[0, 1] = quadrant.shape[0]/2
    subquadrant_indexes[0, 2] = 0
    subquadrant_indexes[0, 3] = quadrant.shape[1]/2
    
    subquadrant_indexes[1, 0] = quadrant.shape[0]/2
    subquadrant_indexes[1, 1] = quadrant.shape[0]
    subquadrant_indexes[1, 2] = 0
    subquadrant_indexes[1, 3] = quadrant.shape[1]/2
    
    subquadrant_indexes[2, 0] = 0
    subquadrant_indexes[2, 1] = quadrant.shape[0]/2
    subquadrant_indexes[2, 2] = quadrant.shape[1]/2
    subquadrant_indexes[2, 3] = quadrant.shape[1]
    
    subquadrant_indexes[3, 0] = quadrant.shape[0]/2
    subquadrant_indexes[3, 1] = quadrant.shape[0]
    subquadrant_indexes[3, 2] = quadrant.shape[1]/2
    subquadrant_indexes[3, 3] = quadrant.shape[1]
    
    return subquadrant_indexes

@jit(nopython = True)
def prepare_system(array, coordinates):
    '''
    Prepare each matrix of a system of equations to compute the best fitting plane
    
    @param array: A 2D NumPy array of values
    @param coordinates: A 3D NumPy array with the spatial coordinate of each cell
                        of array
    
    @return The two matrices required to solve the system
    '''
    valid_indexes = []
    for j in range(array.shape[0]):
        for i in range(array.shape[1]):
            if math.isnan(array[j, i]) == False:
                valid_indexes.append((j, i))
         
    A = np.ones((len(valid_indexes), 3))
    B = np.empty(len(valid_indexes))
    for i in range(len(valid_indexes)):
        A[i, 0] = coordinates[0][valid_indexes[i]]
        A[i, 1] = coordinates[1][valid_indexes[i]]
        B[i] = array[valid_indexes[i]]
            
    return A, B

@jit(nopython = True)
def nanvarplane(array, coordinates, ddof = 0.):
    '''
    Compute the variance of a set of points relative to the best fitting plane
    
    @param array: A 2D NumPy array of values
    @param coordinates: A 3D NumPy array with the spatial coordinate of each cell
                        of array
    @param ddof: The divisor for the variance is N - ddof, where N is the number
                 of points
    
    @return The variance
    '''
    A, B = prepare_system(array, coordinates)
    C = np.linalg.lstsq(A, B)
    best_fit_plane = C[0][0]*coordinates[0] + C[0][1]*coordinates[1] + C[0][2]
    
    return np.nansum((array - best_fit_plane)**2)/(B.size - ddof)

@jit(nopython = True)
def nansize(array):
    '''
    Compute size of an array without the NaNs
    
    @param array: A NumPy array
    '''   
    return array.size - np.isnan(array).sum()

@jit(nopython = True)
def decompose_quadrant(quadrant_values,
                       quadrant_gradient,
                       quadrant_coordinates,
                       variance_threshold,
                       smallest_quadrant_size,
                       largest_quadrant_size,
                       use_variance_best_plane,
                       use_median,
                       max_nan_proportion,
                       sampled_points):
    '''
    Decompose a quadrant into four subquadrants if possible
    
    @param quadrant_values: A 2D NumPy array of quadrant values
    @param quadrant_gradient: A 2D NumPy array of quadrant values used for 
                              testing the variance
    @param quadrant_coordinates: A 3D NumPy array with the spatial coordinate of
                                 each cell of the quadrant
    @param variance_threshold: A variance threshold below which there is no
                               decomposition
    @param smallest_quadrant_size: A size threshold below which there is no
                                   decomposition
    @param largest_quadrant_size: A size threshold above which decomposition is
                                  done independently of the variance value
    @param use_variance_best_plane: True to use the variance relative to the best
                                    fitting plane, false to use the variance
                                    relative to the mean
    @param use_median: True to use the median instead of the mean as output when
                       stopping the decomposition
    @param max_nan_proportion: Maximum proportion of NaN to take a quadrant into
                               account
    @param sampled_points: Points that are sampled through the decomposition
    '''
    subquadrant_indexes = divide_quadrant(quadrant_values)
    
    for i in range(4):
        
        subquadrant_values = quadrant_values[subquadrant_indexes[i, 0]:subquadrant_indexes[i, 1],
                                             subquadrant_indexes[i, 2]:subquadrant_indexes[i, 3]]
        assert 0 not in subquadrant_values.shape, "Empty subquadrant"
        
        if nansize(subquadrant_values)/subquadrant_values.size > 1 - max_nan_proportion:
            subquadrant_gradient = quadrant_gradient[subquadrant_indexes[i, 0]:subquadrant_indexes[i, 1],
                                                     subquadrant_indexes[i, 2]:subquadrant_indexes[i, 3]]
            subquadrant_coordinates = quadrant_coordinates[:,
                                                           subquadrant_indexes[i, 0]:subquadrant_indexes[i, 1],
                                                           subquadrant_indexes[i, 2]:subquadrant_indexes[i, 3]]

            quadrant_size = max(np.max(subquadrant_coordinates[0, 0, :] - subquadrant_coordinates[0, -1, :]),
                                np.max(subquadrant_coordinates[1, :, -1] - subquadrant_coordinates[1, :, 0]))
            variance = np.nan
            if use_variance_best_plane == True and nansize(subquadrant_gradient) > 2:
                variance = nanvarplane(subquadrant_gradient, subquadrant_coordinates)
            else:
                variance = np.nanvar(subquadrant_gradient)

            if (quadrant_size > smallest_quadrant_size
                and (quadrant_size > largest_quadrant_size or variance > variance_threshold)):
                decompose_quadrant(subquadrant_values,
                                   subquadrant_gradient,
                                   subquadrant_coordinates,
                                   variance_threshold,
                                   smallest_quadrant_size,
                                   largest_quadrant_size,
                                   use_variance_best_plane,
                                   use_median,
                                   max_nan_proportion,
                                   sampled_points)
            else:
                subquadrant_center_value = np.nan
                if use_median == True:
                    subquadrant_center_value = np.nanmedian(subquadrant_values)
                else:
                    subquadrant_center_value = np.nanmean(subquadrant_values)
                sampled_points.append(((np.min(subquadrant_coordinates[0]) + np.max(subquadrant_coordinates[0]))/2.,
                                       (np.min(subquadrant_coordinates[1]) + np.max(subquadrant_coordinates[1]))/2.,
                                       subquadrant_center_value))

def downsample_image_with_quadtree(image_array,
                                   coordinates_array,
                                   variance_threshold,
                                   smallest_quadrant_size,
                                   largest_quadrant_size,
                                   gradient_array = None,
                                   use_variance_best_plane = True,
                                   use_median = False,
                                   max_nan_proportion = 1.):
    '''
    Downsample an image using a quadtree
    
    @param image_array: A 2D NumPy array of values
    @param coordinates_array: A 3D NumPy array with the spatial coordinate of
                              each cell of the image
    @param variance_threshold: A variance threshold below which there is no 
                               decomposition
    @param smallest_quadrant_size: A size threshold below which there is no 
                                   decomposition
    @param largest_quadrant_size: A size threshold above which decomposition is
                                  done independently of the variance value
    @param gradient_array: A 2D NumPy array of image values used for testing the
                           variance (if None, the image is used)
    @param use_variance_best_plane: True to use the variance relative to the best
                                    fitting plane, false to use the variance
                                    relative to the mean
    @param use_median: True to use the median instead of the mean as output when
                       stopping the decomposition
    @param max_nan_proportion: Maximum proportion of NaN to take a quadrant into
                               account
    
    @param return A list of sampled points
    '''
    assert len(image_array.shape) == 2, "image_array must be 2D"
    assert len(coordinates_array.shape) == 3, "coordinates_array must be 3D"
    assert coordinates_array.shape[0] == 2, "The first dimension of coordinates_array must have two elements, x and y"
    assert coordinates_array.shape[1:] == image_array.shape, "The last two dimensions of coordinates_array must be the same than the dimensions of image_array"
    if gradient_array is None:
        gradient_array = image_array
    else:
        assert gradient_array.shape == image_array.shape, "gradient_array must have the same shape than image_array"
    assert 0. <= max_nan_proportion <= 1., "max_nan_proportion must be between 0 and 1"
        
    sampled_points = [(-99999., -99999., -99999.)]
    decompose_quadrant(image_array,
                       gradient_array,
                       coordinates_array,
                       variance_threshold,
                       smallest_quadrant_size,
                       largest_quadrant_size,
                       use_variance_best_plane,
                       use_median,
                       max_nan_proportion,
                       sampled_points)

    return sampled_points[1:]

def get_closest_los_vector(x, y, los_vector, los_vector_extent):
    '''
    Get the closest line-of-sight vector values for the given coordinates
    
    @param x: A NumPy array of x coordinates
    @param y: A NumPy array of y coordinates
    @param los_vector: A NumPy array of los vectors
    @param los_vector_extent: The spatial extent of the los vector array
    
    @param return A list of sampled points
    '''
    i, j = transform_to_pixel_coordinates(x,
                                          y,
                                          los_vector_extent[0],
                                          los_vector_extent[1],
                                          los_vector_extent[2],
                                          los_vector_extent[3],
                                          los_vector.shape[-1],
                                          los_vector.shape[-2])
    
    return los_vector[:, j, i]