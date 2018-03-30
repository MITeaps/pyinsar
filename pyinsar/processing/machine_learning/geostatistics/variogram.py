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
from enum import Enum

from numba import jit, vectorize, float64

################################################################################
# 2D experimental variogram
################################################################################

@jit(nopython = True)
def compute_experimental_variogram(value_array,
                                   grid_yx_spacing,
                                   number_of_lags,
                                   lag_unit_distance,
                                   tolerance = None,
                                   sampling = 1.,
                                   no_data_value = -99999):
    '''
    Compute the isotropic experimental variogram on a regular grid
    
    @param value_array: A 2D NumPy array with the variable to study
    @param grid_yx_spacing: The cell size along each axis (y, x)
    @param number_of_lags: Number of lags to compute the variogram
    @param lag_unit_distance: Distance between each lag
    @param tolerance: Tolerance to add a pair of points to a given lag (if None,
                       set to 0.6*lag_unit_distance)
    @param sampling: The proportion of cells of the array to take into account
    @param no_data_value: The no-data value
    
    @return A 2D Numpy array containing for each lag the variogram values before
            dividing by the number of pairs of points, the number of pairs of
            points, and the lag values
    '''
    assert (len(value_array.shape) == 2
            and len(grid_yx_spacing) == 2), "The array must be two-dimensional"

    if tolerance is None:
        tolerance = 0.6*lag_unit_distance
    lags = np.empty(number_of_lags)
    lag_ranges = np.empty((number_of_lags, 2))
    for i in range(number_of_lags):
        lags[i] = (i + 1)*lag_unit_distance
        lag_ranges[i, 0] = max((i + 1)*lag_unit_distance - tolerance, 0.)
        lag_ranges[i, 1] = (i + 1)*lag_unit_distance + tolerance

    variogram_values = np.zeros(number_of_lags)
    number_of_values = np.zeros(number_of_lags)
    
    sampling_nb = int(1/sampling)
    sampled_value_array = value_array[::sampling_nb, ::sampling_nb]
    
    neighborhood_shape = (math.floor(lag_unit_distance*(number_of_lags + 1)/(grid_yx_spacing[0]*sampling_nb)),
                          math.floor(lag_unit_distance*(number_of_lags + 1)/(grid_yx_spacing[1]*sampling_nb)))
    neighborhood_size = (2*neighborhood_shape[0] + 1)*neighborhood_shape[1] + neighborhood_shape[1]
    vario_neighborhood_indexes = np.empty((neighborhood_size,
                                           2),
                                          dtype = np.int64)
    vario_neighborhood_lags = np.full((neighborhood_size,
                                       math.ceil(2*tolerance/lag_unit_distance)),
                                      no_data_value,
                                      dtype = np.int64)
    i_neigh = 0
    for j in range(-neighborhood_shape[0], neighborhood_shape[0] + 1):
        for i in range(0, neighborhood_shape[1] + 1):
            if (i > 0) or (i == 0 and j > 0):
                vario_neighborhood_indexes[i_neigh, 0] = j
                vario_neighborhood_indexes[i_neigh, 1] = i
                distance = math.sqrt((j*grid_yx_spacing[0]*sampling_nb)**2
                                     + (i*grid_yx_spacing[1]*sampling_nb)**2)
                j_lag = 0
                for i_lag in range(len(lag_ranges)):
                    if lag_ranges[i_lag, 0] < distance <= lag_ranges[i_lag, 1]:
                        vario_neighborhood_lags[i_neigh, j_lag] = i_lag
                        j_lag += 1
                i_neigh += 1

    for j in range(sampled_value_array.shape[0]):
        for i in range(sampled_value_array.shape[1]):
            if math.isnan(sampled_value_array[j, i]) == False:
                for i_neigh in range(vario_neighborhood_indexes.shape[0]):
                    cell = (j + vario_neighborhood_indexes[i_neigh, 0],
                            i + vario_neighborhood_indexes[i_neigh, 1])
                    if (0 <= cell[0] < sampled_value_array.shape[0]
                        and 0 <= cell[1] < sampled_value_array.shape[1]
                        and math.isnan(sampled_value_array[cell]) == False):
                        i_lag = 0
                        lag = vario_neighborhood_lags[i_neigh, i_lag]
                        while i_lag < vario_neighborhood_lags.shape[1] and lag != no_data_value:
                            variogram_values[lag] += (1/2.)*(sampled_value_array[j, i] - sampled_value_array[cell])**2
                            number_of_values[lag] += 1
                            i_lag += 1
                            lag = vario_neighborhood_lags[i_neigh, i_lag]
        
    return variogram_values, number_of_values, lags

################################################################################
# 2D theoretical variogram
################################################################################

class VariogramModel(Enum):
    NUGGET = 0
    GAUSSIAN = 1
    SPHERICAL = 2
    EXPONENTIAL = 3

@jit(nopython = True, nogil = True)
def nugget_variogram(reduced_distance, variance_contribution):
    '''
    Compute the value of a variogram with a pure nugget effect
    
    @param reduced_distance: The distance between the two points divided by the
                             variogram range
    @param variance_contribution: The variance for this variogram model
    
    @return The value of the variogram
    '''
    if reduced_distance == 0.:
        return 0.
    return variance_contribution
@jit(nopython = True, nogil = True)
def gaussian_variogram(reduced_distance, variance_contribution):
    '''
    Compute the value of a variogram with a Gaussian model
    
    @param reduced_distance: The distance between the two points divided by the
                             variogram range
    @param variance_contribution: The variance for this variogram model
    
    @return The value of the variogram
    '''
    return variance_contribution*(1. - math.exp(-3*reduced_distance**2))
@jit(nopython = True, nogil = True)
def spherical_variogram(reduced_distance, variance_contribution):
    '''
    Compute the value of a variogram with a spherical model
    
    @param reduced_distance: The distance between the two points divided by the
                             variogram range
    @param variance_contribution: The variance for this variogram model
    
    @return The value of the variogram
    '''
    if reduced_distance <= 1:
        return variance_contribution*(1.5*reduced_distance - 0.5*reduced_distance**3)
    return variance_contribution
@jit(nopython = True, nogil = True)
def exponential_variogram(reduced_distance, variance_contribution):
    '''
    Compute the value of a variogram with an exponential model
    
    @param reduced_distance: The distance between the two points divided by the
                             variogram range
    @param variance_contribution: The variance for this variogram model
    
    @return The value of the variogram
    '''
    return variance_contribution*(1. - math.exp(-3*reduced_distance))

@jit(nopython = True, nogil = True)
def compute_variogram(delta_y,
                      delta_x,
                      vario_models,
                      vario_sills,
                      vario_ranges,
                      rotation_matrix):
    '''
    Compute the value of a (possibly nested) 2D variogram
    
    @param delta_y: The distance between the two points along the y axis
    @param delta_x: The distance between the two points along the x axis
    @param vario_models: The models for the variogram
    @param vario_sills: The sills for the variogram
    @param vario_ranges: The major and minor ranges for the variogram
    @param rotation_matrix: The 2D rotation matrix
    
    @return The value of the variogram
    '''
    variogram_value = 0.
    nugget = 0.
    for vario_model, vario_sill, vario_range in zip(vario_models,
                                                    vario_sills,
                                                    vario_ranges):
        u = delta_x
        v = delta_y
        if vario_range[0] != 0. and vario_range[1] != 0.:
            u = (delta_y*rotation_matrix[0, 0] + delta_x*rotation_matrix[0, 1])/vario_range[0]
            v = (delta_y*rotation_matrix[1, 0] + delta_x*rotation_matrix[1, 1])/vario_range[1]
        reduced_distance = math.sqrt(u**2 + v**2)
        if vario_model == VariogramModel.GAUSSIAN:
            variogram_value += gaussian_variogram(reduced_distance, vario_sill - nugget)
        elif vario_model == VariogramModel.SPHERICAL:
            variogram_value += spherical_variogram(reduced_distance, vario_sill - nugget)
        elif vario_model == VariogramModel.EXPONENTIAL:
            variogram_value += exponential_variogram(reduced_distance, vario_sill - nugget)
        else:
            variogram_value += nugget_variogram(reduced_distance, vario_sill)
        nugget = vario_sill

    return variogram_value

################################################################################
# Vectorized theoretical variogram
################################################################################

@vectorize([float64(float64, float64, float64)])
def vectorized_gaussian_variogram(distance, vario_range, variance_contribution):
    '''
    Compute the value of a variogram with a Gaussian model
    
    @param distance: The distance between the two points
    @param vario_range: The variogram range
    @param variance_contribution: The variance for this variogram model
    
    @return The value of the variogram
    '''
    return variance_contribution*(1. - math.exp(-3*(distance/vario_range)**2))

@vectorize([float64(float64, float64, float64)])
def vectorized_spherical_variogram(distance, vario_range, variance_contribution):
    '''
    Compute the value of a variogram with a spherical model
    
    @param distance: The distance between the two points
    @param vario_range: The variogram range
    @param variance_contribution: The variance for this variogram model
    
    @return The value of the variogram
    '''
    if distance/vario_range <= 1:
        return variance_contribution*(1.5*distance/vario_range - 0.5*(distance/vario_range)**3)
    return variance_contribution

@vectorize([float64(float64, float64, float64)])
def vectorized_exponential_variogram(distance, vario_range, variance_contribution):
    '''
    Compute the value of a variogram with an exponential model
    
    @param distance: The distance between the two points
    @param vario_range: The variogram range
    @param variance_contribution: The variance for this variogram model
    
    @return The value of the variogram
    '''
    return variance_contribution*(1. - math.exp(-3*distance/vario_range))

################################################################################
# Utilities
################################################################################

@jit(nopython = True)
def map_2D_variogram(vario_models,
                     vario_sills,
                     vario_azimuth,
                     vario_ranges,
                     neighborhood_range,
                     map_shape,
                     grid_spacing):
    '''
    Map the variogram values in a 2D neighborhood
    
    @param vario_models: The models for the variogram
    @param vario_sills: The sills for the variogram
    @param vario_azimuth: The azimuth of the variogram's major axis (in degree)
    @param vario_ranges: The major and minor ranges for the variogram
    @param neighborhood_range: The range of the neighborhood, beyond which the
                               cells with already a value are not taken into
                               account (in grid spacing unit)
    @param map_shape: The number of cells along each axis (y, x) for the variogram map
    @param grid_spacing: The cell size along each axis (y, x)
    
    @return The variogram map as a 2D array
    '''
    assert (len(map_shape) == 2
            and len(grid_spacing) == 2
            and len(neighborhood_range) == 2), "The arrays must be two-dimensional"
    assert (len(vario_models) == len(vario_sills)
            and len(vario_models) == len(vario_ranges)), "Some parameters for the (nested) variogram are missing"
    for vario_range in vario_ranges:
        assert len(vario_range) == 2, "A major or minor range is missing from the variogram ranges"
    
    vario_azimuth_rad = vario_azimuth*math.pi/180.
    rotation_matrix = np.empty((2, 2))
    rotation_matrix[0, 0] = math.cos(vario_azimuth_rad)
    rotation_matrix[0, 1] = math.sin(vario_azimuth_rad)
    rotation_matrix[1, 0] = math.sin(vario_azimuth_rad)
    rotation_matrix[1, 1] = -math.cos(vario_azimuth_rad)

    variogram_map = np.full(map_shape, np.nan)

    for j in range(map_shape[0]):
        delta_y = (int(map_shape[0]/2.) - j)*grid_spacing[0]
        for i in range(map_shape[1]):
            delta_x = (i - int(map_shape[1]/2.))*grid_spacing[1]
            ellipse_radius = ((delta_y*rotation_matrix[0, 0]
                               + delta_x*rotation_matrix[0, 1])**2)/(neighborhood_range[0]**2)\
                             + ((delta_y*rotation_matrix[1, 0]
                                 + delta_x*rotation_matrix[1, 1])**2)/(neighborhood_range[1]**2)
            if ellipse_radius <= 1:
                variogram = compute_variogram(delta_y,
                                              delta_x,
                                              vario_models,
                                              vario_sills,
                                              vario_ranges,
                                              rotation_matrix)
                variogram_map[j, i] = variogram
                
    return variogram_map

@jit(nopython = True)
def compute_range_variogram(deltas_y,
                            deltas_x,
                            vario_models,
                            vario_sills,
                            vario_ranges,
                            vario_azimuth = 0.):
    '''
    Compute the variogram values for a range of distances
    
    @param deltas_y: A 1D array of distances between the two points along the y axis
    @param deltas_x: A 1D array of distances between the two points along the x axis
    @param vario_models: The models for the variogram
    @param vario_sills: The sills for the variogram
    @param vario_azimuth: The azimuth of the variogram's major axis (in degree)
    @param vario_ranges: The major and minor ranges for the variogram
    
    @return The variogram values as a 1D array
    '''
    assert (len(deltas_y.shape) == 1
            and len(deltas_x.shape) == 1), "The delta arrays must be one-dimensional"
    assert (deltas_y.shape == deltas_x.shape), "The delta arrays must have the same size"
    assert (len(vario_models) == len(vario_sills)
            and len(vario_models) == len(vario_ranges)), "Some parameters for the (nested) variogram are missing"
    for vario_range in vario_ranges:
        assert len(vario_range) == 2, "A major or minor range is missing from the variogram ranges"
    
    vario_azimuth_rad = vario_azimuth*math.pi/180.
    rotation_matrix = np.empty((2, 2))
    rotation_matrix[0, 0] = math.cos(vario_azimuth_rad)
    rotation_matrix[0, 1] = math.sin(vario_azimuth_rad)
    rotation_matrix[1, 0] = math.sin(vario_azimuth_rad)
    rotation_matrix[1, 1] = -math.cos(vario_azimuth_rad)
    
    variogram = np.zeros(len(deltas_y))
    for i in range(len(deltas_y)):
        value = compute_variogram(deltas_y[i],
                                  deltas_x[i],
                                  vario_models,
                                  vario_sills,
                                  vario_ranges,
                                  rotation_matrix)
        variogram[i] = value
        
    return variogram