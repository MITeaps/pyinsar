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
import random
from enum import Enum
from scipy.special import erfinv
from scipy import stats
# import concurrent.futures

from numba import jit
from numba import prange

from pyinsar.processing.machine_learning.geostatistics.geostatistics_utils import PathType, unflatten_index, standardize
from pyinsar.processing.machine_learning.geostatistics.variogram import *

################################################################################
# Merging secondary data
################################################################################

def merge_secondary_data(secondary_data_array,
                         correlations_with_primary,
                         correlations_between_secondary):
    '''
    Merge several secondary data (see Babak and Deutsch, 2009, doi:10.1016/j.petrol.2009.08.001)
    
    @param secondary_data_array: A 3D array gathering several 2D secondary data
    @param correlations_with_primary: Correlation weights between the main data
                                      and the secondary data
    @param correlations_between_secondary: Correlation weights between the secondary data
    
    @return A 2D array with the merged secondary data and a weight for the merged data
    '''
    assert len(secondary_data_array.shape) == 3, "The secondary data array must be three-dimensional"
    assert len(correlations_with_primary) == secondary_data_array.shape[0], "Some correlations with the primary data are missing"
    assert (len(correlations_between_secondary.shape) == 2
            and correlations_between_secondary.shape[0] == secondary_data_array.shape[0]
            and correlations_between_secondary.shape[1] == secondary_data_array.shape[0]), "Some correlations between secondary data are missing"
    
    correlation_weights = np.linalg.solve(correlations_between_secondary,
                                          correlations_with_primary)
    merged_secondary_data_weight = math.sqrt(np.sum(correlation_weights*correlations_with_primary))
    merged_secondary_data_array = np.sum(correlation_weights[:, None, None]*secondary_data_array,
                                         axis = 0)/merged_secondary_data_weight
    
    return merged_secondary_data_array, merged_secondary_data_weight

################################################################################
# Sequential Gaussian Simulation (SGS)
################################################################################

@jit(nopython = True, nogil = True)
def compute_euclidean_distance(cell_1, cell_2):
    '''
    Compute the 2D Euclidean distance
    
    @param cell_1: The first point
    @param array_shape: The second point
    
    @return The distance
    '''
    return math.sqrt((cell_1[0] - cell_2[0])**2 + (cell_1[1] - cell_2[1])**2)

@jit(nopython = True, nogil = True)
def compute_axis_aligned_ellipse_range(neighborhood_range, neighborhood_azimuth_rad):
    '''
    Compute the extent of an ellipse along the y and x axes
    
    @param neighborhood_range: The major and minor axis length of the ellipse
    @param neighborhood_azimuth_rad: The azimuth giving the orientation of the
                                     major axis
    
    @return The extent
    '''
    uy = neighborhood_range[0]*math.cos(neighborhood_azimuth_rad)
    ux = neighborhood_range[0]*math.sin(neighborhood_azimuth_rad)
    vy = neighborhood_range[1]*math.cos(neighborhood_azimuth_rad + math.pi/2.)
    vx = neighborhood_range[1]*math.sin(neighborhood_azimuth_rad + math.pi/2.)

    return (math.sqrt(uy**2 + vy**2), math.sqrt(ux**2 + vx**2))

@jit(nopython = True, nogil = True)
def compute_axis_aligned_neighborhood_shape(neighborhood_range,
                                            neighborhood_azimuth,
                                            grid_yx_spacing):
    '''
    Compute the shape (in cells) of an ellipse along the y and x axes
    
    @param neighborhood_range: The major and minor axis length of the ellipse
    @param neighborhood_azimuth_rad: The azimuth giving the orientation of the
                                     major axis (in radian)
    @param grid_yx_spacing: The cell size along each axis (y, x)
    
    @return The shape
    '''
    y_aligned_range, x_aligned_range = compute_axis_aligned_ellipse_range(neighborhood_range,
                                                                          neighborhood_azimuth)

    return (math.floor(y_aligned_range/grid_yx_spacing[0]),
            math.floor(x_aligned_range/grid_yx_spacing[1]))

@jit(nopython = True, nogil = True)
def compute_neighborhood_template(neighborhood_range,
                                  grid_yx_spacing,
                                  vario_models,
                                  vario_sills,
                                  vario_ranges,
                                  vario_azimuth_rad,
                                  rotation_matrix,
                                  eps = 0.0001):
    '''
    Compute the template for the neighborhood, i.e. the closest cells in term of
    variogram value and the variogram values for each cell
    
    @param neighborhood_range: The major and minor axis length of the ellipse
    @param grid_yx_spacing: The cell size along each axis (y, x)
    @param vario_models: The variogram models
    @param vario_sills: The variogram sills
    @param vario_ranges: The variogram ranges
    @param vario_azimuth_rad: The azimuth of the variogram's major axis (in radian)
    @param rotation_matrix: The 2D rotation matrix
    @param eps: A factor to include the distance when determining the closest points
    
    @return The closest cells to the center cell of the neighborhood and the
    variogram values within the neighborhood
    '''
    neighborhood_shape = compute_axis_aligned_neighborhood_shape(neighborhood_range,
                                                                 vario_azimuth_rad,
                                                                 grid_yx_spacing)
    max_distance = math.sqrt((neighborhood_shape[0]*grid_yx_spacing[0])**2
                             + (neighborhood_shape[1]*grid_yx_spacing[1])**2)
    
    neighborhood_template = [(-99999., -99999., -99999.)]
    correlation_template = np.ones((4*neighborhood_shape[0] + 1, 4*neighborhood_shape[1] + 1))
    for j in range(-2*neighborhood_shape[0], 2*neighborhood_shape[0] + 1):
        delta_y = j*grid_yx_spacing[0]
        for i in range(-2*neighborhood_shape[1], 2*neighborhood_shape[1] + 1):
            if i != 0 or j != 0:
                delta_x = i*grid_yx_spacing[1]
                ellipse_radius = ((delta_y*rotation_matrix[0, 0]
                                   + delta_x*rotation_matrix[0, 1])**2)/(neighborhood_range[0]**2)\
                                  + ((delta_y*rotation_matrix[1, 0]
                                      + delta_x*rotation_matrix[1, 1])**2)/(neighborhood_range[1]**2)
                variogram = compute_variogram(delta_y,
                                              delta_x,
                                              vario_models,
                                              vario_sills,
                                              vario_ranges,
                                              rotation_matrix)
                correlation_template[2*neighborhood_shape[0] - j,
                                     i + 2*neighborhood_shape[1]] = (vario_sills[-1] - variogram)/vario_sills[-1]
                if ellipse_radius <= 1:
                    distance = math.sqrt(delta_y**2 + delta_x**2)
                    distance_template = variogram/vario_sills[-1] + eps*distance/max_distance
                    neighborhood_template.append((distance_template, j, i))

    return sorted(neighborhood_template[1:]), correlation_template

@jit(nopython = True, nogil = True)
def get_neighborhood(cell_index,
                     simulation_array,
                     neighborhood_template,
                     max_number_data,
                     no_data_value):
    '''
    Get the neighborhood around a given cell, i.e., all the cells that already 
    have a value around the given cell
    
    @param cell_index: The cell index in the simulation grid
    @param simulation_array: The simulation grid
    @param neighborhood_template: The template for the neighborhood, which gives
                                  the closest cells to the current one
    @param max_number_data: The maximum number of data in the neighborhood
    @param no_data_value: The no-data value, which defines the cell to simulate
    
    @return The closest cells to the center cell of the neighborhood
    and the variogram values within the neighborhood
    '''
    neighborhood = [(-99999, -99999)]
    i_neighbor = 0
    while len(neighborhood) - 1 < max_number_data and i_neighbor < len(neighborhood_template):
        j = int(cell_index[0] + neighborhood_template[i_neighbor][1])
        i = int(cell_index[1] + neighborhood_template[i_neighbor][2])
        if (0 <= j < simulation_array.shape[0]
            and 0 <= i < simulation_array.shape[1]
            and simulation_array[j, i] != no_data_value
            and math.isnan(simulation_array[j, i]) == False):
            neighborhood.append((j, i))
        i_neighbor += 1
    return neighborhood[1:]

@jit(nopython = True, nogil = True)
def get_values_matrix(neighborhood, simulation_array):
    '''
    Get the matrix of already simulated values around the cell to estimate
    
    @param neighborhood: The data around the cell to estimate
    @param simulation_array: The simulation grid
    
    @return The matrix
    '''
    matrix = np.zeros((len(neighborhood), 1))
    for i in range(len(neighborhood)):
        matrix[i, 0] = simulation_array[neighborhood[i]]
    return matrix

@jit(nopython = True, nogil = True)
def get_data_to_data_matrix(kriging_method,
                            cell_index,
                            neighborhood,
                            correlation_template,
                            secondary_data_weight):
    '''
    Get the matrix of variogram values between the cells with already a value
    
    @param kriging_method: The kriging method to use (see KrigingMethod)
    @param cell_index: The cell to estimate
    @param neighborhood: The data around the point to estimate
    @param correlation_template: The variogram values in the neighborhood
    @param secondary_data_weight: The weight for the secondary data (math.nan if
                                  no secondary data)
    
    @return The matrix
    '''
    number_of_rows = len(neighborhood)
    if math.isnan(secondary_data_weight) == False:
        number_of_rows += len(neighborhood) + 1
    if kriging_method == KrigingMethod.ORDINARY:
        number_of_rows += 1
        
    matrix = np.zeros((number_of_rows, number_of_rows))
    correlation_template_ranges = (int((correlation_template.shape[0] - 1)/2),
                                   int((correlation_template.shape[1] - 1)/2))
    for j in range(len(neighborhood)):
        for i in range(j, len(neighborhood)):
            matrix[j, i] = correlation_template[neighborhood[j][0] - neighborhood[i][0] + correlation_template_ranges[0],
                                                neighborhood[j][1] - neighborhood[i][1] + correlation_template_ranges[1]]
            if j != i:
                matrix[i, j] = matrix[j, i]
            if math.isnan(secondary_data_weight) == False:
                matrix[len(neighborhood) + j, len(neighborhood) + i] = matrix[j, i]
                matrix[len(neighborhood) + j, i] = secondary_data_weight*matrix[j, i]
                matrix[j, len(neighborhood) + i] = secondary_data_weight*matrix[j, i]
                if j != i:
                    matrix[len(neighborhood) + i, len(neighborhood) + j] = matrix[j, i]
                    matrix[i, len(neighborhood) + j] = secondary_data_weight*matrix[j, i]
                    matrix[len(neighborhood) + i, j] = secondary_data_weight*matrix[j, i]
    if math.isnan(secondary_data_weight) == False:
        for i in range(len(neighborhood)):
            correlation = correlation_template[neighborhood[i][0] - cell_index[0] + correlation_template_ranges[0],
                                               neighborhood[i][1] - cell_index[1] + correlation_template_ranges[1]]
            matrix[i, 2*len(neighborhood)] = secondary_data_weight*correlation
            matrix[2*len(neighborhood), i] = secondary_data_weight*correlation
            matrix[len(neighborhood) + i, 2*len(neighborhood)] = correlation
            matrix[2*len(neighborhood), len(neighborhood) + i] = correlation
        matrix[2*len(neighborhood), 2*len(neighborhood)] = 1
    if kriging_method == KrigingMethod.ORDINARY:
        for j in range(number_of_rows - 1):
            matrix[-1, j] = 1.
            matrix[j, -1] = 1.
        matrix[-1, -1] = 0.
    
    return matrix

@jit(nopython = True, nogil = True)
def get_data_to_unknown_matrix(kriging_method,
                               cell_index,
                               neighborhood,
                               correlation_template,
                               secondary_data_weight):
    '''
    Get the matrix of variogram values between the cells with already a value
    and the cell to estimate
    
    @param kriging_method: The kriging method to use (see KrigingMethod)
    @param cell_index: The cell to estimate
    @param neighborhood: The data around the point to estimate
    @param correlation_template: The variogram values in the neighborhood
    @param secondary_data_weight: The weight for the secondary data (math.nan if
                                  no secondary data)
    
    @return The matrix
    '''
    number_of_rows = len(neighborhood)
    if math.isnan(secondary_data_weight) == False:
        number_of_rows += len(neighborhood) + 1
    if kriging_method == KrigingMethod.ORDINARY:
        number_of_rows += 1

    matrix = np.zeros((number_of_rows, 1))
    correlation_template_ranges = (int((correlation_template.shape[0] - 1)/2),
                                   int((correlation_template.shape[1] - 1)/2))
    for i in range(len(neighborhood)):
        matrix[i, 0] = correlation_template[neighborhood[i][0] - cell_index[0] + correlation_template_ranges[0],
                                            neighborhood[i][1] - cell_index[1] + correlation_template_ranges[1]]
        if math.isnan(secondary_data_weight) == False:
            matrix[len(neighborhood) + i, 0] = secondary_data_weight*matrix[i, 0]
    if math.isnan(secondary_data_weight) == False:
        matrix[2*len(neighborhood), 0] = secondary_data_weight
    if kriging_method == KrigingMethod.ORDINARY:
        matrix[-1, 0] = 1.
        
    return matrix

@jit(nopython = True, nogil = True)
def solve_kriging_system(cell_index,
                         neighborhood,
                         simulation_array,
                         primary_mean,
                         primary_variance,
                         correlation_template,
                         secondary_data_weight,
                         secondary_data_mean,
                         secondary_data_array):
    '''
    Get the matrices of the kriging system and solve it
    
    @param cell_index: The cell to estimate
    @param neighborhood: The data around the point to estimate
    @param simulation_array: The simulation grid
    @param primary_mean: Mean for the variable to estimate (if using a simple
                         kriging, math.nan if using an ordinary kriging)
    @param primary_variance: Variance for the variable to estimate
    @param correlation_template: The variogram values in the neighborhood
    @param secondary_data_weight: The weight for the secondary data (math.nan if
                                  no secondary data)
    @param secondary_data_mean: Mean for the secondary data
    @param secondary_data_array: The secondary data
    
    @return The estimation of the mean and variance for the cell
    '''
    kriging_method = KrigingMethod.SIMPLE
    if math.isnan(primary_mean):
        kriging_method = KrigingMethod.ORDINARY
        primary_mean = 0.
        
    data_matrix = get_data_to_data_matrix(kriging_method,
                                          cell_index,
                                          neighborhood,
                                          correlation_template,
                                          secondary_data_weight)
    unknown_matrix = get_data_to_unknown_matrix(kriging_method,
                                                cell_index,
                                                neighborhood,
                                                correlation_template,
                                                secondary_data_weight)
    kriging_weights = np.linalg.solve(data_matrix, unknown_matrix)

    estimated_mean = primary_mean
    estimated_variance = primary_variance
    primary_values = get_values_matrix(neighborhood, simulation_array)
    for i in range(len(neighborhood)):
        estimated_mean += kriging_weights[i, 0]*(primary_values[i, 0] - primary_mean)
        estimated_variance -= kriging_weights[i, 0]*unknown_matrix[i, 0]
        
    if math.isnan(secondary_data_weight) == False:
        secondary_values = get_values_matrix(neighborhood, secondary_data_array)
        for i in range(len(neighborhood)):
            estimated_mean += kriging_weights[len(neighborhood) + i, 0]*(secondary_values[i, 0]
                                                                         - secondary_data_mean)
            estimated_variance -= kriging_weights[len(neighborhood) + i, 0]*unknown_matrix[len(neighborhood) + i, 0]
        estimated_mean += kriging_weights[2*len(neighborhood), 0]*(secondary_data_array[cell_index]
                                                                   - secondary_data_mean)
        estimated_variance -= kriging_weights[2*len(neighborhood), 0]*unknown_matrix[2*len(neighborhood), 0]
    if kriging_method == KrigingMethod.ORDINARY:
        estimated_variance -= kriging_weights[-1, 0]

    return (estimated_mean, estimated_variance)

class KrigingMethod(Enum):
    SIMPLE = 0
    ORDINARY = 1
@jit(nopython = True)
def run_sgs(data_array,
            grid_yx_spacing,
            vario_models,
            vario_sills,
            vario_azimuth,
            vario_ranges,
            number_realizations = 1,
            path_type = PathType.RANDOM,
            kriging_method = KrigingMethod.SIMPLE,
            neighborhood_range = (math.nan, math.nan),
            max_number_data = 12,
            secondary_data_weight = math.nan,
            secondary_data_array = np.empty((1, 1)),
            seed = 100,
            no_data_value = -99999.):
    '''
    Perform a 2D Sequential Gaussian Simulation (SGS). Secondary data are taken
    into account using an intrinsic collocated cokriging
    (see Babak and Deutsch, 2009, doi:10.1016/j.cageo.2008.02.025)
    
    @param data_array: The primary data (no_data_value for the areas to simulate,
                       math.nan for the areas outside of the simulation domain)
    @param grid_yx_spacing: The cell size along each axis (y, x)
    @param vario_models: The models for the variogram
    @param vario_sills: The sills for the variogram
    @param vario_azimuth: The azimuth of the variogram's major axis (in degree)
    @param vario_ranges: The major and minor ranges for the variogram
    @param number_realizations: The number of realizations to simulate
    @param path_type: The type of path, which determines the order through which
                      the cells are visited (see PathType)
    @param kriging_method: The kriging method (see KrigingMethod)
    @param neighborhood_range: The range of the neighborhood, beyond which the
                               cells with already a value are not taken into
                               account (in grid spacing unit)
    @param max_number_data: The maximum number of data in the neighborhood
    @param secondary_data_weight: The weight for the secondary data (math.nan if
                                  no secondary data)
    @param secondary_data_array: The secondary data
    @param seed: The seed
    @param no_data_value: The no-data value, which defines the cell to simulate
    
    @return The simulated array
    '''
    assert (len(data_array.shape) == 2
            and len(grid_yx_spacing) == 2
            and len(neighborhood_range) == 2
            and len(secondary_data_array.shape) == 2), "The arrays must be two-dimensional"
    assert (len(vario_models) == len(vario_sills)
            and len(vario_models) == len(vario_ranges)), "Some parameters for the (nested) variogram are missing"
    for vario_range in vario_ranges:
        assert len(vario_range) == 2, "A major or minor range is missing from the variogram ranges"
    if math.isnan(secondary_data_weight) == False:
        assert -1 <= secondary_data_weight <= 1, "The secondary data weight must be between -1 and 1"
    assert math.isnan(no_data_value) == False, "The no data value cannot be NaN"
        
    random.seed(seed)
    
    primary_mean = 0.
    if kriging_method == KrigingMethod.ORDINARY:
        primary_mean = math.nan
    primary_variance = vario_sills[-1]
    
    if math.isnan(neighborhood_range[0]) and math.isnan(neighborhood_range[1]):
        neighborhood_range = (vario_ranges[-1][0], vario_ranges[-1][1])
    
    vario_azimuth_rad = vario_azimuth*math.pi/180.
    rotation_matrix = np.empty((2, 2))
    rotation_matrix[0, 0] = math.cos(vario_azimuth_rad)
    rotation_matrix[0, 1] = math.sin(vario_azimuth_rad)
    rotation_matrix[1, 0] = math.sin(vario_azimuth_rad)
    rotation_matrix[1, 1] = -math.cos(vario_azimuth_rad)
    
    neighborhood_template, correlation_template = compute_neighborhood_template(neighborhood_range,
                                                                                grid_yx_spacing,
                                                                                vario_models,
                                                                                vario_sills,
                                                                                vario_ranges,
                                                                                vario_azimuth_rad,
                                                                                rotation_matrix)
    
    simulation_array = np.empty((number_realizations, data_array.shape[0], data_array.shape[1]))
    for i_rez in range(number_realizations):
        simulation_array[i_rez] = data_array
        simulation_path = np.arange(0, simulation_array[i_rez].size, 1)
        if path_type == PathType.RANDOM:
            random.shuffle(simulation_path)

        for flattened_index in simulation_path:
            cell_index = unflatten_index(flattened_index, simulation_array[i_rez].shape)

            if simulation_array[i_rez, cell_index[0], cell_index[1]] == no_data_value:
                neighborhood = get_neighborhood(cell_index,
                                                simulation_array[i_rez],
                                                neighborhood_template,
                                                max_number_data,
                                                no_data_value)
                if len(neighborhood) < 2:
                    if math.isnan(secondary_data_weight) == True:
                        simulation_array[i_rez, cell_index[0], cell_index[1]] = random.gauss(0., 1.)
                    else:
                        mean = secondary_data_weight*secondary_data_array[cell_index]
                        variance = 1. - secondary_data_weight**2
                        simulation_array[i_rez, cell_index[0], cell_index[1]] = random.gauss(mean, variance)
                else:
                    estimated_mean, estimated_variance = solve_kriging_system(cell_index,
                                                                              neighborhood,
                                                                              simulation_array[i_rez],
                                                                              primary_mean,
                                                                              primary_variance,
                                                                              correlation_template,
                                                                              secondary_data_weight,
                                                                              0.,
                                                                              secondary_data_array)
                    simulation_array[i_rez, cell_index[0], cell_index[1]] = random.gauss(estimated_mean,
                                                                                         math.sqrt(estimated_variance))
    
    return simulation_array

@jit(nopython = True, nogil=True)
def simulate_sgs_realization(data_array,
                             path_type,
                             primary_mean,
                             primary_variance,
                             neighborhood_template,
                             correlation_template,
                             max_number_data,
                             secondary_data_weight,
                             secondary_data_array,
                             seed,
                             no_data_value):
    '''
    Perform a single 2D Sequential Gaussian Simulation (SGS). Secondary data are
    taken into account using an intrinsic collocated cokriging
    (see Babak and Deutsch, 2009, doi:10.1016/j.cageo.2008.02.025)
    
    @param data_array: The primary data (no_data_value for the areas to simulate,
                       math.nan for the areas outside of the simulation domain)
    @param path_type: The type of path, which determines the order through which
                      the cells are visited (see PathType)
    @param primary_mean: Mean for the variable to estimate (if using a simple
                         kriging, math.nan if using an ordinary kriging)
    @param primary_variance: Variance for the variable to estimate
    @param neighborhood_template: The neighborhood to a center cell, from the
                                  closest to the farthest
    @param correlation_template: The variogram values in the neighborhood
    @param max_number_data: The maximum number of data in the neighborhood
    @param secondary_data_weight: The weight for the secondary data (math.nan if
                                  no secondary data)
    @param secondary_data_array: The secondary data
    @param seed: The seed
    @param no_data_value: The no-data value, which defines the cell to simulate
    
    @return The simulated array
    '''
    random.seed(seed)
    
    simulation_array = np.copy(data_array)

    simulation_path = np.arange(0, simulation_array.size, 1)
    if path_type == PathType.RANDOM:
        random.shuffle(simulation_path)

    for flattened_index in simulation_path:
        cell_index = unflatten_index(flattened_index, simulation_array.shape)

        if simulation_array[cell_index[0], cell_index[1]] == no_data_value:
            neighborhood = get_neighborhood(cell_index,
                                            simulation_array,
                                            neighborhood_template,
                                            max_number_data,
                                            no_data_value)
            if len(neighborhood) < 2:
                if math.isnan(secondary_data_weight) == True:
                    simulation_array[cell_index[0], cell_index[1]] = random.gauss(0., 1.)
                else:
                    mean = secondary_data_weight*secondary_data_array[cell_index]
                    variance = 1. - secondary_data_weight**2
                    simulation_array[cell_index[0], cell_index[1]] = random.gauss(mean, variance)
            else:
                estimated_mean, estimated_variance = solve_kriging_system(cell_index,
                                                                          neighborhood,
                                                                          simulation_array,
                                                                          primary_mean,
                                                                          primary_variance,
                                                                          correlation_template,
                                                                          secondary_data_weight,
                                                                          0.,
                                                                          secondary_data_array)
                simulation_array[cell_index[0], cell_index[1]] = random.gauss(estimated_mean,
                                                                              math.sqrt(estimated_variance))
    
    return simulation_array

@jit(nopython = True, parallel = True, nogil = True)
def run_parallel_sgs(data_array,
                     grid_yx_spacing,
                     vario_models,
                     vario_sills,
                     vario_azimuth,
                     vario_ranges,
                     number_realizations = 1,
                     path_type = PathType.RANDOM,
                     kriging_method = KrigingMethod.SIMPLE,
                     neighborhood_range = (math.nan, math.nan),
                     max_number_data = 12,
                     secondary_data_weight = math.nan,
                     secondary_data_array = np.empty((1, 1)),
                     seed = 100,
                     nb_threads = 4,
                     no_data_value = -99999.):
    '''
    Perform a 2D Sequential Gaussian Simulation (SGS) with the realizations
    simulated in parallel. Secondary data are taken into account using an
    intrinsic collocated cokriging (see Babak and Deutsch, 2009,
    doi:10.1016/j.cageo.2008.02.025)
    
    @param data_array: The primary data (no_data_value for the areas to simulate,
                       math.nan for the areas outside of the simulation domain)
    @param grid_yx_spacing: The cell size along each axis (y, x)
    @param vario_models: The models for the variogram
    @param vario_sills: The sills for the variogram
    @param vario_azimuth: The azimuth of the variogram's major axis (in degree)
    @param vario_ranges: The major and minor ranges for the variogram
    @param number_realizations: The number of realizations to simulate
    @param path_type: The type of path, which determines the order through which
                      the cells are visited (see PathType)
    @param kriging_method: The kriging method (see KrigingMethod)
    @param neighborhood_range: The range of the neighborhood, beyond which the
                               cells with already a value are not taken into
                               account (in grid spacing unit)
    @param max_number_data: The maximum number of data in the neighborhood
    @param secondary_data_weight: The weight for the secondary data (math.nan if
                                  no secondary data)
    @param secondary_data_array: The secondary data
    @param seed: The seed
    @param no_data_value: The no-data value, which defines the cell to simulate
    
    @return The simulated array
    '''
    assert (len(data_array.shape) == 2
            and len(grid_yx_spacing) == 2
            and len(neighborhood_range) == 2
            and len(secondary_data_array.shape) == 2), "The arrays must be two-dimensional"
    assert (len(vario_models) == len(vario_sills)
            and len(vario_models) == len(vario_ranges)), "Some parameters for the (nested) variogram are missing"
    for i in range(len(vario_ranges)):
        assert len(vario_ranges[i]) == 2, "A major or minor range is missing from the variogram ranges"
    if math.isnan(secondary_data_weight) == False:
        assert -1 <= secondary_data_weight <= 1, "The secondary data weight must be between -1 and 1"
    assert math.isnan(no_data_value) == False, "The no data value cannot be NaN"
    
    primary_mean = 0.
    if kriging_method == KrigingMethod.ORDINARY:
        primary_mean = math.nan
    primary_variance = vario_sills[-1]
    
    if math.isnan(neighborhood_range[0]) and math.isnan(neighborhood_range[1]):
        neighborhood_range = (vario_ranges[-1][0], vario_ranges[-1][1])
    
    vario_azimuth_rad = vario_azimuth*math.pi/180.
    rotation_matrix = np.empty((2, 2))
    rotation_matrix[0, 0] = math.cos(vario_azimuth_rad)
    rotation_matrix[0, 1] = math.sin(vario_azimuth_rad)
    rotation_matrix[1, 0] = math.sin(vario_azimuth_rad)
    rotation_matrix[1, 1] = -math.cos(vario_azimuth_rad)
    
    neighborhood_template, correlation_template = compute_neighborhood_template(neighborhood_range,
                                                                                grid_yx_spacing,
                                                                                vario_models,
                                                                                vario_sills,
                                                                                vario_ranges,
                                                                                vario_azimuth_rad,
                                                                                rotation_matrix)
    
    simulation_array = np.empty((number_realizations, data_array.shape[0], data_array.shape[1]))
    for i_rez in prange(number_realizations):
        simulation_array[i_rez] = simulate_sgs_realization(data_array,
                                                           path_type,
                                                           primary_mean,
                                                           primary_variance,
                                                           neighborhood_template,
                                                           correlation_template,
                                                           max_number_data,
                                                           secondary_data_weight,
                                                           secondary_data_array,
                                                           seed + i_rez,
                                                           no_data_value)
    # with concurrent.futures.ThreadPoolExecutor(max_workers = nb_threads) as executor:
    #     future_to_sim = {executor.submit(simulate_sgs_realization,
    #                                      data_array,
    #                                      path_type,
    #                                      primary_mean,
    #                                      primary_variance,
    #                                      neighborhood_template,
    #                                      correlation_template,
    #                                      max_number_data,
    #                                      secondary_data_weight,
    #                                      secondary_data_array,
    #                                      seed + i_rez,
    #                                      no_data_value): i_rez for i_rez in range(number_realizations)}
    #     i_rez = 0
    #     for rez in concurrent.futures.as_completed(future_to_sim):
    #         try:
    #             simulation_array[i_rez] = rez.result()
    #         except Exception as exc:
    #             print('The simulation %d failed' % i_rez)
    #         i_rez += 1
    
    return simulation_array

################################################################################
# Data transform
################################################################################

def inverse_standard_normal_cdf(x):
    '''
    Compute the inverse of a normal cumulative distribution
    
    @param x: A float or array
    
    @return The values normally distributed
    '''
    return math.sqrt(2.)*erfinv(2.*x - 1.)

@jit(nopython = True)
def _compute_averaged_cumulative_distribution_from_array(value_array,
                                                         unique_sorted_array,
                                                         counts_sorted_array):
    '''
    Compute the cumulative probability distribution, where the cumulative
    probability for a data value is the average between its cumulative
    probability and that of the next lower data value
    
    @param value_array: The array containing the data values
    @param unique_sorted_array: Unique values within value_array sorted in ascending order
    @param counts_sorted_array: Counts for the sorted unique values
    
    @return The cumulative probability for each value of the original array
    '''
    assert len(unique_sorted_array.shape) == 1, "The array of unique values must be one-dimensional"
    assert len(counts_sorted_array.shape) == 1, "The array of counts must be one-dimensional"
    
    probability_array = counts_sorted_array/len(np.where(~np.isnan(value_array))[0])
    cumulative_sum_array = np.cumsum(probability_array)
    
    cumulative_value_array = np.full(value_array.shape, math.nan)
    for index in np.ndindex(value_array.shape):
        if math.isnan(value_array[index]) == False:
            value_index = np.searchsorted(unique_sorted_array, value_array[index])
            if value_index < unique_sorted_array.size:
                prev_cum = 0.
                if value_index > 0:
                    prev_cum = cumulative_sum_array[value_index - 1]
                cumulative_value_array[index] = (cumulative_sum_array[value_index] + prev_cum)/2.

    return cumulative_value_array

def compute_averaged_cumulative_distribution_from_array(value_array):
    '''
    Compute the cumulative probability distribution, where the cumulative
    probability for a data value is the average between its cumulative
    probability and that of the next lower data value
    
    @param value_array: The array containing the data values
    
    @return The cumulative probability for each value of the original array
    '''
    sorted_array = np.sort(value_array[~np.isnan(value_array)].ravel())
    unique_sorted_array, counts_sorted_array = np.unique(sorted_array,
                                                         return_counts = True)
    
    return _compute_averaged_cumulative_distribution_from_array(value_array,
                                                                unique_sorted_array,
                                                                counts_sorted_array)

def normal_score_tranform(value_array):
    '''
    Transform the values of an array to a normal distribution
    
    @param value_array: An array
    
    @return An array with its values normally distributed
    '''
    cumulative_value_array = compute_averaged_cumulative_distribution_from_array(value_array)
    return inverse_standard_normal_cdf(cumulative_value_array)