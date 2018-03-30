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
from numba import jit
from numba import prange

from pyinsar.processing.machine_learning.geostatistics.geostatistics_utils import PathType, VariableType, unflatten_index

@jit(nopython = True)
def is_any_inferior(list_1, list_2):
    '''
    Check if any value of a list is strictly inferior to those of another list 
    (or tuple, or 1D NumPy array)
    
    @param list_1: The first list
    @param list_2: The other list
    
    @return True if a value is strictly inferior, false otherwise
    '''
    for i in range(min(len(list_1), len(list_2))):
        if list_1[i] < list_2[i]:
            return True
    return False

@jit(nopython = True)
def compute_neighborhoods(simulation_array,
                          cell_j,
                          cell_i,
                          max_number_data,
                          neighborhood_shape,
                          rotation_angle_rad,
                          scaling_factor,
                          no_data_value):
    '''
    Find the neighborhood of a cell in the simulation grid, i.e., perform a 
    spiral loop from that cell and save the cells that already have a value.
    The search is done for each variable, and each variable can have a different
    neighborhood
    
    @param simulation_array: A NumPy array representing the simulation grid.
                             It should be a 3D array, with one dimension for
                             the variable(s), and two spatial dimensions
    @param cell_j: Index along the y axis of the central cell in the simulation grid
    @param cell_i: Index along the x axis of the central cell in the simulation grid
    @param max_number_data: The maximal number of data inside a neighborhood for
                            each variable
    @param neighborhood_shape: The maximal coverage of the neighborhood along
                               each axis
    @param rotation_angle_rad: The rotation to apply to each neighborhood (in radian)
    @param scaling_factor: The scaling to apply to each neighborhood for each axis
    @param no_data_value: The no-data value, which defines the cell to simulate
    
    @return The indexes of the neighborhood's cells and the corresponding values
    '''
    number_cells = 1
    d = 1
    m = 1
    j = 1
    i = 0
    neighbor_indexes = np.full((simulation_array.shape[0],
                                np.max(max_number_data),
                                2),
                               no_data_value,
                               dtype = np.int64)
    neighbor_values = np.full((simulation_array.shape[0],
                               np.max(max_number_data)),
                              no_data_value,
                              dtype = np.float64)
    loop_indexes = np.zeros(simulation_array.shape[0], dtype = np.int32)
    array_size = simulation_array.shape[1]*simulation_array.shape[2]
    while (is_any_inferior(loop_indexes, max_number_data)
           and number_cells < array_size):
        while (2*j*d < m
               and is_any_inferior(loop_indexes, max_number_data)
               and number_cells < array_size):
            if (0 <= cell_j + j < simulation_array.shape[1]
                and 0 <= cell_i + i < simulation_array.shape[2]):
                if (-neighborhood_shape[0] < j < neighborhood_shape[0]
                    and -neighborhood_shape[1] < i < neighborhood_shape[1]):
                    for k in range(loop_indexes.size):
                        if (loop_indexes[k] < max_number_data[k]
                            and math.isnan(simulation_array[k, j + cell_j, i + cell_i]) == False
                            and simulation_array[k, j + cell_j, i + cell_i] != no_data_value):
                            neighbor_indexes[k, loop_indexes[k], 0] = round((j*math.cos(-rotation_angle_rad)
                                                                             - i*math.sin(-rotation_angle_rad))/scaling_factor[0])
                            neighbor_indexes[k, loop_indexes[k], 1] = round((j*math.sin(-rotation_angle_rad)
                                                                             + i*math.cos(-rotation_angle_rad))/scaling_factor[1])
                            neighbor_values[k, loop_indexes[k]] = simulation_array[k, j + cell_j, i + cell_i]
                            loop_indexes[k] += 1
                number_cells += 1
            j += d
        while (2*i*d < m
               and is_any_inferior(loop_indexes, max_number_data)
               and number_cells < array_size):
            if (0 <= cell_j + j < simulation_array.shape[1]
                and 0 <= cell_i + i < simulation_array.shape[2]):
                if (-neighborhood_shape[0] < j < neighborhood_shape[0]
                    and -neighborhood_shape[1] < i < neighborhood_shape[1]):
                    for k in range(loop_indexes.size):
                        if (loop_indexes[k] < max_number_data[k]
                            and math.isnan(simulation_array[k, j + cell_j, i + cell_i]) == False
                            and simulation_array[k, j + cell_j, i + cell_i] != no_data_value):
                            neighbor_indexes[k, loop_indexes[k], 0] = round((j*math.cos(-rotation_angle_rad)
                                                                             - i*math.sin(-rotation_angle_rad))/scaling_factor[0])
                            neighbor_indexes[k, loop_indexes[k], 1] = round((j*math.sin(-rotation_angle_rad)
                                                                             + i*math.cos(-rotation_angle_rad))/scaling_factor[1])
                            neighbor_values[k, loop_indexes[k]] = simulation_array[k, j + cell_j, i + cell_i]
                            loop_indexes[k] += 1
                number_cells += 1
            i += d
        d *= -1
        m += 1

    return neighbor_indexes, neighbor_values

@jit(nopython = True)
def compute_continuous_distance(training_image_array,
                                ti_j,
                                ti_i,
                                min_ti_value,
                                max_ti_value,
                                neighbor_indexes,
                                neighbor_values,
                                var_k,
#                                 delta,
                                no_data_value):
    '''
    Compute the distance between two neighborhoods for a continuous variable
    
    @param training_image_array: A NumPy array containing the training image,
                                 from which the simulated values are borrowed. 
                                 It should be a 3D array, with one dimension for
                                 the variable(s), and two spatial dimensions
    @param ti_j: Index along the y axis of the initial cell to visit in the
                 training image
    @param ti_i: Index along the x axis of the initial cell to visit in the
                 training image
    @param min_ti_value: Minimal value of the variable
    @param max_ti_value: Maximal value of the variable
    @param neighbor_indexes: Indexes of the neighborhood from the cell to simulate
    @param neighbor_values: Values of the neighborhood in the simulation grid
    @param var_k: Index of the variable
    @param no_data_value: The no-data value, which defines the cell to simulate
    
    @return The distance
    '''
    if (len(neighbor_indexes[var_k]) > 0
        and neighbor_indexes[var_k, 0, 0] == no_data_value):
        return 0.
        
    distance = 0.
    sum_pattern_distance = 0.
    i = 0
    while (i < len(neighbor_indexes[var_k])
           and neighbor_indexes[var_k, i, 0] != no_data_value
           and 0 <= neighbor_indexes[var_k, i, 0] + ti_j < training_image_array[var_k].shape[0]
           and 0 <= neighbor_indexes[var_k, i, 1] + ti_i < training_image_array[var_k].shape[1]
           and math.isnan(training_image_array[var_k,
                                               neighbor_indexes[var_k, i, 0] + ti_j,
                                               neighbor_indexes[var_k, i, 1] + ti_i]) == False
           and training_image_array[var_k,
                                    neighbor_indexes[var_k, i, 0] + ti_j,
                                    neighbor_indexes[var_k, i, 1] + ti_i] != no_data_value):
        
        distance += (neighbor_values[var_k, i]
                     - training_image_array[var_k,
                                            neighbor_indexes[var_k, i, 0] + ti_j,
                                            neighbor_indexes[var_k, i, 1] + ti_i])**2
        sum_pattern_distance += 1
#         alpha = math.sqrt(neighbor_indexes[var_k, i, 0]**2
#                           + neighbor_indexes[var_k, i, 1]**2)**(-delta)
#         distance += alpha*(neighbor_values[var_k, i]
#                            - training_image_array[var_k,
#                                                   neighbor_indexes[var_k, i, 0] + ti_j,
#                                                   neighbor_indexes[var_k, i, 1] + ti_i])**2
#         sum_pattern_distance += math.sqrt(neighbor_indexes[var_k, i, 0]**2
#                                           + neighbor_indexes[var_k, i, 1]**2)**(-delta)

        i += 1
    if (i < len(neighbor_indexes[var_k])
        and neighbor_indexes[var_k, i, 0] != no_data_value):
        return math.inf
    
    return math.sqrt(distance/(sum_pattern_distance*(max_ti_value - min_ti_value)**2))

@jit(nopython = True)
def compute_discrete_distance(training_image_array,
                              ti_j,
                              ti_i,
                              neighbor_indexes,
                              neighbor_values,
                              var_k,
#                               delta,
                              no_data_value):
    '''
    Compute the distance between two neighborhoods for a discrete variable
    
    @param training_image_array: A NumPy array containing the training image,
                                 from which the simulated values are borrowed. 
                                 It should be a 3D array, with one dimension for
                                 the variable(s), and two spatial dimensions
    @param ti_j: Index along the y axis of the initial cell to visit in the
                 training image
    @param ti_i: Index along the x axis of the initial cell to visit in the
                 training image
    @param neighbor_indexes: Indexes of the neighborhood from the cell to simulate
    @param neighbor_values: Values of the neighborhood in the simulation grid
    @param var_k: Index of the variable
    @param no_data_value: The no-data value, which defines the cell to simulate
    
    @return The distance
    '''
    if (len(neighbor_indexes[var_k]) > 0
        and neighbor_indexes[var_k, 0, 0] == no_data_value):
        return 0.
        
    distance = 0.
    sum_pattern_distance = 0.
    i = 0
    while (i < len(neighbor_indexes[var_k])
           and neighbor_indexes[var_k, i, 0] != no_data_value
           and 0 <= neighbor_indexes[var_k, i, 0] + ti_j < training_image_array[var_k].shape[0]
           and 0 <= neighbor_indexes[var_k, i, 1] + ti_i < training_image_array[var_k].shape[1]
           and math.isnan(training_image_array[var_k,
                                               neighbor_indexes[var_k, i, 0] + ti_j,
                                               neighbor_indexes[var_k, i, 1] + ti_i]) == False
           and training_image_array[var_k,
                                    neighbor_indexes[var_k, i, 0] + ti_j,
                                    neighbor_indexes[var_k, i, 1] + ti_i] != no_data_value):
        
        if (neighbor_values[var_k, i] != training_image_array[var_k,
                                                              neighbor_indexes[var_k, i, 0] + ti_j,
                                                              neighbor_indexes[var_k, i, 1] + ti_i]):
            distance += 1
        sum_pattern_distance += 1
#             distance += (math.sqrt(neighbor_indexes[var_k, i, 0]**2
#                                    + neighbor_indexes[var_k, i, 1]**2)**(-delta))
#         sum_pattern_distance += math.sqrt(neighbor_indexes[var_k, i, 0]**2
#                                           + neighbor_indexes[var_k, i, 1]**2)**(-delta)
        
        i += 1
    if (i < len(neighbor_indexes[var_k])
        and neighbor_indexes[var_k, i, 0] != no_data_value):
        return math.inf
            
    return math.sqrt(distance/sum_pattern_distance)

@jit(nopython = True)
def get_value_from_training_image(training_image_array,
                                  ti_j,
                                  ti_i,
                                  minmax_ti_values,
                                  neighbor_indexes,
                                  neighbor_values,
                                  distance_thresholds,
                                  ti_fraction,
#                                   delta,
                                  no_data_value):
    '''
    Find a value in the training image so that the distance between its
    neighborhood and the neighborhood in the simulation grid is lower than a
    threshold
    
    @param training_image_array: A NumPy array containing the training image,
                                 from which the simulated values are borrowed. 
                                 It should be a 3D array, with one dimension for
                                 the variable(s), and two spatial dimensions
    @param ti_j: Index along the y axis of the initial cell to visit in the
                 training image
    @param ti_i: Index along the x axis of the initial cell to visit in the
                 training image
    @param minmax_ti_values: Min and max value of each variable
    @param neighbor_indexes: Indexes of the neighborhood from the cell to simulate
    @param neighbor_values: Values of the neighborhood in the simulation grid
    @param minmax_ti_values: Min and max value of each variable
    @param distance_thresholds: The distance thresholds for each variable
    @param ti_fraction: The maximal fraction of the training image that can be 
                        covered
    @param no_data_value: The no-data value, which defines the cell to simulate
    
    @return The value
    '''
    new_ti_j = ti_j
    new_ti_i = ti_i
    number_cells = 0
    min_error = math.inf
    min_ti_j = ti_j
    min_ti_i = ti_i
    ti_size = training_image_array.shape[1]*training_image_array.shape[2]
    while (number_cells/ti_size < ti_fraction and min_error > 0.):
        if is_any_nan(training_image_array[:, new_ti_j, new_ti_i]) == False:
            error = 0.
            for k in range(training_image_array.shape[0]):
                if math.isnan(minmax_ti_values[k, 0]) == False:
                    distance = compute_continuous_distance(training_image_array,
                                                           new_ti_j,
                                                           new_ti_i,
                                                           minmax_ti_values[k, 0],
                                                           minmax_ti_values[k, 1],
                                                           neighbor_indexes,
                                                           neighbor_values,
                                                           k,
#                                                            delta,
                                                           no_data_value)
                    error += max(0., (distance - distance_thresholds[k])/distance_thresholds[k])
                else:
                    distance = compute_discrete_distance(training_image_array,
                                                         new_ti_j,
                                                         new_ti_i,
                                                         neighbor_indexes,
                                                         neighbor_values,
                                                         k,
#                                                          delta,
                                                         no_data_value)
                    error += max(0., (distance - distance_thresholds[k])/distance_thresholds[k])
            if error < min_error:
                min_error = error
                min_ti_j = new_ti_j
                min_ti_i = new_ti_i
        
        new_ti_i += 1
        if new_ti_i == training_image_array.shape[2]:
            new_ti_i = 0
            new_ti_j += 1
            if new_ti_j == training_image_array.shape[1]:
                new_ti_j = 0
                
        number_cells += 1
        
    return training_image_array[:, min_ti_j, min_ti_i]

@jit(nopython = True)
def get_minmax_array(array, variable_types):
    '''
    Get the min and max values of one or several variables in a 3D NumPy array,
    the first dimension being the variables, the other two the spatial dimensions
    
    @param array: The array
    @param variable_types: The type of variables in the array (i.e., discrete or
                           continuous)
    
    @return The min and max values of each variable. Discrete variables get NaN.
    '''
    minmax_values = np.empty((array.shape[0], 2))
    for k in range(array.shape[0]):
        if variable_types[k] == VariableType.CONTINUOUS:
            minmax_values[k, 0] = math.inf
            minmax_values[k, 1] = -math.inf
            for j in range(array.shape[1]):
                for i in range(array.shape[2]):
                    if array[k, j, i] < minmax_values[k, 0]:
                        minmax_values[k, 0] = array[k, j, i]
                    if array[k, j, i] > minmax_values[k, 1]:
                        minmax_values[k, 1] = array[k, j, i]
        elif variable_types[k] == VariableType.DISCRETE:
            minmax_values[k, 0] = math.nan
            minmax_values[k, 1] = math.nan
                    
    return minmax_values

@jit(nopython = True)
def is_any_equal(list_1, value):
    '''
    Check if there is a given value in a list (or tuple, or 1D NumPy array)
    
    @param list_1: The list
    @param value: The value
    
    @return True if there is the value, false otherwise
    '''
    for i in range(len(list_1)):
        if list_1[i] == value:
            return True
    return False

@jit(nopython = True)
def is_any_nan(list_1):
    '''
    Check if there is any NaN in a list (or tuple, or 1D NumPy array)
    
    @param list_1: The list
    
    @return True if there is a NaN, false otherwise
    '''
    for i in range(len(list_1)):
        if math.isnan(list_1[i]):
            return True
    return False

@jit(nopython = True)
def run_ds(data_array,
           training_image_array,
           variable_types,
           distance_thresholds,
           ti_fraction,
           max_number_data = 10,
           neighborhood_shape = (math.inf, math.inf),
#            delta = 0.,
           rotation_angle_array = np.empty((1, 1)),
           scaling_factor_array = np.empty((1, 1, 1)),
           number_postproc = 0,
           postproc_factor = 1,
           number_realizations = 1,
           path_type = PathType.RANDOM,
           seed = 100,
           no_data_value = -99999):
    '''
    Perform a 2D Mulitple-Point Simulation (SGS) using the Direct Sampling (DS) 
    method (Mariethoz et al., 2010, doi:10.1029/2008WR007621). More information 
    about the parameters and their impact can be found in Meerschman et al. 
    (2012, 10.1016/j.cageo.2012.09.019)
    
    @param data_array: A NumPy array containing the primary data (and 
                       no_data_value for the areas to simulate, math.nan for the
                       areas outside of the simulation domain). Its shape should
                       match that of the training image
    @param training_image_array: A NumPy array containing the training image,
                                 from which the simulated values are borrowed. 
                                 It should be a 3D array, with one dimension for
                                 the variable(s), and two spatial dimensions
    @param variable_types: The type of variables in the array (i.e., discrete or
                           continuous)
    @param distance_thresholds: The distance thresholds for each variable
    @param ti_fraction: The maximal fraction of the training image that can be 
                        covered
    @param max_number_data: The maximal number of data inside a neighborhood for
                            each variable
    @param neighborhood_shape: The maximal coverage of the neighborhood along
                               each axis
    @param rotation_angle_array: A NumPy array describing the rotation to apply
                                 to each neighborhood (in degree)
    @param scaling_factor_array: A NumPy array describing the scaling to apply
                                 to each neighborhood for each axis
    @param number_postproc: The number of postprocessing steps.
    @param postproc_factor: The factor by which ti_fraction and max_number_data
                            are reduced during postprocessing.
    @param number_realizations: The number of realizations to simulate
    @param path_type: The type of path, which determines the order through which
                      the cells are visited (see PathType)
    @param seed: The seed
    @param no_data_value: The no-data value, which defines the cell to simulate
    
    @return The simulated array
    '''
    assert (len(training_image_array.shape) == 3
            and len(data_array.shape) == 3), "The arrays must be three-dimensional"
    assert data_array.shape[0] == training_image_array.shape[0], "The data and training image arrays should have the same number of variables"
    assert (len(variable_types) == training_image_array.shape[0]
            and len(distance_thresholds) == training_image_array.shape[0]
            and len(max_number_data) == training_image_array.shape[0]), "variable_types, distance_thresholds, and max_number_data should match the number of variables in the training image"
    for i in range(len(distance_thresholds)):
        assert (0. < distance_thresholds[i] <= 1.), "Distance threshold(s) must be between (0., 1.]"
        assert max_number_data[i] >= 0, "Number(s) of data must be positive"
    assert len(neighborhood_shape) == 2, "Neighborhood shape must be two-dimensional"
    if rotation_angle_array.shape != data_array.shape[-2:]:
        rotation_angle_array = np.full(data_array.shape[-2:], 0.)
    else:
        rotation_angle_array *= np.pi/180.
    if scaling_factor_array.shape != (2, data_array.shape[-2], data_array.shape[-1]):
        scaling_factor_array = np.full((2, data_array.shape[-2], data_array.shape[-1]), 1.)
    assert math.isnan(no_data_value) == False, "The no data value cannot be NaN"
            
    random.seed(seed)

    minmax_ti_values = get_minmax_array(training_image_array, variable_types)
    
    simulation_array = np.empty((number_realizations,
                                 data_array.shape[0],
                                 data_array.shape[1],
                                 data_array.shape[2]))
    for i_rez in range(number_realizations):
        simulation_array[i_rez] = data_array
        is_postproc = False
        for i_step in range(1 + number_postproc):
            
            factor = 1
            if is_postproc == True:
                factor = postproc_factor
                
            factored_max_number_data = np.empty(len(max_number_data), dtype = np.int64)
            for i in range(len(max_number_data)):
                factored_max_number_data[i] = int(max_number_data[i]/factor)
        
            simulation_path = np.arange(0, simulation_array[i_rez, 0].size, 1)
            if path_type == PathType.RANDOM:
                random.shuffle(simulation_path)

            for flattened_index in simulation_path:
                cell_j, cell_i = unflatten_index(flattened_index, simulation_array[i_rez, 0].shape)
                if (is_any_equal(simulation_array[i_rez, :, cell_j, cell_i], no_data_value) == True
                    or (is_postproc == True
                        and is_any_nan(simulation_array[i_rez, :, cell_j, cell_i]) == False)):
                    neighbor_indexes, neighbor_values = compute_neighborhoods(simulation_array[i_rez],
                                                                              cell_j,
                                                                              cell_i,
                                                                              factored_max_number_data,
                                                                              neighborhood_shape,
                                                                              rotation_angle_array[cell_j, cell_i],
                                                                              scaling_factor_array[:, cell_j, cell_i],
                                                                              no_data_value)
                    ti_j = random.randint(0, training_image_array.shape[-2] - 1)
                    ti_i = random.randint(0, training_image_array.shape[-1] - 1)
                    simulation_array[i_rez,
                                     :,
                                     cell_j,
                                     cell_i] = get_value_from_training_image(training_image_array,
                                                                             ti_j,
                                                                             ti_i,
                                                                             minmax_ti_values,
                                                                             neighbor_indexes,
                                                                             neighbor_values,
                                                                             distance_thresholds,
                                                                             ti_fraction/factor,
#                                                                              delta,
                                                                             no_data_value)
            is_postproc = True
                
    return simulation_array

@jit(nopython = True, nogil = True)
def simulate_ds_realization(data_array,
                            training_image_array,
                            minmax_ti_values,
                            distance_thresholds,
                            ti_fraction,
                            max_number_data,
                            neighborhood_shape,
                            rotation_angle_array,
                            scaling_factor_array,
                            number_postproc,
                            postproc_factor,
                            path_type,
                            seed,
                            no_data_value):
    '''
    Perform a single 2D 2D Mulitple-Point Simulation (SGS) using the Direct
    Sampling (DS) method (Mariethoz et al., 2010, doi:10.1029/2008WR007621).
    More information about the parameters and their impact can be found in
    Meerschman et al. (2012, 10.1016/j.cageo.2012.09.019)
    
    @param data_array: A NumPy array containing the primary data (and 
                       no_data_value for the areas to simulate, math.nan for the
                       areas outside of the simulation domain). Its shape should
                       match that of the training image
    @param training_image_array: A NumPy array containing the training image,
                                 from which the simulated values are borrowed. 
                                 It should be a 3D array, with one dimension for
                                 the variable(s), and two spatial dimensions
    @param minmax_ti_values: Min and max value of each variable
    @param distance_thresholds: The distance thresholds for each variable
    @param ti_fraction: The maximal fraction of the training image that can be 
                        covered
    @param max_number_data: The maximal number of data inside a neighborhood for
                            each variable
    @param neighborhood_shape: The maximal coverage of the neighborhood along
                               each axis
    @param rotation_angle_array: A NumPy array describing the rotation to apply
                                 to each neighborhood (in degree)
    @param scaling_factor_array: A NumPy array describing the scaling to apply
                                 to each neighborhood for each axis
    @param number_postproc: The number of postprocessing steps.
    @param postproc_factor: The factor by which ti_fraction and max_number_data
                            are reduced during postprocessing.
    @param number_realizations: The number of realizations to simulate
    @param path_type: The type of path, which determines the order through which
                      the cells are visited (see PathType)
    @param seed: The seed
    @param no_data_value: The no-data value, which defines the cell to simulate
    
    @return The simulated array
    '''     
    random.seed(seed)
    
    simulation_array = np.copy(data_array)
    is_postproc = False
    for i_step in range(1 + number_postproc):

        factor = 1
        if is_postproc == True:
            factor = postproc_factor

        factored_max_number_data = np.empty(len(max_number_data), dtype = np.int64)
        for i in range(len(max_number_data)):
            factored_max_number_data[i] = int(max_number_data[i]/factor)

        simulation_path = np.arange(0, simulation_array[0].size, 1)
        if path_type == PathType.RANDOM:
            random.shuffle(simulation_path)

        for flattened_index in simulation_path:
            cell_j, cell_i = unflatten_index(flattened_index, simulation_array[0].shape)
            if (is_any_equal(simulation_array[:, cell_j, cell_i], no_data_value) == True
                or (is_postproc == True
                    and is_any_nan(simulation_array[:, cell_j, cell_i]) == False)):
                neighbor_indexes, neighbor_values = compute_neighborhoods(simulation_array,
                                                                          cell_j,
                                                                          cell_i,
                                                                          factored_max_number_data,
                                                                          neighborhood_shape,
                                                                          rotation_angle_array[cell_j, cell_i],
                                                                          scaling_factor_array[:, cell_j, cell_i],
                                                                          no_data_value)
                ti_j = random.randint(0, training_image_array.shape[-2] - 1)
                ti_i = random.randint(0, training_image_array.shape[-1] - 1)
                simulation_array[:,
                                 cell_j,
                                 cell_i] = get_value_from_training_image(training_image_array,
                                                                         ti_j,
                                                                         ti_i,
                                                                         minmax_ti_values,
                                                                         neighbor_indexes,
                                                                         neighbor_values,
                                                                         distance_thresholds,
                                                                         ti_fraction/factor,
                                                                         no_data_value)
        is_postproc = True
                
    return simulation_array

@jit(nopython = True, parallel = True, nogil = True)
def run_parallel_ds(data_array,
                    training_image_array,
                    variable_types,
                    distance_thresholds,
                    ti_fraction,
                    max_number_data = 10,
                    neighborhood_shape = np.full(2, math.inf),
                    rotation_angle_array = np.empty((1, 1)),
                    scaling_factor_array = np.empty((1, 1, 1)),
                    number_postproc = 0,
                    postproc_factor = 1,
                    number_realizations = 1,
                    path_type = PathType.RANDOM,
                    seed = 100,
                    no_data_value = -99999):
    '''
    Perform a 2D Mulitple-Point Simulation (SGS) using the Direct Sampling (DS) 
    method (Mariethoz et al., 2010, doi:10.1029/2008WR007621) with the
    realizations simulated in parallel. More information about the parameters
    and their impact can be found in Meerschman et al.
    (2012, 10.1016/j.cageo.2012.09.019)
    
    @param data_array: A NumPy array containing the primary data (and 
                       no_data_value for the areas to simulate, math.nan for the
                       areas outside of the simulation domain). Its shape should
                       match that of the training image
    @param training_image_array: A NumPy array containing the training image,
                                 from which the simulated values are borrowed. 
                                 It should be a 3D array, with one dimension for
                                 the variable(s), and two spatial dimensions
    @param variable_types: The type of variables in the array (i.e., discrete or
                           continuous)
    @param distance_thresholds: The distance thresholds for each variable
    @param ti_fraction: The maximal fraction of the training image that can be 
                        covered
    @param max_number_data: The maximal number of data inside a neighborhood for
                            each variable
    @param neighborhood_shape: The maximal coverage of the neighborhood along
                               each axis
    @param rotation_angle_array: A NumPy array describing the rotation to apply
                                 to each neighborhood (in degree)
    @param scaling_factor_array: A NumPy array describing the scaling to apply
                                 to each neighborhood for each axis
    @param number_postproc: The number of postprocessing steps.
    @param postproc_factor: The factor by which ti_fraction and max_number_data
                            are reduced during postprocessing.
    @param number_realizations: The number of realizations to simulate
    @param path_type: The type of path, which determines the order through which
                      the cells are visited (see PathType)
    @param seed: The seed
    @param no_data_value: The no-data value, which defines the cell to simulate
    
    @return The simulated array
    '''
    assert (len(training_image_array.shape) == 3
            and len(data_array.shape) == 3), "The arrays must be three-dimensional"
    assert data_array.shape[0] == training_image_array.shape[0], "The data and training image arrays should have the same number of variables"
    assert (len(variable_types) == training_image_array.shape[0]
            and len(distance_thresholds) == training_image_array.shape[0]
            and len(max_number_data) == training_image_array.shape[0]), "variable_types, distance_thresholds, and max_number_data should match the number of variables in the training image"
    for i in range(len(distance_thresholds)):
        assert (0. < distance_thresholds[i] <= 1.), "Distance threshold(s) must be between (0., 1.]"
        assert max_number_data[i] >= 0, "Number(s) of data must be positive"
    assert len(neighborhood_shape) == 2, "Neighborhood shape must be two-dimensional"
    if rotation_angle_array.shape != data_array.shape[-2:]:
        rotation_angle_array = np.full(data_array.shape[-2:], 0.)
    else:
        rotation_angle_array *= np.pi/180.
    if scaling_factor_array.shape != (2, data_array.shape[-2], data_array.shape[-1]):
        scaling_factor_array = np.full((2, data_array.shape[-2], data_array.shape[-1]), 1.)
    assert math.isnan(no_data_value) == False, "The no data value cannot be NaN"
            
    random.seed(seed)

    minmax_ti_values = get_minmax_array(training_image_array, variable_types)
    
    simulation_array = np.empty((number_realizations,
                                 data_array.shape[0],
                                 data_array.shape[1],
                                 data_array.shape[2]))
    for i_rez in prange(number_realizations):
        simulation_array[i_rez] = simulate_ds_realization(data_array,
                                                          training_image_array,
                                                          minmax_ti_values,
                                                          distance_thresholds,
                                                          ti_fraction,
                                                          max_number_data,
                                                          neighborhood_shape,
                                                          rotation_angle_array,
                                                          scaling_factor_array,
                                                          number_postproc,
                                                          postproc_factor,
                                                          path_type,
                                                          seed + i_rez,
                                                          no_data_value)
    
    return simulation_array