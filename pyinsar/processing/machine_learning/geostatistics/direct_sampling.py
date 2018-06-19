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
def compute_neighborhood_lag_vectors(neighborhood_shape,
                                     grid_yx_spacing,
                                     delta):
    '''
    Compute the lag vectors for the neighborhood, assuming a regular grid
    
    @param neighborhood_shape: The maximal coverage of the neighborhood along
                               each axis
    @param grid_yx_spacing: The cell size along each axis (y, x)
    @param delta: A weight for the neighboring cells during simulation, a high
                  delta giving more influence to the cells closer to the cell to
                  simulate
    
    @return The closest cells to the center cell of the neighborhood and the
            corresponding weighted distance
    '''
    # Sorting in Numba does not take any arguments, so to sort a multidimensional
    # array we first create a list of tuples, and then create the array. This 
    # will need to be changed
    lag_vectors = []
    for j in range(-2*neighborhood_shape[0], 2*neighborhood_shape[0] + 1):
        delta_y = j*grid_yx_spacing[0]
        for i in range(-2*neighborhood_shape[1], 2*neighborhood_shape[1] + 1):
            if j != 0 or i != 0:
                delta_x = i*grid_yx_spacing[1]
                distance = math.sqrt(delta_y**2 + delta_x**2)
                lag_vectors.append((distance, j, i))
                
    sorted_lag_vectors = sorted(lag_vectors)
    lag_vectors = np.empty((int((2*neighborhood_shape[0] + 1)*(2*neighborhood_shape[1] + 1)), 2),
                           dtype = np.int64)
    lag_distances = np.empty(int((2*neighborhood_shape[0] + 1)*(2*neighborhood_shape[1] + 1)))
    for i in range((2*neighborhood_shape[0] + 1)*(2*neighborhood_shape[1] + 1)):
        lag_vectors[i, 0] = int(sorted_lag_vectors[i][1])
        lag_vectors[i, 1] = int(sorted_lag_vectors[i][2])
        lag_distances[i] = sorted_lag_vectors[i][0]**(-delta)

    return lag_vectors, lag_distances

@jit(nopython = True)
def compute_neighborhoods(simulation_array,
                          data_weight_array,
                          cell_j,
                          cell_i,
                          lag_vectors,
                          lag_distances,
                          max_number_data,
                          max_density_data,
                          neighborhood_shape,
                          rotation_angle_rad,
                          scaling_factor,
                          no_data_value):
    '''
    Find the neighborhood of a cell in the simulation grid, i.e., go through
    the pre-computed lag vectors and find the cell with already a value.
    The search is done for each variable, and each variable can have a different
    neighborhood
    
    @param simulation_array: A NumPy array representing the simulation grid.
                             It should be a 3D array, with one dimension for
                             the variable(s), and two spatial dimensions
    @param data_weight_array: A NumPy array representing the conditioning weight.
                              It should be a 3D array, with one dimension for
                              the variable(s), and two spatial dimensions
    @param cell_j: Index along the y axis of the central cell in the simulation grid
    @param cell_i: Index along the x axis of the central cell in the simulation grid
    @param lag_vectors: Index of the closest cells relative to the central cell
    @param lag_distances: Weighted distance of the closest cells relative to the
                          central cell
    @param max_number_data: The maximal number of data inside a neighborhood for
                            each variable
    @param max_density_data: The maximal density of data inside a neighborhood
                             for each variable
    @param neighborhood_shape: The maximal coverage of the neighborhood along
                               each axis
    @param rotation_angle_rad: The rotation to apply to each neighborhood (in radian)
    @param scaling_factor: The scaling to apply to each neighborhood for each axis
    @param no_data_value: The no-data value, which defines the cell to simulate
    
    @return The indexes of the neighborhood's cells; the corresponding values,
            the weighted distances, and the conditioning weights; and the neighbor
            size for each variable
    ''' 
    pre_neighbor_indexes = np.full((simulation_array.shape[0],
                                    np.max(max_number_data),
                                    2),
                                   no_data_value,
                                   dtype = np.int64)
    pre_neighbor_distances = np.full((simulation_array.shape[0],
                                      np.max(max_number_data)),
                                     no_data_value,
                                     dtype = np.float64)
    loop_indexes = np.zeros(simulation_array.shape[0], dtype = np.int32)
    i_neighbor = 0
    number_cells = 0
    neighborhood_size = neighborhood_shape[0]*neighborhood_shape[1]
    for k in range(loop_indexes.size):
        while (loop_indexes[k] < max_number_data[k]
               and i_neighbor < lag_vectors.shape[0]
               and number_cells < neighborhood_size):
            j = lag_vectors[i_neighbor, 0]
            i = lag_vectors[i_neighbor, 1]
            if (0 <= cell_j + j < simulation_array.shape[1]
                and 0 <= cell_i + i < simulation_array.shape[2]):
                if (simulation_array[k, cell_j + j, cell_i + i] != no_data_value
                    and math.isnan(simulation_array[k, cell_j + j, cell_i + i]) == False):
                    pre_neighbor_indexes[k, loop_indexes[k], 0] = j
                    pre_neighbor_indexes[k, loop_indexes[k], 1] = i
                    pre_neighbor_distances[k, loop_indexes[k]] = lag_distances[i_neighbor]
                    loop_indexes[k] += 1
                number_cells += 1
            i_neighbor += 1
        
    # TODO: always get the hard data in the data event, even when a density smaller than 1 is used
    neighbor_indexes = np.full(pre_neighbor_indexes.shape,
                               no_data_value,
                               dtype = np.int64)
    neighbor_values = np.full((pre_neighbor_indexes.shape[0],
                               pre_neighbor_indexes.shape[1],
                               3),
                              no_data_value,
                              dtype = np.float64)
    for k in range(pre_neighbor_indexes.shape[0]):
        step = 1
        if loop_indexes[k]/max_number_data[k] > max_density_data[k]:
            step = int(1/max_density_data[k])
            loop_indexes[k] = int(loop_indexes[k]*max_density_data[k])
        for i in range(loop_indexes[k]):
            neighbor_indexes[k, i, 0] = round((pre_neighbor_indexes[k, i*step, 0]*math.cos(-rotation_angle_rad)
                                               - pre_neighbor_indexes[k, i*step, 1]*math.sin(-rotation_angle_rad))/scaling_factor[0])
            neighbor_indexes[k, i, 1] = round((pre_neighbor_indexes[k, i*step, 0]*math.sin(-rotation_angle_rad)
                                               + pre_neighbor_indexes[k, i*step, 1]*math.cos(-rotation_angle_rad))/scaling_factor[1])
            neighbor_values[k, i, 0] = simulation_array[k,
                                                        pre_neighbor_indexes[k, i*step, 0] + cell_j,
                                                        pre_neighbor_indexes[k, i*step, 1] + cell_i]
            neighbor_values[k, i, 1] = pre_neighbor_distances[k, i*step]
            neighbor_values[k, i, 2] = data_weight_array[k,
                                                         pre_neighbor_indexes[k, i*step, 0] + cell_j,
                                                         pre_neighbor_indexes[k, i*step, 1] + cell_i]

    return neighbor_indexes, neighbor_values, loop_indexes

@jit(nopython = True)
def compute_continuous_distance(training_image_array,
                                ti_j,
                                ti_i,
                                ti_ranges_max,
                                neighbor_indexes,
                                neighbor_values,
                                neighbor_numbers,
                                min_distances,
                                var_k,
                                max_non_matching_proportion,
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
    @param ti_ranges_max: Squared difference between the min and max value of 
                          each variable
    @param neighbor_indexes: Indexes of the neighborhood from the cell to simulate
    @param neighbor_values: Values of the neighborhood in the simulation grid
    @param neighbor_numbers: Number of neighbors for each variable
    @param min_distances: The minimal distance of each variable found so far
    @param var_k: Index of the variable
    @param max_non_matching_proportion: Authorized proportion of non-matching
                                        nodes, i.e., whose distance is below the
                                        threshold for the variable
    @param no_data_value: The no-data value, which defines the cell to simulate
    
    @return The distance
    '''
    if neighbor_numbers[var_k] == 0:
        return 0.
        
    distance = 0.
    sum_pattern_distance = 0.
    non_matching_proportion = 0.
    i = 0
    while (i < neighbor_numbers[var_k]
           and 0 <= neighbor_indexes[var_k, i, 0] + ti_j < training_image_array.shape[1]
           and 0 <= neighbor_indexes[var_k, i, 1] + ti_i < training_image_array.shape[2]
           and non_matching_proportion < max_non_matching_proportion
           and math.isnan(training_image_array[var_k,
                                               neighbor_indexes[var_k, i, 0] + ti_j,
                                               neighbor_indexes[var_k, i, 1] + ti_i]) == False
           and training_image_array[var_k,
                                    neighbor_indexes[var_k, i, 0] + ti_j,
                                    neighbor_indexes[var_k, i, 1] + ti_i] != no_data_value):
        
        distance += neighbor_values[var_k, i, 2]*neighbor_values[var_k, i, 1]*(neighbor_values[var_k, i, 0]
                                                                               - training_image_array[var_k,
                                                                                                      neighbor_indexes[var_k, i, 0] + ti_j,
                                                                                                      neighbor_indexes[var_k, i, 1] + ti_i])**2
        sum_pattern_distance += neighbor_values[var_k, i, 1]
        if math.sqrt(distance/(sum_pattern_distance*ti_ranges_max[var_k])) > min_distances[var_k]:
            non_matching_proportion += 1./neighbor_numbers[var_k]

        i += 1
    if i < neighbor_numbers[var_k]:
        return math.inf
    
    return math.sqrt(distance/(sum_pattern_distance*ti_ranges_max[var_k]))

@jit(nopython = True)
def compute_discrete_distance(training_image_array,
                              ti_j,
                              ti_i,
                              neighbor_indexes,
                              neighbor_values,
                              neighbor_numbers,
                              min_distances,
                              var_k,
                              max_non_matching_proportion,
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
    @param neighbor_numbers: Number of neighbors for each variable
    @param min_distances: The minimal distance of each variable found so far
    @param var_k: Index of the variable
    @param max_non_matching_proportion: Authorized proportion of non-matching
                                        nodes, i.e., whose distance is below the
                                        threshold for the variable
    @param no_data_value: The no-data value, which defines the cell to simulate
    
    @return The distance
    '''
    if neighbor_numbers[var_k] == 0:
        return 0.
        
    distance = 0.
    sum_pattern_distance = 0.
    non_matching_proportion = 0.
    i = 0
    while (i < neighbor_numbers[var_k] 
           and 0 <= neighbor_indexes[var_k, i, 0] + ti_j < training_image_array.shape[1]
           and 0 <= neighbor_indexes[var_k, i, 1] + ti_i < training_image_array.shape[2]
           and non_matching_proportion/neighbor_numbers[var_k] < max_non_matching_proportion
           and math.isnan(training_image_array[var_k,
                                               neighbor_indexes[var_k, i, 0] + ti_j,
                                               neighbor_indexes[var_k, i, 1] + ti_i]) == False
           and training_image_array[var_k,
                                    neighbor_indexes[var_k, i, 0] + ti_j,
                                    neighbor_indexes[var_k, i, 1] + ti_i] != no_data_value):

        if (neighbor_values[var_k, i, 0] != training_image_array[var_k,
                                                                 neighbor_indexes[var_k, i, 0] + ti_j,
                                                                 neighbor_indexes[var_k, i, 1] + ti_i]):
            distance += neighbor_values[var_k, i, 2]*neighbor_values[var_k, i, 1]
        sum_pattern_distance += neighbor_values[var_k, i, 1]
        if distance/sum_pattern_distance > min_distances[var_k]:
            non_matching_proportion += 1.
        
        i += 1
        
    if i < neighbor_numbers[var_k]:
        return math.inf
            
    return distance/sum_pattern_distance

@jit(nopython = True)
def find_closest_cell_in_training_image(training_image_array,
                                        ti_ranges_max,
                                        ti_indices,
                                        ti_index,
                                        neighbor_indexes,
                                        neighbor_values,
                                        neighbor_numbers,
                                        distance_thresholds,
                                        max_non_matching_proportion,
                                        ti_fraction,
                                        no_data_value):
    '''
    Find a cell in the training image so that the distance between its
    neighborhood and the neighborhood in the simulation grid is lower than a
    threshold
    
    @param training_image_array: A NumPy array containing the training image,
                                 from which the simulated values are borrowed. 
                                 It should be a 3D array, with one dimension for
                                 the variable(s), and two spatial dimensions
    @param ti_indices: List of indices of the training image's cells
    @param ti_index: Index of the initial cell to visit in the list of training image's indices
    @param ti_ranges_max: Squared difference between the min and max value of 
                          each variable
    @param neighbor_indexes: Indexes of the neighborhood from the cell to simulate
    @param neighbor_values: Values of the neighborhood in the simulation grid
    @param neighbor_numbers: Number of neighbors for each variable
    @param distance_thresholds: The distance thresholds for each variable
    @param max_non_matching_proportion: Authorized proportion of non-matching
                                        nodes, i.e., whose distance is below the
                                        threshold for the variable
    @param ti_fraction: The maximal fraction of the training image that can be 
                        covered
    @param no_data_value: The no-data value, which defines the cell to simulate
    
    @return The index of the cell
    '''
    new_ti_index = ti_index
    number_indices = 0
    number_valid_cells = 0
    min_error = math.inf
    min_distances = np.full(distance_thresholds.shape, 2.)
    distances = np.copy(min_distances)
    min_ti_index = ti_index
    ti_size = len(ti_indices)
    while (min_error > 0.
           and number_valid_cells/ti_size < ti_fraction
           and number_indices < ti_size):
        error = 0.
        for k in range(training_image_array.shape[0]):
            if math.isnan(ti_ranges_max[k]) == False:
                distances[k] = compute_continuous_distance(training_image_array,
                                                           ti_indices[new_ti_index][0],
                                                           ti_indices[new_ti_index][1],
                                                           ti_ranges_max,
                                                           neighbor_indexes,
                                                           neighbor_values,
                                                           neighbor_numbers,
                                                           min_distances,
                                                           k,
                                                           max_non_matching_proportion,
                                                           no_data_value)
                error += max(0., (distances[k] - distance_thresholds[k])/distance_thresholds[k])
            else:
                distances[k] = compute_discrete_distance(training_image_array,
                                                         ti_indices[new_ti_index][0],
                                                         ti_indices[new_ti_index][1],
                                                         neighbor_indexes,
                                                         neighbor_values,
                                                         neighbor_numbers,
                                                         min_distances,
                                                         k,
                                                         max_non_matching_proportion,
                                                         no_data_value)
                error += max(0., (distances[k] - distance_thresholds[k])/distance_thresholds[k])
        if error < min_error:
            min_error = error
            min_distances = np.copy(distances)
            min_ti_index = new_ti_index
            number_valid_cells += 1
        elif math.isinf(error) == False:
            number_valid_cells += 1
        
        new_ti_index += 1
        if new_ti_index == ti_size:
            new_ti_index = 0
  
        number_indices += 1
    
    return ti_indices[min_ti_index]

@jit(nopython = True)
def prepare_training_image(array, variable_types):
    '''
    Get the squared difference between the min and max values of one or several
    variables in a 3D NumPy array, the first dimension being the variables, the
    other two the spatial dimensions. In addition, get the list of training
    image's indices
    
    @param array: The array
    @param variable_types: The type of variables in the array (i.e., discrete or
                           continuous)
    
    @return The squared difference between the min and max values of each variable.
            Discrete variables get NaN. And the list of indices.
    '''
    ranges_max = np.empty(array.shape[0])
    min_values = np.full(array.shape[0], math.inf)
    max_values = np.full(array.shape[0], -math.inf)
    training_image_indices = []
    for j in range(array.shape[1]):
        for i in range(array.shape[2]):
            are_nan = 0
            for k in range(array.shape[0]):
                if math.isnan(array[k, j, i]) == False:
                    if array[k, j, i] < min_values[k]:
                        min_values[k] = array[k, j, i]
                    if array[k, j, i] > max_values[k]:
                        max_values[k] = array[k, j, i]
                else:
                    are_nan += 1
            if are_nan == 0:
                training_image_indices.append((j, i))

    for k in range(array.shape[0]):
        if variable_types[k] == VariableType.CONTINUOUS:
            ranges_max[k] = (max_values[k] - min_values[k])**2
        elif variable_types[k] == VariableType.DISCRETE:
            ranges_max[k] = math.nan
                    
    return ranges_max, training_image_indices

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
           max_number_data,
           max_density_data,
           neighborhood_shape = (math.inf, math.inf),
           grid_yx_spacing = (1., 1.),
           delta = 0.,
           conditioning_data_weight = 1.,
           max_non_matching_proportion = 1.,
           start_parameter_reduction = 1,
           reduction_factor = 1,
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
    @param max_density_data: The maximal density of data inside a neighborhood
                             for each variable
    @param neighborhood_shape: The maximal coverage of the neighborhood along
                               each axis
    @param grid_yx_spacing: The cell size along each axis (y, x), only useful for
                            an anisotropic grid
    @param delta: A weight for the neighboring cells during simulation, a high
                  delta giving more influence to the cells closer to the cell to
                  simulate
    @param conditioning_data_weight: A weight to favor the reproduction of the
                                     patterns around conditioning data
    @param max_non_matching_proportion: Authorized proportion of non-matching
                                        nodes, i.e., whose distance is below the
                                        threshold for the variable
    @param start_parameter_reduction: When to start parameter reduction during
                                      the simulation, 0 being immediately, 1
                                      being never
    @param reduction_factor: Reduction factor for the distance thresholds, TI 
                             fraction and max number of data during parameter
                             reduction
    @param rotation_angle_array: A NumPy array describing the rotation to apply
                                 to each neighborhood (in degree)
    @param scaling_factor_array: A NumPy array describing the scaling to apply
                                 to each neighborhood for each axis
    @param number_postproc: The number of post-processing steps
    @param postproc_factor: The factor by which ti_fraction and max_number_data
                            are reduced during post-processing
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
            and len(max_number_data) == training_image_array.shape[0]
            and len(max_density_data) == training_image_array.shape[0]), "variable_types, distance_thresholds, max_number_data, and max_density_data should match the number of variables in the training image"
    for i in range(len(distance_thresholds)):
        assert (0. < distance_thresholds[i] <= 1.), "Distance threshold(s) must be between (0., 1.]"
        assert max_number_data[i] >= 0, "Number(s) of data must be positive"
        assert 0. < max_density_data[i] <= 1., "Density(ies) of data must be between 0 and 1"
    assert len(neighborhood_shape) == 2, "Neighborhood shape must be two-dimensional"
    assert 0. < max_non_matching_proportion <= 1., "Non-matching proportion must be between 0 and 1"
    assert 0. < start_parameter_reduction <= 1., "Start parameter reduction must be between 0 and 1"
    if rotation_angle_array.shape != data_array.shape[-2:]:
        rotation_angle_array = np.full(data_array.shape[-2:], 0.)
    else:
        rotation_angle_array = rotation_angle_array*np.pi/180.
    if scaling_factor_array.shape != (2, data_array.shape[-2], data_array.shape[-1]):
        scaling_factor_array = np.full((2, data_array.shape[-2], data_array.shape[-1]), 1.)
    assert math.isnan(no_data_value) == False, "The no data value cannot be NaN"
            
    random.seed(seed)

    ti_ranges_max, ti_indices = prepare_training_image(training_image_array, variable_types)

    neighborhood_shape = (min(data_array.shape[1], neighborhood_shape[0]),
                          min(data_array.shape[2], neighborhood_shape[1]))
    lag_vectors, lag_distances = compute_neighborhood_lag_vectors(neighborhood_shape,
                                                                  grid_yx_spacing,
                                                                  delta)
    
    data_weight_array = np.ones(data_array.shape)
    for index in np.ndindex(data_weight_array.shape):
        if (data_array[index] != no_data_value
            and math.isnan(data_array[index]) == False):
            data_weight_array[index] = conditioning_data_weight
    
    simulation_array = np.empty((number_realizations,
                                 data_array.shape[0],
                                 data_array.shape[1],
                                 data_array.shape[2]))
    for i_rez in range(number_realizations):
        simulation_array[i_rez] = data_array
        is_postproc = False
        for i_step in range(1 + number_postproc):
            
            factor = 1.
            if is_postproc == True:
                factor = postproc_factor
            
            factored_ti_fraction = ti_fraction/factor
            factored_max_number_data = np.empty(len(max_number_data), dtype = np.int64)
            factored_distance_thresholds = np.empty(len(max_number_data))
            for i in range(len(max_number_data)):
                factored_max_number_data[i] = int(max_number_data[i]/factor)
                factored_distance_thresholds[i] = distance_thresholds[i]
        
            simulation_path = np.arange(0, simulation_array[i_rez, 0].size, 1)
            if path_type == PathType.RANDOM:
                random.shuffle(simulation_path)
                
            is_reduced = False

            for i_cell in range(len(simulation_path)):
                
                if is_reduced == False and i_cell/len(simulation_path) > start_parameter_reduction:
                    factored_ti_fraction /= reduction_factor
                    for i in range(factored_max_number_data.size):
                        factored_max_number_data[i] = int(factored_max_number_data[i]/reduction_factor)
                        factored_distance_thresholds[i] = min(factored_distance_thresholds[i]*reduction_factor, 1.)
                    is_reduced = True
                    
                cell_j, cell_i = unflatten_index(simulation_path[i_cell], simulation_array[i_rez, 0].shape)
                if (is_any_equal(simulation_array[i_rez, :, cell_j, cell_i], no_data_value) == True
                    or (is_postproc == True
                        and is_any_equal(data_array[:, cell_j, cell_i], no_data_value) == True
                        and is_any_nan(simulation_array[i_rez, :, cell_j, cell_i]) == False)):
                    neighbor_indexes, neighbor_values, neighbor_numbers = compute_neighborhoods(simulation_array[i_rez],
                                                                                                data_weight_array,
                                                                                                cell_j,
                                                                                                cell_i,
                                                                                                lag_vectors,
                                                                                                lag_distances,
                                                                                                factored_max_number_data,
                                                                                                max_density_data,
                                                                                                neighborhood_shape,
                                                                                                rotation_angle_array[cell_j, cell_i],
                                                                                                scaling_factor_array[:, cell_j, cell_i],
                                                                                                no_data_value)
                    ti_index = random.randrange(0, len(ti_indices))
                    ti_j, ti_i = find_closest_cell_in_training_image(training_image_array,
                                                                     ti_ranges_max,
                                                                     ti_indices,
                                                                     ti_index,
                                                                     neighbor_indexes,
                                                                     neighbor_values,
                                                                     neighbor_numbers,
                                                                     factored_distance_thresholds,
                                                                     max_non_matching_proportion,
                                                                     factored_ti_fraction,
                                                                     no_data_value)
                    for i_val in range(simulation_array.shape[1]):
                        if data_array[i_val, cell_j, cell_i] == no_data_value:
                            simulation_array[i_rez,
                                             i_val,
                                             cell_j,
                                             cell_i] = training_image_array[i_val,
                                                                            ti_j,
                                                                            ti_i]
            is_postproc = True
                
    return simulation_array

@jit(nopython = True, nogil = True)
def simulate_ds_realization(data_array,
                            data_weight_array,
                            training_image_array,
                            ti_ranges_max,
                            ti_indices,
                            distance_thresholds,
                            ti_fraction,
                            max_number_data,
                            max_density_data,
                            lag_vectors,
                            lag_distances,
                            neighborhood_shape,
                            max_non_matching_proportion,
                            start_parameter_reduction,
                            reduction_factor,
                            rotation_angle_array,
                            scaling_factor_array,
                            number_postproc,
                            postproc_factor,
                            path_type,
                            seed,
                            no_data_value):
    '''
    Perform a single 2D Mulitple-Point Simulation (MPS) using the Direct
    Sampling (DS) method (Mariethoz et al., 2010, doi:10.1029/2008WR007621).
    More information about the parameters and their impact can be found in
    Meerschman et al. (2012, 10.1016/j.cageo.2012.09.019)
    
    @param data_array: A NumPy array containing the primary data (and 
                       no_data_value for the areas to simulate, math.nan for the
                       areas outside of the simulation domain). Its shape should
                       match that of the training image
    @param data_weight_array: A NumPy array representing the conditioning weight.
                              It should be a 3D array, with one dimension for
                              the variable(s), and two spatial dimensions
    @param training_image_array: A NumPy array containing the training image,
                                 from which the simulated values are borrowed. 
                                 It should be a 3D array, with one dimension for
                                 the variable(s), and two spatial dimensions
    @param ti_ranges_max: Squared difference between the min and max value of 
                          each variable
    @param distance_thresholds: The distance thresholds for each variable
    @param ti_fraction: The maximal fraction of the training image that can be 
                        covered
    @param max_number_data: The maximal number of data inside a neighborhood for
                            each variable
    @param max_density_data: The maximal density of data inside a neighborhood
                             for each variable
    @param lag_vectors: Index of the closest cells relative to the central cell
    @param lag_distances: Weighted distance of the closest cells relative to the
                          central cell
    @param neighborhood_shape: The maximal coverage of the neighborhood along
                               each axis
    @param max_non_matching_proportion: Authorized proportion of non-matching
                                        nodes, i.e., whose distance is below the
                                        threshold for the variable
    @param start_parameter_reduction: When to start parameter reduction during
                                      the simulation, 0 being immediately, 1
                                      being never
    @param reduction_factor: Reduction factor for the distance thresholds, TI 
                             fraction and max number of data during parameter
                             reduction
    @param rotation_angle_array: A NumPy array describing the rotation to apply
                                 to each neighborhood (in degree)
    @param scaling_factor_array: A NumPy array describing the scaling to apply
                                 to each neighborhood for each axis
    @param number_postproc: The number of post-processing steps
    @param postproc_factor: The factor by which ti_fraction and max_number_data
                            are reduced during post-processing
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

        factor = 1.
        if is_postproc == True:
            factor = postproc_factor
        
        factored_ti_fraction = ti_fraction/factor
        factored_max_number_data = np.empty(len(max_number_data), dtype = np.int64)
        factored_distance_thresholds = np.empty(len(max_number_data))
        for i in range(len(max_number_data)):
            factored_max_number_data[i] = int(max_number_data[i]/factor)
            factored_distance_thresholds[i] = distance_thresholds[i]
    
        simulation_path = np.arange(0, simulation_array[0].size, 1)
        if path_type == PathType.RANDOM:
            random.shuffle(simulation_path)
            
        is_reduced = False

        for i_cell in range(len(simulation_path)):
            
            if is_reduced == False and i_cell/len(simulation_path) > start_parameter_reduction:
                factored_ti_fraction /= reduction_factor
                for i in range(factored_max_number_data.size):
                    factored_max_number_data[i] = int(factored_max_number_data[i]/reduction_factor)
                    factored_distance_thresholds[i] = min(factored_distance_thresholds[i]*reduction_factor, 1.)
                is_reduced = True
                
            cell_j, cell_i = unflatten_index(simulation_path[i_cell], simulation_array[0].shape)
            if (is_any_equal(simulation_array[:, cell_j, cell_i], no_data_value) == True
                or (is_postproc == True
                    and is_any_equal(data_array[:, cell_j, cell_i], no_data_value) == True
                    and is_any_nan(simulation_array[:, cell_j, cell_i]) == False)):
                neighbor_indexes, neighbor_values, neighbor_numbers = compute_neighborhoods(simulation_array,
                                                                                            data_weight_array,
                                                                                            cell_j,
                                                                                            cell_i,
                                                                                            lag_vectors,
                                                                                            lag_distances,
                                                                                            factored_max_number_data,
                                                                                            max_density_data,
                                                                                            neighborhood_shape,
                                                                                            rotation_angle_array[cell_j, cell_i],
                                                                                            scaling_factor_array[:, cell_j, cell_i],
                                                                                            no_data_value)
                ti_index = random.randrange(0, len(ti_indices))
                ti_j, ti_i = find_closest_cell_in_training_image(training_image_array,
                                                                 ti_ranges_max,
                                                                 ti_indices,
                                                                 ti_index,
                                                                 neighbor_indexes,
                                                                 neighbor_values,
                                                                 neighbor_numbers,
                                                                 factored_distance_thresholds,
                                                                 max_non_matching_proportion,
                                                                 factored_ti_fraction,
                                                                 no_data_value)
                for i_val in range(simulation_array.shape[0]):
                    if data_array[i_val, cell_j, cell_i] == no_data_value:
                        simulation_array[i_val,
                                         cell_j,
                                         cell_i] = training_image_array[i_val,
                                                                        ti_j,
                                                                        ti_i]
        is_postproc = True
                
    return simulation_array

@jit(nopython = True, parallel = True, nogil = True)
def run_parallel_ds(data_array,
                    training_image_array,
                    variable_types,
                    distance_thresholds,
                    ti_fraction,
                    max_number_data,
                    max_density_data,
                    neighborhood_shape = (math.inf, math.inf),
                    grid_yx_spacing = (1., 1.),
                    delta = 0.,
                    conditioning_data_weight = 1.,
                    max_non_matching_proportion = 1.,
                    start_parameter_reduction = 1,
                    reduction_factor = 1,
                    rotation_angle_array = np.empty((1, 1)),
                    scaling_factor_array = np.empty((1, 1, 1)),
                    number_postproc = 0,
                    postproc_factor = 1,
                    number_realizations = 1,
                    path_type = PathType.RANDOM,
                    seed = 100,
                    no_data_value = -99999):
    '''
    Perform a 2D Mulitple-Point Simulation (MPS) using the Direct Sampling (DS) 
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
    @param max_density_data: The maximal density of data inside a neighborhood
                             for each variable
    @param neighborhood_shape: The maximal coverage of the neighborhood along
                               each axis
    @param grid_yx_spacing: The cell size along each axis (y, x), only useful for
                            an anisotropic grid
    @param delta: A weight for the neighboring cells during simulation, a high
                  delta giving more influence to the cells closer to the cell to
                  simulate
    @param conditioning_data_weight: A weight to favor the reproduction of the
                                     patterns around conditioning data
    @param max_non_matching_proportion: Authorized proportion of non-matching
                                        nodes, i.e., whose distance is below the
                                        threshold for the variable
    @param start_parameter_reduction: When to start parameter reduction during
                                      the simulation, 0 being immediately, 1
                                      being never
    @param reduction_factor: Reduction factor for the distance thresholds, TI 
                             fraction and max number of data during parameter
                             reduction
    @param rotation_angle_array: A NumPy array describing the rotation to apply
                                 to each neighborhood (in degree)
    @param scaling_factor_array: A NumPy array describing the scaling to apply
                                 to each neighborhood for each axis
    @param number_postproc: The number of post-processing steps
    @param postproc_factor: The factor by which ti_fraction and max_number_data
                            are reduced during post-processing
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
            and len(max_number_data) == training_image_array.shape[0]
            and len(max_density_data) == training_image_array.shape[0]), "variable_types, distance_thresholds, max_number_data, and max_density_data should match the number of variables in the training image"
    for i in range(len(distance_thresholds)):
        assert (0. < distance_thresholds[i] <= 1.), "Distance threshold(s) must be between (0., 1.]"
        assert max_number_data[i] >= 0, "Number(s) of data must be positive"
        assert 0. < max_density_data[i] <= 1., "Density(ies) of data must be between 0 and 1"
    assert len(neighborhood_shape) == 2, "Neighborhood shape must be two-dimensional"
    assert 0. < max_non_matching_proportion <= 1., "Non-matching proportion must be between 0 and 1"
    assert 0. < start_parameter_reduction <= 1., "Start parameter reduction must be between 0 and 1"
    if rotation_angle_array.shape != data_array.shape[-2:]:
        rotation_angle_array = np.full(data_array.shape[-2:], 0.)
    else:
        rotation_angle_array = rotation_angle_array*np.pi/180.
    if scaling_factor_array.shape != (2, data_array.shape[-2], data_array.shape[-1]):
        scaling_factor_array = np.full((2, data_array.shape[-2], data_array.shape[-1]), 1.)
    assert math.isnan(no_data_value) == False, "The no data value cannot be NaN"
            
    random.seed(seed)

    ti_ranges_max, ti_indices = prepare_training_image(training_image_array, variable_types)

    updated_neighborhood_shape = np.empty(2)
    updated_neighborhood_shape[0] = min(data_array.shape[1], neighborhood_shape[0])
    updated_neighborhood_shape[1] = min(data_array.shape[2], neighborhood_shape[1])
    lag_vectors, lag_distances = compute_neighborhood_lag_vectors(updated_neighborhood_shape,
                                                                  grid_yx_spacing,
                                                                  delta)
    
    data_weight_array = np.ones(data_array.shape)
    for index in np.ndindex(data_weight_array.shape):
        if (data_array[index] != no_data_value
            and math.isnan(data_array[index]) == False):
            data_weight_array[index] = conditioning_data_weight
    
    simulation_array = np.empty((number_realizations,
                                 data_array.shape[0],
                                 data_array.shape[1],
                                 data_array.shape[2]))
    for i_rez in prange(number_realizations):
        simulation_array[i_rez] = simulate_ds_realization(data_array,
                                                          data_weight_array,
                                                          training_image_array,
                                                          ti_ranges_max,
                                                          ti_indices,
                                                          distance_thresholds,
                                                          ti_fraction,
                                                          max_number_data,
                                                          max_density_data,
                                                          lag_vectors,
                                                          lag_distances,
                                                          updated_neighborhood_shape,
                                                          max_non_matching_proportion,
                                                          start_parameter_reduction,
                                                          reduction_factor,
                                                          rotation_angle_array,
                                                          scaling_factor_array,
                                                          number_postproc,
                                                          postproc_factor,
                                                          path_type,
                                                          seed + i_rez,
                                                          no_data_value)
    
    return simulation_array