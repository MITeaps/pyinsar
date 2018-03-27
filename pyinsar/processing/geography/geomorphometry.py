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

@jit(nopython = True)
def add_symmetric_border(array, border_size = 1):
    '''
    Add a symmetric border to a 2D array
    
    @param array: The array
    @param border_size: The size of the border
    
    @return The expended array
    '''
    assert len(array.shape) == 2, "The array must be two-dimensional"
    assert border_size > 0, "The border size must be strictly higher than 0"
    
    bordered_array = np.full((array.shape[0] + 2*border_size, 
                              array.shape[1] + 2*border_size), 
                             np.nan)
    bordered_array[border_size:-border_size, border_size:-border_size] = array
    
    bordered_array[border_size:-border_size, -border_size:] = array[:, -1:-border_size - 1:-1]
    bordered_array[border_size:-border_size, border_size - 1::-1] = array[:, 0:border_size]
    bordered_array[0:border_size, :] = bordered_array[2*border_size - 1:border_size - 1:-1, :]
    bordered_array[-border_size:, :] = bordered_array[-border_size - 1:-2*border_size - 1:-1, :]
    
    return bordered_array

@jit(nopython = True)
def compute_gradient_at_cell(array, j, i, grid_yx_spacing, axis = 1):
    '''
    Compute Horn's gradient for a given cell of an array
    
    @param array: The array
    @param j: The index of the cell along the y axis
    @param i: The index of the cell along the x axis
    @param grid_yx_spacing: The cell size, which is considered fixed for the entire array
    @param axis: the axis along which the gradient is computed (0: y; 1: x)
    
    @return The gradient value for the cell
    '''
    assert len(array.shape) == 2 and len(grid_yx_spacing) == 2, "The array must be two-dimensional"
    assert 0 <= j < array.shape[0] and 0 <= i < array.shape[1], "The cell is outside the array"
    assert axis == 0 or axis == 1, "Invalid axis"
    
    cell_1 = (j + 1, i + 1)
    cell_2 = (j + 1, i - 1)
    cell_3 = (j, i + 1)
    cell_4 = (j, i - 1)
    cell_5 = (j - 1, i + 1)
    cell_6 = (j - 1, i - 1)
    distance = grid_yx_spacing[1]
    if axis == 0:
        cell_2 = (j - 1, i + 1)
        cell_3 = (j + 1, i)
        cell_4 = (j - 1, i)
        cell_5 = (j + 1, i - 1)
        distance = grid_yx_spacing[0]

    return ((array[cell_1] - array[cell_2]) +
            2*(array[cell_3] - array[cell_4]) +
            (array[cell_5] - array[cell_6]))/(8.*distance)

@jit(nopython = True)
def compute_horne_slope(array, 
                        grid_yx_spacing):
    '''
    Compute Horn's slope of a 2D array with a fixed cell size
    
    @param array: The array
    @param grid_yx_spacing: The cell size, which is considered fixed for the entire array
    
    @return The slope (in degree)
    '''
    assert len(array.shape) == 2 and len(grid_yx_spacing) == 2, "The array must be two-dimensional"

    array = add_symmetric_border(array)
    
    slope_array = np.full(array.shape, np.nan)

    for j in range(1, slope_array.shape[0] - 1):
        for i in range(1, slope_array.shape[1] - 1):
            dx = compute_gradient_at_cell(array, j, i,
                                          grid_yx_spacing)
            dy = compute_gradient_at_cell(array,j, i,
                                          grid_yx_spacing, 0)
            slope_array[j, i] = math.atan(math.sqrt(dx*dx + dy*dy))*180./math.pi
                
    return slope_array[1:-1, 1:-1]