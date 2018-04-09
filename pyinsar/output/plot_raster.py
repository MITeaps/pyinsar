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

from numba import jit
import matplotlib.pyplot as plt
from ipywidgets import interact, BoundedIntText

def average_minmax_slices(array, axis = 0):
    '''
    Get the averaged minimal and maximal values for the slices of a NumPy array
    along a given axis
    
    @param array: A NumPy array
    @param axis: An axis
    
    @return The averaged minimal and maximal values
    '''
    assert 0 <= axis < len(array.shape), "The array doesn't have such axis"

    mean_min = 0.
    mean_max = 0.
    for i in range(array.shape[axis]):
        s = [slice(j, j + 1) if j == axis else slice(None) for j in range(3)]
        sliced_array = array[s].squeeze()
        mean_min += np.nanmin(sliced_array)
        mean_max += np.nanmax(sliced_array)
    
    return mean_min/array.shape[axis], mean_max/array.shape[axis]

def plot_interactive_slicing(array,
                             slice_index,
                             axis = 0,
                             cmap = 'viridis',
                             extent = None,
                             clabel = '',
                             xlabel = '',
                             ylabel = ''):
    '''
    Plot 2D slices of a 3D NumPy array, with the possibility to loop through and
    visualize the slices
    
    @param array: A 3D NumPy array
    @param slice_index: The initial slice to show
    @param axis: The axis perpendicular to the slices
    @param cmap: A colormap for the array
    @param extent: The spatial extent of the slice
    @param clabel: A label for the colorbar
    @param xlabel: A label for the x axis
    @param ylabel: A label for the y axis
    '''
    
    mean_min, mean_max = average_minmax_slices(array, axis = axis)
    
    s = [slice(slice_index, slice_index + 1) if i == axis else slice(None) for i in range(3)]
    sliced_array = array[s].squeeze()
    
    figure = plt.figure()
    subfigure = figure.add_subplot(111)

    raster_map = plt.imshow(np.ma.masked_invalid(sliced_array), extent = extent,
                            cmap = cmap, interpolation = 'None', rasterized = True,
                            vmin = mean_min, vmax = mean_max,
                            zorder = 0)

    raster_map_colorbar = plt.colorbar(raster_map)
    raster_map_colorbar.set_label(clabel)

    subfigure.set_xlabel(xlabel)
    subfigure.set_ylabel(ylabel)

    plt.show()
    
    def update_imshow(new_slice_index):
        s = [slice(new_slice_index, new_slice_index + 1) if i == axis else slice(None) for i in range(3)]
        sliced_array = array[s].squeeze()
        raster_map.set_data(sliced_array)
    
    interactive_plot = interact(update_imshow,
                                new_slice_index = BoundedIntText(value = slice_index,
                                                                 min = 0,
                                                                 max = array.shape[axis] - 1,
                                                                 step = 1,
                                                                 description = 'Slice',
                                                                 continuous_update = False))

    plt.draw()