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
from ipywidgets import interact, interactive, BoundedIntText, Dropdown, HBox

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
                             model_array = None,
                             axis = 0,
                             cmap = 'viridis',
                             extent = None,
                             clabel = '',
                             xlabel = '',
                             ylabel = '',
                             figsize = None,
                             update_colorbar = False):
    '''
    Plot 2D slices of a 3D NumPy array, with the possibility to loop through and
    visualize the slices
    
    @param array: A 3D NumPy array
    @param slice_index: The initial slice to show
    @param model_array: A 2D NumPy array to compare to the realizations
    @param axis: The axis perpendicular to the slices
    @param cmap: A colormap for the array
    @param extent: The spatial extent of the slice
    @param clabel: A label for the colorbar
    @param xlabel: A label for the x axis
    @param ylabel: A label for the y axis
    @param figsize: A 2D tuple for the figure size
    @param update_colorbar: True if the colorbar must be updated between each
                            slice, false otherwise
    '''
    s = [slice(slice_index, slice_index + 1) if i == axis else slice(None) for i in range(3)]
    sliced_array = array[s].squeeze()
    
    min_value = 0.
    max_value = 1.
    if update_colorbar == False:
        if model_array is None:
            min_value, max_value = average_minmax_slices(array, axis = axis)
        else:
            min_value = np.nanmin(model_array)
            max_value = np.nanmax(model_array)
    else:
        min_value = np.nanmin(sliced_array)
        max_value = np.nanmax(sliced_array)
    
    number_plots = 1
    if model_array is not None:
        number_plots = 2
    
    figure, subplots = plt.subplots(1,
                                    number_plots,
                                    sharex = True, 
                                    sharey = True, 
                                    figsize = figsize)
     
    subplot = subplots
    if model_array is not None:
        subplot = subplots[1]
        
        model_raster_map = subplots[0].imshow(np.ma.masked_invalid(model_array), 
                                              extent = extent,
                                              cmap = cmap, interpolation = 'None',
                                              rasterized = True,
                                              vmin = min_value,
                                              vmax = max_value,
                                              zorder = 0)
        subplots[0].set_xlabel(xlabel)
        subplots[0].set_ylabel(ylabel)

    raster_map = subplot.imshow(np.ma.masked_invalid(sliced_array),
                                extent = extent,
                                cmap = cmap, interpolation = 'None',
                                rasterized = True,
                                vmin = min_value,
                                vmax = max_value,
                                zorder = 0)
    raster_map_colorbar = plt.colorbar(raster_map, ax = subplots)
    raster_map_colorbar.set_label(clabel)
    if model_array is None:
        subplot.set_ylabel(ylabel)
    subplot.set_xlabel(xlabel)

    plt.show()
    
    def update_imshow(new_slice_index):
        s = [slice(new_slice_index, new_slice_index + 1) if i == axis else slice(None) for i in range(3)]
        sliced_array = array[s].squeeze()
        raster_map.set_data(sliced_array)
        if update_colorbar == True:
            raster_map.set_clim(vmin = np.nanmin(sliced_array),
                                vmax = np.nanmax(sliced_array))
            if model_array is not None:
                model_raster_map.set_clim(vmin = np.nanmin(sliced_array),
                                          vmax = np.nanmax(sliced_array))
    
    interactive_plot = interact(update_imshow,
                                new_slice_index = BoundedIntText(value = slice_index,
                                                                 min = 0,
                                                                 max = array.shape[axis] - 1,
                                                                 step = 1,
                                                                 description = 'Slice',
                                                                 continuous_update = False))

    plt.draw()

def plot_interactive_multiple_slicing(array,
                                      axes,
                                      slice_indexes,
                                      model_array = None,
                                      cmap = 'viridis',
                                      update_colorbar = False,
                                      vmin = 0.,
                                      vmax = 1.,
                                      extent = None,
                                      clabel = '',
                                      xlabel = '',
                                      ylabel = '',
                                      figsize = None):
    '''
    Plot 2D slices of a ND NumPy array, with the possibility to loop through and
    visualize the slices
    
    @param array: A ND NumPy array
    @param axes: A list of N - 2 elements that contains the indexes of the axes 
                 to slice
    @param slice_indexes: A list of N - 2 elements that contains the initial 
                          slice to show for each axis
    @param model_array: A 2D NumPy array to compare to the slices
    @param cmap: A colormap for the array
    @param update_colorbar: True if the colorbar must be updated between each 
                            slice, false otherwise
    @param v_min: The minimal value of the colorbar if update_colorbar if false 
                  and no model is used
    @param v_max: The maximal value of the colorbar if update_colorbar if false 
                  and no model is used
    @param extent: The spatial extent of the slice
    @param clabel: A label for the colorbar
    @param xlabel: A label for the x axis
    @param ylabel: A label for the y axis
    @param figsize: A 2D tuple for the figure size
    '''
    assert len(set(axes)) == len(axes), "Axes must be unique"
    assert len(slice_indexes) == len(axes), "Slice indexes and axes should have the same length"
    assert array.ndim - len(axes) == 2, "The plotted array should be two-dimensional"
    
    s = [slice(None) for i in range(array.ndim)]
    for axis, slice_index in zip(axes, slice_indexes):
        s[axis] = slice(slice_index, slice_index + 1)
    sliced_array = array[s].squeeze()
    
    if update_colorbar == False:
        if model_array is not None:
            vmin = np.nanmin(model_array)
            vmax = np.nanmax(model_array)
    else:
        vmin = np.nanmin(sliced_array)
        vmax = np.nanmax(sliced_array)
    
    number_plots = 1
    if model_array is not None:
        number_plots = 2
    
    figure, subplots = plt.subplots(1,
                                    number_plots,
                                    sharex = True, 
                                    sharey = True, 
                                    figsize = figsize)
     
    subplot = subplots
    if model_array is not None:
        subplot = subplots[1]
        
        model_raster_map = subplots[0].imshow(np.ma.masked_invalid(model_array), 
                                              extent = extent,
                                              cmap = cmap, interpolation = 'None',
                                              rasterized = True,
                                              vmin = vmin,
                                              vmax = vmax,
                                              zorder = 0)
        subplots[0].set_xlabel(xlabel)
        subplots[0].set_ylabel(ylabel)

    raster_map = subplot.imshow(np.ma.masked_invalid(sliced_array),
                                extent = extent,
                                cmap = cmap, interpolation = 'None',
                                rasterized = True,
                                vmin = vmin,
                                vmax = vmax,
                                zorder = 0)
    raster_map_colorbar = plt.colorbar(raster_map, ax = subplots)
    raster_map_colorbar.set_label(clabel)
    if model_array is None:
        subplot.set_ylabel(ylabel)
    subplot.set_xlabel(xlabel)

    plt.show()
    
    def update_imshow(new_axis, new_slice_index):
        s[new_axis] = slice(new_slice_index, new_slice_index + 1)
        sliced_array = array[s].squeeze()
        raster_map.set_data(sliced_array)
        if update_colorbar == True:
            raster_map.set_clim(vmin = np.nanmin(sliced_array),
                                vmax = np.nanmax(sliced_array))
            if model_array is not None:
                model_raster_map.set_clim(vmin = np.nanmin(sliced_array),
                                          vmax = np.nanmax(sliced_array))


    for axis, slice_index in zip(axes, slice_indexes):
        axis_widget = Dropdown(options = [axis],
                               value = axis,
                               description = 'Axis',
                               disabled = False)
        slice_widget = BoundedIntText(value = slice_index,
                                      min = 0,
                                      max = array.shape[axis] - 1,
                                      step = 1,
                                      description = 'Slice',
                                      continuous_update = False)
        widget = interactive(update_imshow,
                             new_axis = axis_widget,
                             new_slice_index = slice_widget)
        box = HBox(widget.children)
        display(box)