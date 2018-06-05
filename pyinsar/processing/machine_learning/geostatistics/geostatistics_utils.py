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

from enum import Enum
import numpy as np

from numba import jit

class VariableType(Enum):
    DISCRETE = 0
    CONTINUOUS = 1

class PathType(Enum):
    LINEAR = 0
    RANDOM = 1

@jit(nopython = True, nogil = True)
def unflatten_index(flattened_index, array_shape):
    '''
    Unflatten an index for a 2D array
    
    @param flattened_index: The flattened index (i.e., a single integer)
    @param array_shape: The shape of the array for the two dimensions (j, i)
    
    @return The 2D index (j, i)
    '''
    j = int((flattened_index/array_shape[1])%array_shape[0])
    i = int(flattened_index%array_shape[1])
    return (j, i)

def standardize(x):
    '''
    Reduce and center a float or array
    
    @param x: The float or array
    
    @return A float or array
    '''
    return (x - np.nanmean(x))/np.nanstd(x)

def normalize(x):
    '''
    Reduce and center a float or array
    
    @param x: The float or array
    
    @return A float or array
    '''
    x_min = np.nanmin(x)
    x_max = np.nanmax(x)
    
    return (x - x_min)/(x_max - x_min)