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

def wrap(x, to_2pi = False):
    '''
    Wrap a float or an array
    
    @param x: The float or array
    @param to_2pi: If True, wrap to [0, 2pi) instead of [-pi, pi]
    
    @return The wrapped array (in radian between -pi and pi)
    '''
    if to_2pi == True:
        return np.mod(x, 2.*np.pi)
    return np.angle(np.exp(1j*(x)))

def crop_array_from_center(array, crop_shape):
    '''
    Crop an array along its borders
    
    @param array: The array
    @param crop_shape: The number of cells to remove along the y and x axes
    
    @return The cropped array
    '''
    slices = []
    for i in range(len(crop_shape)):
        start = array.shape[i]//2 - crop_shape[i]//2
        end = start + crop_shape[i]
        slices.append(slice(start, end))
        
    return array[slices]