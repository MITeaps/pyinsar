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
from urllib.request import urlopen

def open_sgems_file(file_location):
    '''
    Open an SGEMS file containing one or several variables in an array
    
    @param file_location: The location of the file
    
    @return A NumPy array
    '''
    with open(file_location) as sgems_file:

        head = [next(sgems_file).rstrip('\n') for i in range(2)]
        array_shape = [int(i) for i in reversed(head[0].split(' '))]
        number_var = int(head[1])
        var_names = [next(sgems_file).rstrip('\n') for i in range(number_var)]

        flattened_index = 0
        var_array = np.empty([number_var] + array_shape)
        for line in sgems_file:
            index = np.unravel_index(flattened_index, array_shape)
            values = list(filter(None, line.rstrip('\n').split(' ')))
            for i_var in range(number_var):
                var_array[i_var][index] = float(values[i_var])

            flattened_index += 1
            
        return var_array

def open_sgems_file_from_url(file_url):
    '''
    Open an SGEMS file containing one or several variables in an array from the file's URL
    
    @param file_url: The URL of the file
    
    @return A NumPy array
    '''
    with urlopen(file_url) as sgems_url:
        sgems_file = iter([line.decode('utf-8').rstrip('\n\r') for line in sgems_url.readlines()])

        head = [next(sgems_file) for i in range(2)]
        array_shape = [int(i) for i in reversed(head[0].split(' '))]
        number_var = int(head[1])
        var_names = [next(sgems_file) for i in range(number_var)]

        flattened_index = 0
        var_array = np.empty([number_var] + array_shape)
        for line in sgems_file:
            index = np.unravel_index(flattened_index, array_shape)
            values = list(filter(None, line.split(' ')))
            for i_var in range(number_var):
                var_array[i_var][index] = float(values[i_var])

            flattened_index += 1
            
        return var_array