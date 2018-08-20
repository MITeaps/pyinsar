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

import os.path
import numpy as np
from urllib.request import urlopen

################################################################################
# Import GACOS runs
################################################################################

def read_rsc_header_file(file_path):
    '''
    Read the rsc header file from GACOS data
    
    @param file_location: The path to the file
    
    @return A dictionary containing the header's information
    '''
    header_dict = {}
    with open(file_path) as header_file:
        for line in header_file:
            line_list = line.rstrip('\n').split(' ')
            key = line_list[0]
            value = line_list[-1]
            try:
                value = float(value)
            except ValueError:
                pass
            header_dict[key] = value
            
    return header_dict

def open_gacos_tropospheric_delays(tropodelay_header_path):
    '''
    Open a topospheric delay map computed by the Generic Atmospheric Correction Online Service for InSAR (GACOS)
    
    @param tropodelay_header_path: Path to the header file (.ztd.rsc or .dztd.rsc)
    
    @return A NumPy array containing the topospheric delay in meters
            and a tuple containing the extent of the array
    '''
    split_tropodelay_header_path = tropodelay_header_path.split('.')
    assert (split_tropodelay_header_path[-1] == 'rsc'
            and (split_tropodelay_header_path[-2] == 'ztd'
                 or split_tropodelay_header_path[-2] == 'dztd')), 'Incorrect input format, must be .ztd.rsc or .dztd.rsc'
    assert os.path.exists(tropodelay_header_path) == True, "Header %r doesn't exist" % header_path
    tropodelay_file_path = '.'.join(split_tropodelay_header_path[0:-1])
    assert os.path.exists(tropodelay_file_path) == True, "File %r doesn't exist" % file_path
        
    header_dict = read_rsc_header_file(tropodelay_header_path)

    tropodelay_array = np.fromfile(tropodelay_file_path, dtype = np.float32)
    tropodelay_array = tropodelay_array.reshape((int(header_dict['FILE_LENGTH']),
                                                 int(header_dict['WIDTH'])))
    tropodelay_extent = (header_dict['X_FIRST'] - 0.5*header_dict['X_STEP'],
                         header_dict['X_FIRST'] + header_dict['X_STEP']*(header_dict['WIDTH'] - 1) + 0.5*header_dict['X_STEP'],
                         header_dict['Y_FIRST'] + header_dict['Y_STEP']*(header_dict['FILE_LENGTH'] - 1) + 0.5*header_dict['Y_STEP'],
                         header_dict['Y_FIRST'] - 0.5*header_dict['Y_STEP'])
    
    return tropodelay_array, tropodelay_extent

################################################################################
# Import SGEMS files
################################################################################

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
