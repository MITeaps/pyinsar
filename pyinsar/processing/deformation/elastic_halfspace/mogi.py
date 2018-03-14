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

def compute_mogi_source_displacement(source_x,
                                     source_y,
                                     source_depth,
                                     source_radius,
                                     poisson_ratio,
                                     pressurization,
                                     shear_modulus,
                                     xx_array,
                                     yy_array):
    '''
    Compute the surface displacements for a Mogi source, i.e., a spheroidal
    pressure source
    
    @param source_x: x cooordinate for the source's center
    @param source_y: y cooordinate for the source's center
    @param source_radius: Source's radius
    @param poisson_ratio: Poisson's ratio
    @param pressurization: Change of pressure applied to the source
    @param shear_modulus: Shear modulus
    @param xx_array: x cooordinate for the domain within a 2D array
    @param yy_array: y cooordinate for the domain within a 2D array
    
    @return The surface displacement field
    '''
    source_distance_array = np.sqrt((xx_array - source_x)**2
                                    + (yy_array - source_y)**2
                                    + source_depth**2)
    
    source_change_in_volume = (np.pi*pressurization*source_radius**3)/shear_modulus
    source_strength = (1 - poisson_ratio)*source_change_in_volume/(np.pi)
    
    displacement_array = np.zeros((3, xx_array.shape[0], xx_array.shape[1]))
    displacement_array[0,:,:] = source_strength*(xx_array - source_x)/(source_distance_array**3)
    displacement_array[1,:,:] = source_strength*(yy_array - source_y)/(source_distance_array**3)
    displacement_array[2,:,:] = source_strength*source_depth/(source_distance_array**3)
    
    return displacement_array