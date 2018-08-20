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

def compute_closed_pipe_displacement(closed_pipe_x,
                                     closed_pipe_y,
                                     closed_pipe_depth_1,
                                     closed_pipe_depth_2,
                                     closed_pipe_radius,
                                     poisson_ratio,
                                     pressurization,
                                     shear_modulus,
                                     xx_array,
                                     yy_array):
    '''
    Compute the surface displacements for a closed pipe
    
    @param closed_pipe_x: x cooordinate for the pipe's center
    @param closed_pipe_y: y cooordinate for the pipe's center
    @param closed_pipe_depth_1: Pipe's top depth
    @param closed_pipe_depth_2: Pipe's bottom depth
    @param closed_pipe_radius: Pipe's radius
    @param poisson_ratio: Poisson's ratio
    @param pressurization: Change of pressure applied to the pipe
    @param shear_modulus: Shear modulus
    @param xx_array: x cooordinate for the domain within a 2D array
    @param yy_array: y cooordinate for the domain within a 2D array
    
    @return The surface displacement field
    '''
    closed_pipe_horizontal_distance_array = np.sqrt((xx_array - closed_pipe_x)**2 + (yy_array - closed_pipe_y)**2)
    closed_pipe_distance_1_array = np.sqrt(closed_pipe_horizontal_distance_array**2 + closed_pipe_depth_1**2)
    closed_pipe_distance_2_array = np.sqrt(closed_pipe_horizontal_distance_array**2 + closed_pipe_depth_2**2)

    displacement_array = np.zeros((3, xx_array.shape[0], xx_array.shape[1]))
    
    b = (pressurization*closed_pipe_radius**2)/(4*shear_modulus)
    displacement_array[0,:,:] = b*((closed_pipe_depth_1**3)/(closed_pipe_distance_1_array**3)
                                   + 2*closed_pipe_depth_1*(5*poisson_ratio - 3)/closed_pipe_distance_1_array
                                   + ((5*closed_pipe_depth_2**3)*(1 - 2*poisson_ratio)
                                      - 2*closed_pipe_depth_2*(closed_pipe_horizontal_distance_array**2)*(5*poisson_ratio - 3))/(closed_pipe_distance_2_array**3))*(xx_array/closed_pipe_horizontal_distance_array**2)
    displacement_array[1,:,:] = b*((closed_pipe_depth_1**3)/(closed_pipe_distance_1_array**3)
                                   + 2*closed_pipe_depth_1*(5*poisson_ratio - 3)/closed_pipe_distance_1_array
                                   + ((5*closed_pipe_depth_2**3)*(1 - 2*poisson_ratio)
                                      - 2*closed_pipe_depth_2*(closed_pipe_horizontal_distance_array**2)*(5*poisson_ratio - 3))/(closed_pipe_distance_2_array**3))*(yy_array/closed_pipe_horizontal_distance_array**2)
    displacement_array[2,:,:] = -b*((closed_pipe_depth_1**2)/(closed_pipe_distance_1_array**3)
                                    + 2*(5*poisson_ratio - 2)/closed_pipe_distance_1_array
                                    + ((3 - 10*poisson_ratio)*closed_pipe_depth_2**2 + 2*(5*poisson_ratio - 2)*closed_pipe_horizontal_distance_array**2)/(closed_pipe_distance_2_array**3))
    
    return displacement_array

def compute_open_pipe_displacement(open_pipe_x,
                                   open_pipe_y,
                                   open_pipe_depth_0,
                                   open_pipe_depth_1,
                                   open_pipe_depth_2,
                                   open_pipe_radius,
                                   poisson_ratio,
                                   pressurization,
                                   shear_modulus,
                                   xx_array,
                                   yy_array):
    '''
    Compute the surface displacements for an open pipe
    
    @param open_pipe_x: x cooordinate for the pipe's center
    @param open_pipe_y: y cooordinate for the pipe's center
    @param open_pipe_depth_0: Pipe's top depth with minimal pressurization
    @param open_pipe_depth_1: Pipe's top depth with maximal pressurization
    @param open_pipe_depth_2: Pipe's bottom depth
    @param open_pipe_radius: Pipe's radius
    @param poisson_ratio: Poisson's ratio
    @param pressurization: Change of pressure applied to the pipe
    @param shear_modulus: Shear modulus
    @param xx_array: x cooordinate for the domain within a 2D array
    @param yy_array: y cooordinate for the domain within a 2D array
    
    @return The surface displacement field
    '''  
    open_pipe_horizontal_distance_array = np.sqrt((xx_array - open_pipe_x)**2 + (yy_array - open_pipe_y)**2)
    open_pipe_distance_0_array = np.sqrt(open_pipe_horizontal_distance_array**2 + open_pipe_depth_0**2)
    open_pipe_distance_1_array = np.sqrt(open_pipe_horizontal_distance_array**2 + open_pipe_depth_1**2)
    open_pipe_distance_2_array = np.sqrt(open_pipe_horizontal_distance_array**2 + open_pipe_depth_2**2)

    b = open_pipe_radius*pressurization/shear_modulus

    displacement_array = np.zeros((3, xx_array.shape[0], xx_array.shape[1]))
    
    displacement_array[0,:,:] = (b*open_pipe_radius/2.)*((open_pipe_depth_1**3)/(open_pipe_distance_1_array**3)
                                                         - 2*open_pipe_depth_1*(1 + poisson_ratio)/open_pipe_distance_1_array
                                                         + ((open_pipe_depth_2**3)*(1 + 2*poisson_ratio)
                                                            + 2*open_pipe_depth_2*(open_pipe_horizontal_distance_array**2)*(1 + poisson_ratio))/(open_pipe_distance_2_array**3))*(xx_array/open_pipe_horizontal_distance_array**2)
    displacement_array[1,:,:] = (b*open_pipe_radius/2.)*((open_pipe_depth_1**3)/(open_pipe_distance_1_array**3)
                                                         - 2*open_pipe_depth_1*(1 + poisson_ratio)/open_pipe_distance_1_array
                                                         + ((open_pipe_depth_2**3)*(1 + 2*poisson_ratio)
                                                            + 2*open_pipe_depth_2*(open_pipe_horizontal_distance_array**2)*(1 + poisson_ratio))/(open_pipe_distance_2_array**3))*(yy_array/open_pipe_horizontal_distance_array**2)
    displacement_array[2,:,:] = -(b*open_pipe_radius/2.)*((open_pipe_depth_1**2)/(open_pipe_distance_1_array**3)
                                                          - 2*poisson_ratio/open_pipe_distance_1_array
                                                          + (-open_pipe_depth_2**2 + 2*(open_pipe_distance_2_array**2)*poisson_ratio)/(open_pipe_distance_2_array**3))

    displacement_array[0,:,:] += (b*open_pipe_radius/2.)*(-(open_pipe_depth_0**3)/(open_pipe_distance_0_array**3)
                                                          + 2*poisson_ratio/open_pipe_distance_0_array
                                                          + (open_pipe_depth_1**2 - 2*(open_pipe_depth_1**2 + open_pipe_horizontal_distance_array**2)*poisson_ratio)/(open_pipe_distance_1_array**3))*(xx_array/open_pipe_depth_1)
    displacement_array[1,:,:] += (b*open_pipe_radius/2.)*(-(open_pipe_depth_0**3)/(open_pipe_distance_0_array**3)
                                                          + 2*poisson_ratio/open_pipe_distance_0_array
                                                          + (open_pipe_depth_1**2 - 2*(open_pipe_depth_1**2 + open_pipe_horizontal_distance_array**2)*poisson_ratio)/(open_pipe_distance_1_array**3))*(yy_array/open_pipe_depth_1)
    displacement_array[2,:,:] += -(b*open_pipe_radius/2.)*((open_pipe_depth_0**3)/(open_pipe_distance_0_array**3)
                                                           - (open_pipe_depth_1**3)/(open_pipe_distance_1_array**3)
                                                           - 2*poisson_ratio/open_pipe_distance_1_array
                                                           + open_pipe_depth_1*(2*poisson_ratio - 1)/open_pipe_distance_1_array
                                                           + open_pipe_depth_0*(1 - 2*poisson_ratio)/open_pipe_distance_0_array
                                                           + (2*poisson_ratio - 1)*np.log(open_pipe_depth_0 + open_pipe_distance_0_array)
                                                           - (2*poisson_ratio - 1)*np.log(open_pipe_depth_1 + open_pipe_distance_1_array))*(1/open_pipe_depth_1)
    
    return displacement_array
