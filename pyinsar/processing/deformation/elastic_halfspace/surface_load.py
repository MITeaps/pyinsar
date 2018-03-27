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
from scipy.special import hyp2f1

def compute_uniform_disk_load_displacement(disk_x,
                                           disk_y,
                                           disk_radius,
                                           poisson_ratio,
                                           pressure,
                                           shear_modulus,
                                           xx_array,
                                           yy_array):
    '''
    Compute the surface displacements for a uniform disk load
    
    @param disk_x: x cooordinate for the disk's center
    @param disk_y: y cooordinate for the disk's center
    @param disk_radius: Disk's radius
    @param poisson_ratio: Poisson's ratio
    @param pressure: Pressure applied by the disk
    @param shear_modulus: Shear modulus
    @param xx_array: x cooordinate for the domain within a 2D array
    @param yy_array: y cooordinate for the domain within a 2D array
    
    @return The surface displacement field
    '''
    horizontal_distance_array = np.sqrt((xx_array - disk_x)**2 + (yy_array - disk_y)**2)
    
    displacement_array = np.zeros((3, xx_array.shape[0], xx_array.shape[1]))
    
    constant_term = -pressure*(1 - 2*poisson_ratio)/(4*shear_modulus*disk_radius)
    displacement_array[0,:,:][horizontal_distance_array <= disk_radius] = constant_term*horizontal_distance_array[horizontal_distance_array <= disk_radius]*xx_array[horizontal_distance_array <= disk_radius]
    displacement_array[1,:,:][horizontal_distance_array <= disk_radius] = constant_term*horizontal_distance_array[horizontal_distance_array <= disk_radius]*yy_array[horizontal_distance_array <= disk_radius]
    displacement_array[2,:,:][horizontal_distance_array <= disk_radius] = constant_term*(4*(disk_radius**2)*(1 - poisson_ratio)*hyp2f1(1/2., -1/2., 1., (horizontal_distance_array[horizontal_distance_array <= disk_radius]**2)/disk_radius**2))/(1 - 2*poisson_ratio)
    
    constant_term = -pressure*(disk_radius**2)*(1 - 2*poisson_ratio)/(4*shear_modulus)
    displacement_array[0,:,:][horizontal_distance_array > disk_radius] = constant_term*xx_array[horizontal_distance_array > disk_radius]/horizontal_distance_array[horizontal_distance_array > disk_radius]**2
    displacement_array[1,:,:][horizontal_distance_array > disk_radius] = constant_term*yy_array[horizontal_distance_array > disk_radius]/horizontal_distance_array[horizontal_distance_array > disk_radius]**2
    displacement_array[2,:,:][horizontal_distance_array > disk_radius] = constant_term*(2*(1 - poisson_ratio)*hyp2f1(1/2., 1/2., 2., (disk_radius**2)/horizontal_distance_array[horizontal_distance_array > disk_radius]**2))/((1 - 2*poisson_ratio)*horizontal_distance_array[horizontal_distance_array > disk_radius])

    return displacement_array