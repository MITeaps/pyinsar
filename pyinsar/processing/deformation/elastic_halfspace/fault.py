# The MIT License (MIT)
# Copyright (c) 2018 Massachusetts Institute of Technology
#
# Author: Cody Rude
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

from pyinsar.processing.utilities.generic import rotate, translate
from .okada import compute_okada_displacement

from tqdm import trange
import numpy as np


class Fault(object):
    '''
    Model a fault as a collection of small okada faults
    '''
    def __init__(self, x_center, y_center, depth, length, width, strike, dip, num_elements_length, num_elements_width,
                 poisson_ratio = 0.25):
        '''
        Initialize Fault object

        @param x_center: x centroid of fault
        @param y_center: y centroid of fault
        @param depth: Depth to centroid of fault
        @param length: Length of fault (along strike)
        @param width: Width of fault (along dip)
        @param dip: Dip angle
        @param num_elements_length: Number of elements in the length direction
        @param num_elements_width: Number of elements in the widht direction
        @param poisson_ratio: Poisson ratio
        '''
        self.x_center = x_center
        self.y_center = y_center
        self.depth = depth
        self.length = length
        self.width = width
        self.strike = strike
        self.dip = dip
        self.poisson_ratio = 0.25

        # Generate rectangular centroids
        self.cell_width = self.width / num_elements_width
        self.cell_length = self.length / num_elements_length

        cell_x_coords = np.linspace(self.cell_length/2 - self.length/2, self.length/2 - self. cell_length/2, num_elements_length)
        cell_z_coords = np.linspace(self.cell_width/2 - self.width/2, self.width/2 - self.cell_width/2, num_elements_width)

        cell_x_centroids, cell_z_centroids = np.meshgrid(cell_x_coords, cell_z_coords)

        cell_centroids = np.zeros([3, len(cell_x_centroids.ravel())])
        cell_centroids[0,:] = cell_x_centroids.ravel()
        cell_centroids[2,:] = cell_z_centroids.ravel()

        # Save unrotated centers for making slip matrices
        self.unrotated_x = cell_x_centroids
        self.unrotated_y = cell_z_centroids

        # Change coordinate system
        x_angle = np.pi/2 - self.dip
        z_angle = -self.strike - np.pi/2

        cell_centroids = rotate(cell_centroids, 0, 0, x_angle)
        cell_centroids = rotate(cell_centroids, z_angle, 0, 0)

        cell_centroids = translate(cell_centroids, x_center, y_center, -depth)

        self.cell_centroids = cell_centroids



    def generateDeformation(self, slip, rake, x_coords, y_coords):
        '''
        Generate surface deformations from fault

        @param slip: 2d array of slip with size (num_elements_width, num_elements_length)
        @param rake: Scalar Rake value
        @param x_coords: 2d array of x coordinates
        @param y_coords: 2d array of y coordinates

        @return Surface deformations at specificed coordinates
        '''

        deformation = np.zeros([3,*x_coords.shape])
        slip_ravel = slip.ravel()

        for index in trange(len(slip_ravel)):
            x_center = self.cell_centroids[0,index]
            y_center = self.cell_centroids[1,index]
            depth = -self.cell_centroids[2,index]
            slip_value = slip_ravel[index]


            deformation += compute_okada_displacement(fault_centroid_x = x_center,
                                                      fault_centroid_y = y_center,
                                                      fault_centroid_depth = depth,
                                                      fault_strike = self.strike,
                                                      fault_dip = self.dip,
                                                      fault_length = self.cell_length,
                                                      fault_width = self.cell_width,
                                                      fault_rake = rake,
                                                      poisson_ratio = self.poisson_ratio,
                                                      fault_open = 0,
                                                      xx_array = x_coords,
                                                      yy_array = y_coords,
                                                      fault_slip = slip_value)

        return deformation
