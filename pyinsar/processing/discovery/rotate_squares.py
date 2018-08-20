# The MIT License (MIT)
# Copyright (c) 2018 Massachusetts Institute of Technology
#
# Authors: Cody Rude
# This software is part of the NSF DIBBS Project "An Infrastructure for
# Computer Aided Discovery in Geoscience" (PI: V. Pankratius) and
# NASA AIST Project "Computer-Aided Discovery of Earth Surface
# Deformation Phenomena" (PI: V. Pankratius)
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


from collections import OrderedDict

from skdiscovery.data_structure.framework.base import PipelineItem
from skdiscovery.utilities.patterns.image_tools import generateSquaresAroundPoly

import numpy as np
import scipy as sp

from pyinsar.processing.utilities import insar_simulator_utils

def rotateSquare(image, square, angle, order):
    '''
    Rotate a subsection of an image defined by a shapely square

    @param image: Full image containing subsection to be rotated
    @param square: Shapely square
    @param angle: Angle of rotation
    @param order: Order of spline interpolation

    '''
    x_start, y_start, x_end, y_end = np.rint(square.bounds).astype(np.int)
    size = x_end - x_start

    half_size = np.ceil(size/2.0).astype(np.int)
    x_slice = slice(x_start - half_size, x_end + half_size)
    y_slice = slice(y_start - half_size, y_end + half_size)
    
    rotated_image = sp.ndimage.rotate(image[y_slice, x_slice], angle,cval=np.nan, reshape=True, order=order)
    return insar_simulator_utils.crop_array_from_center(rotated_image, (size, size))


class RotateSquares(PipelineItem):
    '''
    Generate new images by rotating subsections of data defined by Shapely squares
    '''

    def __init__(self, str_description, ap_paramList, square_result_name, angles, clean=True):
        '''
        Initialize RotateSquares object

        @param str_description: String describing class
        @param ap_paramList[SplineOrder]: Spline order used in interpolation
        @param square_result_name: Name of pipeline item that contains the Shapely squares
        @param angles: Angles used when rotating squares
        @param clean: Remove any squares that contain NaN's
        '''

        self._angles = angles

        self._square_result_name = square_result_name

        self._clean = clean

        super(RotateSquares, self).__init__(str_description, ap_paramList)


    def process(self, obj_data):
        '''
        Generate rotated images based on Shapely squares

        @param obj_data: Image data wrapper
        '''

        data_result = OrderedDict()
        meta_result = OrderedDict()


        square_dict = obj_data.getResults()[self._square_result_name]

        spline_order = self.ap_paramList[0]()

        for label, data in obj_data.getIterator():
            square_list = square_dict[label]

            i = 0
            for square in square_list:
                for angle in self._angles:
                    new_data = rotateSquare(data, square, angle, spline_order)

                    if not self._clean or np.count_nonzero(np.isnan(new_data)) == 0:
                        new_label = 'Image {}, Square {:d}, rotated {:f} degrees'.format(label, i, angle)
                        data_result[new_label] = new_data
                        meta_result[new_label] = OrderedDict()
                        meta_result[new_label]['angle'] = angle
                        meta_result[new_label]['square'] = square
                        meta_result[new_label]['original_image'] = label
                        i += 1

                    

        
        obj_data.update(data_result)
        obj_data.updateMetadata(meta_result)
