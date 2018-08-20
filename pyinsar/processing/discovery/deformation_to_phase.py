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

# Scikit Discovery imports
from skdiscovery.data_structure.framework.base import PipelineItem

# Pyinsar imports
from pyinsar.processing.utilities import insar_simulator_utils

# Standard library imports
from collections import OrderedDict

# 3rd party imports
import numpy as np

class DeformationToPhase(PipelineItem):
    '''
    Convert deformation to phas
    '''

    def __init__(self, str_description, ap_paramList, xx, yy):
        '''
        Initialize Deformation to Phase pipeline item
    
        @param str_description: String description of item
        @param ap_paramList[track_angle] = Auto param of the track angle
        @param ap_paramList[min_ground_range_1] = Auto param of min_ground_range_1
        @param ap_paramList[height_1] = Auto param of height_1
        @param ap_paramList[is_right_looking] = Auto param of is_right_looking (boolean)
        @param ap_paramList[wavelength] = Auto param of the wavelength for converting deformation to phase
        @param ap_paramList[k] = Auto param of k
        @param xx = x coordinates
        @param yy = y coordinates
        '''

        self._xx = xx
        self._yy = yy

        super(DeformationToPhase, self).__init__(str_description, ap_paramList)

    def process(self, obj_data):
        """
        Convert deformations in a data wrapper to phases

        @param obj_data: Image data wrapper
        """

        track_angle = self.ap_paramList[0]()
        min_ground_range_1 = self.ap_paramList[1]()
        height_1 = self.ap_paramList[2]()
        is_right_looking = self.ap_paramList[3]()
        wavelength = self.ap_paramList[4]()
        k = self.ap_paramList[5]()
        
        for label, data in obj_data.getIterator():
            phase = insar_simulator_utils.generate_interferogram_from_deformation(track_angle = track_angle,
                                                                                  min_ground_range_1 = min_ground_range_1,
                                                                                  height_1 = height_1,
                                                                                  is_right_looking = is_right_looking,
                                                                                  wavelength = wavelength,
                                                                                  k = k,
                                                                                  deformation = data,
                                                                                  xx = self._xx,
                                                                                  yy = self._yy)

            obj_data.updateData(label, phase)
            

            
