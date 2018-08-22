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
from pyinsar.processing.utilities.insar_simulator_utils import change_in_range_to_phase, phase_to_change_in_range

# Standard library imports
from collections import OrderedDict

# 3rd party imports
import numpy as np



class LOS_Deformation_Phase(PipelineItem):
    """
    Converts between LOS deformation and phase

    *** In Development ***
    """

    def __init__(self, str_description, wavelength, k=2, convert_target='los', channel_index = None):
        """
        Initialize LOS Deformation Phase item

        @param str_description: String describing item
        @param wavelength: Radar wavelength 
        @param k: Number of radar passes
        @param convert_target: Convert to 'los' or 'phase'
        @param channel_index: Which channel index to use (None if there is no channel axis)
        """

        self.wavelength = wavelength
        self.convert_target = convert_target
        self.channel_index = channel_index

        super(LOS_Deformation_Phase, self).__init__(str_description)

    def process(self, obj_data):
        """
        Convert between LOS deformation and phase

        @param obj_data: Image Wrapper
        """

        if self.convert_target == 'los':
            convert_function  = change_in_range_to_phase
        elif self.convert_target == 'phase':
            convert_function = phase_to_change_in_range
        else:
            raise RuntimeError('Conversion target "{}" not understood'.format(self.convert_target))

        for label, data in obj_data.getIterator():

            if channel_index is None:
                data = convert_function(data, self.wavelength, self.k)

            else:
                data[channel_index, ...] = convert_function(data[channel_index, ...], self.wavelength, self.k)

            obj_data.updateData(label, data)
