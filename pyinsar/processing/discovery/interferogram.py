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

# Standard library imports
from collections import OrderedDict

# 3rd party imports
from more_itertools import pairwise
import numpy as np

class Interferogram(PipelineItem):
    ''' Create Inteferogram from SLC data'''

    def __init__(self, str_description, pairing='neighbor'):
        """
        Initialize Interferogram item

        @param str_description: String describing item
        @param pairing: How to pair SLC images. Currently only 'neighbor' is accepted'
        """
        self._pairing = pairing

        super(Interferogram, self).__init__(str_description)

    def process(self, obj_data):
        """
        Create interferograms from SLC images in an image wrapper

        @param obj_data: Image wrapper containing SLC images
        """

        data_dict = OrderedDict()
        metadata_dict = OrderedDict()

        if self._pairing == 'neighbor':
            data_iterator = pairwise(obj_data.getIterator())


        master_label = next(obj_data.getIterator())[0]

        for (label1, image1), (label2, image2) in data_iterator:
            new_label = label1 + ' ' + label2
            data_dict[new_label] = np.angle(image1 * np.conj(image2))
            metadata_dict[new_label] = OrderedDict()
            metadata_dict[new_label]['image1'] = obj_data.info(label1)
            metadata_dict[new_label]['image2'] = obj_data.info(label2)
            if 'Geolocation' in obj_data.info(master_label):
                metadata_dict[new_label]['Geolocation'] = obj_data.info(master_label)['Geolocation']
            metadata_dict[new_label]['Wavelength'] = obj_data.info(label1)['Wavelength']

        obj_data.data = data_dict
        obj_data.meta_data = metadata_dict
