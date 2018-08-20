# The MIT License (MIT)
# Copyright (c) 2017 Massachusetts Institute of Technology
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

# Standard library imports
from collections import OrderedDict

# scikit discovery imports
from skdiscovery.data_structure.framework.base import PipelineItem

# Pyinsar imports
from pyinsar.processing.utilities.generic import coherence

# Scikit data access imports
from skdaccess.utilities.support import progress_bar


class Coherence(PipelineItem):
    ''' Calculate coherence between single-look complex SAR images '''

    def __init__(self, str_description, window, pairing='neighbor', use_progress_bar = False):
        '''
        Initialize coherence pipeline item

        @param str_description: Short string describing item
        @param window: Tuple indicating the y and x window size
        @param pairing: How to pair slc images. "neighbor" computes
                        coherence between neighboring images

        @param use_progress_bar: Display progress using a progress bar
        '''

        self.window = window
        self.pairing = pairing
        self.use_progress_bar = use_progress_bar


        super(Coherence, self).__init__(str_description,[])


    def process(self, obj_data):
        '''
        Compute the coherency between two

        @param obj_data: Data wrapper
        '''

        results_dict = OrderedDict()

        if self.pairing == 'neighbor':
            first_image_it = obj_data.getIterator()
            second_image_it = obj_data.getIterator()

            next(second_image_it)

            for (label1, image1), (label2, image2) in progress_bar(zip(first_image_it, second_image_it),
                                                                   total = len(obj_data)-1,
                                                                   enabled = self.use_progress_bar):

                results_dict[label2] = coherence(image1, image2, self.window)


        obj_data.addResult(self.str_description, results_dict)
