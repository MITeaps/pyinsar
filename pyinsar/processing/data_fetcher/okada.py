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

from skdaccess.framework.data_class import DataFetcherBase, ImageWrapper
from pyinsar.processing.deformation.elastic_halfspace.okada import compute_okada_displacement

class DataFetcher(DataFetcherBase):
    """
    Generates data from an Okada model
    """
    def __init__(self, ap_paramList, xx_array, yy_array, verbose=False):
        """
        Initialize Okada DataFetcher

        @param ap_paramList[fault_centroid_x]: x centroid
        @param ap_paramList[fault_centroid_y]: y centroid
        @param ap_paramList[fault_centroid_depth]: Fault depth
        @param ap_paramList[fault_strike]: Fault strike
        @param ap_paramList[fault_dip]: Fault dip
        @param ap_paramList[fault_length]: Fault Length
        @param ap_paramList[fault_width]: Fault width
        @param ap_paramList[fault_rake]: Fault rake
        @param ap_paramList[fault_slip]: Fault slip
        @param ap_paramList[fault_open]: Fault open
        @param ap_paramList[poisson_ratio]: Poisson ratio
        @param xx_array: Array of x coordinates
        @param yy_array: Array of y coordinates
        @param verbose: Print out extra information
        """
        self._xx_array = xx_array
        self._yy_array = yy_array

        super(DataFetcher, self).__init__(ap_paramList, verbose)


    def output(self):
        """
        Output deformation in an image wrapper

        @return Deformation in an Image wrapper 
        """

        metadata_dict = OrderedDict()
        data_dict = OrderedDict()

        parameter_list = [
            'fault_centroid_x',
            'fault_centroid_y',
            'fault_centroid_depth',
            'fault_strike',
            'fault_dip',
            'fault_length',
            'fault_width',
            'fault_rake',
            'fault_slip',
            'fault_open',
            'poisson_ratio',
        ]

        kwargs = OrderedDict()

        for index, param in enumerate(parameter_list):
            kwargs[param] = self.ap_paramList[index]()



        deformation = compute_okada_displacement(**kwargs,
                                                 xx_array = self._xx_array,
                                                 yy_array = self._yy_array)



        data_dict['deformation'] = deformation
        metadata_dict['deformation'] = kwargs

        return ImageWrapper(data_dict, meta_data = metadata_dict)
