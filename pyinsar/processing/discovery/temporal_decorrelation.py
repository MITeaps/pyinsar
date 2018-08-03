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
from pyinsar.processing.machine_learning.geostatistics.sequential_gaussian_simulation import run_sgs, compute_averaged_cumulative_distribution_from_array


# Standard library imports
from collections import OrderedDict
import random

# 3rd party imports
import numpy as np
import scipy as sp

class TemporalDecorrelation(PipelineItem):
    '''
    Pipeline item to add temporal decorrelation to some phase
    '''

    def __init__(self, str_description, ap_paramList, grid_yx_spacing, wavelength, seed=None, save_noise=False):
        '''
        Initialize Temporal Decorrelation pipeline item

        @param str_description: String description of item
        @param ap_paramList[vario_models] = Auto list of SGS models
        @param ap_paramList[vario_sills] = Auto list of SGS sills
        @param ap_paramList[vario_azimuth] = Auto param of SGS azimuth
        @param ap_paramList[vario_ranges] = Auto list of SGS ranges
        @param ap_paramList[max_num_data] = Auto param of the max size of the neighborhood
        @param ap_paramList[decorrelation_mean] = Auto param of the decorrelation mean in the same units as the wavelength
        @param ap_paramList[decorrelation_std] = Auto param of decorrelation standard deviation in the same units as the wavelength
        @param grid_yx_spacing: The y,x grid spacing
        @param wavelength: Wavelength for converting to phase (from path length)
        @param seed: Seed to use when generating noise
        @param save_noise: Boolean indicating whether or not to save a copy of the noise in the results
        '''

        self._grid_yx_spacing = grid_yx_spacing

        self._seed = seed
        self._wavelength = wavelength
        self._save_noise = save_noise

        super(TemporalDecorrelation, self).__init__(str_description, ap_paramList)

    def process(self, obj_data):
        """
        Add temporal decorrelation to a phase image

        @param obj_data: Image data wrapper
        """

        vario_models = self.ap_paramList[0]()
        vario_sills = self.ap_paramList[1]()
        vario_azimuth = self.ap_paramList[2]()

        vario_ranges = self.ap_paramList[3]()
        max_num_data = self.ap_paramList[4]()

        decorrelation_mean = self.ap_paramList[5]()
        decorrelation_std = self.ap_paramList[6]()


        if self._save_noise:
            my_noise = OrderedDict()

        for label, data in obj_data.getIterator():

            data_array = np.full_like(data, -99999)


            # run_sgs requires an integer seed
            # randomly creating one if necessary
            if self._seed is None:
                sys_random = random.SystemRandom()
                seed = sys_random.randint(0,2**32 -1)
            else:
                seed = self._seed
            
            raw_temporal_decorrelation = run_sgs(data_array,
                                                 self._grid_yx_spacing,
                                                 vario_models,
                                                 vario_sills,
                                                 vario_azimuth,
                                                 vario_ranges,
                                                 max_number_data = max_num_data,
                                                 seed = seed)

            cumulative_frequency = compute_averaged_cumulative_distribution_from_array(raw_temporal_decorrelation[0])

            temporal_decorrelation = sp.stats.laplace.ppf(cumulative_frequency,
                                                       loc = decorrelation_mean,
                                                       scale = decorrelation_std)

            temporal_decorrelation = insar_simulator_utils.change_in_range_to_phase(temporal_decorrelation, wavelength = self._wavelength)

            if self._save_noise:
                my_noise[label] = temporal_decorrelation

            obj_data.updateData(label, data + temporal_decorrelation)

        if self._save_noise:
            obj_data.addResult(self.str_description, my_noise)
