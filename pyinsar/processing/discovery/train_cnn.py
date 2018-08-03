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

# Standard library imports
from collections import OrderedDict

# Pyinsar imports
from pyinsar.processing.utilities import ann

# Scikit Discovery imports
from skdiscovery.data_structure.framework.base import PipelineItem


# 3rd party imports
import numpy as np


class TrainCNN(PipelineItem):
    """ Train a CNN """

    def __init__(self, str_description, cnn_network_dir, batch_size, config=None):
        """
        Initialize TrainCNN item

        @param str_description: String describing item
        @param cnn_network_dir: Strining containing the directiory where the CNN is stored
        @param batch_size: Batch size to use when training data
        @param config: Dictinoary of extra options to use with the tensorflow session
        """

        self.cnn_network_dir = cnn_network_dir
        self.batch_size = batch_size
        self.config = config
        super(TrainCNN, self).__init__(str_description, [])

    def process(self, obj_data):
        """
        Training CNN using data in Image wrapper

        @param obj_data: Image wrapper
        """

        for label, data in obj_data.getIterator():

            data_labels = obj_data.info(label)['Labels']

            ann.train(image_data = data,
                      image_labels = data_labels,
                      model_dir=self.cnn_network_dir,
                      batch_size=self.batch_size,
                      num_epochs=1,
                      shuffle = False,
                      config=self.config)
