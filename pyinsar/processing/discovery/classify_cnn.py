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
from skdiscovery.utilities.patterns.image_tools import divideIntoSquares



# 3rd party imports
import numpy as np


class ClassifyCNN(PipelineItem):
    """ Train a CNN """

    def __init__(self, str_description, cnn_network_dir, batch_size=2000, config=None, compare_labels=False,
                 stride = None, size=None):
        """
        Initialize TrainCNN item

        @param str_description: String describing item
        @param cnn_network_dir: Strining containing the directiory where the CNN is stored
        @param batch_size: Batch size to use when classifying with Tensorflow
        @param config: Additional session configuration dictionary
        @param compare_labels: Compare measured labels with labels stored in metadata
        @param stride: Distance between images if it necessary to cut image into tiles
        @param size: Size of images to feed into CNN
        """
        assert stride is None and size is None or \
               stride is not None and size is not None, \
               'Either both or neither stride and size should be None'

        self.cnn_network_dir = cnn_network_dir
        self.batch_size = batch_size
        self.config = config
        self.compare_labels = compare_labels
        self.stride = stride
        self.size = size

        super(ClassifyCNN, self).__init__(str_description, [])

    def process(self, obj_data):
        """
        Classify data using a CNN using data in Image wrapper

        @param obj_data: Image wrapper
        """
        results = OrderedDict()
        for label, data in obj_data.getIterator():
            results[label] = OrderedDict()

            if self.stride is not None:
                extents, processed_data = divideIntoSquares(data, self.size, self.stride)

                print(processed_data.shape)

            else:
                processed_data = data
                extents = np.array([[0, data.shape[-2], 0, data.shape[-1]]])


            labels = ann.classify(image_data = processed_data,
                                  model_dir = self.cnn_network_dir,
                                  batch_size = self.batch_size,
                                  config = self.config)


            results[label]['labels'] = labels
            results[label]['extents']  = extents

            if self.compare_labels:

                given_labels = info['Labels']

                fraction_correct = np.count_nonzero(given_labels == labels) / len(labels)

                results[label]['fraction_correct'] = fraction_correct




        obj_data.addResult(self.str_description, results)
