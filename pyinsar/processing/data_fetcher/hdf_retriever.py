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
from pyinsar.processing.utilities.machine_learning import DataRetriever
from pyinsar.processing.utilities.generic import subarray_slice

# Scikit Data Access imports
from skdaccess.framework.data_class import DataFetcherBase, ImageWrapper

# 3rd party imports
import numpy as np

class DataFetcher(DataFetcherBase):
    """
    Data fetcher for retrieving hdf image data made for training in convolutional neural networks
    """

    def __init__(self, filename_list, label_list, size, dtype, num_chunks, index):
        """
        Initialize TrainCNN item

        @param filename_list: List of hdf retriever files
        @param label_list: Label for each file
        @param size: Image shape
        @param dtype: Data type to return
        @param num_chunks: Number of chunks to read in at at time. This is necessary due
                           to a performance issue with h5py
        @param num_training_items: Number of items in each dataset to use for training
        @param num_validation_items: Number of items from each dataset to use for validation
        @param num_testing_items: Number of items in each dataset to use for testing
        """
        self._filename_list = filename_list
        self._label_list = label_list
        self._size = size
        self._dtype = dtype
        self._num_chunks = num_chunks
        self._index = index
        
        self._current_chunk = 0
        
        super(DataFetcher, self).__init__()

        assert self._index.shape[0] % self._num_chunks == 0, "Number of training items must be divisible by number of chunks"
        self._items_per_chunk = self._index.shape[0] // self._num_chunks

        data_retriever = DataRetriever(self._filename_list, self._label_list, self._size, self._dtype)
        num_images = data_retriever.get_num_images()
        num_labels = len(self._label_list)

    def perturb(self):
        self._current_chunk = (self._current_chunk + 1) % self._num_chunks

    def output(self):

        data_retriever = DataRetriever(self._filename_list, self._label_list, self._size, self._dtype)

        data = OrderedDict()
        metadata = OrderedDict()

        data_label = 'chunk_' + str(self._current_chunk)

        data_slice = subarray_slice(self._current_chunk, self._items_per_chunk)

        data[data_label] = data_retriever.get_images(self._index[data_slice,:])
        metadata[data_label] = OrderedDict()
        metadata[data_label]['Num_Chunks'] = self._num_chunks
        metadata[data_label]['Current_Chunk'] = self._current_chunk
        metadata[data_label]['Labels'] = self._index[data_slice, 0]

        return ImageWrapper(data, -1, metadata)


    def randomizeIndex(self):
        """
        Shuffle training index
        """
        np.random.shuffle(self._index)
