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

class DataRetriever(DataFetcherBase):
    """
    Data fetcher for retrieving hdf image data made for training in convolutional neural networks
    """

    def __init__(self, filename_list, label_list, size, dtype, num_chunks, num_training_items, num_validation_items, num_testing_items):
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
        self._num_training_items = num_training_items
        self._num_validation_items = num_validation_items
        self._num_testing_items = num_testing_items
        
        self._current_chunk = 0
        
        super(HDFDataRetriever, self).__init__()

        assert self._num_training_items % self._num_chunks == 0, "Number of training items must be divisible by number of chunks"
        self._items_per_chunk = (self._num_training_items * len(self._label_list)) // self._num_chunks

        data_retriever = DataRetriever(self._filename_list, self._label_list, self._size, self._dtype)
        num_images = data_retriever.get_num_images()
        num_labels = len(self._label_list)

        # self._full_index = np.zeros((np.sum(num_images[:,1]), 2), dtype=np.int)

        # start_image_index = 0
        # for num_images_index, label in enumerate(self._label_list):
        #     end_image_index = start_image_index + num_images[num_images_index, 1]
        #     self._full_index[start_image_index:end_image_index, 0] = label
        #     self._full_index[start_image_index:end_image_index, 1] = np.arange(end_image_index-start_image_index)
        #     start_image_index = end_image_index



        self._training_index = np.zeros((num_labels * self._num_training_items, 2), dtype = np.int)
        self._validiation_index = np.zeros((num_labels * self._num_validation_items, 2), dtype = np.int)
        self._testing_index = np.zeros((num_labels * self._num_testing_items, 2), dtype = np.int)


        def add_indices(index_array, image_index, label_index, label, num_items):
            """
            Add indices to an index array

            @param index_array: Array being used as the index
            @param image_index: Indices of imaging to insert into the index array
            @param label_index: Index of the current label
            @param label: Label of the current image index
            """
            my_slice = subarray_slice(label_index, num_items)
            index_array[my_slice, 0] = label
            index_array[my_slice, 1] = image_index

        def image_index_slice(num_skip_items, num_items):
            """
            Select num_items from an array after skipping num_skip_items

            @param num_skip_items: Number of items to skip
            @param num_items: Number of items to select
            @return slice that starts after num_skip_items and is num_items long
            """
            return slice(num_skip_items, num_items + num_skip_items)


        for label_index, label in enumerate(self._label_list):
            image_index = np.arange(num_images[label_index, 1])
            np.random.shuffle(image_index)

            training_image_slice = image_index_slice(0, self._num_training_items)
            validation_image_slice = image_index_slice(self._num_training_items, self._num_validation_items)
            testing_image_slice = image_index_slice(self._num_training_items + self._num_validation_items, self._num_testing_items)

            add_indices(self._training_index, image_index[training_image_slice], label_index, label, self._num_training_items)
            add_indices(self._validiation_index, image_index[validation_image_slice], label_index, label, self._num_validation_items)
            add_indices(self._testing_index, image_index[testing_image_slice], label_index, label, self._num_testing_items)

        np.random.shuffle(self._training_index)

    def perturb(self):
        self._current_chunk = (self._current_chunk + 1) % self._num_chunks

    def output(self):

        data_retriever = DataRetriever(self._filename_list, self._label_list, self._size, self._dtype)

        data = OrderedDict()
        metadata = OrderedDict()

        data_label = 'chunk_' + str(self._current_chunk)

        data_slice = subarray_slice(self._current_chunk, self._items_per_chunk)

        data[data_label] = data_retriever.get_images(self._training_index[data_slice,:])
        metadata[data_label] = OrderedDict()
        metadata[data_label]['Num_Chunks'] = self._num_chunks
        metadata[data_label]['Current_Chunk'] = self._current_chunk
        metadata[data_label]['Labels'] = self._training_index[data_slice, 0]

        return ImageWrapper(data, -1, metadata)

