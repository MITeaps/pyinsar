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

# PyInSAR imports
from pyinsar.processing.data_fetcher.hdf_retriever import DataFetcher
from pyinsar.processing.utilities.machine_learning import DataRetriever
from pyinsar.processing.utilities.generic import subarray_slice

# Third party imports
import numpy as np

def build_data_fetchers(filename_list, label_list, size, dtype, num_training_data_per_label, num_validation_data_per_label, num_testing_data_per_label, num_chunks):
    """
    Build training, validation, and testing HDF Retriever Data Fetchers

    @param filename_list: List of HDF file names
    @param label_list: List of labels for each HDF file
    @param num_training_data_per_label: Number of training data for each label
    @param num_validation_data_per_label: Number of validation data for each label
    @param num_testing_data_per_label: Number of testing data for each label

    @return List of HDF Retriever Data fetchers. One for training, one for validation, 
            and one for testing
    """

    data_retriever = DataRetriever(filename_list, label_list, size, dtype)

    num_labels = len(label_list)

    training_index = np.zeros((num_labels * num_training_data_per_label, 2), dtype = np.int)
    validation_index = np.zeros((num_labels * num_validation_data_per_label, 2), dtype = np.int)
    testing_index = np.zeros((num_labels * num_testing_data_per_label, 2), dtype = np.int)

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

    num_images = data_retriever.get_num_images()

    for label_index, label in enumerate(label_list):
        image_index = np.arange(num_images[label_index, 1])
        np.random.shuffle(image_index)

        training_image_slice = image_index_slice(0, num_training_data_per_label)
        validation_image_slice = image_index_slice(num_training_data_per_label, num_validation_data_per_label)
        testing_image_slice = image_index_slice(num_training_data_per_label + num_validation_data_per_label,
                                                num_testing_data_per_label)

        add_indices(training_index,
                    image_index[training_image_slice],
                    label_index,
                    label,
                    num_training_data_per_label)
        
        add_indices(validation_index,
                    image_index[validation_image_slice],
                    label_index,
                    label,
                    num_validation_data_per_label)
        
        add_indices(testing_index,
                    image_index[testing_image_slice],
                    label_index,
                    label,
                    num_testing_data_per_label)

    np.random.shuffle(training_index)

    data_fetcher_dict = OrderedDict()

    data_fetcher_names = ['training', 'validation', 'testing']
    data_fetcher_indices = [training_index, validation_index, testing_index]

    for index, label in zip(data_fetcher_indices, data_fetcher_names):
        data_fetcher_dict[label] = DataFetcher(filename_list, label_list, size, dtype, num_chunks, index)

    return data_fetcher_dict
