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

# Pyinsar imports
from pyinsar.processing.utilities.generic import generateMatplotlibRectangle

# Scikit Discovery imports
from skdiscovery.data_structure.framework.base import PipelineItem

# 3rd party imports
import matplotlib.pyplot as plt
import matplotlib as mpl



class ShowCNNClasses(PipelineItem):
    """
    Dispay CNN Classifications on segments of an image
    """

    def __init__(str_description, class_name, colors):
        """
        Initialize ShowCNNClassesItem

        @param str_description: String name of item
        @param class_name: Name of classes
        @param colors: List of colors containing a color for each class
        """
        self.class_name = class_name
        self.colors = colors

        super(ShowCNNClasses, self).__init__(str_description)

    def process(self, obj_data):
        """
        Show the images with classifications

        @param obj_data: Image data wrapper
        """
        for label, data in obj_data.getIterator():

            extents = obj_data.info(label)[self.class_name]['extents']
            labels = obj_data.info(label)[self.class_name]['labels']

            ax = plt.axes()

            for label, color in zip(labels, self.colors):
                patch_collection = mpl.collections.PathCollection([generateMatplotlibRectangle(extent) for extent in extents[labels == label]],
                                                                  facecolor=self, alpha = 0.5)
                ax.add_collection(patch_collection)
