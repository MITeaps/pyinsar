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

class GDAL_DataFetcher(DataFetcherBase):
    """
    Data fetcher for loading interferograms produced compatiable with GDAL
    """

    def __init__(self, filename_list, label_list, verbose=False):
        """
        Initialize ISCE data fetcher

        @param filename_list: List of filenames of ISCE interferograms
        @param label_list: List of strings containing names for the interferograms
        @param verbose: Print extra information
        """
        self._filename_list = filename_list
        self._label_list = label_list

        super(ISCE_DataFetcher, self).__init__([], verbose)


    def output(self):
        """
        Load GDAL data

        @return Image data wrapper
        """

        data_dict = OrderedDict()
        meta_dict = OrderedDict()

        for label, filename in zip(self._label_list, self._filename_list):
            ds = import_georaster.open_georaster(filename)

            data_dict[label] = ds.ReadAsArray()

            meta_dict[label] = OrderedDict()
            meta_dict[label]['WKT'] = ds.GetProjection()
            meta_dict[label]['GeoTransform'] = ds.GetGeoTransform()

        return ImageWrapper(data_dict, meta_data = meta_dict)
