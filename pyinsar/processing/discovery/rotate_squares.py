from collections import OrderedDict

from skdiscovery.data_structure.framework.base import PipelineItem
from skdiscovery.utilities.patterns.image_tools import generateSquaresAroundPoly

import numpy as np
import scipy as sp

from pyinsar.processing.utilities import insar_simulator_utils

def rotateSquare(image, square, angle, order):
    x_start, y_start, x_end, y_end = np.rint(square.bounds).astype(np.int)
    size = x_end - x_start

    half_size = np.ceil(size/2.0).astype(np.int)
    x_slice = slice(x_start - half_size, x_end + half_size)
    y_slice = slice(y_start - half_size, y_end + half_size)
    
    rotated_image = sp.ndimage.rotate(image[y_slice, x_slice], angle,cval=np.nan, reshape=True, order=order)
    return insar_simulator_utils.crop_array_from_center(rotated_image, (size, size))


class RotateSquares(PipelineItem):
    def __init__(self, str_description, ap_paramList, square_result_name, angles, clean=True):

        self._angles = angles

        self._square_result_name = square_result_name

        self._clean = clean

        super(RotateSquares, self).__init__(str_description, ap_paramList)


    def process(self, obj_data):

        data_result = OrderedDict()
        meta_result = OrderedDict()


        square_dict = obj_data.getResults()[self._square_result_name]

        spline_order = self.ap_paramList[0]()

        for label, data in obj_data.getIterator():
            square_list = square_dict[label]

            i = 0
            for square in square_list:
                for angle in self._angles:
                    new_data = rotateSquare(data, square, angle, spline_order)

                    if not self._clean or np.count_nonzero(np.isnan(new_data)) == 0:
                        new_label = 'Image {}, Square {:d}, rotated {:f} degrees'.format(label, i, angle)
                        data_result[new_label] = new_data
                        meta_result[new_label] = OrderedDict()
                        meta_result[new_label]['angle'] = angle
                        meta_result[new_label]['square'] = square
                        meta_result[new_label]['original_image'] = label
                        i += 1

                    

        
        obj_data.update(data_result)
        obj_data.updateMetadata(meta_result)
