from collections import OrderedDict

from skdaccess.framework.data_class import DataFetcherBase, ImageWrapper
from pyinsar.processing.deformation.elastic_halfspace.okada import compute_okada_displacement

class DataFetcher(DataFetcherBase):
    '''
    Generates data from an Okada model
    '''
    def __init__(self, ap_paramList, xx_array, yy_array, verbose=False):
        '''
        Initialize DataFetcher

        @param dictionary where the keys are filenames and the values are the dataset names
        @param verbose: Output extra debug information
        '''
        self._xx_array = xx_array
        self._yy_array = yy_array

        super(DataFetcher, self).__init__(ap_paramList, verbose)


    def output(self):

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
