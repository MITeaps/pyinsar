from collections import OrderedDict

from skdaccess.framework.data_class import DataFetcherBase, ImageWrapper
from pyinsar.processing.deformation.elastic_halfspace.okada import compute_okada_displacement

class DataFetcher(DataFetcherBase):
    """
    Generates data from an Okada model
    """
    def __init__(self, ap_paramList, xx_array, yy_array, verbose=False):
        """
        Initialize Okada DataFetcher

        @ap_paramList[fault_centroid_x]: x centroid
        @ap_paramList[fault_centroid_y]: y centroid
        @ap_paramList[fault_centroid_depth]: Fault depth
        @ap_paramList[fault_strike]: Fault strike
        @ap_paramList[fault_dip]: Fault dip
        @ap_paramList[fault_length]: Fault Length
        @ap_paramList[fault_width]: Fault width
        @ap_paramList[fault_rake]: Fault rake
        @ap_paramList[fault_slip]: Fault slip
        @ap_paramList[fault_open]: Fault open
        @ap_paramList[poisson_ratio]: Poisson ratio
        @xx_array: Array of x coordinates
        @yy_array: Array of y coordinates
        @verbose: Print out extra information
        """
        self._xx_array = xx_array
        self._yy_array = yy_array

        super(DataFetcher, self).__init__(ap_paramList, verbose)


    def output(self):
        """
        Output deformation in an image wrapper

        @return Deformation in an Image wrapper 
        """

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
