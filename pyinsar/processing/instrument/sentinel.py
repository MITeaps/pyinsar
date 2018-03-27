# Package imports
from .methods import OrbitInterpolation
from .utilities.generic import phase_shift, findClosestTime

# Standard library imports
from collections import OrderedDict
import xml.etree.ElementTree as ET

# 3rd party imports
import cv2
import numpy as np
import pandas as pd
from scipy.constants import c


def transform_slc(slc, deramped_phase, transformation_matrix):
    '''
    @param slc: Input slc
    @param ramp_phase: Phase to be removed before the transformation and to be readded afterwards
    @param transformation_matrix: A 2x3 transformation matrix to be used by warpAffine by opencv

    @return transformed slc
    '''
    deramped_shifted = cv2.warpAffine(deramped_phase, transformation_matrix, None, cv2.INTER_LANCZOS4)
    
    deramped_slc = phase_shift(slc, deramped_phase)

    burst_slc_shifted =      cv2.warpAffine(deramped_slc.real, transformation_matrix, None, cv2.INTER_LANCZOS4) \
                        + 1j*cv2.warpAffine(deramped_slc.imag, transformation_matrix, None, cv2.INTER_LANCZOS4)
        
    return phase_shift(burst_slc_shifted, -deramped_shifted)

def selectValidLines(data, tree,cut=True):
    '''
    Extract burst information from SLC

    @param data: Input SLC data
    @param tree: Metadata as an ElementTree
    @param cut: Remove invalid lines
    
    @return A list containing individual images of each burst
    '''
    burst_info_list = tree.findall('swathTiming/burstList/burst')
    lines_per_burst = int(tree.find('swathTiming/linesPerBurst').text)
    burst_list = []
    for index, burst in enumerate(burst_info_list):
        if cut==True:
            valid_lines = [int(val) for val in burst.find('firstValidSample').text.split(' ')]
            valid_lines = np.array(valid_lines) != -1
        
            burst_list.append(data[index*lines_per_burst:(index+1)*lines_per_burst,:][valid_lines,:])
        else:
            burst_list.append(data[index*lines_per_burst:(index+1)*lines_per_burst,:])

    return burst_list

class DerampPolynomial(object):
    '''
    Polynomial used for quantities relating to deramping sentinel
    '''
    def __init__(self, t0, coeff_list, slant_range_time_interval, slant_range_time):
        '''
        Initialize Deramp Polynomial object

        @param t0: Starting time
        @param coeff_list: List of coefficients
        @param slant_range_time_interval: Time between range samples
        @param slant_range_time: Two way slant range time
        '''
        self._t0 = t0
        self._coeff_list = coeff_list
        self._slant_range_time_interval = slant_range_time_interval
        self._slant_range_time = slant_range_time
        
    def __call__(self, t):
        '''
        Evaluate the polynomial
        
        @param t: Input time
        
        @return Value of polynomial at time t
        '''
        res = 0
        for index, coeff in enumerate(self._coeff_list):
            res += coeff*(t*self._slant_range_time_interval + self._slant_range_time - self._t0)**index
        return res

class SentinelRamp(object):
    '''
    Calcuate the combined ramp and modulated phase in Sentinel

    This class was created following the guide at:
    https://sentinel.esa.int/documents/247904/1653442/Sentinel-1-TOPS-SLC_Deramping
    '''
    
    def __init__(self, metadata):
        '''
        Initialize Sentiel Ramp

        @param metadata: Elementree containing the SLC metadata
        '''
        
        self._metadata = metadata
        tree = self._metadata['Tree']
        
        burst_list = tree.findall('swathTiming/burstList/burst')
        az_fm_rate_list = tree.findall('generalAnnotation/azimuthFmRateList/azimuthFmRate')
        doppler_centroid_list = tree.findall('dopplerCentroid/dcEstimateList/dcEstimate')
        
        self._num_bursts = len(burst_list)

        
        self._vel_interp = OrbitInterpolation(self._metadata['Orbit'],interp_target='velocity')
        
        self._lines_per_burst = int(tree.find('swathTiming/linesPerBurst').text)
        self._samples_per_burst = int(tree.find('swathTiming/samplesPerBurst').text)
        
        self._az_time_interval = float(tree.find('imageAnnotation/imageInformation/azimuthTimeInterval').text)        
        self._az_steering_rate = np.deg2rad(float(tree.find('generalAnnotation/productInformation/azimuthSteeringRate').text))
        
        radar_freq = float(tree.find('generalAnnotation/productInformation/radarFrequency').text)
        self._radar_lambda = c/radar_freq
        
        self._slant_range_time = float(tree.find('imageAnnotation/imageInformation/slantRangeTime').text)
        self._slant_range_time_interval = 1/float(tree.find('generalAnnotation/productInformation/rangeSamplingRate').text)
        
        self._doppler_centroid_scanning_rate_list = list(map(self._dopplerCentroidRate, burst_list))
        self._doppler_fm_rate_list = list(map(self._dopplerFMRate, az_fm_rate_list))
        self._doppler_centroid_frequency_list = list(map(self._dopplerCentroidFrequency, doppler_centroid_list))

        
#     def __call__(self, lines, samples, invert=False):
        
#         piecewise_functions = [lambda lines, samples: self._derampedPhase(lines, samples, index) for index in range(self._num_bursts)]
        
#         np.piecewise()
        
        
#    def _derampedPhase(self, lines, samples, index):
    def __call__(self, lines, samples, index):
        '''
        Calculate the phase change from the Sentinel ramp and modulation

        @param lines: Index of lines
        @param samples: Index of samples
        @param index: Burst index

        @return Phase due to ramp and modulation
        '''
        centroid_tops = self._dopplerCentroidTops(samples, 
                                                  self._doppler_fm_rate_list[index], 
                                                  self._doppler_centroid_scanning_rate_list[index])
        
        doppler_centroid_frequency_func = self._doppler_centroid_frequency_list[index]
        
        zero_doppler_azimuth_time = self._zeroDopplerAzimuthTime(lines)
        ref_zero_doppler_azimuth_time = self._referenceZeroDopplerAzimuthTime(samples,
                                                                              doppler_centroid_frequency_func,
                                                                              self._doppler_fm_rate_list[index])
        
        doppler_centroid_frequency = doppler_centroid_frequency_func(samples)
        
        return  -np.pi * centroid_tops * (zero_doppler_azimuth_time - ref_zero_doppler_azimuth_time)**2 \
               - 2*np.pi * doppler_centroid_frequency * (zero_doppler_azimuth_time - ref_zero_doppler_azimuth_time)

        
    def _dopplerCentroidRate(self, burst):
        '''
        Generate a function to calculate doppler centroid rate from scanning the antenna

        @param burst: Elementree containing burst information

        @return Function to comptue the doppler centroid rate for a given range index
        '''
        az_start_time = pd.to_datetime(burst.find('azimuthTime').text)
        
        az_time_mid_burst =   az_start_time \
                            + pd.to_timedelta(self._az_time_interval*self._lines_per_burst/2,'s')
            
        speed = np.linalg.norm(self._vel_interp(az_time_mid_burst))

        return self._az_steering_rate * 2 * speed / self._radar_lambda
    
    def _dopplerFMRate(self, az_fm_rate):
        '''
        Generate a function for calculating the doppler FM rate

        @param az_fm_rate: Elementree containing metadat for the azimuth FM rate
        
        @return Function for coluationg the doppler FM rate for a given range index
        '''
        doppler_fm_rate_t0 = float(az_fm_rate.find('t0').text)
        doppler_fm_rate_coeffs = [float(az_fm_rate.find(label).text) for label in ['c0','c1','c2']]
        
        return DerampPolynomial(doppler_fm_rate_t0, 
                                doppler_fm_rate_coeffs, 
                                self._slant_range_time_interval,
                                self._slant_range_time)
    
    def _dopplerCentroidTops(self, samples, doppler_fm_rate, doppler_centroid_rate_from_scanning_antenna):
        '''
        Generate a function for computing the centroid rate in TOPS data

        @param samples: Input range samples 
        @param doppler_fm_rate: Function for computing the doppler FM rate
        @param doppler_centroid_rate_from_scanning_antenna: Function for computing the doppler centroid rate from
        scanning the antenna

        @return: The doppler centroid
        '''

    
        return (doppler_fm_rate(samples) * doppler_centroid_rate_from_scanning_antenna) \
              / (doppler_fm_rate(samples) - doppler_centroid_rate_from_scanning_antenna)
        
        
        
    def _dopplerCentroidFrequency(self, doppler_centroid_estimate):
        '''
        Functino for computing the dopplber centroid frequency

        @param doppler_centroid_estimate: Elementree metadata of the doppler centroid estaimte
        
        @return Function for computing the doppler centroid frequency
        '''
        doppler_centroid_coeffs = [float(poly) for poly in doppler_centroid_estimate.find('dataDcPolynomial').text.split(' ')]
        doppler_centroid_t0 = float(doppler_centroid_estimate.find('t0').text)
        return DerampPolynomial(doppler_centroid_t0, 
                          doppler_centroid_coeffs,
                          self._slant_range_time_interval,
                          self._slant_range_time)
    
    
    def _zeroDopplerAzimuthTime(self, lines):
        '''
        Calculates the zero doppler azimuth time for a line 

        Time is centered on the middle of the burst

        @param lines: Lines which the zero doppler time will be calculated.
        @return: Zero doppler azimuth time centered on the middle of the burst
        '''
        lines = lines % self._lines_per_burst
        
        return lines*self._az_time_interval - self._az_time_interval * self._lines_per_burst/2
        
    def _referenceZeroDopplerAzimuthTime(self, samples, doppler_centroid_frequency, doppler_fm_rate):
        '''
        Beam crossing time for a given range

        @param samples: Input samples
        @param doppler_centroid_frequency: Function for calculating the doppler centroid frequency
        @param doppler_fm_rate: Function for calculating the doppler FM rate
        '''
        beam_center_crossing_time = lambda samples: -doppler_centroid_frequency(samples) / doppler_fm_rate(samples)
        return beam_center_crossing_time(samples) - beam_center_crossing_time(0)
    

def retrieveAzimuthTime(in_tree):
    '''
    Retrieves the zero azimuth time for all the lines in the data

    @param in_tree: SLC Metadata as an ElementTree
    
    @return Pandas series of azimuth times for each line
    '''

    azimuth_time_interval = float(in_tree.find('imageAnnotation/imageInformation/azimuthTimeInterval').text)
    lines_per_burst = int(in_tree.find('swathTiming/linesPerBurst').text)
    num_bursts = len(in_tree.findall('swathTiming/burstList/burst'))
    num_lines = lines_per_burst * num_bursts
    azimuth_time = pd.Series(index=np.arange(num_lines), dtype='datetime64[ns]')
    first_overlap_indicies = []
    last_overlap_indicies = []
    split_indicies = []

    for index, burst in enumerate(in_tree.findall('swathTiming/burstList/burst')):

        start_index = index*lines_per_burst
        end_index = (index+1)*lines_per_burst
        index_slice = slice(start_index,  end_index)

        first_valid_samples = burst.find('firstValidSample')    
        start_date = pd.to_datetime(burst.find('azimuthTime').text)
        date_offsets = pd.to_timedelta(np.arange(lines_per_burst) * azimuth_time_interval, 's')
        azimuth_time[index_slice] = start_date + date_offsets
        if index != 0:
            starting_indicies = [findClosestTime(azimuth_time[:start_index], start_date), start_index]
            first_overlap_indicies.append(starting_indicies)
            previous_burst_split = int(round(np.average([starting_indicies[0], start_index-1])))
            current_burst_split = findClosestTime(azimuth_time[start_index:], azimuth_time[previous_burst_split])

            split_indicies.append([previous_burst_split, current_burst_split])
        
    # return azimuth_time.astype('datetime64[ns, UTC]')

    line_index = np.arange(0, split_indicies[0][0], dtype='int')
    for index in range(0, len(split_indicies)-1):
        line_index = np.concatenate([line_index, np.arange(split_indicies[index][1], split_indicies[index+1][0], dtype='int')])
    line_index = np.concatenate([line_index, np.arange(split_indicies[-1][1], num_lines)])
        
    
    return azimuth_time, line_index, split_indicies


def readGeoLocation(in_geolocation_tree, azimuth_time):
    '''
    Read in geolocation data

    @param in_geolocation_tree: ElementTree containing geolocation data
    @param azimuth time: Azimuth time for each line

    @return longitude information from metadata
    '''
    azimuth_time = azimuth_time.reset_index(drop=True)
    num_entries = int(in_geolocation_tree.attrib['count'])
    lines = np.zeros(num_entries)
    samples = np.zeros(num_entries)
    latitudes = np.zeros(num_entries)
    longitudes = np.zeros(num_entries)
    for index, point in enumerate(in_geolocation_tree):
        lat = float(point.find('latitude').text)
        lon = float(point.find('longitude').text)
        sample = float(point.find('pixel').text)
        time = pd.to_datetime(point.find('azimuthTime').text)
        line = findClosestTime(azimuth_time, time)
        lines[index] = line
        samples[index] = sample
        latitudes[index] = lat
        longitudes[index] = lon
        
    results = OrderedDict()
    results['Lines'] = lines
    results['Samples'] = samples
    results['Latitude'] = latitudes
    results['Longitude'] = longitudes

    return results


def parseSatelliteData(in_satellite_file):
    '''
    Parse Sentinel satelllite data

    @param in_sentinel_file: Satellite orbit filename

    @return DataFrame of orbit information
    '''
    satellite_tree = ET.parse(in_satellite_file)
    
    names = ['TAI', 'UTC', 'UT1','Absolute_Orbit', 'X', 'Y', 'Z', 'VX', 'VY', 'VZ', 'Quality']
    time_converter = lambda x: pd.to_datetime(x[4:])
    converters = [time_converter, time_converter, time_converter, int, float, float, float, 
                  float, float, float, lambda x: x]
    tmp_data = []

    for orbit in satellite_tree.findall('Data_Block/List_of_OSVs/OSV'):
        row = []
        for name, converter in zip(names, converters):
            row.append(converter(orbit.find(name).text))
        tmp_data.append(row)

    return pd.DataFrame(tmp_data, columns=names)
