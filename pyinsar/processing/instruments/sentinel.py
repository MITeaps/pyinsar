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


# Package imports
from pyinsar.processing.utilities.generic import phase_shift, find_closest_time, OrbitInterpolation

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
    @param deramped_phase: Phase to be removed before the transformation and to be readded afterwards
    @param transformation_matrix: A 2x3 transformation matrix to be used by warpAffine by opencv

    @return transformed slc
    '''
    deramped_shifted = cv2.warpAffine(deramped_phase, transformation_matrix, None, cv2.INTER_LANCZOS4)

    deramped_slc = phase_shift(slc, deramped_phase)

    burst_slc_shifted =      cv2.warpAffine(deramped_slc.real, transformation_matrix, None, cv2.INTER_LANCZOS4) \
                        + 1j*cv2.warpAffine(deramped_slc.imag, transformation_matrix, None, cv2.INTER_LANCZOS4)

    return phase_shift(burst_slc_shifted, -deramped_shifted), deramped_shifted

def find_overlapping_valid_lines(metadata_tree):
    '''
    Determine which lines between bursts overlap

    @param metadata_tree: Sentinel metadata XML tree

    @return List of overlapping index ranges
    '''

    burst_info_list = metadata_tree.findall('swathTiming/burstList/burst')
    lines_per_burst = int(metadata_tree.find('swathTiming/linesPerBurst').text)
    valid_lines = get_valid_lines(metadata_tree)
    times, line_index, split_index = retrieve_azimuth_time(metadata_tree)

    indices = []

    for index in range(1,len(burst_info_list)):
        burst_start_index = lines_per_burst * index
        valid_burst_start_index = burst_start_index + np.argmax(valid_lines[burst_start_index:])

        valid_burst_end_index = (burst_start_index-1) - np.argmax(valid_lines[:burst_start_index-1][::-1])

        start_index = find_closest_time(times[:burst_start_index], times[valid_burst_start_index])
        end_index = find_closest_time(times[burst_start_index:], times[valid_burst_end_index])

        indices.append( ((start_index, valid_burst_end_index), (valid_burst_start_index, end_index)) )

    return indices


def get_valid_lines(metadata_tree, per_burst = False):
    '''
    Retrieve all lines that contain some valid data

    @param metadata_tree: Sentinel XML metadata tree
    @param per_burst: Retrieve the burst data as seperate arrays

    @return Sentinel data for all lines that are valid
    '''

    burst_info_list = metadata_tree.findall('swathTiming/burstList/burst')
    lines_per_burst = int(metadata_tree.find('swathTiming/linesPerBurst').text)

    valid_lines = []

    for burst in burst_info_list:
        valid_lines_in_burst = [int(val) for val in burst.find('firstValidSample').text.split(' ')]
        if per_burst:
            valid_lines.append(np.array(valid_lines_in_burst) != -1)
        else:
            valid_lines += valid_lines_in_burst

    if per_burst:
        return valid_lines
    else:
        return np.array(valid_lines) != -1


def select_valid_lines(data, tree, cut=True):
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

class RampPolynomial(object):
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

    def __init__(self, metadata, modulation=True):
        '''
        Initialize Sentiel Ramp

        @param metadata: ElemenTree containing the SLC metadata
        @param modulation: Whether to include modulation in the ramp
        '''

        self._metadata = metadata
        self.modulation = modulation

        tree = self._metadata['Tree']

        burst_list = tree.findall('swathTiming/burstList/burst')
        az_fm_rate_list = tree.findall('generalAnnotation/azimuthFmRateList/azimuthFmRate')
        doppler_centroid_list = tree.findall('dopplerCentroid/dcEstimateList/dcEstimate')

        self._num_bursts = len(burst_list)


        self._interp = OrbitInterpolation(self._metadata['Orbit'])

        self._lines_per_burst = int(tree.find('swathTiming/linesPerBurst').text)
        self._samples_per_burst = int(tree.find('swathTiming/samplesPerBurst').text)

        self._az_time_interval = float(tree.find('imageAnnotation/imageInformation/azimuthTimeInterval').text)
        self._az_steering_rate = np.deg2rad(float(tree.find('generalAnnotation/productInformation/azimuthSteeringRate').text))

        radar_freq = float(tree.find('generalAnnotation/productInformation/radarFrequency').text)
        self._radar_lambda = c/radar_freq

        self._slant_range_time = float(tree.find('imageAnnotation/imageInformation/slantRangeTime').text)
        self._slant_range_time_interval = 1/float(tree.find('generalAnnotation/productInformation/rangeSamplingRate').text)

        self._doppler_centroid_scanning_rate_list = list(map(self._doppler_centroid_rate, burst_list))
        self._doppler_fm_rate_list = list(map(self._doppler_fm_rate, az_fm_rate_list))
        self._doppler_centroid_frequency_list = list(map(self._doppler_centroid_frequency, doppler_centroid_list))


    def __call__(self, lines, samples, index):
        '''
        Calculate the phase change from the Sentinel ramp and modulation

        @param lines: Index of lines
        @param samples: Index of samples
        @param index: Burst index (starts at 0)

        @return Phase due to ramp and modulation
        '''
        centroid_tops = self._doppler_centroid_tops(samples,
                                                    self._doppler_fm_rate_list[index],
                                                    self._doppler_centroid_scanning_rate_list[index])

        doppler_centroid_frequency_func = self._doppler_centroid_frequency_list[index]

        zero_doppler_azimuth_time = self._zero_doppler_azimuth_time(lines)
        ref_zero_doppler_azimuth_time = self._reference_zero_doppler_azimuth_time(samples,
                                                                                  doppler_centroid_frequency_func,
                                                                                  self._doppler_fm_rate_list[index])

        doppler_centroid_frequency = doppler_centroid_frequency_func(samples)

        ramp_phase = -np.pi * centroid_tops * (zero_doppler_azimuth_time - ref_zero_doppler_azimuth_time)**2
        modulation_phase = -2*np.pi * doppler_centroid_frequency * (zero_doppler_azimuth_time - ref_zero_doppler_azimuth_time)

        if self.modulation == True:
            return ramp_phase + modulation_phase
        else:
            return ramp_phase



    def _doppler_centroid_rate(self, burst):
        '''
        Generate a function to calculate doppler centroid rate from scanning the antenna

        @param burst: Elementree containing burst information

        @return Function to comptue the doppler centroid rate for a given range index
        '''
        az_start_time = pd.to_datetime(burst.find('azimuthTime').text)

        az_time_mid_burst =   az_start_time \
                            + pd.to_timedelta(self._az_time_interval*self._lines_per_burst/2,'s')

        speed = np.linalg.norm(self._interp(az_time_mid_burst, interp='velocity'))

        return self._az_steering_rate * 2 * speed / self._radar_lambda

    def _doppler_fm_rate(self, az_fm_rate):
        '''
        Generate a function for calculating the doppler FM rate

        @param az_fm_rate: Elementree containing metadat for the azimuth FM rate

        @return Function for coluationg the doppler FM rate for a given range index
        '''
        doppler_fm_rate_t0 = float(az_fm_rate.find('t0').text)
        children_names = set(child.tag for child in az_fm_rate.getchildren())

        coeff_labels = ['c0','c1','c2']

        if set(coeff_labels).issubset(children_names):
            doppler_fm_rate_coeffs = [float(az_fm_rate.find(label).text) for label in coeff_labels]

        elif 'azimuthFmRatePolynomial' in children_names:
            doppler_fm_rate_coeffs = [float(val) for val in az_fm_rate.find('azimuthFmRatePolynomial').text.split(' ')]

        else:
            raise ValueError("Cannot find azimuth FM rate polynomials in metadata")


        return RampPolynomial(doppler_fm_rate_t0,
                                doppler_fm_rate_coeffs,
                                self._slant_range_time_interval,
                                self._slant_range_time)

    def _doppler_centroid_tops(self, samples, doppler_fm_rate, doppler_centroid_rate_from_scanning_antenna):
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



    def _doppler_centroid_frequency(self, doppler_centroid_estimate):
        '''
        Functino for computing the dopplber centroid frequency

        @param doppler_centroid_estimate: Elementree metadata of the doppler centroid estaimte

        @return Function for computing the doppler centroid frequency
        '''
        doppler_centroid_coeffs = [float(poly) for poly in doppler_centroid_estimate.find('dataDcPolynomial').text.split(' ')]
        doppler_centroid_t0 = float(doppler_centroid_estimate.find('t0').text)
        return RampPolynomial(doppler_centroid_t0,
                          doppler_centroid_coeffs,
                          self._slant_range_time_interval,
                          self._slant_range_time)


    def _zero_doppler_azimuth_time(self, lines):
        '''
        Calculates the zero doppler azimuth time for a line

        Time is centered on the middle of the burst

        @param lines: Lines which the zero doppler time will be calculated.
        @return: Zero doppler azimuth time centered on the middle of the burst
        '''
        lines = lines % self._lines_per_burst

        return lines*self._az_time_interval - self._az_time_interval * self._lines_per_burst/2

    def _reference_zero_doppler_azimuth_time(self, samples, doppler_centroid_frequency, doppler_fm_rate):
        '''
        Beam crossing time for a given range

        @param samples: Input samples
        @param doppler_centroid_frequency: Function for calculating the doppler centroid frequency
        @param doppler_fm_rate: Function for calculating the doppler FM rate
        '''
        beam_center_crossing_time = lambda samples: -doppler_centroid_frequency(samples) / doppler_fm_rate(samples)
        return beam_center_crossing_time(samples) - beam_center_crossing_time(0)


def retrieve_azimuth_time(in_tree):
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

        start_date = pd.to_datetime(burst.find('azimuthTime').text)
        date_offsets = pd.to_timedelta((np.arange(lines_per_burst) * azimuth_time_interval)*1e9, 'ns')
        azimuth_time[index_slice] = start_date + date_offsets
        if index != 0:
            starting_indicies = [find_closest_time(azimuth_time[:start_index], start_date), start_index]
            first_overlap_indicies.append(starting_indicies)
            previous_burst_split = int(round(np.average([starting_indicies[0], start_index-1])))
            current_burst_split = find_closest_time(azimuth_time[start_index:], azimuth_time[previous_burst_split])

            split_indicies.append([previous_burst_split, current_burst_split])

    # return azimuth_time.astype('datetime64[ns, UTC]')

    line_index = np.arange(0, split_indicies[0][0], dtype='int')
    for index in range(0, len(split_indicies)-1):
        line_index = np.concatenate([line_index, np.arange(split_indicies[index][1], split_indicies[index+1][0], dtype='int')])
    line_index = np.concatenate([line_index, np.arange(split_indicies[-1][1], num_lines)])


    return azimuth_time, line_index, split_indicies


def read_geolocation(tree):
    '''
    Read in geolocation data

    @param tree: Sentinel metadata as an ElementTree

    @return Geolocation metadata
    '''
    geolocation_tree = tree.find('geolocationGrid/geolocationGridPointList')
    num_entries = int(geolocation_tree.attrib['count'])

    metadata_names = ['azimuthTime', 'slantRangeTime', 'line', 'pixel',
                      'latitude', 'longitude', 'height', 'incidenceAngle',
                      'elevationAngle']

    new_metadata_names = ['Azimuth Times', 'Slant Range Times', 'Lines',
                          'Samples', 'Latitudes', 'Longitudes', 'Heights',
                          'Elevation Angle']

    dtypes = ['datetime64[ns]', 'float', 'int', 'int', 'float', 'float',
              'float', 'float']

    conversions = [pd.to_datetime, float, int, int, float, float, float,
                  float]

    results = OrderedDict()


    for new_name, dtype in zip(new_metadata_names, dtypes):
        results[new_name] = np.zeros(num_entries, dtype=dtype)

    for index, point in enumerate(geolocation_tree):
        for old_name, new_name, conversion in zip(metadata_names, new_metadata_names, conversions):
            results[new_name][index] = conversion(point.find(old_name).text)

    return results

def update_geolocation_lines(tree, azimuth_times, geolocation_data):
    '''
    Update which line is associated with geolocation data using azimuth times

    @param tree: Sentinel XML metadata
    @param azimuth_times: Azimuth times
    @param geolocation_data: Geolocation data read in by read_geolocation

    @return New lines for the geolocation data
    '''
    range_sampling_interval = 1/float(tree.find('generalAnnotation/productInformation/rangeSamplingRate').text)

    azimuth_times = azimuth_times.reset_index(drop=True)

    lines = np.zeros_like(geolocation_data['Lines'])

    for index, (az_time, sample) in enumerate(zip(geolocation_data['Azimuth Times'],geolocation_data['Samples'])):
        lines[index] = find_closest_time(azimuth_times,
                                         az_time - pd.to_timedelta((sample * range_sampling_interval/2) * 1e9, 'ns'))

    return lines


def get_sentinel_extents(geolocation, offset=0.0):
    """
    Get the extents (latitude and longitude) of a sentinel-1 image given its geolocation information

    @param geolocation: Geolocation data read in by read_geolocation
    @param offset: Extra offset to add to the extent

    @return Latitude and longitude extents of a sentinel-1
    """
    lat_min = np.min(geolocation['Latitudes']) - offset
    lon_min = np.min(geolocation['Longitudes']) - offset
    lat_max = np.max(geolocation['Latitudes']) + offset
    lon_max = np.max(geolocation['Longitudes']) + offset

    return lat_min, lat_max, lon_min, lon_max
