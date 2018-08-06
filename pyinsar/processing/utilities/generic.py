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
from urllib.parse import urlencode
import json
import re

# 3rd party imports
from six.moves.urllib.request import urlopen
import cv2
import numpy as np
import pandas as pd
from osgeo import osr, gdal
import shapely as shp
import shapely.geometry
import shapely.wkt


import warnings
with warnings.catch_warnings():
    warnings.simplefilter('ignore', FutureWarning)
    import statsmodels.api as sm

from scipy.signal import convolve
from scipy.interpolate import interp1d
# from geodesy import wgs84
from sklearn.linear_model import RANSACRegressor


def get_image_extents(geotransform, shape):
    """
    Get extents of in projection coordinates

    @param geotransform: Geo transform for converting between pixel and projected coordinates
    @param shape: Shape of image
    """
    georaster_x_size = shape[1]
    georaster_y_size = shape[0]
    xmin = geotransform[0]
    ymax = geotransform[3]
    xmax = xmin + geotransform[1]*georaster_x_size + geotransform[2]*georaster_y_size
    ymin = ymax + geotransform[4]*georaster_x_size + geotransform[5]*georaster_y_size

    return (xmin, xmax, ymin, ymax)


def proj4StringToDictionary(proj4_string):
    '''
    Convert a proj4 string into a dictionary

    Statements with no value are given a value of None

    @param proj4_string: Proj4 string
    @return Dictionary containing proj4 parameters as a OrderedDict
    '''
    proj4_dict = OrderedDict(re.findall('\+([^ ]+)=([^ ]+)', proj4_string))
    all_params = re.findall('\+([^ =]+)', proj4_string)

    for param in all_params:
        if param not in proj4_dict:
            proj4_dict[param] = None

    return proj4_dict


# def getCartopyProjection(in_wkt):
#     my_spatial = osr.SpatialReference()
#     my_spatial.ImportFromWkt(in_wkt)

#     proj4_string = my_spatial.ExportToProj4()

#     proj4_params_dict = proj4StringToDictionary(proj4_string)

#     if 'datum' in proj4_params_dict:
#         datum = proj4_params_dict['datum']
#         del proj4_params_dict['datum']

#     else:
#         datum = None

#     if 'ellps' in proj4_params_dict:
#         ellipse = proj4_params_dict['ellps']
#         del proj4_params_dict['ellps']

#     else:
#         ellipse='WGS84'

#     globe = ccrs.Globe(datum=datum, ellipse=ellipse)

#     return ccrs.Projection(**proj4_params_dict, globe=globe)


def sorted_alphanumeric(l):
    '''
    Sort a list of strings with numbers

    @param l: The list

    @return The sorted list
    '''
    convert = lambda text: int(text) if text.isdigit() else text
    alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]

    return sorted(l, key = alphanum_key)

def phase_shift(data, phase):
    '''
     Apply a phase shift to data

    @param data: Input data
    @param phase: Input phase

    @return data shifted by phase
    '''
    return  (data.real*np.cos(phase) + data.imag*np.sin(phase)) \
            + 1j*(-data.real*np.sin(phase) + data.imag*np.cos(phase))

def find_closest_time(time, date):
    '''
    Find the closest time to a date

    @param time: Pandas series of datetimes
    @param date: Input date

    @return Index of closest time to date
    '''
    return np.abs(time-date).idxmin()


def rotate(col_vectors, az, ay, ax, dtype=np.float64):
    '''
    Rotate 3 dimensional column vectors

    @param col_vectors: Array of column vectors
    @param az: Angle for rotation about the z axis
    @param ay: Angle for rotation about the y axis
    @param ax: Angle for rotation about the x axis
    @param dtype: Data type to use

    @param return Rotated vectors
    '''
    rz = np.array([[np.cos(az), -np.sin(az), 0], [np.sin(az), np.cos(az), 0], [0, 0, 1]], dtype=dtype)
    ry = np.array([[np.cos(ay), 0, np.sin(ay)], [0, 1, 0], [-np.sin(ay), 0, np.cos(ay)]], dtype=dtype)
    rx = np.array([[ 1, 0, 0], [0, np.cos(ax), -np.sin(ax)], [0, np.sin(ax), np.cos(ax)]], dtype=dtype)

    rot = rx @ ry @ rz

    return rot @ col_vectors

def translate(col_vectors, delta_x, delta_y, delta_z):
    '''
    Translate 3 dimensional column vectors

    @param col_vectors: Array of column vectors
    @param delta_x: Move this many units in the x direction
    @param delta_y: Move this many units in the y direction
    @param delta_z: Move this many units in the z direction

    @return Translated vectors
    '''
    col_vectors = col_vectors.copy()
    col_vectors[0,:] += delta_x
    col_vectors[1,:] += delta_y
    col_vectors[2,:] += delta_z

    return col_vectors


class OrbitInterpolation(object):
    '''
    Class for interpolating satellite positions
    '''

    def __init__(self, orbit_data, time_name = 'UTC'):
        '''
        Initilaize orbit interpolation object

        @param orbit_data: Orbit position data
        @param time_name: Name of time column name in Orbit position data. Set this to None
                          to use the data frame index
        '''

        self._orbit_data = orbit_data
        if time_name == None:
            self._start_date = self._orbit_data.index[0]
            self._elapsed_time = (self._orbit_data.index - self._start_date).total_seconds()

        else:
            self._start_date = orbit_data[time_name].iloc[0]
            self._elapsed_time = (pd.DatetimeIndex(self._orbit_data[time_name]) - self._start_date).total_seconds()

        self._pos_labels = ['X','Y','Z']
        self._vel_labels = ['VX', 'VY', 'VZ']

        self._pos_interp_functions = OrderedDict()
        self._vel_interp_functions = OrderedDict()

        for label in self._pos_labels:
            self._pos_interp_functions[label] = interp1d(self._elapsed_time, self._orbit_data[label], 'cubic')

        for label in self._vel_labels:
            self._vel_interp_functions[label] = interp1d(self._elapsed_time, self._orbit_data[label], 'cubic')

    def get_start_date(self):
        '''
        Get starting date used in the interpolation

        @return Starting date
        '''
        return self._start_date


    def __call__(self, in_time, in_datetime=True, interp='position'):
        '''
        Compute the satellites position or velocity

        @param in_time: Time of interest
        @param in_datetime: Input is a datetime object (otherwise it's assumed its seconds from start date)
        @param interp: Interpolate "position" or "velocity"

        @return Satellite position or velocity at in_time
        '''

        if in_datetime:
            elapsed_time = (in_time - self._start_date) / pd.to_timedelta(1,'s')
        else:
            elapsed_time = in_time

        results = []

        if interp == 'position':
            label_list = self._pos_labels
            data_dict = self._pos_interp_functions
        elif interp == 'velocity':
            label_list = self._vel_labels
            data_dict = self._vel_interp_functions
        else:
            raise ValueError('Interp type {} not understood'.format(interp))

        for label in label_list:
            tmp_res = data_dict[label](elapsed_time)
            if tmp_res.ndim == 0:
                tmp_res = tmp_res.reshape(-1)[0]
            results.append(tmp_res)

        return np.array(results).T


def coherence(s1, s2, window, topo_phase = 0):
    '''
    This function computes the coherence between two SLCs

    The coherence is estimated using an equation presented in
    InSAR processing: a practical approach, equation 2.7

    @param s1: The first single look complex image
    @param s2: The second single look complex image
    @param window: Tuple specifing y, and x window size
    @param topo_phase: Change in phase due to topography

    @return Numpy array of the coherence
    '''

    kernel = np.ones(window, dtype=s1.dtype)

    numerator = s1 * np.conj(s2) * np.exp(-1j * topo_phase)
    numerator = convolve(numerator, kernel, mode='same', method='direct')

    denom_1 = convolve(s1 * np.conj(s1), kernel, mode='same', method='direct')
    denom_2 = convolve(s2 * np.conj(s2), kernel, mode='same', method='direct')

    denominator = np.sqrt(denom_1 * denom_2)


    return numerator / denominator


def scale_image(input_data, vmin=None, vmax=None):

    if vmin==None or vmax==None:
        stddev = sm.robust.mad(input_data.ravel())
        middle = np.median(input_data.ravel())

    if vmin == None:
        vmin = middle - 1*stddev

    if vmax == None:
        vmax = middle + 1*stddev

    input_data = input_data.astype(np.float)
    input_data[input_data<vmin] = vmin
    input_data[input_data>vmax] = vmax

    input_data = np.round((input_data - vmin) * 255 / (vmax-vmin)).astype(np.uint8)

    return input_data


def keypoints_align(img1, img2, max_matches=40, invert=True):
    """
    *** In Development *** Determine transformation matrix for aligning images

    @param img1: First image
    @param img2: Second image
    @param max_matches: Maximum number of matches between the two images
    @param invert: Invert the transformation matrix

    @return: Transformation matrix that connects two images
    """

    def buildMatchedPoints(in_matches, query_kp, train_kp):
        query_index = [match.queryIdx for match in in_matches]
        train_index = [match.trainIdx for match in in_matches]

        sorted_query_kp = [query_kp[i] for i in query_index]
        sorted_train_kp = [train_kp[i] for i in train_index]


        query_positions = [[kp.pt[0], kp.pt[1]] for kp in sorted_query_kp]
        train_positions = [[kp.pt[0], kp.pt[1]] for kp in sorted_train_kp]

        return query_positions, train_positions

    orb = cv2.ORB.create()
    slc1_keypoints = orb.detectAndCompute(img1, None)
    slc2_keypoints = orb.detectAndCompute(img2, None)

    bfmatcher = cv2.BFMatcher_create(cv2.NORM_HAMMING, crossCheck=True)
    matches = bfmatcher.match(slc1_keypoints[1], slc2_keypoints[1])

    matches.sort(key=lambda x: x.distance)

    cut_matches = matches[:max_matches]
    s1_coords, s2_coords = buildMatchedPoints(cut_matches, slc1_keypoints[0], slc2_keypoints[0])

    s1_coords = pd.DataFrame(s1_coords, columns=['Range','Azimuth'])
    s2_coords = pd.DataFrame(s2_coords, columns=['Range','Azimuth'])

    distances = pd.Series([mp.distance for mp in cut_matches],name='Distance')

    rg_res = sm.WLS(s2_coords['Range'],sm.add_constant(s1_coords.loc[:,['Range']]),
                weights=1.0/distances**2).fit()

    ransac_range = RANSACRegressor()
    ransac_range.fit(s1_coords.loc[:,['Range']], s2_coords.loc[:,['Range']])
    rg_scale = ransac_range.estimator_.coef_[0].item()
    rg_const = ransac_range.estimator_.intercept_

    ransac_azimuth = RANSACRegressor()
    ransac_azimuth.fit(s1_coords.loc[:,['Azimuth']], s2_coords.loc[:,['Azimuth']])
    az_scale = ransac_azimuth.estimator_.coef_[0].item()
    az_const = ransac_azimuth.estimator_.intercept_

    az_res = sm.WLS(s2_coords['Azimuth'],sm.add_constant(s1_coords.loc[:,['Azimuth']]),
                     weights=1.0/distances**2).fit()

    transformation_matrix = np.array([[rg_scale, 0,        rg_const],
                                      [0,        az_scale, az_const],
                                      [0,        0,        1]])

    if invert==True:
        transformation_matrix = np.linalg.inv(transformation_matrix)


    return(transformation_matrix[:2,:])


class FindNearestPixel(object):
    '''
    Find the nearest given a time
    '''
    def __init__(self, aztime, start_date):
        '''
        Initialize FindNearestPixel

        @param aztime: Input azimuth time series
        @param start_date: The starting date to use when compting the nearest pixel
        '''
        self._aztime = aztime
        self._start_date = start_date

        self._interp_func = interp1d(((aztime-start_date) / pd.to_timedelta('1s')).as_matrix(), aztime.index, kind='linear')

    def __call__(self, in_time):
        '''
        Find the pixel closest to in_time.

        The time is converted to a datetime based on the start_date used
        to create this object

        @param in_time: Input time
        @return: Pixel that is closest to the input time
        '''
        res = self._interp_func(in_time)
        if np.ndim(res) == 0:
            res = res.item()

        return res



def subarray_slice(index, num_items):
    """
    Returns a slice that selects for selecting a chunk out of an array

    @param index: Which chunk to select
    @param num_items: Number of items in a chunk
    @return A slice for selecting index*num_items to (index+1)*num_items
    """
    return slice(index * num_items, (index+1) * num_items)


def find_data_asf(lat, lon, processingLevel='SLC', platform='Sentinel-1A,Sentinel-1B',
                  **kwargs):
    """
    Search Alaska Satellite Facility for data

    @param lat: Latitude
    @param lon: Longitude
    @param processingLevel: Processing level of data
    @param platform: Instrument to search
    @param kwargs: All additional kwargs will be used to search ASF
                   See https://www.asf.alaska.edu/get-data/learn-by-doing/
    @returns: List of available data matching the search criteria
    """
    baseurl = 'https://api.daac.asf.alaska.edu/services/search/param?'

    if 'intersectsWith' in kwargs:
        raise RuntimeWarning('Ignoring lat/lon as intersectsWith keyword was supplied')

    else:
        point = shp.geometry.Point(lon, lat)
        kwargs['intersectsWith'] = point.to_wkt()

    if 'output' in kwargs:
        raise RuntimeWarning("Keyword 'output' ignored")


    kwargs['output'] = 'json'
    kwargs['processingLevel'] = 'SLC'
    kwargs['platform'] = platform

    search = urlencode(kwargs)
    search_url = baseurl+search

    with urlopen(baseurl+search) as urldata:
        data = json.load(urldata)

    return data[0]


def _get_key(data):
    """
    Retrieve the key for a particular image

    @param data: Dictionary of information from the Alaska Satellite Facility
    @return Dictionary key for data
    """
    return data['track'], data['frameNumber']



def select_max_matched_data(sentinel_data_list):
    """
    Select the data that can be combined into an interferogram

    The particular frame and track that maximizes the number
    of useable data is chosen

    @param sentinel_data_list:
    @returns:
    """

    def add_to_key(key):
        """
        Count the number of overlapping images

        @param key: Key to use to identify image location
        """

        if key not in max_keys:
            max_keys[key] = 1
        else:
            max_keys[key] += 1

    max_keys = OrderedDict()
    for data in sentinel_data_list:
        add_to_key(_get_key(data))

    max_orbit = None
    max_count = -1
    for orbit, count in max_keys.items():
        if max_count < count:
            max_count = count
            max_orbit = orbit

    final_data_list = []
    for data in sentinel_data_list:
        if _get_key(data) == max_orbit:
            final_data_list.append(data)

    return final_data_list


def match_data(sentinel_data_list):
    """
    Seperate into sets of overlapping data

    Seperates based on relative orbit, track, and frame

    @param sentinel_data_list: List of information for different images
    @return: Dictionary of lists of overlapping data
    """

    def add_info(data, data_dict):
        """
        Add information about image to a dictionary

        @param data: Input data about an image
        @param data_dict: Dictionary to store data
        """
        key = _get_key(data)
        if key not in data_dict:
            data_dict[key] = []

        data_dict[key].append(data)

    organized_data_dict = OrderedDict()
    for data in sentinel_data_list:
        add_info(data, organized_data_dict)

    return organized_data_dict



def find_earthquake_pairs(organized_data, date):
    """
    Select image pairs around a specified date

    @param organized_data: Dictionary of information about data that
                           has been organized into overlapping images

    @param date: Date of the event of interest
    @return Dictionary containing lists of pairs of images around the specified event
    """

    if isinstance(date, str):
        date = pd.to_datetime(date)

    return_data_dict = OrderedDict()
    remove_list = []
    for label, data in organized_data.items():
        if len(data) > 1:
            date_array = np.array([pd.to_datetime(info['sceneDate']) for info in data])
            sorted_index = np.argsort(date_array)
            date_array = date_array[sorted_index]

            date_index = np.searchsorted(date_array, date)
            if date_index != 0 and date_index != len(date_array):
                first_image_index = sorted_index[date_index-1]
                second_image_index = sorted_index[date_index]

                return_data_dict[label] = []
                return_data_dict[label].append(data[first_image_index])
                return_data_dict[label].append(data[second_image_index])

    return return_data_dict


def generateMatplotlibRectangle(extent, **kwargs):
    """
    Generate a matplotlib rectangle from a extents

    @param extent: Container holding the extent (x_min, x_max, y_min, y_max)
    @param kwargs: Extra keyword arguments passed to matplotlib.patches.Rectangle

    @return Matplotlib rectangle
    """
    xy = [extent[0], extent[2]]
    width = extent[1] - extent[0]
    height = extent[3] - extent[2]

    return mpl.patches.Rectangle(xy, width, height, **kwargs)


def project_insar_data(in_dataset, lon_center, lat_center, interpolation=gdal.GRA_Cubic,
                       no_data_value=np.nan, data_type=gdal.GDT_Float64):
    """
    Project InSAR data using GDAL

    @param in_dataset: GDAL data set to be projected
    @param lon_center: Longitude center of projecting
    @param lat_center: Latitude center of projecting
    @param interpolation: What kind of interpolation to use (GDAL Flags)
    @param no_data_value: What value to use in the case of no data
    @param data_type: Resulting data type (GDAL flag)

    @return array containing projected data
    """

    spatial = osr.SpatialReference()
    spatial.ImportFromProj4(f'+proj=tmerc +lat_0={lat_center} +lon_0={lon_center} +datum=WGS84 +ellps=WGS84 +k=0.9996 +no_defs')
    reprojected_dataset = reproject_georaster(georaster=in_dataset,
                                              interpolation_method=interpolation,
                                              new_cell_sizes=[100,100],
                                              new_projection_wkt=spatial.ExportToWkt(),
                                              no_data_value=no_data_value,
                                              data_type=data_type)
    return reprojected_dataset.ReadAsArray()
