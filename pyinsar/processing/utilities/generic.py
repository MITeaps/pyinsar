# Standard library imports
from collections import OrderedDict

# 3rd party imports
import cv2
import numpy as np
import pandas as pd

import warnings
with warnings.catch_warnings():
    warnings.simplefilter('ignore', FutureWarning)
    import statsmodels.api as sm

from scipy.signal import convolve
from scipy.interpolate import interp1d
from geodesy import wgs84
from sklearn.linear_model import RANSACRegressor



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
