# Standard library imports
from collections import OrderedDict

# 3rd party imports
import numpy as np
import pandas as pd
from scipy.signal import convolve
from scipy.interpolate import interp1d
from geodesy import wgs84
import numpy as np

def phaseShift(data, phase):
    '''
     Apply a phase shift to data

    @param data: Input data
    @param phase: Input phase

    @return data shifted by phase
    '''
    return  (data.real*np.cos(phase) + data.imag*np.sin(phase)) \
            + 1j*(-data.real*np.sin(phase) + data.imag*np.cos(phase))

def findClosestTime(time, date):
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

    def __init__(self, orbit_data, interp_target = 'positions', time_name = 'UTC'):
        '''
        Initilaize orbit interpolation object

        @param orbit_data: Orbit position data
        @param interp_target: Type of interpolation. Can be positions or velocity.
        @param time_name: Name of time column name in Orbit position data. Set this to None
                          to use the data frame index
        '''

        self._orbit_data = orbit_data
        if time_name == None:
            self._start_date = self_orbit_data.index[0]
            self._elapsed_time = (self._orbit_data.index - self._start_date).total_seconds()

        else:
            self._start_date = orbit_data[time_name].iloc[0]
            self._elapsed_time = (pd.DatetimeIndex(self._orbit_data[time_name]) - self._start_date).total_seconds()

        self._interp_target = interp_target

        if self._interp_target == 'positions':
            self._labels = ['X','Y','Z']

        elif self._interp_target == 'velocity':
            self._labels = ['VX', 'VY', 'VZ']

        else:
            raise ValueError('Interpolation target ' + self._interp_target + ' not understood')

        self._interp_functions = OrderedDict()

        for label in self._labels:
            self._interp_functions[label] = interp1d(self._elapsed_time, self._orbit_data[label], 'cubic')


    def __call__(self, in_time):
        '''
        Compute the satellites position or velocity

        @param in_time: Time of interest

        @return Satellite position or velocity at in_time
        '''

        elapsed_time = (in_time - self._start_date) / pd.to_timedelta(1,'s')

        results = []

        for label in self._labels:
            tmp_res = self._interp_functions[label](elapsed_time)
            if tmp_res.ndim == 0:
                tmp_res = tmp_res.reshape(-1)[0]
            results.append(tmp_res)

        return results


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

