# The MIT License (MIT)
# Copyright (c) 2018 Massachusetts Institute of Technology
#
# Author: Cody Rude
# This software has been created in projects supported by the US National
# Science Foundation and NASA (PI: Pankratius)
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
from pyinsar.processing.deformation.elastic_halfspace.fault import Fault
from pyinsar.processing.utilities.generic import translate, rotate

# Standard library imports
import re

# 3rd party imports
from scipy.io import loadmat
from scipy.interpolate import SmoothBivariateSpline
import numpy as np

from skdaccess.utilities.image_util import SplineGeolocation

def read_srcmod_data(srcmod_data, dtype=np.float64, skip_sanity_check=False):
    '''
    *** In Development *** Generate faults of okada sources from src mod mat files.

    Note: Only single segment models with a single time window are currently supported

    @param srcmod_data: src mod data read in from the .mat file
    @param dtype: Data type to use
    @param skip_sanity_check: Skip checks to ensure data was interpreted properly (Used for debugging)
    
    @return List of faults objects, list of slips, list of rakes
    '''

    # Number of segments
    num_segments = int(srcmod_data['invSEGM'])

    if num_segments != 1:
        raise NotImplementedError('Only single segment srcmods are currently supported')


    # lat = data['geoLAT'].ravel()
    # lon = data['geoLON'].ravel()

    # x = data['geoX'].ravel()
    # y = data['geoY'].ravel()

    # lat_spline = SmoothBivariateSpline(y, x, lat, s=0)
    # lon_spline = SmoothBivariateSpline(y, x, lon, s=0)

    # x_spline = SmoothBivariateSpline(lat, lon, x, s=0)
    # y_spline = SmoothBivariateSpline(lat, lon, y, s=0)

    # geolocation = SplineGeolocation(lat_spline=lat_spline,
    #                                 lon_spline=lon_spline,
    #                                 x_spline=x_spline,
    #                                 y_spline=y_spline)


    lat = srcmod_data['geoLAT']
    lon = srcmod_data['geoLON']

    # Size of the subfaults
    Dz, Dx = srcmod_data['invDzDx']

    # Number of the subfault
    Nz, Nx = srcmod_data['invNzNx'].astype(int)

    # Number of time windows
    num_time_windows = int(srcmod_data['invNoTW'])

    def get_data_from_time_windows(data, prefix, alternate_column, num_windows, Nx, Nz, alternate_value=None):
        if prefix + 'TW1' in data.keys():
            new_data = np.zeros([num_time_windows, Nz, Nx], dtype=dtype)

            for num in range(1, num_windows+1):
                new_data[num-1,:,:] = np.fliplr(np.flipud(data[prefix+'TW' + str(num)]))

        elif alternate_column in data.keys():
            new_data = np.fliplr(np.flipud(data[alternate_column]))
            new_data = new_data.reshape(1, *new_data.shape)

        else:
            new_data = np.ones([1, Nz, Nx], dtype=dtype)
            new_data[0,:,:] = data[alternate_value]

        return new_data


    fault_slip = get_data_from_time_windows(srcmod_data, 'slip', 'slipSPL', num_time_windows, Nx, Nz) / 100
    fault_rake = np.deg2rad(get_data_from_time_windows(srcmod_data, 'rake', 'RAKE', num_time_windows, Nx, Nz, 'srcARake'))


    

    # Location of the hypocenter in the along strike/down dip
    # coordinate system
    center_x, center_z = srcmod_data['srcHypXZ']

    # Distance from the surface to the top of the fault
    z_top = srcmod_data['srcZ2top']

    # Width and length of fault
    width, length = srcmod_data['srcDimWL']

    # Fault properties
    fault_strike = np.deg2rad(srcmod_data['srcAStke'])
    fault_dip = np.deg2rad(srcmod_data['srcDipAn'])

    ### Generate fault ###
    # Hypocenter vector in the coordinate system where the centroid of
    # the fault is at 0,0,0
    hypx_col_vec = np.array([[center_z - width / 2], 
                             [ center_x - length/2],
                             [                   0]], dtype=dtype)

    # SRCMOD's defintion of the centroid (center of cell in x, top in
    # z)
    # This used for comparing with the positions in the srcmod file
    compare_hypx_col_vec = np.array([[-Dz/2],
                                     [    0],
                                     [    0]], dtype=dtype)

    # Rotate the fault to put in a new coordinate system (x->EW,
    # y->NS)
    hypo_center_coords = rotate(rotate(hypx_col_vec, 0, fault_dip, 0, dtype=dtype), -fault_strike, 0, 0, dtype=dtype)

    # Rotating the srcmod's definition of a centroid
    compare_center_coords = rotate(rotate(compare_hypx_col_vec, 0, fault_dip, 0, dtype=dtype), -fault_strike, 0, 0, dtype=dtype)

    # Determine the centroid of the fault
    fault_center = -hypo_center_coords
    fault_center[2] = np.sin(fault_dip)*width/2 + z_top
    fault_center *= 1000

    # Create fault
    fault = Fault(*fault_center, length*1000, width*1000,
                  fault_strike, fault_dip, Nx, Nz, dtype=dtype)

    # Determine the centroids of each subfault that srcmod uses
    compare_with_provided_centers = translate(fault.cell_centroids[:,::-1]/1000, *compare_center_coords)


    if not skip_sanity_check and not np.allclose(compare_with_provided_centers, np.stack([ srcmod_data['geoX'].ravel(),
                                                                 srcmod_data['geoY'].ravel(),
                                                                 -srcmod_data['geoZ'].ravel()]), atol=1e-3):

        raise RuntimeError("Unable to recreate srcmod's centroids!")

    ### Finished generating fault ###

    return [fault], [fault_slip], [fault_rake]
