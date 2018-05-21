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
from pyproj import Proj

from skdaccess.utilities.image_util import SplineGeolocation

def read_srcmod_data(input_src_mat_filename):
    '''
    Generate faults of okada sources from src mod mat files.

    Note: Only single segment models with a single time window are currently supported

    @param input_src_mat_filename: Filename or filelike object of srcmod mat file
    
    @return List of faults objects, list of slips, list of rakes
    '''
    matdata = loadmat(input_src_mat_filename)

    data_keys = []

    # Find all non private keys
    for key in matdata.keys():
        if re.match('__', key) is None:
            data_keys.append(key)

    # Make sure there is only on data key
    if len(data_keys) != 1:
        raise ValueError('Too many keys in mat file')


    data = matdata[data_keys[0]][0,0]

    # Number of segments
    num_segments = data['invSEGM'][0,0].astype(int)

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


    


    lat = data['geoLAT']
    lon = data['geoLON']

    proj = Proj(proj='gnom', lat_0=lat, lon_0=lon)

    # Size of the subfaults
    Dz, Dx = data['invDzDx'][0,:]

    # Number of the subfaults
    Nz, Nx = data['invNzNx'][0,:].astype(int)

    # Number of time windows
    num_time_windows = data['invNoTW'].astype(int)

    def get_data_from_time_windows(data, prefix, alternate_column, num_windows, Nx, Nz, alternate_value=None):
        if prefix + 'TW1' in data.dtype.names:
            new_data = np.zeros([num_time_windows, Nz, Nx])

            for num in range(1, num_windows+1):
                new_data[num-1,:,:] = np.fliplr(np.flipud(data[prefix+'TW' + str(num)]))

        elif alternate_column in data.dtype.names:
            new_data = np.fliplr(np.flipud(data[alternate_column]))
            new_data = new_data.reshape(1, *new_data.shape)

        else:
            new_data = np.ones([1, Nz, Nx])
            new_data[0,:,:] = data[alternate_value][0,0]

        return new_data


    fault_slip = get_data_from_time_windows(data, 'slip', 'slipSPL', num_time_windows, Nx, Nz) / 100
    fault_rake = np.deg2rad(get_data_from_time_windows(data, 'rake', 'RAKE', num_time_windows, Nx, Nz, 'srcARake'))


    

    # Location of the hypocenter in the along strike/down dip
    # coordinate system
    center_x, center_z = data['srcHypXZ'][0,:]

    # Distance from the surface to the top of the fault
    z_top = data['srcZ2top'][0,0]

    # Width and length of fault
    width, length = data['srcDimWL'][0,:]

    # Fault properties
    fault_strike = np.deg2rad(data['srcAStke'][0,0])
    fault_dip = np.deg2rad(data['srcDipAn'][0,0])


    ### Generate fault ###
    # Hypocenter vector in the coordinate system where the centroid of
    # the fault is at 0,0,0
    hypx_col_vec = np.array([[center_z - width / 2], 
                             [ center_x - length/2],
                             [                   0]])

    # SRCMOD's defintion of the centroid (center of cell in x, top in
    # z)
    # This used for comparing with the positions in the srcmod file
    compare_hypx_col_vec = np.array([[-Dz/2],
                                     [    0],
                                     [    0]])

    # Rotate the fault to put in a new coordinate system (x->EW,
    # y->NS)
    hypo_center_coords = rotate(rotate(hypx_col_vec, 0, fault_dip, 0), -fault_strike, 0, 0)

    # Rotating the srcmod's definition of a centroid
    compare_center_coords = rotate(rotate(compare_hypx_col_vec, 0, fault_dip, 0), -fault_strike, 0, 0)

    # Determine the centroid of the fault
    fault_center = -hypo_center_coords
    fault_center[2] = np.sin(fault_dip)*width/2 + z_top
    fault_center *= 1000

    # Create fault
    fault = Fault(*fault_center, length*1000, width*1000,
                  fault_strike, fault_dip, Nx, Nz)

    # Determine the centroids of each subfault that srcmod uses
    compare_with_provided_centers = translate(fault.cell_centroids[:,::-1]/1000, *compare_center_coords)

    if not np.allclose(compare_with_provided_centers, np.stack([ data['geoX'].ravel(), 
                                                                 data['geoY'].ravel(), 
                                                                -data['geoZ'].ravel()])):

        raise RuntimeError("Unable to recreate srcmod's centroids!")

    ### Finished generating fault ###

    return [fault], [fault_slip], [fault_rake], proj

    
    
