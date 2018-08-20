# The MIT License (MIT)
# Copyright (c) 2018 Massachusetts Institute of Technology
#
# Author: Guillaume Rongier
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

import numpy as np
from skimage.filters import threshold_li

def wrap(x, to_2pi = False):
    '''
    Wrap a float or an array

    @param x: The float or array
    @param to_2pi: If True, wrap to [0, 2pi) instead of [-pi, pi]

    @return The wrapped array (in radian between -pi and pi)
    '''
    if to_2pi == True:
        return np.mod(x, 2.*np.pi)
    return np.mod(x + np.pi, 2.*np.pi) - np.pi

def crop_array_from_center(array, crop_shape):
    '''
    Crop an array along its borders

    @param array: The array
    @param crop_shape: The number of cells to remove along the y and x axes

    @return The cropped array
    '''
    slices = []
    for i in range(len(crop_shape)):
        start = array.shape[i]//2 - crop_shape[i]//2
        end = start + crop_shape[i]
        slices.append(slice(start, end))

    return array[slices]


def mask_deformation(deformation, threshold_function = threshold_li):
    '''
    Mask image using a threshold function

    @param deformation: Deformation to mask
    @param threshold_function: Function to calculate the threshold value
    @return Masked image
    '''

    mask = np.zeros_like(deformation, dtype=np.bool)
    for i in range(deformation.shape[0]):
        thresh = threshold_function(np.abs(deformation[i,:,:]))
        mask[i, np.abs(deformation[i,:,:]) < thresh] = True


    mask = np.all(mask, axis=0)

    deformation_masked = deformation.copy()
    deformation_masked[:,mask] = np.nan

    return deformation_masked


def calc_bounding_box(image, threshold_function = threshold_li):
    '''
    Calcluate the bounding box around an image using the li threshold

    @param image: Input image
    @param threshold_function: Threshold function to use
    @return Extents of a bounding box around the contents in the image (x_min, x_max, y_min, y_max)
    '''
    thresh = threshold_function(image)
    thresh_image = np.where(image < thresh, 0, 1)

    return retrieve_bounds(thresh_image)


def retrieve_bounds(thresh_image):
    """
    Retrieve the bounds of an image that has been thesholded

    @param thresh_image: Image filled with ones for valid and zeros for invalid
    @return: Extents of a rectangle around valid data (x_start, x_end, y_start, y_end)
    """
    column_maximums = np.max(thresh_image, axis=0)
    row_maximums = np.max(thresh_image, axis=1)
    x_start = np.argmax(column_maximums)
    x_end = len(column_maximums) - np.argmax(column_maximums[::-1])

    y_start = np.argmax(row_maximums)
    y_end = len(row_maximums) - np.argmax(row_maximums[::-1])

    return x_start, x_end, y_start, y_end


def crop_nans(image):
    """
    Shrink image by removing nans

    @param image: Input image
    @return: Image cropped around valid data
    """
    thresh_image = ~np.isnan(image)

    x_start, x_end, y_start, y_end = retrieve_bounds(thresh_image)

    return image[y_start:y_end, x_start:x_end]


def determine_deformation_bounding_box(deformations, largest_box=True, **kwargs):
    '''
    Calculate the extent of the deformation in image coordinates

    @param deformations: Input deformations
    @param largest_box: Choose a bounding max that encomposses all selected values in all dimensions
    @param kwargs: Any additional keyword arguments passed to calc_bounding_box

    @return Extents deformations (x_min, x_max, y_min, y_max)
    '''
    bounds = np.stack([calc_bounding_box(np.abs(deformations[i,:,:]), **kwargs) for i in range(3)])
    if largest_box:
        return np.min(bounds[:,0]), np.max(bounds[:,1]), np.min(bounds[:,2]), np.max(bounds[:,3])
    else:
        return np.max(bounds[:,0]), np.min(bounds[:,1]), np.max(bounds[:,2]), np.min(bounds[:,3])


def determine_x_y_bounds(deformations, x_array, y_array, offset=5000, **kwargs):
    '''
    Determine the x and y coordinates of the extent of the deformation

    @param deformations: Input deformations
    @param x_array: x coordinates
    @param y_array: y coordinatse
    @param offset: Size to extend the extents of the box
    @param kwargs: Any additional keyword arguments passed to determine_deformation_bounding_box

    @return  Extents of the deformation plus the offset (x_min, x_max, y_min, y_max)
    '''

    bounding_box = determine_deformation_bounding_box(deformations, **kwargs)
    x_start, x_end = x_array[bounding_box[2:], bounding_box[:2]]
    y_start, y_end = y_array[bounding_box[2:], bounding_box[:2]]

    if y_start > y_end:
        tmp = y_start
        y_start = y_end
        y_end = tmp


    return x_start - offset, x_end + offset, y_start - offset, y_end + offset


def generate_interferogram_from_deformation(track_angle,
                                            min_ground_range_1,
                                            height_1,
                                            is_right_looking,
                                            wavelength,
                                            k,
                                            deformation,
                                            xx, yy,
                                            projected_topography=None,
                                            min_ground_range_2 = None,
                                            height_2 = None):
    '''
    Generate an interferogram from deformations

    @param track_angle: Satellite track angle
    @param min_ground_range_1: Minimum ground range to deformations for first pass
    @param height_1: Height of satellite for first pass
    @param is_right_looking: The satellite is looking to the right
    @param wavelength: Wavelength of the signal
    @param k: number of passes (1 or 2)
    @param deformation: map of deformation
    @param xx: x coordinates of deformation
    @param yy: y coordinates of deformation
    @param projected_topography: Elevation data
    @param min_ground_range_2: Minimum ground range to deformations for second pass
    @param height_2: Height of satellite for second pass

    @return Inteferogram due to the deformations
    '''

    rad_track_angle = track_angle

    cross_track_distance = xx * np.cos(rad_track_angle) - yy * np.sin(rad_track_angle)

    if is_right_looking:
        phi = 2 * np.pi - track_angle
        cross_track_distance *= -1.

    else:
        phi = np.pi - track_angle

    cross_track_deformation = deformation[0,:,:].astype(np.float64) * np.cos(phi) + deformation[1,:,:].astype(np.float64) * np.sin(phi)

    if height_2 is None:
        height_2 = height_1

    if min_ground_range_2 is None:
        min_ground_range_2 = min_ground_range_1

    if projected_topography is not None:
        corrected_height_1 = height_1 - projected_topography
        corrected_height_2 = height_2 - projected_topography
    else:
        corrected_height_1 = height_1
        corrected_height_2 = height_2

    corrected_height_2 -= deformation[2,:,:].astype(np.float64)

    cross_track_distance -= cross_track_distance.min()

    ground_range_1 = cross_track_distance + min_ground_range_1
    ground_range_2 = cross_track_distance + min_ground_range_2 + cross_track_deformation

    slant_range_1 = np.sqrt(corrected_height_1**2 + ground_range_1**2)
    slant_range_2 = np.sqrt(corrected_height_2**2 + ground_range_2**2)

    phase = change_in_range_to_phase(slant_range_1 - slant_range_2, wavelength)

    return phase

def old_generate_interferogram_from_deformation(track_angle,
                                                min_ground_range,
                                                height,
                                                is_right_looking,
                                                wavelength,
                                                k,
                                                deformation,
                                                xx, yy,
                                                projected_topography=None):
    '''
    Generate an interferogram from deformations

    @param track_angle: Satellite track angle
    @param min_ground_range: Minimum ground range to deformations
    @param height: Height of satellite
    @param is_right_looking: The satellite is looking to the right
    @param wavelength: Wavelength of the signal
    @param k: number of passes (1 or 2)
    @param deformation: map of deformation
    @param xx: x coordinates of deformation
    @param yy: y coordinates of deformation
    @param projected_topography: Elevation data

    @return Inteferogram due to the deformations
    '''

    rad_track_angle = track_angle

    cross_track_distance = xx * np.cos(rad_track_angle) - yy * np.sin(rad_track_angle)

    if is_right_looking:
        phi = 2 * np.pi - track_angle

        cross_track_distance *= -1.

    else:
        phi = np.pi - track_angle

    if projected_topography is not None:
        heights = height - projected_topography
    else:
        heights = height

    cross_track_distance -= cross_track_distance.min()

    ground_range = cross_track_distance + min_ground_range

    rad_look_angle = np.arctan2(ground_range, heights)

    theta = np.pi - rad_look_angle

    x = np.sin(theta) * np.cos(phi)
    y = np.sin(theta) * np.sin(phi)
    z = np.cos(theta)

    look_vectors = np.stack([x, y, z])

    los_deformation = np.sum(look_vectors * deformation, axis=0)

    phase = 2. * np.pi * k * los_deformation / wavelength

    return phase


def change_in_range_to_phase(los_deformation, wavelength, k=2):
    '''
    Compute phase from change in range

    @param los_deformation: Change in distance along line of site
    @param wavelength: Wavelength of radar
    @param k: Number of passes

    @return phase due to change in
    '''
    return -2. * np.pi * k * los_deformation / wavelength

def phase_to_change_in_range(phase, wavelength, k=2):
    '''
    Compute change in range from phase

    @param phase: Input phase
    @param wavelength: Wavelength of radar
    @param k: Number of passes

    @return Change in range
    '''
    return -phase * wavelength / (2 * np.pi * k)
