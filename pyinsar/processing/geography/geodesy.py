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

import math
import numpy as np

from numba import jit

################################################################################
# Geodesy on a sphere
################################################################################

def compute_great_circle_distance_and_bearing(rad_longitude_1, rad_latitude_1, 
                                              rad_longitude_2, rad_latitude_2, 
                                              planet_radius):
    '''
    Compute the distance and initial bearing between two points on a sphere
    
    @param rad_longitude_1: Longitude of the first point (in radian)
    @param rad_latitude_1: Latitude of the first point (in radian)
    @param rad_longitude_2: Longitude of the second point (in radian)
    @param rad_latitude_2: Latitude of the second point (in radian)
    @param planet_radius: Radius of the considered planet (same unit as the distance)
    
    @return The Haversine distance and the initial bearing (in radian)
    '''
    delta_longitude = rad_longitude_1 - rad_longitude_2
    delta_latitude = rad_latitude_1 - rad_latitude_2
    
    root = np.sin(delta_latitude/2.)**2 + np.cos(rad_latitude_1)*np.cos(rad_latitude_2)*np.sin(delta_longitude/2.)**2
    distance = 2*planet_radius*np.arcsin(np.sqrt(root))
    
    rad_bearing = np.arctan2(np.sin(delta_longitude)*np.cos(rad_latitude_2), 
                             np.cos(rad_latitude_1)*np.sin(rad_latitude_2) - np.sin(rad_latitude_1)*np.cos(rad_latitude_2)*np.cos(delta_longitude))
    rad_bearing = -rad_bearing%(2*np.pi)
    
    return distance, rad_bearing

def compute_lonlat_from_distance_bearing(rad_longitude_1,
                                         rad_latitude_1,
                                         distance,
                                         rad_bearing,
                                         planet_radius):
    '''
    Compute the longitude and latitude of a point given another point and the
    distance and initial bearing between two points on a sphere
    
    @param rad_longitude_1: Longitude of the first point (in radian)
    @param rad_latitude_1: Latitude of the first point (in radian)
    @param distance: Haversine distance between the two points
    @param rad_bearing: Initial bearing between the two points (in radian)
    @param planet_radius: Radius of the considered planet (same unit as the distance)
    
    @return The longitude and latitude of the second point (in radian)
    '''
    angular_distance = distance/planet_radius
    rad_latitude_2 = np.arcsin(np.sin(rad_latitude_1)*np.cos(angular_distance)
                               + np.cos(rad_latitude_1)*np.sin(angular_distance)*np.cos(rad_bearing))
    rad_longitude_2 = rad_longitude_1 + np.arctan2(np.sin(rad_bearing)*np.sin(angular_distance)*np.cos(rad_latitude_1),
                                                   np.cos(angular_distance) - np.sin(rad_latitude_1)*np.sin(rad_latitude_2))
    
    return rad_longitude_2, rad_latitude_2

################################################################################
# Geodesy on an oblate spheroid
################################################################################

@jit(nopython = True)
def direct_vincenty_formula(rad_lon_1, rad_lat_1,
                            distance, rad_bearing_1,
                            a, f,
                            eps = 1e-12):
    '''
    Compute the longitude and latitude of a point given another point and the
    distance and initial bearing between two points on an oblate spheroid using
    Vincenty's direct formula, which relies on an optimization scheme
    
    @param rad_lon_1: Longitude of the first point (in radian)
    @param rad_lat_1: Latitude of the first point (in radian)
    @param distance: Haversine distance between the two points
    @param rad_bearing_1: Initial bearing between the two points (in radian)
    @param a: Length of the semi-major axis of the considered planet (same unit as the distance)
    @param f: Flattening of the considered planet
    @param eps: Wanted precision for the longitude and latitude
    
    @return The longitude and latitude of the second point (in radian)
    '''
    b = a*(1. - f)
    tan_reduced_rad_lat_1 = (1 - f)*math.tan(rad_lat_1)
    cos_reduced_rad_lat_1 = 1/math.sqrt(1 + tan_reduced_rad_lat_1**2)
    sin_reduced_rad_lat_1 = tan_reduced_rad_lat_1*cos_reduced_rad_lat_1
    sigma_1 = math.atan2(tan_reduced_rad_lat_1, math.cos(rad_bearing_1))
    sin_bearing = cos_reduced_rad_lat_1*math.sin(rad_bearing_1)
    sq_cos_bearing = 1 - sin_bearing**2
    sq_u = sq_cos_bearing*(a**2 - b**2)/b**2
#     A = 1 + sq_u*(4096. + sq_u*(-768. + sq_u*(320. - 175.*sq_u)))/16384.
#     B = sq_u*(256. + sq_u*(-128. + sq_u*(74 - 47*sq_u)))/1024.
    k_1 = (math.sqrt(1 + sq_u) - 1)/(math.sqrt(1 + sq_u) + 1)
    A = (1. + k_1**2/4.)/(1. - k_1)
    B = k_1*(1. - 3.*k_1**2/8.)
    
    sigma = distance/(b*A)
    sigma_prime = math.inf
    cos_2sigma_m = math.cos(2.*sigma_1 + sigma)
    while abs(sigma - sigma_prime) > eps:
        cos_2sigma_m = math.cos(2.*sigma_1 + sigma)
        delta_sigma = B*math.sin(sigma)*(cos_2sigma_m + B*(math.cos(sigma)*(-1. + 2.*cos_2sigma_m**2)
                                                           - B*cos_2sigma_m*(-3. + 4.*math.sin(sigma)**2)*(-3. + 4.*cos_2sigma_m**2)/6.)/4.)
        sigma_prime = sigma
        sigma = distance/(b*A) + delta_sigma
    
    temp = sin_reduced_rad_lat_1*math.sin(sigma) - cos_reduced_rad_lat_1*math.cos(sigma)*math.cos(rad_bearing_1)
    rad_lat_2 = math.atan2(sin_reduced_rad_lat_1*math.cos(sigma) + cos_reduced_rad_lat_1*math.sin(sigma)*math.cos(rad_bearing_1),
                           (1. - f)*math.sqrt(sin_bearing**2 + temp**2))
    lon = math.atan2(math.sin(sigma)*math.sin(rad_bearing_1),
                     cos_reduced_rad_lat_1*math.cos(sigma) - sin_reduced_rad_lat_1*math.sin(sigma)*math.cos(rad_bearing_1))
    C = f*sq_cos_bearing*(4. + f*(4. - 3.*sq_cos_bearing))/16.
    L = lon - (1. - C)*f*sin_bearing*(sigma + C*math.sin(sigma)*(cos_2sigma_m + C*math.cos(sigma)*(-1. + 2.*cos_2sigma_m**2)))
    rad_lon_2 = (rad_lon_1 + L + 3.*math.pi)%(2.*math.pi) - math.pi
    
    return (rad_lon_2, rad_lat_2)

@jit(nopython = True)
def direct_vincenty_formula_for_array(rad_longitude_1_array,
                                      rad_latitude_1_array,
                                      distance_array,
                                      rad_bearing_1,
                                      a,
                                      f,
                                      eps = 1e-12):
    '''
    Compute the longitude and latitude of several points given some other points
    and the distance and initial bearing between two points on an oblate spheroid
    using Vincenty's direct formula, which relies on an optimization scheme
    
    @param rad_longitude_1_array: Longitude of the first points (in radian)
    @param rad_latitude_1_array: Latitude of the first points (in radian)
    @param distance: Haversine distance between the points
    @param rad_bearing_1: Initial bearing between the points (in radian)
    @param a: Length of the semi-major axis of the considered planet (same unit as the distance)
    @param f: Flattening of the considered planet
    @param eps: Wanted precision for the longitude and latitude
    
    @return The longitude and latitude of the second points (in radian)
    '''
    rad_longitude_2_array = np.full(rad_longitude_1_array.shape, np.nan)
    rad_latitude_2_array = np.full(rad_latitude_1_array.shape, np.nan)
    for index in np.ndindex(rad_latitude_2_array.shape):
        if (math.isnan(rad_longitude_1_array[index]) == False
            and math.isnan(rad_latitude_1_array[index]) == False):
            rad_longitude_2_array[index], rad_latitude_2_array[index] = direct_vincenty_formula(rad_longitude_1_array[index],
                                                                                                rad_latitude_1_array[index],
                                                                                                distance_array[index],
                                                                                                rad_bearing_1,
                                                                                                a,
                                                                                                f,
                                                                                                eps = eps)

    return (rad_longitude_2_array, rad_latitude_2_array)

@jit(nopython = True)
def update_lambda(Lambda, reduced_rad_lat_1, reduced_rad_lat_2, diff_lon, f):
    '''
    Update the parameter lambda of Vincenty's inverse formula
    '''
    sin_reduced_rad_lat_1 = math.sin(reduced_rad_lat_1)
    cos_reduced_rad_lat_1 = math.cos(reduced_rad_lat_1)
    sin_reduced_rad_lat_2 = math.sin(reduced_rad_lat_2)
    cos_reduced_rad_lat_2 = math.cos(reduced_rad_lat_2)
    sin_lambda = math.sin(Lambda)
    cos_lambda = math.cos(Lambda)
    
    sin_sigma = math.sqrt((cos_reduced_rad_lat_2*sin_lambda)**2
                          + (cos_reduced_rad_lat_1*sin_reduced_rad_lat_2
                             - sin_reduced_rad_lat_1*cos_reduced_rad_lat_2*cos_lambda)**2)

    cos_sigma = sin_reduced_rad_lat_1*sin_reduced_rad_lat_2 + cos_reduced_rad_lat_1*cos_reduced_rad_lat_2*cos_lambda
    sigma = math.atan2(sin_sigma, cos_sigma)
    sin_alpha = cos_reduced_rad_lat_1*cos_reduced_rad_lat_2*sin_lambda/sin_sigma
    sq_cos_alpha = 1. - sin_alpha**2
    cos_2sigma_m = 0.
    if sq_cos_alpha != 0.:
        cos_2sigma_m = cos_sigma - 2.*sin_reduced_rad_lat_1*sin_reduced_rad_lat_2/sq_cos_alpha
    C = f*sq_cos_alpha*(4 + f*(4. - 3.*sq_cos_alpha))/16.
    Lambda = diff_lon + (1. - C)*f*sin_alpha*(sigma + C*sin_sigma*(cos_2sigma_m
                                                                  + C*cos_sigma*(-1. + 2.*cos_2sigma_m**2)))
    
    return Lambda, sq_cos_alpha, sin_sigma, cos_sigma, cos_2sigma_m, sigma

@jit(nopython = True)
def inverse_vincenty_formula(rad_lon_1, rad_lat_1,
                             rad_lon_2, rad_lat_2,
                             a, f,
                             eps = 1e-12, max_iter = 200):
    '''
    Compute the distance and initial bearing between two points on an oblate
    spheroid using Vincenty's inverse formula, which relies on an optimization scheme
    
    @param rad_lon_1: Longitude of the first point (in radian)
    @param rad_lat_1: Latitude of the first point (in radian)
    @param rad_lon_2: Longitude of the second point (in radian)
    @param rad_lat_2: Latitude of the second point (in radian)
    @param a: Length of the semi-major axis of the considered planet (same unit as the distance)
    @param f: Flattening of the considered planet
    @param eps: Wanted precision for the longitude and latitude
    @param max_iter: Maximum number of iterations
    
    @return The distance and the initial bearing
    '''
    if rad_lon_1 == rad_lon_2 and rad_lat_1 == rad_lat_2:
        return 0., 0.

    b = a*(1. - f)
    reduced_rad_lat_1 = math.atan((1 - f)*math.tan(rad_lat_1))
    reduced_rad_lat_2 = math.atan((1 - f)*math.tan(rad_lat_2))
    diff_lon = rad_lon_2 - rad_lon_1
    
    Lambda = diff_lon
    prev_lambda = math.inf
    sigma = math.inf
    cos_2sigma_m = math.inf
    sq_cos_alpha = math.inf
    sin_sigma = math.inf
    cos_sigma = math.inf
    iteration = 0
    while abs(Lambda - prev_lambda) > eps and iteration < max_iter:
        prev_lambda = Lambda
        Lambda, sq_cos_alpha, sin_sigma, cos_sigma, cos_2sigma_m, sigma = update_lambda(Lambda,
                                                                                        reduced_rad_lat_1,
                                                                                        reduced_rad_lat_2,
                                                                                        diff_lon, f)
        iteration += 1

    sq_u = sq_cos_alpha*(a**2 - b**2)/b**2
#     A = 1 + sq_u*(4096. + sq_u*(-768. + sq_u*(320. - 175.*sq_u)))/16384.
#     B = sq_u*(256. + sq_u*(-128. + sq_u*(74 - 47*sq_u)))/1024.
    k_1 = (math.sqrt(1 + sq_u) - 1.)/(math.sqrt(1. + sq_u) + 1.)
    A = (1. + k_1**2/4.)/(1 - k_1)
    B = k_1*(1. - 3.*k_1**2/8.)
    delta_sigma = B*sin_sigma*(cos_2sigma_m + B*(cos_sigma*(-1. + 2.*cos_2sigma_m**2)
                                                 - B*cos_2sigma_m*(-3. + 4.*sin_sigma**2)*(-3. + 4.*cos_2sigma_m**2)/6.)/4.)
    distance = b*A*(sigma - delta_sigma)
    
    rad_bearing_1 = math.atan2(math.cos(reduced_rad_lat_2)*math.sin(Lambda),
                               math.cos(reduced_rad_lat_1)*math.sin(reduced_rad_lat_2)
                               - math.sin(reduced_rad_lat_1)*math.cos(reduced_rad_lat_2)*math.cos(Lambda))

    return distance, rad_bearing_1

@jit(nopython = True)
def inverse_vincenty_formula_for_array(rad_longitude_1,
                                       rad_latitude_1,
                                       rad_longitude_2_array,
                                       rad_latitude_2_array,
                                       a,
                                       f,
                                       eps = 1e-12,
                                       max_iter = 200):
    '''
    Compute the distances and initial bearings between some points on an oblate
    spheroid using Vincenty's inverse formula, which relies on an optimization scheme
    
    @param rad_longitude_1: Longitude of the first point (in radian)
    @param rad_latitude_1: Latitude of the first point (in radian)
    @param rad_longitude_2_array: Longitude of the second points (in radian)
    @param rad_latitude_2_array: Latitude of the second points (in radian)
    @param a: Length of the semi-major axis of the considered planet (same unit as the distance)
    @param f: Flattening of the considered planet
    @param eps: Wanted precision for the longitude and latitude
    @param max_iter: Maximum number of iterations
    
    @return The distance and the initial bearing (in radian)
    '''
    distance_array = np.full(rad_longitude_2_array.shape, np.nan)
    rad_bearing_array = np.full(rad_longitude_2_array.shape, np.nan)
    for index in np.ndindex(rad_latitude_2_array.shape):
        if (math.isnan(rad_longitude_2_array[index]) == False
            and math.isnan(rad_latitude_2_array[index]) == False):
            distance_array[index], rad_bearing_array[index] = inverse_vincenty_formula(rad_longitude_1,
                                                                                       rad_latitude_1,
                                                                                       rad_longitude_2_array[index],
                                                                                       rad_latitude_2_array[index],
                                                                                       a,
                                                                                       f,
                                                                                       eps = eps,
                                                                                       max_iter = max_iter)
            
    return distance_array, rad_bearing_array

@jit(nopython = True)
def compute_point_to_line_distance_on_ellipsoid(rad_point_lon,
                                                rad_point_lat,
                                                rad_geodesic_origin_lon,
                                                rad_geodesic_origin_lat,
                                                rad_geodesic_bearing,
                                                a,
                                                f,
                                                eps = 1e-12,
                                                max_iter = 200):
    '''
    Compute the perpendicular distance between a point and a geodesic
    (i.e., a line) on an ellipsoid, using an optimization scheme 
    (see Baselga and Martinez-Llario, 2017, doi: 10.1007/s11200-017-1020-z)
    
    @param rad_point_lon: Longitude of the point (in radian)
    @param rad_point_lat: Latitude of the point (in radian)
    @param rad_geodesic_origin_lon: Longitude of the geodesic's origin (in radian)
    @param rad_geodesic_origin_lat: Latitude of the geodesic's origin (in radian)
    @param rad_geodesic_bearing: Geodesic's bearing from its origin (in radian)
    @param a: Length of the semi-major axis of the considered planet (same unit as the distance)
    @param f: Flattening of the considered planet
    @param eps: Wanted precision for the longitude and latitude
    @param max_iter: Maximum number of iterations
    
    @return The distance
    '''
    intersect_lon = rad_geodesic_origin_lon
    intersect_lat = rad_geodesic_origin_lat
    along_distance = math.inf
    cross_distance = math.inf
    iteration = 0
    while abs(along_distance) > eps and iteration < max_iter:
        distance_from_origin, bearing_from_origin = inverse_vincenty_formula(intersect_lon,
                                                                             intersect_lat,
                                                                             rad_point_lon,
                                                                             rad_point_lat,
                                                                             a,
                                                                             f,
                                                                             eps = eps,
                                                                             max_iter = max_iter)
        A = abs(bearing_from_origin - rad_geodesic_bearing)
        cross_distance = a*math.asin(math.sin(distance_from_origin/a)*math.sin(A))
        along_distance = 2.*a*math.atan(math.tan((distance_from_origin - cross_distance)/(2.*a))*math.sin((math.pi/2. + A)/2.)/math.sin((math.pi/2. - A)/2.))
        intersect_lon, intersect_lat = direct_vincenty_formula(intersect_lon,
                                                               intersect_lat,
                                                               along_distance,
                                                               rad_geodesic_bearing,
                                                               a,
                                                               f,
                                                               eps = eps)
        iteration += 1
        
    return cross_distance

@jit(nopython = True)
def compute_point_to_line_distance_for_array(rad_longitude_1,
                                             rad_latitude_1,
                                             rad_longitude_2_array,
                                             rad_latitude_2_array,
                                             rad_bearing,
                                             a,
                                             f,
                                             eps = 1e-12,
                                             max_iter = 200):
    '''
    Compute the perpendicular distance between a point and several geodesics
    (i.e., lines) on an ellipsoid, using an optimization scheme
    (see Baselga and Martinez-Llario, 2017, doi: 10.1007/s11200-017-1020-z)
    
    @param rad_longitude_1: Longitude of the point (in radian)
    @param rad_latitude_1: Latitude of the point (in radian)
    @param rad_longitude_2_array: Longitudes of the geodesics' origins (in radian)
    @param rad_latitude_2_array: Latitudes of the geodesics' origins (in radian)
    @param rad_bearing: Geodesics' bearing from their origins (in radian)
    @param a: Length of the semi-major axis of the considered planet (same unit as the distance)
    @param f: Flattening of the considered planet
    @param eps: Wanted precision for the longitude and latitude
    @param max_iter: Maximum number of iterations
    
    @return The distance
    '''
    distance_array = np.full(rad_longitude_2_array.shape, np.nan)
    for index in np.ndindex(rad_latitude_2_array.shape):
        if (math.isnan(rad_longitude_2_array[index]) == False
            and math.isnan(rad_latitude_2_array[index]) == False):
            distance_array[index] = compute_point_to_line_distance_on_ellipsoid(rad_longitude_2_array[index],
                                                                                rad_latitude_2_array[index],
                                                                                rad_longitude_1,
                                                                                rad_latitude_1,
                                                                                rad_bearing,
                                                                                a,
                                                                                f,
                                                                                eps = eps,
                                                                                max_iter = max_iter)
            
    return distance_array