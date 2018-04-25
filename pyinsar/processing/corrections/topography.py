import numpy as np
from geodesy import wgs84

def ellipsoidal_earth_slant_ranges(azimuth_time, latlon, orbit_interp,
                                   start_x, end_x, start_y, end_y):
    '''
    Compute slant ranges assuming no topography

    @param azimuth_time: Pandas time series data conatining the time of each azimuth line
    @param latlon: Function to compute latitude and longitude for each pixel coordinate
    @param orbit_interp: Function to compute satellite positions
    @param start_x: Starting x pixel
    @param end_x: Ending pixel x pxiel
    @param start_y: Starting y pixel
    @param end_y: Endying y pixel

    @return Slant range distance to each pixel
    '''

    geo_to_cart = np.vectorize(wgs84.geodesic_to_cartesian)

    x,y = np.meshgrid(np.arange(start_x, end_x), np.arange(start_y, end_y))

    lat, lon = latlon(y,x)

    lines = lat.shape[0]
    samples = lat.shape[1]

    dates = azimuth_time[start_y:end_y]

    sat_positions = np.stack(orbit_interp(dates).T, axis=1)

    flat_earth_positions = np.stack(geo_to_cart(lat.ravel(), lon.ravel(), 0), axis=1)

    distance_vectors = np.repeat(sat_positions,samples, axis=0) - flat_earth_positions

    return np.linalg.norm(distance_vectors, axis=1).reshape(lines, samples), sat_positions
