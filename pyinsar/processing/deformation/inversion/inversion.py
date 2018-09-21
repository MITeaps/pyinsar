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

import random
import math
import numpy as np

from scipy import optimize
from sklearn.externals.joblib import Parallel, delayed

from pyinsar.processing.deformation.elastic_halfspace import okada

def misfit_function(fault_parameters,
                    fault_open,
                    poisson_ratio,
                    x,
                    y,
                    los_vectors,
                    unwrapped_interferograms,
                    interferogram_weights):
    '''
    Misfit function for the inversion of an Okada model
    
    @param fault_parameters: The fault parameters to find (centroid x, y, and depth, strike, dip, length, width, rake, slip)
    @param fault_open: Opening of the fault (fixed)
    @param poisson_ratio: Poisson ratio (fixed)
    @param x: A NumPy array of x coordinates to compute the Okada model
    @param y: A NumPy array of y coordinates to compute the Okada model
    @param los_vectors: A NumPy array of line-of-sight vectors for each coordinates
    @param unwrapped_interferograms: A NumPy array of unwrapped interferogram values to match for each coordinates
    @param interferogram_weights: A weight for each considered interferogram in the misfit function
    
    @return The misfit value
    '''
    displacement = okada.compute_okada_displacement(fault_parameters[0],
                                                    fault_parameters[1],
                                                    fault_parameters[2],
                                                    fault_parameters[3],
                                                    fault_parameters[4],
                                                    fault_parameters[5],
                                                    fault_parameters[6],
                                                    fault_parameters[7],
                                                    fault_parameters[8],
                                                    fault_open,
                                                    poisson_ratio,
                                                    x,
                                                    y)

    misfit = 0.
    for i in range(los_vectors.shape[0]):
        los_displacement = np.nansum(displacement*los_vectors[i], axis = 0)
        misfit += interferogram_weights[i]*np.nansum((los_displacement - unwrapped_interferograms[i])**2)

    return math.sqrt(misfit/np.count_nonzero(~np.isnan(unwrapped_interferograms)))

def okada_inversion_with_monte_carlo_restarts(fault_centroid_x_initial_bounds,
                                              fault_centroid_y_initial_bounds,
                                              fault_centroid_depth_initial_bounds,
                                              fault_strike_initial_bounds,
                                              fault_dip_initial_bounds,
                                              fault_length_initial_bounds,
                                              fault_width_initial_bounds,
                                              fault_rake_initial_bounds,
                                              fault_slip_initial_bounds,
                                              fault_open,
                                              poisson_ratio,
                                              x,
                                              y,
                                              unwrapped_interferograms,
                                              los_vectors,
                                              number_restarts,
                                              interferogram_weights = None,
                                              xtol = 0.0001,
                                              ftol = 0.0001,
                                              maxiter = None,
                                              maxfun = None,
                                              seed = 100):
    '''
    Perform the inversion of an Okada model using a downhill simplex algorithm with Monte Carlo restarts to fit InSAR interferograms
    
    @param fault_centroid_x_initial_bounds: Initial bounds for the x cooordinate for the fault's centroid
    @param fault_centroid_y_initial_bounds: Initial bounds for the y cooordinate for the fault's centroid
    @param fault_centroid_depth_initial_bounds: Initial bounds for the depth of the fault's centroid
    @param fault_strike_initial_bounds: Initial bounds for the strike of the fault ([0 - 2pi], in radian)
    @param fault_dip_initial_bounds: Initial bounds for the dip of the fault ([0 - pi/2], in radian)
    @param fault_length_initial_bounds: Initial bounds for the length of the fault (same unit as x and y)
    @param fault_width_initial_bounds: Initial bounds for the width of the fault (same unit as x and y)
    @param fault_rake_initial_bounds: Initial bounds for the rake of the fault ([-pi - pi], in radian)
    @param fault_slip_initial_bounds: Initial bounds for the slipe of the fault (same unit as x and y)
    @param fault_open: opening of the fault (same unit as x and y)
    @param poisson_ratio: Poisson's ratio
    @param x: x cooordinate for the domain within a NumPy array
    @param y: y cooordinate for the domain within a NumPy array
    @param unwrapped_interferograms: A NumPy array of unwrapped interferogram values to match for each coordinates
    @param los_vectors: A NumPy array of line-of-sight vectors for each coordinates
    @param number_restarts: Number of Monte Carlo restarts
    @param interferogram_weights: A weight for each considered interferogram in the misfit function
    @param xtol: Absolute error in xopt between iterations that is acceptable for convergence
    @param ftol: Absolute error in func(xopt) between iterations that is acceptable for convergence
    @param maxiter: Maximum number of iterations to perform
    @param maxfun: Maximum number of function evaluations to make
    @param seed: Random seed
    
    @return The parameters for the best run
    '''
    random.seed(seed)
    
    if interferogram_weights is None:
        interferogram_weights = np.ones(unwrapped_interferograms.shape[0])
    
    best_results = (None, np.inf)
    for i in range(number_restarts):
#         print('Inversion: ', i + 1, '/', number_restarts, sep = '')
    
        fault_parameters = [random.uniform(fault_centroid_x_initial_bounds[0], fault_centroid_x_initial_bounds[1]),
                            random.uniform(fault_centroid_y_initial_bounds[0], fault_centroid_y_initial_bounds[1]),
                            random.uniform(fault_centroid_depth_initial_bounds[0], fault_centroid_depth_initial_bounds[1]),
                            random.uniform(fault_strike_initial_bounds[0], fault_strike_initial_bounds[1]),
                            random.uniform(fault_dip_initial_bounds[0], fault_dip_initial_bounds[1]),
                            random.uniform(fault_length_initial_bounds[0], fault_length_initial_bounds[1]),
                            random.uniform(fault_width_initial_bounds[0], fault_width_initial_bounds[1]),
                            random.uniform(fault_rake_initial_bounds[0], fault_rake_initial_bounds[1]),
                            random.uniform(fault_slip_initial_bounds[0], fault_slip_initial_bounds[1])]
        
        results = optimize.fmin(misfit_function,
                                fault_parameters,
                                args = (fault_open,
                                        poisson_ratio,
                                        x,
                                        y,
                                        los_vectors,
                                        unwrapped_interferograms,
                                        interferogram_weights),
                                xtol = xtol,
                                ftol = ftol,
                                maxiter = maxiter,
                                maxfun = maxfun,
                                full_output = True)
        
        if results[1] < best_results[1]:
            best_results = results
            
    return best_results

def okada_inversion_with_parallel_monte_carlo_restarts(fault_centroid_x_initial_bounds,
                                                       fault_centroid_y_initial_bounds,
                                                       fault_centroid_depth_initial_bounds,
                                                       fault_strike_initial_bounds,
                                                       fault_dip_initial_bounds,
                                                       fault_length_initial_bounds,
                                                       fault_width_initial_bounds,
                                                       fault_rake_initial_bounds,
                                                       fault_slip_initial_bounds,
                                                       fault_open,
                                                       poisson_ratio,
                                                       x,
                                                       y,
                                                       unwrapped_interferograms,
                                                       los_vectors,
                                                       number_restarts,
                                                       number_chunks,
                                                       interferogram_weights = None,
                                                       xtol = 0.0001,
                                                       ftol = 0.0001,
                                                       maxiter = None,
                                                       maxfun = None,
                                                       seed = 100):
    '''
    Perform the inversion of an Okada model using a downhill simplex algorithm with parallel Monte Carlo restarts to fit InSAR interferograms
    
    @param fault_centroid_x_initial_bounds: Initial bounds for the x cooordinate for the fault's centroid
    @param fault_centroid_y_initial_bounds: Initial bounds for the y cooordinate for the fault's centroid
    @param fault_centroid_depth_initial_bounds: Initial bounds for the depth of the fault's centroid
    @param fault_strike_initial_bounds: Initial bounds for the strike of the fault ([0 - 2pi], in radian)
    @param fault_dip_initial_bounds: Initial bounds for the dip of the fault ([0 - pi/2], in radian)
    @param fault_length_initial_bounds: Initial bounds for the length of the fault (same unit as x and y)
    @param fault_width_initial_bounds: Initial bounds for the width of the fault (same unit as x and y)
    @param fault_rake_initial_bounds: Initial bounds for the rake of the fault ([-pi - pi], in radian)
    @param fault_slip_initial_bounds: Initial bounds for the slipe of the fault (same unit as x and y)
    @param fault_open: opening of the fault (same unit as x and y)
    @param poisson_ratio: Poisson's ratio
    @param x: x cooordinate for the domain within a NumPy array
    @param y: y cooordinate for the domain within a NumPy array
    @param unwrapped_interferograms: A NumPy array of unwrapped interferogram values to match for each coordinates
    @param los_vectors: A NumPy array of line-of-sight vectors for each coordinates
    @param number_restarts: Number of Monte Carlo restarts
    @param interferogram_weights: A weight for each considered interferogram in the misfit function
    @param xtol: Absolute error in xopt between iterations that is acceptable for convergence
    @param ftol: Absolute error in func(xopt) between iterations that is acceptable for convergence
    @param maxiter: Maximum number of iterations to perform
    @param maxfun: Maximum number of function evaluations to make
    @param seed: Random seed
    
    @return The parameters for the best run
    '''
    chunk_size = math.ceil(number_restarts/number_chunks)
    chunks = [min(chunk_size, number_restarts - i) for i in range(0, number_restarts, chunk_size)]

    results = Parallel(n_jobs = number_chunks)(delayed(okada_inversion_with_monte_carlo_restarts)(fault_centroid_x_initial_bounds,
                                                                                                  fault_centroid_y_initial_bounds,
                                                                                                  fault_centroid_depth_initial_bounds,
                                                                                                  fault_strike_initial_bounds,
                                                                                                  fault_dip_initial_bounds,
                                                                                                  fault_length_initial_bounds,
                                                                                                  fault_width_initial_bounds,
                                                                                                  fault_rake_initial_bounds,
                                                                                                  fault_slip_initial_bounds,
                                                                                                  fault_open,
                                                                                                  poisson_ratio,
                                                                                                  x,
                                                                                                  y,
                                                                                                  unwrapped_interferograms,
                                                                                                  los_vectors,
                                                                                                  chunk,
                                                                                                  interferogram_weights,
                                                                                                  xtol,
                                                                                                  ftol,
                                                                                                  maxiter,
                                                                                                  maxfun,
                                                                                                  seed + i_chunk)
                                               for i_chunk, chunk in enumerate(chunks))
    
    best_index = np.argmin(list(zip(*results))[1])
        
    return results[best_index]