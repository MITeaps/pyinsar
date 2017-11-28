import numpy as np
from scipy.signal import convolve

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
