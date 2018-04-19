from scipy.interpolate import CubicSpline, UnivariateSpline
import pandas as pd

# These functions are all under development

def vapor_pressure(T):
    '''
    Under development
    '''
    return 6.1037*np.exp(17.641 * (T-273.15) / (T - 29.88))

def N(P, T, RH, k1=77.6, k2=23.3, k3=3.75E5):
    '''
    Under development
    '''
    return (k1*(P/T)) + (k2*(vapor_pressure(T)*RH/T) + k3 * (vapor_pressure(T)*RH/(T**2)))


def N_h(h, P, T, RH, k1=77.6, k2=23.3, k3=3.75E5):
    '''
    Under development
    '''
    return N(P(h), T(h), RH(h), k1, k2, k3)


def compute_delays(h, P, T, RH):
    '''
    Under development
    '''
    pressure = CubicSpline(h, P, bc_type='natural', extrapolate=True)
    temperature = CubicSpline(h, T+273.15, bc_type='natural', extrapolate=True)
    rh_index = ~pd.isnull(RH)
    relative_humidity = UnivariateSpline(h[rh_index],RH[rh_index], k=3)

    d_tropo = lambda x: 1e-6 * quad(N_h, x, 9e3, args=(pressure, temperature,relative_humidity))[0]

    x = np.arange(0,9000,100)
    return pd.Series(np.fromiter(map(d_tropo,x),dtype=np.float,count=len(x)), index=x)
