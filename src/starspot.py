import numpy as np

__author__ = "Angela Santos"

Oeq = 2*np.pi/25.

class Spot():

    #-------------------------------------------------cgs units
    rsun = 6.955e10 
    inclination = 70.
    Io = 1.
    Cs = 0.67

    nbeta = 90
    ntheta = 45#8000, 10000, 5000 


    def __init__():
        cycle_length = 11.            # years
        total_time = 4.               # years
        time_step = total_time/365.25/24
        cad = (1/365.25)/time_step

        initial_time = 1996.3525
        first_minimum = 1996.35
        overlap_length = 1.0

        sunspotnum_coeffs = {}
        sunspotnum_coeffs['a1'] = 0.26301229/6.
        sunspotnum_coeffs['t0'] = 1995.1020
        sunspotnum_coeffs['b1'] = 4.7786238
        sunspotnum_coeffs['c1'] = -0.2908343

        area_logn = {}
        area_logn['mean'] = 4.0374632
        area_logn['sigma'] = 1.0521994

        evolution_coeffs = {}
        evolution_coeffs['growth'] = {}
        evolution_coeffs['decay'] = {}
        evolution_coeffs['growth']['gamma1'] = 0.001
        evolution_coeffs['growth']['gamma2'] = 0.2
        evolution_coeffs['decay']['gamma1'] = 0.001
        evolution_coeffs['decay']['gamma2'] = 0.2

        mean_latitude_coeffs = {}
        mean_latitude_coeffs['L0'] = 34.9987
        mean_latitude_coeffs['t0'] = 1995.1150
        mean_latitude_coeffs['damping'] = 7.5

        sigmaL_coeffs = {}
        sigmaL_coeffs['a_sigma'] = 0.0998435
        sigmaL_coeffs['b_sigma'] = 1.20313
        sigmaL_coeffs['c_sigma'] = -0.952859
        sigmaL_coeffs['Pc'] = cycle_length
        

        gw_constant = 10.
        gw_correction = [5., 6.2591e-3]

        wr = np.array([14.713, -2.396, -1.787])
        alpha = 0.2

        const = Oeq*alpha*np.sin(np.radians(90.))**2
        wr2 = [alpha, Oeq*365.25]


