import numpy as np

__author__ = "Angela Santos"
__all__ = ["Spot", "Star"]

Oeq = 2*np.pi/25.
year2day = 365.25
day2hour = 24
rsun = 6.955e10
cycle_length = 11.            # years
total_time = 4.               # years
time_step = total_time/year2day/day2hour
cad = (1/year2day)/time_step
gw_constant = 10.
gw_corr_coeffs = [5., 6.2591e-3]

wr = np.array([14.713, -2.396, -1.787])
alpha = 0.2

0eq = 2*np.pi/25. # ?? what is this?
const = Oeq*alpha*np.sin(np.radians(90.))**2
wr2 = [alpha, Oeq*year2day]


class Spot():
    __all__ = ["evolve",
               "get_lifetime",
               "get_rotation"]
    __attributes__ = ["initial_time", "first_minimum", "overlap_length",
                      "sunspotnum_coeffs", "area_logn", "evolution_coeffs",
                      "mean_latitude_coeffs", "sigmaL_coeffs"]
    def __init__():

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

    def evolve(self, max_spot_area):
        """Evolves a starspot based on the growth and decay factors.
        
        Inputs:
        -------
        max_spot_area - np.float64
            Maximum starspot area
            Unit: MSH (Millionth of Solar Hemisphere)

        Outputs:
        --------
        spot_area_list - list(dtype=np.float64)
            List containing area of spot as a function of time for 
            each time_step.
        """
        self.lifetime = self.get_lifetime(max_spot_area) * year2day

        # decay loop
        spot_area = max_spot_area*1.0
        spot_area_list = [spot_area]
        for ti in np.arange(1, self.lifetime+1, self.time_step*year2day):
            gamma1 = self.evolution_coeffs['decay']['gamma1']
            gamma2 = self.evolution_coeffs['decay']['gamma2']
            d_area = (np.exp(gamma1) * spot_area**gamma2)/self.cad
            spot_area -= d_area
            spot_area_list.append(spot_area)

        tg = self.lifetime - len(spot_area_list)/self.cad

        # growth loop
        spot_area = max_spot_area*1.0
        for ti in np.arange(0, tg, self.time_step*year2day):
            gamma1 = self.evolution_coeffs['growth']['gamma1']
            gamma2 = self.evolution_coeffs['growth']['gamma2']
            d_area = (np.exp(gamma1) * spot_area**gamma2)/self.cad
            spot_area -= d_area
            spot_area_list.insert(0, spot_area)
        return spot_area_list

    def get_lifetime(self, max_spot_area):
        """Computes the lifetime of starspots based on the 
        Gnevyshev-Waldneier (GW) rule.

        Inputs:
        -------
        max_spot_area - np.float64
            Maximum area of starspot
            Unit: MSH (Millionth of Solar Hemisphere)
                - non-dimensional quantity

        Returns:
        --------
        spot_lifetime - np.float64
            Lifetime of the given starspot.
            Unit: years
        """
        if max_spot_area < 85:
            spot_lifetime = gw_corr_coeffs[0]*np.exp(gw_corr_coeffs[1]*max_spot_area)/year2day
        else:
            spot_lifetime = max_spot_area/gw_constant/year2day
        return spot_lifetime

    def get_rotation(self, w_equator):
        """Gets the rotation speed of the starspot 
        considering a differential rotation profile of the star.
        
        Inputs:
        -------
        w_equator - np.float64
            Rotation rate at the equator.
            Unit:

        Returns:
        -------
        Rotation rate at the starspot latitude
        """
        return w_equator*(1. - self.wr2*np.sin(self.latitude)**2)



class Star():
    #-------------------------------------------------cgs units
    inclination = np.radian(70.)
    Io = 1.
    Cs = 0.67

    nbeta = 90
    ntheta = 45#8000, 10000, 5000 

    def __init__():
        

