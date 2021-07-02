import numpy as np

__author__ = "Angela Santos, Samarth Kashyap"
__all__ = ["Spot", "Star"]

Oeq = 2*np.pi/25.
year2day = 365.25
day2hour = 24
rsun = 6.955e10
cycle_length = 11.            # years
total_time = 1.               # years
time_step = total_time/year2day/day2hour
cad = (1/year2day)/time_step
gw_constant = 10.
gw_corr_coeffs = [5., 6.2591e-3]

wr = np.array([14.713, -2.396, -1.787])
alpha = 0.2

equator_rot_rate = 2*np.pi/25.
const = Oeq*alpha*np.sin(np.radians(90.))**2
wr2 = [alpha, equator_rot_rate*year2day]


class Spot():
    __all__ = ["evolve",
               "get_lifetime",
               "get_rotation"]

    __attributes__ = ["spot_id", "latitude", "longitude",
                      "initial_time", "first_minimum", "overlap_length",
                      "area_logn", "evolution_coeffs",
                      "mean_latitude_coeffs", "sigmaL_coeffs"]

    def __init__(self,
                 latitude=np.radians(90.),
                 longitude=np.radians(45.),
                 spot_id=0, no_evolution=False,
                 len_time=None):
        self.time_step = time_step
        self.latitude = latitude
        self.longitude = longitude
        self.spot_id = int(spot_id)

        area_logn = {}
        area_logn['mean'] = 4.0374632
        area_logn['sigma'] = 1.0521994
        self.area_logn = area_logn

        evolution_coeffs = {}
        evolution_coeffs['growth'] = {}
        evolution_coeffs['decay'] = {}
        evolution_coeffs['growth']['gamma1'] = 0.001
        evolution_coeffs['growth']['gamma2'] = 0.2
        evolution_coeffs['decay']['gamma1'] = 0.001
        evolution_coeffs['decay']['gamma2'] = 0.2
        self.evolution_coeffs = evolution_coeffs

        mean_latitude_coeffs = {}
        mean_latitude_coeffs['L0'] = 34.9987
        mean_latitude_coeffs['t0'] = 1995.1150
        mean_latitude_coeffs['damping'] = 7.5
        self.mean_latitude_coeffs = mean_latitude_coeffs

        sigmaL_coeffs = {}
        sigmaL_coeffs['a_sigma'] = 0.0998435
        sigmaL_coeffs['b_sigma'] = 1.20313
        sigmaL_coeffs['c_sigma'] = -0.952859
        sigmaL_coeffs['Pc'] = cycle_length
        self.sigmaL_coeffs = sigmaL_coeffs

        self.isevolved = False
        self.latitude_list = None
        self.longitude_list = None
        self.spot_area_list = None
        self.time_list = None
        self.no_evolution = no_evolution
        self.len_time = len_time

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
        if self.isevolved:
            return self.spot_area_list

        if self.no_evolution:
            longitude_list = self.rotate_nonevolving_spots(max_spot_area)
            return self.spot_area_list

        self.lifetime = self.get_lifetime(max_spot_area) * year2day

        latitude_list = []
        longitude_list = []
        spot_area_list = []
        time_list = []

        spot_area = 1e-4 * max_spot_area
        time = 0.0
        time_step_days = self.time_step * year2day
        # growth loop
        for ti in np.arange(1, self.lifetime+1, time_step_days):
            gamma1 = self.evolution_coeffs['growth']['gamma1']
            gamma2 = self.evolution_coeffs['growth']['gamma2']
            d_area = (np.exp(gamma1) * spot_area**gamma2)/cad
            spot_area += d_area
            if spot_area > max_spot_area: break
            self.latitude = self.latitude
            self.longitude = (self.longitude +
                              self.get_rotation(equator_rot_rate)*time_step) % (2*np.pi)
            time += time_step_days
            latitude_list.append(self.latitude)
            longitude_list.append(self.longitude)
            spot_area_list.append(spot_area)
            time_list.append(time)

        # decay loop
        for ti in np.arange(0, self.lifetime+1, time_step_days):
            gamma1 = self.evolution_coeffs['decay']['gamma1']
            gamma2 = self.evolution_coeffs['decay']['gamma2']
            d_area = (np.exp(gamma1) * spot_area**gamma2)/cad
            spot_area -= d_area
            if spot_area < 0: break
            self.latitude = self.latitude
            self.longitude = (self.longitude +
                              self.get_rotation(equator_rot_rate)*time_step) % (2*np.pi)
            time += time_step_days
            latitude_list.append(self.latitude)
            longitude_list.append(self.longitude)
            spot_area_list.append(spot_area)
            time_list.append(time)

        self.isevolved = True
        self.latitude_list = latitude_list
        self.longitude_list = longitude_list
        self.spot_area_list = spot_area_list
        self.time_list = time_list
        return spot_area_list

    def evolve_old(self, max_spot_area):
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
        if self.isevolved:
            return self.spot_area_list

        self.lifetime = self.get_lifetime(max_spot_area) * year2day

        latitude_list = []
        longitude_list = []
        spot_area_list = []
        time_list = []

        time = 0.0
        time_step_days = self.time_step * year2day

        # decay loop
        spot_area = max_spot_area*1.0
        spot_area_list = [spot_area]
        for ti in np.arange(1, self.lifetime+1, self.time_step*year2day):
            gamma1 = self.evolution_coeffs['decay']['gamma1']
            gamma2 = self.evolution_coeffs['decay']['gamma2']
            d_area = (np.exp(gamma1) * spot_area**gamma2)/cad
            spot_area -= d_area
            if spot_area < 0: break
            self.latitude = self.latitude
            self.longitude = (self.longitude +
                              self.get_rotation(equator_rot_rate)*time_step) % (2*np.pi)
            time += time_step_days
            latitude_list.append(self.latitude)
            longitude_list.append(self.longitude)
            spot_area_list.append(spot_area)
            time_list.append(time)

        tg = self.lifetime - len(spot_area_list)/cad

        # growth loop
        spot_area = max_spot_area*1.0
        for ti in np.arange(0, tg, self.time_step*year2day):
            gamma1 = self.evolution_coeffs['growth']['gamma1']
            gamma2 = self.evolution_coeffs['growth']['gamma2']
            d_area = (np.exp(gamma1) * spot_area**gamma2)/cad
            spot_area -= d_area
            if spot_area < 0: break

            self.latitude = self.latitude
            self.longitude = (self.longitude +
                              self.get_rotation(equator_rot_rate)*time_step) % (2*np.pi)
            time -= time_step_days
            latitude_list.insert(0, self.latitude)
            longitude_list.insert(0, self.longitude)
            spot_area_list.insert(0, spot_area)
            time_list.insert(0, time)

        self.isevolved = True
        self.latitude_list = latitude_list
        self.longitude_list = longitude_list
        self.spot_area_list = spot_area_list
        self.time_list = time_list
        return spot_area_list


    def rotate_nonevolving_spots(self, max_spot_area):
        """Rotates the non-evolving spots using the provided
        differentially rotating profile of the star.

        """
        if self.isevolved:
            return self.spot_area_list

        latitude_list = []
        longitude_list = []
        spot_area_list = []
        time_list = []

        for idx in range(self.len_time):
            self.longitude = (self.longitude +
                              self.get_rotation(equator_rot_rate)*time_step) % (2*np.pi)
            time_list.append(idx*time_step)
            longitude_list.append(self.longitude)
            latitude_list.append(self.latitude)
            spot_area_list.append(max_spot_area)

        self.isevolved = True
        self.latitude_list = latitude_list
        self.longitude_list = longitude_list
        self.spot_area_list = spot_area_list
        self.time_list = time_list

        return longitude_list

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

    def get_rotation(self, equator_rot_rate):
        """Gets the rotation speed of the starspot 
        considering a differential rotation profile of the star.

        Inputs:
        -------
        equator_rot_rate - np.float64
            Rotation rate at the equator.
            Unit:

        Returns:
        -------
        Rotation rate at the starspot latitude
        """
        return equator_rot_rate*(1. - wr2[0]*np.sin(self.latitude)**2)*year2day


class Star():
    Io = 1.
    Cs = 0.67

    nbeta = 90
    ntheta = 45#8000, 10000, 5000 

    initial_time = 1996.3525
    first_minimum = 1996.35
    overlap_length = 1.0

    latitude_mean = np.radians(15.)
    latitude_sigma = np.radians(5.)

    def __init__(self, inclination=70., no_evolution=False):
        self.inclination = np.radians(inclination)
        self.spot_count = 0
        self.spot_dict = None
        self.no_evolution = no_evolution

        # Setting fixed number of non-evolving spots
        self.num_spots = 2 if self.no_evolution else None

        sunspotnum_coeffs = {}
        sunspotnum_coeffs['a1'] = 0.26301229/6.
        sunspotnum_coeffs['t0'] = 1995.1020
        sunspotnum_coeffs['b1'] = 4.7786238
        sunspotnum_coeffs['c1'] = -0.2908343
        self.sunspotnum_coeffs = sunspotnum_coeffs
        self.max_spot_area = 1e1

        time_arr = np.arange(self.initial_time, self.initial_time+total_time, time_step)
        self.time_arr = time_arr
        self.light_curve = np.zeros_like(self.time_arr)

    def simulate_spots(self):
        spot_dict = {}
        spot_id = 0
        time_arr = np.arange(self.initial_time, self.initial_time+total_time, time_step)
        spot_exists_flag = np.zeros_like(time_arr, dtype=np.bool)

        if self.no_evolution:
            longitudes = np.random.uniform(0, 2*np.pi, self.num_spots)
            latitudes = np.random.normal(self.latitude_mean, self.latitude_sigma, self.num_spots)
            for idx in range(self.num_spots):
                spot_id = self.spot_counter()
                spot = Spot(latitude=latitudes[idx],
                            longitude=longitudes[idx],
                            spot_id=spot_id,
                            no_evolution=True,
                            len_time=len(time_arr))
                spot.evolve(self.max_spot_area)
                spot_dict[f'{spot_id}'] = {}
                spot_dict[f'{spot_id}']['time'] = np.array(spot.time_list) + self.initial_time
                spot_dict[f'{spot_id}']['area'] = np.array(spot.spot_area_list)
                spot_dict[f'{spot_id}']['latitude'] = np.array(spot.latitude_list)
                spot_dict[f'{spot_id}']['longitude'] = np.array(spot.longitude_list)
                spot_dict[f'{spot_id}']['spot_exists'] = spot_exists_flag + True
            self.spot_dict = spot_dict
        else:
            for tidx, time in enumerate(time_arr):
                # Nm = self.nspots(time)
                num_spots = np.random.poisson(0.0006)
                # num_spots = np.random.poisson(Nm)
                longitudes = np.random.uniform(0, 2*np.pi, num_spots)
                latitudes = np.random.normal(self.latitude_mean, self.latitude_sigma, num_spots)
                if num_spots < 0: continue
                for idx in range(num_spots):
                    spot_id = self.spot_counter()
                    spot = Spot(latitude=latitudes[idx],
                                longitude=longitudes[idx],
                                spot_id=spot_id)
                    spot.evolve(self.max_spot_area)
                    spot_dict[f'{spot_id}'] = {}
                    spot_dict[f'{spot_id}']['time'] = spot.time_list + time
                    spot_dict[f'{spot_id}']['area'] = np.array(spot.spot_area_list)
                    spot_dict[f'{spot_id}']['latitude'] = np.array(spot.latitude_list)
                    spot_dict[f'{spot_id}']['longitude'] = np.array(spot.longitude_list)

                    len_time = len(spot_dict[f'{spot_id}']['time'])
                    spot_dict[f'{spot_id}']['spot_exists'] = spot_exists_flag*True
                    spot_dict[f'{spot_id}']['spot_exists'][tidx:tidx+len_time] = True
                # if spot_id % 100 == 0: print(f'spot_id = {spot_id}')
            self.spot_dict = spot_dict

    def spot_counter(self):
        """Counter for starspots"""
        self.spot_count = self.spot_count + 1
        return self.spot_count

    def nspots(self, time):
        """Number of starspots, given time from epoch of minima.

        Inputs:
        -------
        time - np.float64
            time at which spots are observed
            Unit: day

        Returns:
        --------
        num_spots - int
            Number of spots found at given time
        """
        a1 = self.sunspotnum_coeffs['a1']
        t0 = self.sunspotnum_coeffs['t0']
        b1 = self.sunspotnum_coeffs['b1']
        c1 = self.sunspotnum_coeffs['c1']
        num_spots = a1*(time - t0)**3/(np.exp((time - t0)**2/b1**2) - c1)
        return num_spots

    def proj_mu(inc, theta, beta):
        pmu = np.cos(math.radians(inc))*np.cos(math.radians(theta))+\
            np.sin(math.radians(inc))*np.sin(math.radians(theta))*np.cos(math.radians(beta))
        return pmu

    def limb_dark(Io, mu):
        ia = 0.5287
        ib = 0.2175
        Imu = Io*(1.0 - ia*(1.0 - mu) +
                  ib*(1.0 - mu)**2)
        return Imu

    def flux_p(Io, mu, dS):
        fp = limb_dark(Io,mu)*mu
        return fp

    def dflux_s(Io, mu, Rsun, dS):
        if 0 <= mu <= 1:
            fp = flux_p(Io,mu,dS)
            fs = fp*(1.-Cs)*dS/math.pi/Rsun**2
        else:
            fs = 0
        return fs

    def tbstar(tspot,bspot,latspot):
        t1 = np.cos(tspot)*np.cos(latspot)
        t2 = np.sin(tspot)*np.cos(bspot)*np.sin(latspot)
        tstar = math.acos(t1+t2)
        sintstar = np.sqrt(1.-(t1+t2)**2)
        if latspot == 0:
            bstar = bspot
            tstar = tspot
        bstar = math.asin(np.sin(tspot)*np.sin(bspot)/sintstar)
        return math.degrees(tstar),math.degrees(bstar)



