import numpy as np
from params import starParams

__author__ = "Samarth Kashyap, Angela Santos"
__all__ = ["Spot", "Star"]

sp = starParams(type='star')


NAX = np.newaxis

year2day = 365.25
day2hour = 24
rsun = 6.955e10
cycle_length = 11.            # years
total_time = 1.               # years
time_step = total_time/year2day/day2hour
cad = (1/year2day)/time_step
gw_constant = 10.
gw_corr_coeffs = [5., 6.2591e-3]
initial_time = 1996.3525

wr = np.array([14.713, -2.396, -1.787])
alpha = 0.2

equator_rot_rate = 2*np.pi/25.
const = equator_rot_rate*alpha*np.sin(np.radians(90.))**2
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

        # coefficients that govern the evolution of spots
        evolution_coeffs = {}
        evolution_coeffs['growth'] = {}
        evolution_coeffs['decay'] = {}
        evolution_coeffs['growth']['gamma1'] = 0.001
        evolution_coeffs['growth']['gamma2'] = 0.2
        evolution_coeffs['decay']['gamma1'] = 0.001
        evolution_coeffs['decay']['gamma2'] = 0.2
        self.evolution_coeffs = evolution_coeffs

        # coefficients to determine latitude of spots
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
            time += time_step
            if time >= total_time:
                break
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
            time += time_step
            if time >= total_time:
                break
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
    Cs = 0.67

    nbeta = 90
    ntheta = 45#8000, 10000, 5000 

    initial_time = 1996.3525
    first_minimum = 1996.35
    overlap_length = 1.0

    latitude_mean = np.radians(15.)
    latitude_sigma = np.radians(5.)

    def __init__(self, inclination=90., no_evolution=False,
                 max_spot_area=1e3, Io=1.):
        self.Io = Io
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
        self.max_spot_area = max_spot_area

        self.spotref_latnum = 10
        self.spotref_longnum = 30
        self.spotref_longitudes = np.linspace(0, 2*np.pi, self.spotref_longnum) - np.pi/2.0

        time_arr = np.arange(self.initial_time, self.initial_time+total_time, time_step)
        self.time_arr = time_arr
        self.light_curve = np.zeros_like(self.time_arr)

    def simulate_spots(self):
        spot_dict = {}
        spot_id = 0
        time_arr = np.arange(self.initial_time, self.initial_time+total_time, time_step)
        spot_exists_flag = np.zeros_like(time_arr, dtype=np.bool)

        if self.no_evolution:
            # longitudes = np.random.uniform(0, 2*np.pi, self.num_spots)
            # latitudes = np.random.normal(self.latitude_mean,
            #                              self.latitude_sigma,
            #                              self.num_spots)
            longitudes = np.linspace(0, 2*np.pi, self.num_spots)
            latitudes = np.linspace(self.latitude_mean - self.latitude_sigma,
                                    self.latitude_mean + self.latitude_sigma,
                                    self.num_spots)
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
                spot_dict[f'{spot_id}']['area'] = np.array(spot.spot_area_list)*1e-6*np.pi*2*rsun**2
                spot_dict[f'{spot_id}']['latitude'] = np.array(spot.latitude_list)
                spot_dict[f'{spot_id}']['longitude'] = np.array(spot.longitude_list) - np.pi/2
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
                    spot_dict[f'{spot_id}']['area'] = np.array(spot.spot_area_list)*1e-6*np.pi*2*rsun**2
                    spot_dict[f'{spot_id}']['latitude'] = np.array(spot.latitude_list)
                    spot_dict[f'{spot_id}']['longitude'] = np.array(spot.longitude_list) - np.pi/2

                    mask_time = spot_dict[f'{spot_id}']['time'] <= self.initial_time + total_time
                    len_time = len(spot_dict[f'{spot_id}']['time'])
                    spot_dict[f'{spot_id}']['time'] = spot_dict[f'{spot_id}']['time'][mask_time]
                    spot_dict[f'{spot_id}']['area'] = spot_dict[f'{spot_id}']['area'][mask_time]
                    spot_dict[f'{spot_id}']['latitude'] = spot_dict[f'{spot_id}']['latitude'][mask_time]
                    spot_dict[f'{spot_id}']['longitude'] = spot_dict[f'{spot_id}']['longitude'][mask_time]
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

    def projected_mu(self, latitude, longitude):
        """Computes the projected mu; mu = cos(xi) where xi is the angle between
        the light-of-sight and the normal to area element.
        Eq(6.4) from Santos (2017) PhD. Thesis

        Inputs:
        -------
        latitude - np.ndarray(ndim=1, dtype=np.float64)
            colatitude of the starspot
            Unit: radians
        longitude - np.ndarray(ndim=1, dtype=np.float64)
            longitude of the starspot
            Unit: radians

        Returns:
        --------
        pmu - np.ndarray(ndim=1, dtype=np.float64)
            pmu = cos(xi)
        """
        pmu = np.cos(self.inclination)*np.cos(latitude)+\
            np.sin(self.inclination)*np.sin(latitude)*np.cos(longitude)
        return pmu

    def limb_dark(self, mu):
        """Returns limb darkened intensity.

        Inputs:
        -------
        mu - np.float64
            mu = np.cos(xi); xi is the angle between normal to area element
              and the line-of-sight vector

        Returns:
        --------
        Imn - np.float64
            limb darkened intensity
        """
        ia = 0.5287
        ib = 0.2175
        Imu = self.Io*(1.0 - ia*(1.0 - mu) +
                       ib*(1.0 - mu)**2)
        return Imu

    def flux_p(self, mu):
        fp = self.limb_dark(mu) * mu
        return fp

    def dflux_s(self, mu, area_element):
        fp = self.flux_p(mu)
        fs = fp * (1.0 - self.Cs) * area_element/np.pi/rsun**2
        return fs

    def convert_to_starref(self, spotref_lat, spotref_lon, starref_spotlat):
        t1 = np.cos(spotref_lat)*np.cos(starref_spotlat)
        t2 = np.sin(spotref_lat)*np.cos(spotref_lon)*np.sin(starref_spotlat)
        starref_lat = np.arccos(t1+t2)
        sintstar = np.sqrt(1.0 - (t1+t2)**2)
        # if starref_lat == 0:
        #     starref_long = spotref_lon
        #     starref_lat = spotref_lat
        starref_long = np.arcsin(np.sin(spotref_lat)*np.sin(spotref_lon)/sintstar)
        return starref_lat, starref_long

    def compute_light_curve(self):
        for idx, spot in self.spot_dict.items():
            areas = spot['area']
            spot_time = spot['time']
            start_time_idx = np.argmin(abs(self.time_arr - spot_time.min()))
            len_time = len(spot_time)
            spotref_maxlats = np.arcsin(np.sqrt(areas/np.pi)/rsun)

            # lat X lon X time
            spotref_lats = np.linspace(0, spotref_maxlats, self.spotref_latnum)[:, NAX, :]
            spotref_lons = self.spotref_longitudes[NAX, :, NAX]

            dlats = spotref_lats[1, 0, :] - spotref_lats[0, 0, :]
            dlons = self.spotref_longitudes[1] - self.spotref_longitudes[0]

            dlats = dlats[NAX, NAX, :]
            starref_lat, starref_lon = self.convert_to_starref(spotref_lats, spotref_lons,
                                                               spot['latitude'][NAX, NAX, :])
            starref_lon += spot['longitude'][NAX, NAX, :]
            proj_mu = self.projected_mu(starref_lat, starref_lon)
            area_element = (rsun**2) * np.sin(spotref_lats) * dlons * dlats
            flux_change = self.dflux_s(proj_mu, area_element)
            mask_mu = (proj_mu <= 1) * (proj_mu >= 0)
            flux_change = flux_change * mask_mu
            self.light_curve[start_time_idx:
                             start_time_idx+len_time] += flux_change.sum(axis=0).sum(axis=0)
        return spotref_maxlats

    def compute_fft(self, data, time):
        fft_data = np.fft.fft(data)
        fft_freq = np.fft.fftfreq(len(data), d=time[1]-time[0])
        pow_data = abs(fft_data)**2
        mask_pos = fft_freq > 0
        fft_freq = fft_freq[mask_pos]
        pow_data = pow_data[mask_pos]
        pow_data /= pow_data.max()
        return pow_data, fft_freq


