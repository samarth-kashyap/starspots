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

