import numpy as np

class starParams():

    year2day = 365.25
    day2hour = 24
    rsun = 6.955e10
    gw_constant = 10.
    gw_corr_coeffs = [5., 6.2591e-3]
    alpha = 0.2

    def __init__(self, type='solar', prot=25., cycle_length=11.,
             initial_time=1996.3525):
        self.equator_rot_rate = 2*np.pi/prot
        self.const = self.equator_rot_rate*self.alpha*np.sin(np.radians(90.))**2
        self.wr2 = [self.alpha, self.equator_rot_rate*self.year2day]

        if type == 'solar':
            self.cycle_length = 11.            # years
            self.total_time = 4.               # years
            self.time_step = self.total_time/self.year2day/self.day2hour
            self.cad = (1/self.year2day)/self.time_step
            self.initial_time = 1996.3525
            self.wr = np.array([14.713, -2.396, -1.787])

        elif type == 'starspot':
            self.cycle_length = cycle_length   # years
            self.total_time = 4.               # years
            self.time_step = self.total_time/self.year2day/self.day2hour
            self.cad = (1/self.year2day)/self.time_step
            self.initial_time = initial_time
