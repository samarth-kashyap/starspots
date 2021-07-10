class params(type='solar', prot=25., cycle_length=11.,
             initial_time=1996.3525):
    year2day = 365.25
    day2hour = 24
    rsun = 6.955e10
    gw_constant = 10.
    gw_corr_coeffs = [5., 6.2591e-3]
    equator_rot_rate = 2*np.pi/prot
    alpha = 0.2
    const = equator_rot_rate*alpha*np.sin(np.radians(90.))**2
    wr2 = [alpha, equator_rot_rate*year2day]

    if type == 'solar':
        cycle_length = 11.            # years
        total_time = 4.               # years
        time_step = total_time/year2day/day2hour
        cad = (1/year2day)/time_step
        initial_time = 1996.3525
        wr = np.array([14.713, -2.396, -1.787])

    elif type == 'starspot':
        cycle_length = cycle_length   # years
        total_time = 4.               # years
        time_step = total_time/year2day/day2hour
        cad = (1/year2day)/time_step
        initial_time = initial_time
