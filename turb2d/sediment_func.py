import numpy as np
"""
Empirical functions for calculating models of sediment dynamics
"""


def get_ew(U, Ch, R, g, umin=0.01, out=None):
    """ calculate entrainment coefficient of ambient water to a turbidity
        current layer

        Parameters
        ----------
        U : ndarray, float
           Flow velocities of a turbidity current.
        Ch : ndarray, float
           Flow height times sediment concentration of a turbidity current.
        R : float
           Submerged specific density of sediment
        g : float
           gravity acceleration
        umin: float
           minimum threshold value of velocity to calculate water entrainment

        out : ndarray
           Outputs

        Returns
        ---------
        e_w : ndarray, float
           Entrainment coefficient of ambient water

    """
    if out is None:
        out = np.zeros(U.shape)

    Ri = np.zeros(U.shape)
    flowing = np.where(U > umin)
    Ri[flowing] = R * g * Ch[flowing] / U[flowing] ** 2
    out = 0.075 / np.sqrt(1 + 718.0*Ri ** 2.4)  # Parker et al. (1987)

    return out


def get_ws(R, g, Ds, nu):
    """ Calculate settling velocity of sediment particles
        on the basis of Ferguson and Church (1982)

    Return
    ------------------
    ws : settling velocity of sediment particles [m/s]

    """

    # Coefficients for natural sands
    C_1 = 18.0
    C_2 = 1.0

    ws = R * g * Ds ** 2 / (C_1 * nu + (0.75 * C_2 * R * g * Ds ** 3) ** 0.5)

    return ws

def get_det_rate(ws, Ch_i, h, det_coef=1.0, out=None):
    """ calculate water detrainment rate
    Parameters
    ------------
    ws : float
        settling velocity of sediment particles
    Ch_i : float
        sediment concentration of a turbidity current at wet nodes
    h : float
        flow depth of a turbidity current
    det_coef : float
        detrainment coefficient
    out : ndarray
        Outputs
    """

    if out is None:
        out = np.zeros(h.shape)

    # if there are no wet nodes, water detrainment is set to zero.
    if Ch_i.size  == 0:
        out = 0.0
    # if there are wet nodes, water entrainment is calculated at each wet nodes. 
    else:
        # calculate weighted mean settlig velocity
        weighted_mean_ws = np.mean(ws*Ch_i/h, axis=0)
        # coefficient of detrainment rate
        out = det_coef*weighted_mean_ws

    return out

def get_es(R, g, Ds, nu, u_star, U, h, Ch, S, r0, salt, p_gp1991, function="GP1991field", out=None):
    """ Calculate entrainment rate of basal sediment to suspension using
        empirical functions proposed by Garcia and Parker (1991),
        van Rijn (1984), or Dorrell (2018)

        Parameters
        --------------
        R : float
            submerged specific density of sediment (~1.65 for quartz particle)
        g : float
            gravity acceleration
        Ds : float
            grain size
        nu : float
            kinematic viscosity of water
        u_star : ndarray
            flow shear velocity
        U: ndarray
            layer-averaged flow velocity
        h: ndarray
            flow depth
        Ch: ndarray
            Volume of suspended sediment in the flow
        r0: float
            Ratio of near-bed concentration to layer-averaged concentration
        S: ndarray
            slope of the bed
        salt: bool
            True if the flow is saline, False if the flow is fresh
        p_gp1991: float
            coefficient in Garcia and Parker (1991)
        function : string, optional
            Name of emprical function to be used.

            'GP1991exp' is a function of Garcia and Parker (1991)
             in original form. This is suitable for experimental scale.

            'GP1991field' is Garcia and Parker (1991)'s function with
            a coefficient (0.1) to limit the entrainment rate. This is suitable
            for the natural scale.

        out : ndarray
            Outputs (entrainment rate of basal sediment)

        Returns
        ---------------
        out : ndarray
            dimensionless entrainment rate of basal sediment into
            suspension



    """
    if out is None:
        out = np.zeros([len(Ds), len(u_star)])

    if function == "GP1991field":
        # p=1.0 in original paper
        out, flow_power, Phi = _gp1991(R, g, Ds, nu, u_star, U, h, p=p_gp1991, out=out)
    elif function == "GP1991exp":
        # p=0.1 in original paper
        out, flow_power, Phi = _gp1991(R, g, Ds, nu, u_star, U, h, p=p_gp1991, out=out)
    elif function=='wright_and_parker(2004)':
        _wright_and_parker(R, g, Ds, nu, u_star, sigma=0.52, w_k=4.0 * 10**-5, slope_inside=2.4*10**-5, out=None)
    elif function=='Fukuda_etal_2023':
        out, flow_power, Phi = _fukuda_etal_2023(u_star, U, g, R, h, Ds, nu, r0, out=out)
    elif function=='Leeuw_2020':
        out, flow_power, Phi = _leeuw_2020(u_star=u_star, U=U, Ch=Ch, g=g, R=R, h=h, Ds=Ds, nu=nu, salt=salt, S=S, out=out)
    else:
        raise ValueError("Please enter the correct entrainment function")


    return out, flow_power, Phi


def _gp1991(R, g, Ds, nu, u_star, U, h, p=1.0, out=None):
    """ Calculate entrainment rate of basal sediment to suspension
        Based on Garcia and Parker (1991)

        Parameters
        --------------
        u_star : ndarray
            flow shear velocity
        out : ndarray
            Outputs (entrainment rate of basal sediment)

        Returns
        ---------------
        out : ndarray
            dimensionless entrainment rate of basal sediment into
            suspension
    """
    if out is None:
        out = np.zeros([len(Ds), u_star.shape])

    # basic parameters
    ws = get_ws(R, g, Ds, nu)

    # calculate subordinate parameters
    Rp = np.sqrt(R * g * Ds) * Ds / nu
    sus_index = u_star / ws

    # coefficients for calculation
    a = 1.3 * 10 ** -7
    alpha_1 = np.zeros(Rp.shape)
    alpha_2 = np.zeros(Rp.shape)
    for i in range(len(Rp)):
        if Rp[i] > 2.36:
            alpha_1[i] = 1.0
            alpha_2[i] = 0.6
        elif Rp[i] <= 2.36:
            alpha_1[i] = 0.586
            alpha_2[i] = 1.23

    # calculate entrainment rate
    Z = alpha_1 * sus_index * Rp ** alpha_2
    out[:, :] = p * a * Z ** 5 / (1 + (a / 0.3) * Z ** 5)
    P_f = u_star**2*(np.abs(U))
    N_f = g*R*h*ws
    flow_power = P_f/N_f
    phi = out

    return out, flow_power, phi

def _wright_and_parker(R, g, Ds, nu, u_star, sigma, w_k, slope_inside, out=None\
):

    if out is None:
        out = np.zeros(u_star.shape)

    a = 7.8 * 10**-7
    me = 1.0
    Rp = np.sqrt(R * g * Ds) * Ds / nu
    kshi=1-0.288*sigma
    Ds50=Ds

    if Rp>2.36:
        alpha_1=1.0
        alpha_2=0.6
    elif Rp<=2.36:
        alpha_1=0.586
        alpha_2=1.23

    Z = alpha_1 * kshi * (u_star/w_k) * Rp**alpha_2 * slope_inside*0.08 * (Ds/Ds50)**0.2
    out[:] = me * a * Z**5 / (1 + (a / 0.3) * Z**5)

    return out

def _fukuda_etal_2023(u_star, U, g, R, h, Ds, nu, r0, out=None):
    """Calculate sediment entrainment rate based on fukuda et al. (2023).
    First, depth-averaged concentration is calculated. 
    Sediment entrainment rate (basal sediment concentration) is calculated using cb = r0*C.
    """

    if out is None:
        out = np.zeros([len(Ds), u_star.shape])

    ws = get_ws(R, g, Ds, nu)

    P_f = u_star**2*(np.abs(U))
    N_f = g*R*h*ws
    flow_power = P_f/N_f
    phi = (5.6*10**(-3))*flow_power**(0.36)

    out[:, :] = r0*phi

    return out, flow_power, phi

def _leeuw_2020(u_star, U, Ch, g, R, h, Ds, nu, salt, S, out=None):
    """This is a method for calculation of sediment entrainment rate based on Leeuw (2020).
   Two parameter model is employed."""
    if salt is True:
        C_i = Ch[:4, :]/h
    elif salt is False:
        C_i = Ch/h 
    C_T = np.sum(C_i, axis=0)
    Fr = U/np.sqrt(R*g*C_T*h)
    ws = get_ws(R, g, Ds, nu)
    # Z = (u_star/ws)**0.945 * Fr - 0.05
    # Z[Z < 0.0] = 0.0
    ks = 2*np.mean(Ds[:4, :])
    h_sk = U**(3/2) * ks**(1/4) / (8.1**(2/3) * (R*C_T*g*S)**(3/4))
    u_star_skin = np.sqrt(R*C_T*g*h_sk*S)
    # out[:, :] = 7.04 * 10**-4 * (u_star/ws)**1.71 * Fr**1.81
    # out[:, :] = 7.04 * 10**-4 * (Z**1.81) / (1 + 3 * (7.04 * 10**-4 * Z**1.81))
    out[:, :] = (4.74 * 10**-4) * ((u_star_skin/ws)**1.77) * Fr**1.18
    P_f = u_star**2*(np.abs(U))
    N_f = g*R*h*ws
    flow_power = P_f/N_f
    phi = out

    return out, flow_power, phi

def get_bedload(u_star, Ds, R=1.65, g=9.81, function="MPM", out=None):
    """Get bedload discharge from empirical formulation

       Parameters
       ------------------------------
       u_star: 1d ndarray
          friction velocity

       Ds: 1d ndarray
          grain diameters

       R: float, optional
          Submerged specific density of sediment particles.
          Default is 1.65

       g: float, optional
          gravity acceleration.
          Default is 9.81

       function: str, optional
          Function name for prediting bedload discharge
          Default is "MPM". Other options are:
          "WP2006": Wong and Parker (2006)

       out: 1d ndarray
          Outputs (1d array of sediment bedload discharge)

       Returns
       ---------------
       out : ndarray
         1d array of sediment bedload discharge
       
    """

    if out is None:
        out = np.zeros([len(Ds), len(u_star)])

    if function == "MPM":
        _MPM(u_star, Ds, R, g, a=8.0, b=1.5, out=out)
    elif function == "WP2006":
        _MPM(u_star, Ds, R, g, a=4.93, b=1.6, out=out)
    else:
        _MPM(u_star, Ds, R, g, a=8.0, b=1.5, out=out)

    return out

def _MPM(u_star, Ds, R=1.65, g=9.81, a=8.0, b=1.5, out=None):
    """Bedload prediction by Meyer=Peter and
       Muller (1948)-type equations

       Parameters
       ------------------------------
       u_star: 1d ndarray
          friction velocity

       Ds: 1d ndarray
          grain diameters

       R: float, optional
          Submerged specific density of sediment particles.
          Default is 1.65

       g: float, optional
          gravity acceleration.
          Default is 9.81

       a: float, optional
          coefficient used in the MPM equation
          Default is 8.0

       b: float, optional
          exponent used in the MPM-type equation
          
       out: 1d ndarray
          Outputs (1d array of sediment bedload discharge)

       Returns
       ---------------
       out : ndarray
         1d array of sediment bedload discharge
       
    """

    if out is None:
        out = np.zeros([len(Ds), u_star.shape])

    tau_c = 0.047

    tau_star_c = u_star * u_star / (R * g * Ds) - tau_c

    tau_star_c = np.where(
        tau_star_c > 0.0,
        tau_star_c,
        0.0
    )

    out[:, :] = a * tau_star_c ** b * np.sqrt(R * g * Ds ** 3)

    return out
