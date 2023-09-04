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
    out = 0.075 / np.sqrt(1 + 718.0 + Ri ** 2.4)  # Parker et al. (1987)

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


def get_es(R, g, Ds, nu, u_star, U, h, r0, function="GP1991field", out=None):
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
        _gp1991(R, g, Ds, nu, u_star, p=0.1, out=out)
    if function == "GP1991exp":
        _gp1991(R, g, Ds, nu, u_star, p=1.0, out=out)
    if function=='wright_and_parker(2004)':
        _wright_and_parker(R, g, Ds, nu, u_star, sigma=0.52, w_k=4.0 * 10**-5, slope_inside=2.4*10**-5, out=None)
    if function=='Fukuda_etal_2023':
        _fukuda_etal_2023(u_star, U, g, R, h, Ds, nu, r0, out=out)
    if function=='Leeuw_2020':
        _leeuw_2020(u_star, U, g, R, h, Ds, nu, out=out)


    return out


def _gp1991(R, g, Ds, nu, u_star, p=1.0, out=None):
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
    a = 7.8 * 10 ** -7
    alpha = 0.6

    # calculate entrainment rate
    Z = sus_index * Rp ** alpha
    out[:, :] = p * a * Z ** 5 / (1 + (a / 0.3) * Z ** 5)

    return out

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

    return out

def _leeuw_2020(u_star, U, g, R, h, Ds, nu, out=None):
    """This is a method for calculation of sediment entrainment rate based on Leeuw (2020).
   Two parameter model using u_star (not using u_star_skin) is employed."""
    Fr = np.abs(U)/(g*h)**0.5
    ws = get_ws(R, g, Ds, nu)
    A = 4.74*10**-4
    P1 = u_star/ws
    P2 = Fr
    e1 = 1.71
    e2 = 1.18

    out[:, :] = A*P1**e1*P2**e2

    return out