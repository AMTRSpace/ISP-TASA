"""Lambert impactor functions"""


import numpy as np
import pykep as pk
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib import cm
from numba import njit
import numbers


def get_lc_traj_singleColor(xs, ys, cs, vmin, vmax, cmap, lw=0.8):
    """
    Get line collection object for a trajectory with a single color.
    Color is based on a colormap defined by vmin ~ vmax

    For plotting, run:
        line = ax.add_collection(lc)
        fig.colorbar(line, ax=ax, label="Colorbar label")

    Args:
        xs (np.array): array-like object of x-coordinates of the trajectory
        ys (np.array): array-like object of y-coordinates of the trajectory
        cs (float or np.array): float or array-like object of color-values along the coordinates
        vmin (float): minimum bound on colorbar
        vmax (float): maximum bound on colorbar
        cmap (str): colormap, e.g. 'viridis'
        lw (float): linewidth of trajectory

    Returns:
        (obj): line collection object
    """
    # check if cs is a float, and if it is convert it to an array
    if isinstance(cs, numbers.Real) == True:
    	cs = cs * np.ones((len(xs),))

    # generate segments
    points = np.array([ xs , ys ]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    plt_color = cs

    # create color bar
    norm = plt.Normalize( vmin, vmax )
    lc = LineCollection(segments, cmap=cmap, norm=norm)
    # Set the values used for colormapping
    lc.set_array( plt_color )
    lc.set_linewidth( lw )
    return lc


@njit
def get_rv_perifocal(mu, rp, ra, theta):
    """Compute r and v vectors in perifocal frame"""
    # compute orbital shape parameters
    e = (ra-rp)/(ra+rp)                # eccentricity
    a = rp / (1-e)                     # semimajor axis
    vp = np.sqrt(mu*(2/rp - 1/a))      # periapsis speed
    h = rp * vp                        # angular momentum
    # compute r and v vectors in perifocal frame
    ct = np.cos(theta)
    st = np.sin(theta)
    rPF = h**2/(mu*(1+e*ct)) * np.array([ct, st, 0.0])
    vPF = mu/h * np.array([-st, e+ct, 0.0])
    return rPF,vPF
    

def acos_safe(theta):
    return np.arccos(max(-1, min(theta, 1)))


def get_impactor_transfer(mu, rp, ra, theta_dep, r_surface, verbose=False):
    """Get impactor transfer by solving Lambert problem
    
    Args:
        mu (float): gravitational parameter
        rp (float): periapsis radius of orbiter
        ra (float): apoapsis radius of orbiter
        theta_dep (float): departure true anomaly of impactor
        r_surface (float): impact radius
        verbose (bool): verbosity flag
    
    Returns:
        (tuple): r1, v1_lamb, v2_lamb, DV1, tof, rp_vec_transfer, validity boolean
    """
    assert theta_dep > 0, "theta_dep must be larger than 0"
    # get initial and final positions
    r1, v1 = get_rv_perifocal(mu, rp, ra, theta_dep)
    r2 = np.array([r_surface, 0.0, 0.0])

    # transfer time
    e = (ra-rp)/(ra+rp)                # eccentricity
    a = (rp+ra)/2                      # semimajor axis
    period = 2*np.pi*np.sqrt(a**3/mu)
    E0 = 2*np.arctan(np.sqrt((1-e)/(1+e)) * np.tan(theta_dep/2))
    M0 = E0 - e*np.sin(E0)
    if M0 < 0:
        M0 = 2*np.pi + M0
    t0 = M0/(2*np.pi) * period
    tof = period - t0

    # solve Lambert problem
    l = pk.lambert_problem(r1, r2, tof, mu=mu, cw=False, max_revs=0)
    v1_lamb = l.get_v1()[0]
    v2_lamb = l.get_v2()[0]
    DV1 = v1_lamb - v1                 # transfer delta-V

    # compute aop of transfer
    energy_transfer = np.linalg.norm(v1_lamb)**2/2 - mu/np.linalg.norm(r1)
    a_transfer = -mu/(2*energy_transfer)
    hvec = np.cross(r1, v1_lamb)
    h_transfer = np.linalg.norm(hvec)
    evec = np.cross(v1_lamb,hvec)/mu - r1/np.linalg.norm(r1)
    e_transfer = np.linalg.norm(evec)
    rp_transfer = a_transfer * (1 - e_transfer)
    aop_transfer = np.arccos(evec[0]/e_transfer)
    if evec[1] < 0:
        aop_transfer *= -1
    valid = aop_transfer >= 0.0
    rp_vec_transfer = rp_transfer * np.array([np.cos(aop_transfer), np.sin(aop_transfer)])

    # print message
    if verbose:
        print(f"Transfer location M0 = {np.rad2deg(M0):1.3f} deg")
        print(f"Transfer time = {tof:1.4f} ({tof/period:1.4f} of period)")
    return r1, v1_lamb, v2_lamb, DV1, tof, rp_vec_transfer, valid


def get_transfer_history(mu,r0,v0,tof,steps=100):
    rs, vs = np.zeros((3,steps)), np.zeros((3,steps))
    times = np.linspace(0.0, tof,steps)
    for idx,t in enumerate(times):
        rs[:,idx], vs[:,idx] = pk.propagate_lagrangian(r0=r0, v0=v0, tof=t, mu=mu)
    return rs, vs


def plot_circle(center, radius, ax, steps=100, color='k', linewidth=0.5):
    thetas = np.linspace(0.0, 2*np.pi, steps)
    rs = np.zeros((2,steps))
    for idx,theta in enumerate(thetas):
        rs[:,idx] = np.array([center[0] + radius*np.cos(theta), center[1] + radius*np.sin(theta)])
    ax.plot(rs[0,:], rs[1,:], color=color, linewidth=linewidth)
    return


def plot_perifocal_orbit(mu, rp, ra, steps=100, ax=None, color='dodgerblue', linewidth=0.5):
    thetas = np.linspace(0.0, 2*np.pi, steps)
    rs = np.zeros((3,steps))
    for idx,theta in enumerate(thetas):
        rs[:,idx],_ = get_rv_perifocal(mu, rp, ra, theta)
    if ax is not None:
        ax.plot(rs[0,:], rs[1,:], color=color, linewidth=linewidth)
    return rs


def rv_to_fpa(rvec,vvec,mu):
    """Compute flight path angle from r and v vectors"""
    kep_elts = pk.ic2par(rvec, vvec, mu)    # Keplerian elements a,e,i,W,w,E
    e = kep_elts[1]
    ta = 2 * np.arctan( np.sqrt((1-e)/(1+e)) * np.tan(kep_elts[5]/2) )
    fpa = e * np.sin(ta) / (1 + e*np.cos(ta))
    return fpa


