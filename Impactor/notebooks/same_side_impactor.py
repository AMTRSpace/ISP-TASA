"""Conceptual development for impactor on the same side of the Moon via perilune lowering"""


import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib import cm
import numbers
import numpy as np
from tqdm.auto import tqdm


# define constants
MU_MOON = 4905          # km^3/s^2
R_MOON = 1737           # km


def get_lc_traj_singleColor(xs, ys, cs, vmin, vmax, cmap, lw=0.8):
    """
    Get line collection object for a trajectory with a single color based on a colormap defined by vmin ~ vmax

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


def plot_circle(ax, center, radius, color='grey', lw=0.85, label=None):
    thetas = np.linspace(0, 2*np.pi, 500)
    xs = center[0] + radius * np.cos(thetas)
    ys = center[1] + radius * np.sin(thetas)
    ax.plot(xs, ys, color=color, lw=lw, label=label)
    return


def visvisa(mu, r, a):
    return np.sqrt(mu*(2/r - 1/a))

def change_periapsis(mu, ra, rp0, rp1):
    sma0 = (ra + rp0) / 2
    va0 = visvisa(mu, ra, sma0)   # velocity at apolune
    sma1 = (ra + rp1) / 2
    va1 = visvisa(mu, ra, sma1)   # velocity at apolune
    DV = va1 - va0
    return DV

class PerifocalOrbit:
    def __init__(self, mu, sma, ecc):
        self.mu = mu
        self.sma = sma
        self.ecc = ecc

        # compute angular momentum
        p = sma * (1 - ecc**2)
        self.h = np.sqrt(self.mu * p)
        return
    
    def perifocal_position(self, theta):
        return self.h**2/self.mu /(1 + self.ecc*np.cos(theta)) * np.array([np.cos(theta), np.sin(theta), 0])
    
    def perifocal_positions(self, thetas):
        return np.array([self.perifocal_position(theta) for theta in thetas])
    
    def perifocal_velocity(self, theta):
        return self.mu/self.h * np.array([-np.sin(theta), self.ecc + np.cos(theta), 0])
    
    def perifocal_velocities(self, thetas):
        return np.array([self.perifocal_velocity(theta) for theta in thetas])
    
    def rnorm2theta(self, rnorm):
        """Compute true anomaly from position vector norm"""
        return np.arccos(1/self.ecc * (self.h**2/(self.mu * rnorm) - 1))
    
    def get_impact_conditions(self, impact_radius, vr_positive = False):
        """Get impact conditions at a given radius"""
        theta_impact = self.rnorm2theta(impact_radius)
        if vr_positive is False:
            theta_impact = 2*np.pi - theta_impact
        rvec_impact = self.perifocal_position(theta_impact)
        vvec_impact = self.perifocal_velocity(theta_impact)

        # compute impact angle
        impact_angle = np.arccos(
            np.dot(rvec_impact, vvec_impact) / (np.linalg.norm(rvec_impact) * np.linalg.norm(vvec_impact))
        ) - np.pi/2
        return theta_impact, rvec_impact, vvec_impact, impact_angle
    

def single_case_demo(verbose = True):
    # initial orbit
    rp0 = R_MOON + 10.0
    ra0 = 25e3
    sma0 = (rp0 + ra0) / 2
    va0 = visvisa(MU_MOON, ra0, sma0)   # velocity at apolune

    # final orbit
    rp1 = 1000.0
    ra1 = ra0
    sma1 = (rp1 + ra1) / 2
    va1 = visvisa(MU_MOON, ra1, sma1)   # velocity at apolune

    # maneuver
    DV = change_periapsis(MU_MOON, ra0, rp0, rp1)
    if verbose:
        print(f"Maneuver at apolune of {ra1:1.0f} km to change perilune from {rp0:1.0f} km to {rp1:1.0f} km: {DV * 1e3:1.2f} m/s")

    # plot
    initial_orbit = PerifocalOrbit(MU_MOON, sma0, (ra0 - rp0) / (ra0 + rp0))
    final_orbit = PerifocalOrbit(MU_MOON, sma1, (ra1 - rp1) / (ra1 + rp1))

    r_initial = initial_orbit.perifocal_positions(np.linspace(0, 2*np.pi, 500))
    r_final = final_orbit.perifocal_positions(np.linspace(0, 2*np.pi, 1500))

    _,r_impact,_,impact_angle = final_orbit.get_impact_conditions(R_MOON)

    fig, ax = plt.subplots(1, 1, figsize=(8,5))
    plot_circle(ax, [0,0], R_MOON, color='grey', label='Moon')
    ax.plot(r_initial[:,0], r_initial[:,1], label=f"Initial orbit (rp = {rp0:1.0f} km)", lw=0.75)
    ax.plot(r_final[:,0], r_final[:,1], label=f"Final orbit (rp = {rp1:1.0f} km)", lw=0.75)
    ax.scatter(-ra1, 0, color='g', marker='^', label=f"DV {DV * 1e3:1.2f} m/s")  
    ax.scatter(r_impact[0], r_impact[1], color='r', marker="x", label=f"Impact (angle = {np.rad2deg(impact_angle):1.2f} deg)")
    ax.set_aspect('equal')
    ax.set(xlabel="x, km", ylabel="y, km")
    ax.set_title(f"Impactor path in perifocal frame (apolune radius = {ra0:1.0f} km)")
    ax.legend()
    return


def plot_design_space():
    rp0 = R_MOON + 10.0                         # we fix perilune radius before maneuver
    ra0_grid = np.linspace(8e3, 70e3, 20)       # we grid through apolune radius before maneuver
    rp1_grid = np.linspace(100, 1500, 100)      # we grid through perilune radius after maneuver

    # prepare plot
    fig, axs = plt.subplots(3,1,figsize=(9,8))#, constrained_layout = True)
    for ax in axs:
        ax.grid(True, alpha=0.5)
    
    # iterate through ra0 values
    for ra0 in ra0_grid:
        # compute DV
        DVs = np.array([change_periapsis(MU_MOON, ra0, rp0, rp1) for rp1 in rp1_grid])
        axs[0].plot(rp1_grid, np.abs(DVs)*1e3, alpha=0)
        lc = get_lc_traj_singleColor(rp1_grid, np.abs(DVs)*1e3, ra0/1e3, vmin=min(ra0_grid)/1e3, vmax=max(ra0_grid)/1e3, cmap='viridis', lw=0.8)
        line = axs[0].add_collection(lc)

        # get impact condition
        final_orbits = [PerifocalOrbit(MU_MOON, (rp1 + ra0) / 2,
                                       (ra0 - rp1) / (ra0 + rp1)) for rp1 in rp1_grid]
        angles = np.zeros(len(rp1_grid),)
        vimpacts = np.zeros(len(rp1_grid))
        for idx, _final_orbit in enumerate(final_orbits):
            _,_,vvec_impact,impact_angle = _final_orbit.get_impact_conditions(R_MOON)
            angles[idx] = impact_angle
            vimpacts[idx] = np.linalg.norm(vvec_impact)

        # plot impact angle
        axs[1].plot(rp1_grid, np.rad2deg(angles), alpha=0)
        lc = get_lc_traj_singleColor(rp1_grid, np.rad2deg(angles), ra0/1e3, vmin=min(ra0_grid)/1e3, vmax=max(ra0_grid)/1e3, cmap='viridis', lw=0.8)
        line = axs[1].add_collection(lc)

        # plot velopcities
        axs[2].plot(rp1_grid, vimpacts, alpha=0)
        lc = get_lc_traj_singleColor(rp1_grid, vimpacts, ra0/1e3, vmin=min(ra0_grid)/1e3, vmax=max(ra0_grid)/1e3, cmap='viridis', lw=0.8)
        line = axs[2].add_collection(lc)
    
    # formatting
    axs[0].set(xlabel="Post-maneuver perilune radius, km", ylabel="DV, m/s", yscale="log")
    axs[1].set(xlabel="Post-maneuver perilune radius, km", ylabel="Impact angle, deg")
    axs[2].set(xlabel="Post-maneuver perilune radius, km", ylabel="Impact velocity, km/s")

    # fig.colorbar(line, ax=axs[0], label="Apolune radius, km")
    fig.suptitle(f"Impactor with pre-maneuver perilune radius {rp0:1.0f} km")
    fig.subplots_adjust(right=0.82, left=0.075, top=0.95, bottom=0.07, hspace=0.25)
    cb_ax = fig.add_axes([0.85, 0.07, 0.02, 0.85])   # adjust values to fit nicely in plot
    cbar = fig.colorbar(line, cax=cb_ax, label="Apolune radius, $10^3$ km")  # im0 is the scatter to fit the colorbar on
    fig.savefig(f"plots/sameside_impactor_rp0_{rp0:1.0f}km.png", dpi=300)
    return


if __name__ == "__main__":
    single_case_demo()
    plot_design_space()
    plt.show()