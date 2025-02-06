
import numpy as np
from numba import jit, prange
from scipy import integrate
from scipy.special import eval_legendre
import excursion_set_functions as es
import warnings

__all__ = ['from_rRAdec_to_XYZ','from_rThetaPhi_to_XYZ','from_XYZ_to_rRAdec','from_RAdec_to_theta_phi',
           'HubbleFactorNomalized','Inverse_HubbleFactorNomalized','ComovingDistanceOverh','StrHminSec']


def StrHminSec(DeltaT):
    hh = int(DeltaT / 3600)
    minutes = int(DeltaT / 60) - hh * 60
    sec = DeltaT % 60
    return str(hh) + ' h ' + str(minutes) + ' min ' + str(sec) + ' sec.'


@jit(nopython=True)
def from_rRAdec_to_XYZ(r,RA,dec):
    x = r * np.cos(dec * np.pi / 180.) * np.sin(RA * np.pi / 180.)
    y = r * np.cos(dec * np.pi / 180.) * np.cos(RA * np.pi / 180.)
    z = r * np.sin(dec * np.pi / 180.)
    return x, y, z



@jit(nopython=True)
def from_rThetaPhi_to_XYZ(r,theta,phi):
    x = r * np.sin(theta) * np.cos(phi)
    y = r * np.sin(theta) * np.sin(phi)
    z = r * np.cos(theta)
    return x, y, z


@jit(nopython=True)
def from_XYZ_to_rRAdec_3arrs(x,y,z):
    r = np.sqrt(x*x + y*y + z*z)
    RA = np.arctan(x/y) * 180. / np.pi
    DEC = np.arcsin(z/r) * 180. / np.pi

    mask = (y < 0)
    RA[(x < 0) & mask] -= 180.
    RA[(x >= 0) & mask] += 180.
    mask[:] = (y == 0)
    RA[mask] += 90. * np.sign(x[mask])

    return r, np.remainder(RA,360), DEC

@jit(nopython=True)
def from_XYZ_to_rRAdec_1arr(xyz):
    rRADEC = np.empty(xyz.shape)
    rRADEC[:,0] = np.sqrt(np.sum(xyz * xyz,axis=1))
    rRADEC[:,1] = np.arctan(xyz[:,0] / xyz[:,1]) * 180. / np.pi

    mask = (xyz[:,1] < 0)
    rRADEC[(xyz[:,0] < 0) & mask,1] -= 180.
    rRADEC[(xyz[:,0] >= 0) & mask,1] += 180.
    mask[:] = (xyz[:,1] == 0)
    rRADEC[mask,1] += 90. * np.sign(xyz[mask,0])

    rRADEC[:,2] = np.arcsin(xyz[:,2] / rRADEC[:,0]) * 180. / np.pi

    return rRADEC

def from_XYZ_to_rRAdec(x,y=None,z=None):
    if (y is None):
        if (z is None):
            return from_XYZ_to_rRAdec_1arr(x)
        else:
            raise ValueError('accepted either 1 shape (N,3) array or 3 shape (N,) arrays')
    return from_XYZ_to_rRAdec_3arrs(x,y,z)

def from_RAdec_to_theta_phi(RA,dec):
    return 0.5 * np.pi - dec * np.pi / 180., RA * np.pi / 180.

@jit(nopython=True)
def HubbleFactorNomalized(Z,OmegaM,w0,wa):
    a = 1./(Z+1.)
    OhmDE = 1. - OmegaM
    return (OmegaM * (a ** -3.) + OhmDE * (a ** (-3. * (1. + w0 + wa))) * np.exp(-3. * wa * (1. - a))) ** 0.5

@jit(nopython=True)
def Inverse_HubbleFactorNomalized(Z,OmegaM,w0,wa):
    a = 1./(Z+1.)
    OhmDE = 1. - OmegaM
    return (OmegaM * (a ** -3.) + OhmDE * (a ** (-3. * (1. + w0 + wa))) * np.exp(-3. * wa * (1. - a)))**-0.5

def ComovingDistanceOverh_explicit(Z,OmegaM,w0,wa):
    if np.isscalar(Z):
        return integrate.quad(Inverse_HubbleFactorNomalized,0.,Z,args=(OmegaM,w0,wa))[0]*2997.92458
    out_dist = np.empty(np.prod(Z.shape))
    with np.nditer(Z,flags=['c_index',]) as it:
        for x in it:
            out_dist[it.index] = integrate.quad(Inverse_HubbleFactorNomalized,0.,x,args=(OmegaM,w0,wa))[0]*2997.92458
    return out_dist




class ComovingDistanceOverh:
    def __init__(self,OmegaM,w0,wa,z_bins_init=None,check_all=True):
        if z_bins_init == None:
            z_bins = np.linspace(-0.2,5.,521)
        elif check_all:
            z_bins = np.sort(z_bins_init)
        else:
            z_bins = z_bins_init
        self.check_all = check_all
        self.z_min = z_bins[0]
        self.z_max = z_bins[-1]
        self.inverse_cHoverH0 = es.spline.cubic_spline(z_bins,Inverse_HubbleFactorNomalized(z_bins,OmegaM,w0,wa)*2997.92458)


    def check_range(self,z):
        if np.all((z < self.z_min) | (z > self.z_max)):
            warnings.warn('some input redshift values are out the inizialization interval: '+str(self.z_min)+' to '+str(self.z_max)+
                          '.\nFor z values out of initialization range the results may be inaccurate.') 

    def get_dist(self,z):
        if self.check_all:
            self.check_range(z)
        return self.inverse_cHoverH0.get_integral(0.,z)
    



class RedshiftFromComovingDistanceOverh:
    def __init__(self,OmegaM,w0,wa,z_bins_init=None,check_all=True):
        if z_bins_init == None:
            z_bins = np.linspace(-0.2,5.,521)
        elif check_all:
            z_bins = np.sort(z_bins_init)
        else:
            z_bins = z_bins_init
        self.check_all = check_all
        self.z_min = z_bins[0]
        self.z_max = z_bins[-1]
        comov_arr = es.spline.cubic_spline(z_bins,Inverse_HubbleFactorNomalized(z_bins,OmegaM,w0,wa)*2997.92458).get_integral(0.,z_bins)
        self.inverse_cHoverH0 = es.spline.cubic_spline(comov_arr,z_bins)
        self.dist_min = comov_arr[0]
        self.dist_max = comov_arr[-1]


    def check_range(self,x):
        if np.all((x < self.dist_min) | (x > self.dist_max)):
            warnings.warn('some input values are out the inizialization interval: '+str(self.dist_min)+' to '+str(self.dist_max)+
                          '.\nFor comoving distance values out of initialization range the results may be inaccurate.') 

    def get_redshift(self,x):
        if self.check_all:
            self.check_range(x)
        return self.inverse_cHoverH0.get_values(x)
    


    
