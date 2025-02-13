
import numpy as np
from numba import jit, prange
from scipy import integrate
from scipy.special import eval_legendre
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



@jit(nopython=True)
def Solve_tridiagonal_system(a,b,c,d,w,g,p):
    #a = Lower Diag, b = Main Diag, c = Upper Diag, d = solution vector
    n = len(d)
    #w= np.zeros(n-1)
    #g= np.zeros(n,)
    #p = np.zeros(n)

    w[0] = c[0]/b[0]
    g[0] = d[0]/b[0]

    for i in range(1,n-1):
        w[i] = c[i]/(b[i] - a[i-1]*w[i-1])
    for i in range(1,n):
        g[i] = (d[i] - a[i-1]*g[i-1])/(b[i] - a[i-1]*w[i-1])
    p[n-1] = g[n-1]
    for i in range(n-1,0,-1):
        p[i-1] = g[i-1] - w[i-1]*p[i]


@jit(nopython=True)
def Tridiagonal_elements_for_k_not_a_knot(x,y,a, b, c, d):
    #a = Lower Diag, b = Main Diag, c = Upper Diag, d = solution vector
    #n = len(x)
    #a = np.zeros(n-1)
    #b = np.zeros(n)
    #c = np.zeros(n-1)
    #d = np.zeros(n)
    f_first = - 1. / (x[2] - x[1]) ** 2
    b[0] = 1. / (x[1] - x[0]) ** 2
    c[0] = 1. / (x[1] - x[0]) ** 2 - 1. / (x[2] - x[1]) ** 2
    d[0] = 2. * ((y[1] - y[0]) / (x[1] - x[0]) ** 3 - (y[2] - y[1]) / (x[2] - x[1]) ** 3)
    a[-1] = 1. / (x[-1] - x[-2]) ** 2 - 1. / (x[-2] - x[-3]) ** 2
    b[-1] = 1. / (x[-1] - x[-2]) ** 2
    f_last = - 1. / (x[-2] - x[-3]) ** 2
    d[-1] = 2. * ((y[-1] - y[-2]) / (x[-1] - x[-2]) ** 3 - (y[-2] - y[-3]) / (x[-2] - x[-3]) ** 3)
    for i in range(1,len(x)-1):
        a[i-1] = 1. / (x[i] - x[i-1])
        b[i] = 2. / (x[i] - x[i-1]) + 2. / (x[i+1] - x[i])
        c[i] = 1. / (x[i+1] - x[i])
        d[i] = 3. * ((y[i] - y[i-1]) / (x[i] - x[i-1]) ** 2 + (y[i+1] - y[i]) / (x[i+1] - x[i]) ** 2)
    b[0] += -f_first * a[0] / c[1]
    c[0] += -f_first * b[1] / c[1]
    d[0] += -f_first * d[1] / c[1]
    a[-1] += -f_last * b[-2] / a[-2]
    b[-1] += -f_last * c[-1] / a[-2]
    d[-1] += -f_last * d[-2] / a[-2]

@jit(nopython=True)
def Tridiagonal_elements_for_k_not_a_knot_left(x,y,a, b, c, d,second_der_xn):
    f_first = - 1. / (x[2] - x[1]) ** 2
    b[0] = 1. / (x[1] - x[0]) ** 2
    c[0] = 1. / (x[1] - x[0]) ** 2 - 1. / (x[2] - x[1]) ** 2
    d[0] = 2. * ((y[1] - y[0]) / (x[1] - x[0]) ** 3 - (y[2] - y[1]) / (x[2] - x[1]) ** 3)
    a[-1] = 1. / (x[-1] - x[-2])
    b[-1] = 2. / (x[-1] - x[-2])
    d[-1] = 3. * (y[-1] - y[-2]) / (x[-1] - x[-2]) ** 2 + second_der_xn / 2.
    for i in range(1,len(x)-1):
        a[i-1] = 1. / (x[i] - x[i-1])
        b[i] = 2. / (x[i] - x[i-1]) + 2. / (x[i+1] - x[i])
        c[i] = 1. / (x[i+1] - x[i])
        d[i] = 3. * ((y[i] - y[i-1]) / (x[i] - x[i-1]) ** 2 + (y[i+1] - y[i]) / (x[i+1] - x[i]) ** 2)
    b[0] += -f_first * a[0] / c[1]
    c[0] += -f_first * b[1] / c[1]
    d[0] += -f_first * d[1] / c[1]

@jit(nopython=True)
def cubic_spline_coeffs(x,y):
    #a = Lower Diag, b = Main Diag, c = Upper Diag, d = solution vector
    n = len(x)
    Coeff = np.zeros((len(x)-1,4))
    a = np.zeros(n-1)
    b = np.zeros(n)
    c = np.zeros(n-1)
    d = np.zeros(n)
    w = np.zeros(n-1)
    g = np.zeros(n)
    k = np.zeros(n)

    Tridiagonal_elements_for_k_not_a_knot(x, y, a, b, c, d)
    Solve_tridiagonal_system(a, b, c, d, w, g, k)

    A = k[:-1] * (x[1:] - x[:-1]) - (y[1:] - y[:-1])
    B = -k[1:] * (x[1:] - x[:-1]) + (y[1:] - y[:-1])
    Coeff[:,0] = y[:-1]
    Coeff[:,1] = A + y[1:] - y[:-1]
    Coeff[:,2] = B - 2.*A
    Coeff[:,3] = A - B
    return Coeff


@jit(nopython=True)
def get_integral_scalar_array(x1, x_eval, x, coeffs):
    len_out = x_eval.shape[0]
    y_out = np.empty(len_out)

    i_out = 0
    len_x_mn2 = x.shape[0] - 2

    integr_offset = 0.
    while ((x1 >= x[i_out+1]) & (i_out < len_x_mn2)):
        delta_x = x[i_out+1] - x[i_out]
        integr_offset += (coeffs[i_out][0] + coeffs[i_out][1] / 2. +
                          coeffs[i_out][2] / 3. + coeffs[i_out][3] / 4.) * delta_x
        i_out += 1
    
    delta_x = x[i_out+1] - x[i_out]
    t = (x1 - x[i_out]) / delta_x
    
    integr_offset += (coeffs[i_out,0] * t + 
                      coeffs[i_out,1] * t * t / 2. + 
                      coeffs[i_out,2] * t * t * t / 3. + 
                      coeffs[i_out,3] * t * t * t * t / 4.) * delta_x

    i_out = 0
    integr_incremental = 0
    while ((x1 >= x[i_out+1]) & (i_out < len_x_mn2)):
        delta_x = x[i_out+1] - x[i_out]
        integr_incremental += (coeffs[i_out][0] + coeffs[i_out][1] / 2. +
                               coeffs[i_out][2] / 3. + coeffs[i_out][3] / 4.) * delta_x
        i_out += 1
    
    sort_ind = np.argsort(x_eval)

    for i in sort_ind:
        while ((x_eval[i] >= x[i_out+1]) & (i_out < len_x_mn2)):
            integr_incremental += (coeffs[i_out][0] + coeffs[i_out][1] / 2. +
                                   coeffs[i_out][2] / 3. + coeffs[i_out][3] / 4.) * delta_x
            i_out += 1
            delta_x = x[i_out+1] - x[i_out]
        
        
        t = (x_eval[i] - x[i_out]) / delta_x
        y_out[i] =  integr_incremental + (coeffs[i_out][0] * t + 
                                          coeffs[i_out][1] * t * t / 2. + 
                                          coeffs[i_out][2] * t * t * t / 3. + 
                                          coeffs[i_out][3] * t * t * t * t / 4.) * delta_x - integr_offset
    return y_out


class ComovingDistanceOverh:
    def __init__(self,OmegaM,w0,wa,z_bins_init=None,check_all=True):
        if z_bins_init == None:
            self.z_bins = np.linspace(-0.2,5.,521)
        elif check_all:
            self.z_bins = np.sort(z_bins_init)
        else:
            self.z_bins = z_bins_init
        self.check_all = check_all
        self.z_min = self.z_bins[0]
        self.z_max = self.z_bins[-1]
        self.coeffs = cubic_spline_coeffs(self.z_bins,Inverse_HubbleFactorNomalized(self.z_bins,OmegaM,w0,wa)*2997.92458)


    def check_range(self,z):
        if np.all((z < self.z_min) | (z > self.z_max)):
            warnings.warn('some input redshift values are out the inizialization interval: '+str(self.z_min)+' to '+str(self.z_max)+
                          '.\nFor z values out of initialization range the results may be inaccurate.') 

    def get_dist(self,z):
        if self.check_all:
            self.check_range(z)
        return get_integral_scalar_array(0., z, self.z_bins, self.coeffs) #self.inverse_cHoverH0.get_integral(0.,z)
    



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
        coeffs = cubic_spline_coeffs(z_bins,Inverse_HubbleFactorNomalized(z_bins,OmegaM,w0,wa)*2997.92458)
        self.comov_arr = get_integral_scalar_array(0., z_bins, z_bins, coeffs)
        #comov_arr = es.spline.cubic_spline(z_bins,Inverse_HubbleFactorNomalized(z_bins,OmegaM,w0,wa)*2997.92458).get_integral(0.,z_bins)

        self.coeffs = cubic_spline_coeffs(self.comov_arr,z_bins)
        #self.inverse_cHoverH0 = es.spline.cubic_spline(comov_arr,z_bins)
        self.dist_min = self.comov_arr[0]
        self.dist_max = self.comov_arr[-1]


    def check_range(self,x):
        if np.all((x < self.dist_min) | (x > self.dist_max)):
            warnings.warn('some input values are out the inizialization interval: '+str(self.dist_min)+' to '+str(self.dist_max)+
                          '.\nFor comoving distance values out of initialization range the results may be inaccurate.') 

    def get_redshift(self,x):
        if self.check_all:
            self.check_range(x)
        return get_integral_scalar_array(0., x, self.comov_arr, self.coeffs) 
    


    
