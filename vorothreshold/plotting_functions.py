import numpy as np
from scipy.interpolate import CubicSpline
from scipy.optimize import minimize

def void_contour_2D(xyz_trs_in_vd,nbins=-1,npts_plot=-1,ij=(0,1)):
    # xyz_trs_in_vd: positions of galaxies in void
    # nbins: number of bins for interpolation. If nbins < 1 it will be automatically computed.
    # npts_plot: length of the output arrays. If npts_plot < 1 it will be automatically set to 101.
    # ij: projection axis, default value is (0,1), corresponding to the (x,y) plane. 0 -> x, 1-> y, 2 -> z

    if nbins < 1:
        nbins = max(5,int(1.3 * len(xyz_trs_in_vd) ** 0.5))
    if npts_plot < 1:
        npts_plot = nbins * 5
        
    xy_center = np.mean(xyz_trs_in_vd[:,ij],axis=0)
    delta_xy = xyz_trs_in_vd[:,ij] - xy_center
    dist_gal_proj = np.sqrt(np.sum(np.square(delta_xy),axis=1))
    
    phi = np.sign(delta_xy[:,1]) * np.arccos(delta_xy[:,0] / dist_gal_proj)

    phi_nodes = np.linspace(-np.pi,np.pi,nbins)

    def fun(dist_interp_in):
        dist_interp = np.zeros(dist_interp_in.shape[0]+1)
        dist_interp[:-1] = dist_interp_in
        dist_interp[-1] = dist_interp_in[0]
        interp = CubicSpline(phi_nodes,dist_interp,bc_type='periodic')
        out = interp(phi)
        
        return np.sum(out)

    def constr_func(dist_interp_in):
        dist_interp = np.zeros(dist_interp_in.shape[0]+1)
        dist_interp[:-1] = dist_interp_in
        dist_interp[-1] = dist_interp_in[0]
        interp = CubicSpline(phi_nodes,dist_interp,bc_type='periodic')
        return interp(phi) - dist_gal_proj

    cons = ({'type': 'ineq', 'fun': constr_func})

    x0 = np.random.uniform(np.min(dist_gal_proj),np.max(dist_gal_proj),nbins-1)
    
    res = minimize(fun, x0, method='SLSQP', constraints=cons)


    dist_interp = np.zeros(res.x.shape[0]+1)
    dist_interp[:-1] = res.x
    dist_interp[-1] = res.x[0]
    interp = CubicSpline(phi_nodes,dist_interp,bc_type='periodic')
    phi_plot = np.linspace(-np.pi,np.pi,101)
    

    return xy_center[0]+np.cos(phi_plot)*interp(phi_plot), xy_center[1]+np.sin(phi_plot)*interp(phi_plot)