import numpy as np
from scipy.interpolate import CubicSpline
from scipy.optimize import minimize

def void_contour_2D(xyz_trs_in_vd,nbins=-1,npts_plot=-1,ij=(0,1)):
    # xyz_center: void center position
    # xyz_trs_in_vd: positions of galaxies in void
    # nbins: number of bins for interpolation. If nbins < 1 it will be automatically computed.
    # npts_plot: length of the ounput arrays. If npts_plot < 1 it will be automatically computed.
    # ij: projection axis, default value are (0,1), corresponding to the (x,y) plane

    if nbins < 1:
        nbins = max(5,int(1.3 * len(xyz_trs_in_vd) ** 0.5))
    if npts_plot < 1:
        npts_plot = nbins * 5
        
    #print(nbins,int(len(xyz_trs_in_vd) ** 0.5),xyz_trs_in_vd.shape)
    xyz_center = np.mean(xyz_trs_in_vd,axis=0)
    delta_xyz = xyz_trs_in_vd - xyz_center
    dist_gal_proj = np.sqrt(np.sum(np.square(delta_xyz[:,ij]),axis=1))
    
    phi = np.sign(delta_xyz[:,ij[1]]) * np.arccos(delta_xyz[:,ij[0]] / dist_gal_proj)

    phi_nodes = np.linspace(-np.pi,np.pi,nbins)

    def fun(dist_interp_in):
        dist_interp = np.zeros(dist_interp_in.shape[0]+1)
        dist_interp[:-1] = dist_interp_in
        dist_interp[-1] = dist_interp_in[0]
        interp = CubicSpline(phi_nodes,dist_interp,bc_type='periodic')
        #mask = interp(phi) >= dist_gal_proj
        out = interp(phi) - dist_gal_proj
        
        return np.sum(out)

    def constr_func(dist_interp_in):
        dist_interp = np.zeros(dist_interp_in.shape[0]+1)
        dist_interp[:-1] = dist_interp_in
        dist_interp[-1] = dist_interp_in[0]
        interp = CubicSpline(phi_nodes,dist_interp,bc_type='periodic')
        #mask = interp(phi) >= dist_gal_proj
        return interp(phi) - dist_gal_proj

    cons = ({'type': 'ineq', 'fun': constr_func})

    x0 = np.random.uniform(np.min(dist_gal_proj),np.max(dist_gal_proj),nbins-1)
    
    res = minimize(fun, x0, method='SLSQP', #bounds=bnds,
                constraints=cons)


    dist_interp = np.zeros(res.x.shape[0]+1)
    dist_interp[:-1] = res.x
    dist_interp[-1] = res.x[0]
    interp = CubicSpline(phi_nodes,dist_interp,bc_type='periodic')
    phi_plot = np.linspace(-np.pi,np.pi,101)
    

    return xyz_center[0]+np.cos(phi_plot)*interp(phi_plot),xyz_center[1]+np.sin(phi_plot)*interp(phi_plot)