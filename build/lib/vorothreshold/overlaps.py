import numpy as np
import os
import time
from numba.core import types
from numba.typed import Dict
from numba import jit, prange, set_num_threads, get_num_threads, get_thread_id


def StrHminSec(DeltaT):
    hh = int(DeltaT / 3600)
    minutes = int(DeltaT / 60) - hh * 60
    sec = DeltaT % 60
    return str(hh) + ' h ' + str(minutes) + ' min ' + str(sec) + ' sec.'


@jit(nopython=True)
def is_in_arr(ar1,ar2):
    mask = np.zeros(len(ar1), dtype=np.bool_)
    for a in ar2:
        mask |= (ar1 == a)
    return mask


@jit(nopython=True)
def overlapping_fraction_core(
    id_loop, id_ord, ids_ovlp, Vol_ovlp, Vol_ovlp_frac, IDS_voids, XYZ_voids, VolVoids, Ncells, 
    max_dist_vds, R_max, Ids_voro_dict, voro_vol, ind_vox, ngrid, voxel_side, max_num_tracers):
    
    id_void = IDS_voids[id_loop]
    
    XYZ_ref = XYZ_voids[id_void,:]
    #Vol = VolVoids[id_loop]

    Ncells_ref = int(Ncells[id_void]) + int(round(Ncells[id_void]%1))

    dist_max = max_dist_vds[id_void] + R_max
    dist2_max = dist_max * dist_max
    # initialize arrays:
    max_num_vox_for_sphere = int(4 * 3.1416 / 3 * (int((dist_max-1) / voxel_side) + 1 + 0.5 * np.sqrt(3))**3) + 1
    max_num_vox_for_sphere = max(max_num_vox_for_sphere, 27)


    half_n_vox_side = int((dist_max-1) / voxel_side) + 1
    
    ijk_in_sphere = np.empty((max_num_vox_for_sphere,3),dtype=np.int_)
    vox_dist_from_xyz = np.empty(3,dtype=np.float_)
    xyz_vox_unit = XYZ_ref / voxel_side
    ijk_vox_void_center = xyz_vox_unit.astype(np.int_)
    
    i_in = max(int(xyz_vox_unit[0] - half_n_vox_side),0)
    i_out = min(int(xyz_vox_unit[0] + half_n_vox_side + 1),ngrid)
    j_in = max(int(xyz_vox_unit[1] - half_n_vox_side),0)
    j_out = min(int(xyz_vox_unit[1] + half_n_vox_side + 1),ngrid)
    k_in = max(int(xyz_vox_unit[2] - half_n_vox_side),0)
    k_out = min(int(xyz_vox_unit[2] + half_n_vox_side + 1),ngrid)

    ids_to_expore = np.zeros(max_num_tracers,dtype=np.int_)
    #ids_ovlp = np.zeros(max_num_tracers,dtype=np.int_)
    #Vol_ovlp = np.zeros(max_num_tracers)
    #Vol_ovlp_frac = np.zeros(max_num_tracers)

    progr = 0
    r2_vox_unit = dist2_max / (voxel_side * voxel_side)
    for i in range(i_in,i_out):
        for j in range(j_in,j_out):
            for k in range(k_in,k_out):
                ijk_in_sphere[progr,0] = i
                ijk_in_sphere[progr,1] = j
                ijk_in_sphere[progr,2] = k

                #// Compute the distance between the center and the closest vertex of voxel i,j,k:
                #// if i < xyz_vox_unit[0] (voxels at left) compare the distance wrt i+1, 
                #// i.e. right vertex of voxel instead of the left one
                vox_dist_from_xyz[0] = i + int(i < xyz_vox_unit[0]) - xyz_vox_unit[0]
                vox_dist_from_xyz[1] = j + int(j < xyz_vox_unit[1]) - xyz_vox_unit[1]
                vox_dist_from_xyz[2] = k + int(k < xyz_vox_unit[2]) - xyz_vox_unit[2]
                #// Compute the square of the distance. The boolean condition is to an exact computation 
                #// of the minimum distance between the center and the voxel:
                #// if , e.g. i == ijk_vox_void_center[0] the minimum distance occurs at the side, not at vertex,
                #// i.e. the minimu radius is has the same x coord of the center, therefore the x projection
                #// of the radius is 0, i.e. (i != ijk_vox_void_center[0])=0.
                #// This algorithm automatically select the voxel to which the center belong, 
                #// independently on the radius and voxel size
                vox_dist_from_xyz[0] *= vox_dist_from_xyz[0] * int(i != ijk_vox_void_center[0])
                vox_dist_from_xyz[1] *= vox_dist_from_xyz[1] * int(j != ijk_vox_void_center[1])
                vox_dist_from_xyz[2] *= vox_dist_from_xyz[2] * int(k != ijk_vox_void_center[2])

                progr += (vox_dist_from_xyz[0] + vox_dist_from_xyz[1] + vox_dist_from_xyz[2]) < r2_vox_unit
                
    progr_sphere = 0
    #// select voxels intersecting with the sphere centered in the void center
    for id_vox in range(progr):
        
        i_tmp = ijk_in_sphere[id_vox,0]
        j_tmp = ijk_in_sphere[id_vox,1]
        k_tmp = ijk_in_sphere[id_vox,2]
        id_vox_tmp = i_tmp * ngrid * ngrid + j_tmp * ngrid + k_tmp
        #if (i_tmp == 0) & (j_tmp == 2) & (k_tmp == 1):
        #    print(id_vox_tmp)
                            
        for id_ptr in range(ind_vox[id_vox_tmp],ind_vox[id_vox_tmp+1]):
            id_trs = IDS_voids[id_ptr]
            ids_to_expore[progr_sphere] = id_trs
            dist2 = np.sum(np.square(XYZ_voids[id_trs,:] - XYZ_ref))
            progr_sphere += int((dist2 <= dist2_max) & (VolVoids[id_trs] <= VolVoids[id_void]) & (id_trs != id_void))

    progr_ovlp = 0
    for id_trs in ids_to_expore[:progr_sphere]:
        ids_ovlp[id_ord,progr_ovlp] = id_trs
        Ncells_loop = int(Ncells[id_trs]) + int(round(Ncells[id_trs]%1))
        mask_in_arr = is_in_arr(Ids_voro_dict[id_trs][:Ncells_loop],Ids_voro_dict[id_void][:Ncells_ref])
        Vol_ovlp[id_ord,progr_ovlp] = np.sum(voro_vol[(Ids_voro_dict[id_trs][:Ncells_loop])[mask_in_arr]])
        Vol_ovlp_frac[id_ord,progr_ovlp] = Vol_ovlp[id_ord,progr_ovlp] / np.sum(voro_vol[Ids_voro_dict[id_trs][:Ncells_loop]])
        progr_ovlp += int(Vol_ovlp[id_ord,progr_ovlp] > 0.)

    #return ids_ovlp, Vol_ovlp, Vol_ovlp_frac, progr_ovlp
    return progr_ovlp
        


#@jit(nopython=True) #,parallel=True)
def overlapping_fraction_loop(
    IDS_voids, id_soring, XYZ_voids, VolVoids, Ncells, max_dist_vds, Ids_voro_dict, voro_vol, ind_vox, ngrid, Lbox, voxel_side):
    Rmax = np.max(max_dist_vds)
    Nvoids = IDS_voids.shape[0]

    N_poisson_in_sphere=1.2
    #max_num_tracers = int(4 * 3.1416 / 3 * (4 * R2max) ** 1.5 * XYZ_voids.shape[0] / (Lbox * Lbox * Lbox) * N_poisson_in_sphere) 
    max_num_tracers = min(Nvoids,int(4 * 3.1416 / 3 * (2*Rmax) ** 3 *  np.max(ind_vox[1:] - ind_vox[:-1]) * (ngrid / Lbox) ** 3 * N_poisson_in_sphere))

    ids_ovlp = np.zeros((Nvoids,max_num_tracers),dtype=np.int_)
    Vol_ovlp = np.zeros((Nvoids,max_num_tracers))
    Vol_ovlp_frac = np.zeros((Nvoids,max_num_tracers))
    num_ovlps = np.zeros(Nvoids,dtype=np.int_)

    #ids_to_explore = IDS_voids[Ncells[IDS_voids] > 1.]
    #for id_loop in prange(ids_to_explore.shape[0]):
        #num_ovlps[ids_to_explore[id_loop]] = overlapping_fraction_core(
        #    ids_to_explore[id_loop], ids_ovlp[ids_to_explore[id_loop],:], Vol_ovlp[ids_to_explore[id_loop],:], Vol_ovlp_frac[ids_to_explore[id_loop],:], 
        #    IDS_voids, XYZ_voids, VolVoids, Ncells, R2vds, R2max, Ids_voro_dict, voro_vol, ind_vox, ngrid, voxel_side, max_num_tracers)
    for id_loop in range(Nvoids):
        
        #print(id_loop,'/',IDS_voids.shape[0])
        id_ord = id_soring[id_loop]
        num_ovlps[id_ord] = overlapping_fraction_core(
            id_loop, id_ord, ids_ovlp, Vol_ovlp, Vol_ovlp_frac, 
            IDS_voids, XYZ_voids, VolVoids, Ncells, max_dist_vds, Rmax, Ids_voro_dict, voro_vol, ind_vox, ngrid, voxel_side, max_num_tracers)
    return ids_ovlp, Vol_ovlp, Vol_ovlp_frac, num_ovlps



@jit(nopython=True,parallel=True)
def compute_max_dist2(Ncells,XYZ_voids,XYZ_voro,id_selected,Ids_voro_dict):
    Nvoids = XYZ_voids.shape[0]
    dist2_max = np.zeros(Nvoids)
    #id_out = np.arange(Nvoids)[Ncells>=1]
    for i in prange(id_selected.shape[0]):
        #print(i,id_selected.shape[0])
        iv = id_selected[i]
        Ncells_loop = int(Ncells[iv]) + int((Ncells[iv]%1) > 0)
        #print(iv,Ncells[iv],Ncells_loop,)
        dist2_max[iv] = np.max(np.sum(np.square(XYZ_voro[Ids_voro_dict[iv][:Ncells_loop],:] - XYZ_voids[iv,:]),axis=1))
    return dist2_max



@jit(nopython=True,parallel=True)
def compute_max_dist_deg(Ncells,XYZ_voids,XYZ_voro,id_selected,Ids_voro_dict):
    dist_vds = np.sqrt(np.sum(np.square(XYZ_voids),axis=1))
    Nvoids = XYZ_voids.shape[0]
    dist_ang_max = np.zeros(Nvoids)
    #id_out = np.arange(Nvoids)[Ncells>=1]
    for i in prange(id_selected.shape[0]):
        #print(i,id_selected.shape[0])
        iv = id_selected[i]
        Ncells_loop = int(Ncells[iv]) + int((Ncells[iv]%1) > 0)
        dist_voro = np.sqrt(np.sum(np.square(XYZ_voro[Ids_voro_dict[iv][:Ncells_loop]]),axis=1))
        #print(iv,Ncells[iv],Ncells_loop,)
        dist_ang_max[iv] = np.max(np.arccos(np.sum(XYZ_voro[Ids_voro_dict[iv][:Ncells_loop],:] * XYZ_voids[iv,:],axis=1) / (dist_voro * dist_vds[iv])))
    return dist_ang_max
        


@jit(nopython=True)
def order_ids_tracers_in_voxels(
    xyz_trs, ngrid, Lbox):

    num_tracers = xyz_trs.shape[0]
    n_trs_vox = np.zeros(ngrid*ngrid*ngrid,dtype=np.int64) # array containing the number of tracer for each voxel
    
    #xyz_trs_out = np.empty(xyz_trs.shape,dtype=xyz_trs.dtype)
    id_trs_out = np.arange(num_tracers)

    voxel_side_inv = ngrid / Lbox
    
    ix = 0 
    iy = 0
    iz = 0
    for i_tr in range(num_tracers):
        ix = int(xyz_trs[i_tr,0] * voxel_side_inv) #- int(xyz_trs[i_tr][0]>=Lbox)
        iy = int(xyz_trs[i_tr,1] * voxel_side_inv) #- int(xyz_trs[i_tr][1]>=Lbox)
        iz = int(xyz_trs[i_tr,2] * voxel_side_inv) #- int(xyz_trs[i_tr][2]>=Lbox)
        n_trs_vox[ix * ngrid * ngrid + iy * ngrid + iz] += 1
    
    ind_vox = np.empty(ngrid*ngrid*ngrid+1,dtype=np.int64)
    ind_vox[0] = 0
    for i in range(ngrid*ngrid*ngrid):
        ind_vox[i+1] = ind_vox[i] + n_trs_vox[i]
    
    for i_tr in range(num_tracers-1,-1,-1):
        ix = int(xyz_trs[i_tr,0] * voxel_side_inv) #- int(xyz_trs[i_tr][0]>=Lbox)
        iy = int(xyz_trs[i_tr,1] * voxel_side_inv) #- int(xyz_trs[i_tr][1]>=Lbox)
        iz = int(xyz_trs[i_tr,2] * voxel_side_inv) #- int(xyz_trs[i_tr][2]>=Lbox)
        id_grid = ix * ngrid * ngrid + iy * ngrid + iz
        #permutation_ind[i_tr] = ind_vox[id_grid]  + n_trs_vox[id_grid] - 1
        id_new = ind_vox[id_grid]  + n_trs_vox[id_grid] - 1
        #xyz_trs_out[id_new,:] = xyz_trs[i_tr,:]
        id_trs_out[id_new] = i_tr

        n_trs_vox[id_grid] -= 1
        
    return id_trs_out, ind_vox

@jit(nopython=True)
def order_ids_tracers_selected_in_voxels(
    xyz_trs, id_selection, ngrid, Lbox):

    num_tracers = id_selection.shape[0]
    n_trs_vox = np.zeros(ngrid*ngrid*ngrid,dtype=np.int64) # array containing the number of tracer for each voxel
    
    #xyz_trs_out = np.empty(xyz_trs.shape,dtype=xyz_trs.dtype)
    id_trs_out = np.arange(num_tracers)
    id_soring = np.arange(num_tracers)
    #id_trs_orig = np.arange(num_tracers)

    voxel_side_inv = ngrid / Lbox
    
    ix = 0 
    iy = 0
    iz = 0
    for i_sel in range(num_tracers):
        i_tr = id_selection[i_sel]
        ix = int(xyz_trs[i_tr,0] * voxel_side_inv) #- int(xyz_trs[i_tr][0]>=Lbox)
        iy = int(xyz_trs[i_tr,1] * voxel_side_inv) #- int(xyz_trs[i_tr][1]>=Lbox)
        iz = int(xyz_trs[i_tr,2] * voxel_side_inv) #- int(xyz_trs[i_tr][2]>=Lbox)
        n_trs_vox[ix * ngrid * ngrid + iy * ngrid + iz] += 1
    
    ind_vox = np.empty(ngrid*ngrid*ngrid+1,dtype=np.int64)
    ind_vox[0] = 0
    for i in range(ngrid*ngrid*ngrid):
        ind_vox[i+1] = ind_vox[i] + n_trs_vox[i]
    
    for i_sel in range(num_tracers-1,-1,-1):
        i_tr = id_selection[i_sel]
        ix = int(xyz_trs[i_tr,0] * voxel_side_inv) #- int(xyz_trs[i_tr][0]>=Lbox)
        iy = int(xyz_trs[i_tr,1] * voxel_side_inv) #- int(xyz_trs[i_tr][1]>=Lbox)
        iz = int(xyz_trs[i_tr,2] * voxel_side_inv) #- int(xyz_trs[i_tr][2]>=Lbox)
        id_grid = ix * ngrid * ngrid + iy * ngrid + iz
        #permutation_ind[i_tr] = ind_vox[id_grid]  + n_trs_vox[id_grid] - 1
        id_new = ind_vox[id_grid] + n_trs_vox[id_grid] - 1
        #xyz_trs_out[id_new,:] = xyz_trs[i_tr,:]
        id_trs_out[id_new] = i_tr
        id_soring[id_new] = i_sel
        #id_trs_orig[i_sel] = id_new

        n_trs_vox[id_grid] -= 1
        
    return id_trs_out, id_soring, ind_vox


def overlapping_fraction(
    xyz_vds, vol_vds, Ncells, xyz_voro, vol_voro, IDs_in_voids, Lbox=-1.,pbc=False,lightcone=False,
    ngrid=-1,nthreads=-1,verbose=True,Omega_rad=-1.,id_selected=None):
    # xyz_vds: dim (num_voids,3) numpy array containing void centers
    # vol_vds: dim (num_voids,) numpy array containing void volumes
    # Ncells: dim (num_voids,) numpy array containing the (fractional) number of voronoi cell in each void
    # xyz_voro: dim (num_tracers,3) numpy array containing tracers coordinates
    # vol_voro: dim (num_tracers,) numpy array Voronoi cell volumes
    # IDs_in_voids: dict containing the IDs of all Voronoi cells building up each void
    # Lbox: side of simulation box
    # pbc: if True consider periodic boundary condition at 0 and Lbox
    # nthreads: number of threads to use for parallel computation. If nthreads=-1, this function automatically uses all the available CPUs
    # verbose: if True the function will print logs

    verboseprint = print if verbose else lambda *a, **k: None

    verboseprint("\nIDs_in_sphere started.",flush=True)

    if id_selected is None:
        verboseprint("\nid_selected not passed.",flush=True)
        id_selected = np.arange(vol_vds.shape[0])[vol_vds > 0.]


    if nthreads <= 0:
        try:
            nthreads = int(os.environ["OMP_NUM_THREADS"])
        except:
            nthreads = get_num_threads()

    set_num_threads(min(nthreads,id_selected.shape[0]))

    verboseprint('\n    nthreads set to',nthreads,flush=True)


    R_max = compute_max_dist2(Ncells,xyz_vds,xyz_voro,id_selected,IDs_in_voids)**0.5


    verboseprint('\n    R_max computed. Max val =',(np.max(R_max)),flush=True)


    if (Lbox < 0):
        lightcone = True
        if pbc:
            print("    WARNING: Lbox not passed and pbc = True. We suggest either to pass Lbox or set pbc = False.",flush=True)
            verboseprint("    Computing Lbox using xyz_vds as reference:",flush=True)
        else:
            verboseprint("\n    Lbox not passed, using xyz_vds as reference:",flush=True)
    if lightcone:
        if Lbox > 0:
            verboseprint("\n    Lbox passed but lightcone set to True. Lbox ignored.",flush=True)

        offset = np.min(xyz_vds[id_selected,:],axis=0)
        max_values = np.max(xyz_vds[id_selected,:],axis=0)
        Lbox = np.max(max_values - offset)

        
        verboseprint("    min(xyz_vds) =",*offset,flush=True)
        verboseprint("    max(xyz_vds) =",*max_values,flush=True)
        verboseprint("    Lbox =",Lbox,flush=True)

    if ngrid < 0:
        #ngrid = max(int(round(5 * Lbox / np.sqrt(np.max(R2_max)))),3)
        ngrid = max(int(round(Lbox / np.max(R_max) )),3)

        verboseprint("\n    ngrid not passed. Set to optimal value:",ngrid,flush=True)



    voxel_side = Lbox / ngrid
    
    verboseprint('\n    order_ids_tracers_selected_in_voxels started',flush=True)
    
    t0 = time.time()

    #IDs_vds_ordered, voxel_ptr = order_ids_tracers_in_voxels(xyz_vds, ngrid, Lbox)
    
    IDs_vds_ordered, id_soring, voxel_ptr = order_ids_tracers_selected_in_voxels(xyz_vds - offset,id_selected, ngrid, Lbox)
    
    dt = time.time() - t0
    verboseprint("    done,",StrHminSec(dt),flush=True)

    verboseprint("\n    computation started (periodic-boundaries condition "+['off','on'][int(pbc)]+")",flush=True)
    t0 = time.time()
    ids_ovlp, Vol_ovlp, Vol_ovlp_frac, num_ovlps = overlapping_fraction_loop(
        IDs_vds_ordered, id_soring, xyz_vds - offset, vol_vds, Ncells, R_max, IDs_in_voids, vol_voro, voxel_ptr, ngrid, Lbox, voxel_side)
    dt = time.time() - t0
    verboseprint("    done,",StrHminSec(dt),'\n',flush=True)



    return ids_ovlp, Vol_ovlp, Vol_ovlp_frac, num_ovlps



@jit(nopython=True)
def select_overlaps_old(frac_threshold,id_selected, ids_ovlp, Vol_ovlp_frac, num_ovlps):
    Ntot = id_selected.shape[0]
    id_out = np.arange(Ntot)
    ind = 0
    while ind < Ntot:
        id_sel = id_out[ind]
        iv_ref = id_selected[id_sel]
        #iv_ref = id_out[ind]
        for j in range(num_ovlps[id_sel]):
            iv_ovlp = ids_ovlp[id_sel,j]
            if (Vol_ovlp_frac[id_sel,j] > frac_threshold):
                #print(ind,id_sel,iv_ref,ids_ovlp[id_sel,j],Vol_ovlp_frac[id_sel,j])
                ii = 0
                while (ii < Ntot-1) & (id_selected[id_out[ii]] != iv_ovlp):
                    #while (id_out[ii] != iv_ovlp) & (ii < Ntot):
                    ii += 1
                ii += int((ii == Ntot-1) & (id_selected[id_out[ii]] != iv_ovlp))

                id_out[ii:-1] = id_out[ii+1:]
                Ntot -= 1
        ind += 1
    return id_out[:Ntot]


@jit(nopython=True)
def select_overlaps(frac_threshold, id_selected, order_id_selected, ids_ovlp, Vol_ovlp_frac, num_ovlps):
    Ntot = order_id_selected.shape[0]
    id_out = np.arange(Ntot)
    ind = 0
    while ind < Ntot:
        id_sel = id_out[ind]
        iv_ref = order_id_selected[id_sel]
        #iv_ref = id_out[ind]
        for j in range(num_ovlps[iv_ref]):
            iv_ovlp = ids_ovlp[iv_ref,j]
            if (Vol_ovlp_frac[iv_ref,j] > frac_threshold):
                ii = 0
                while (ii < Ntot-1) & (id_selected[order_id_selected[id_out[ii]]] != iv_ovlp):
                    #while (id_out[ii] != iv_ovlp) & (ii < Ntot):
                    ii += 1
                ii += int((ii == Ntot-1) & (id_selected[order_id_selected[id_out[ii]]] != iv_ovlp))

                id_out[ii:-1] = id_out[ii+1:]
                Ntot -= 1
        ind += 1
    #return id_selected[order_id_selected[id_out[:Ntot]]]
    return order_id_selected[id_out[:Ntot]]




@jit(nopython=True)
def select_overlaps_explicit(frac_threshold,id_selected, ids_ovlp, Vol_ovlp_frac, num_ovlps):
    Ntot = id_selected.shape[0]
    id_out = np.copy(id_selected)
    ind = 0
    while ind < Ntot:
        iv_ref = id_out[ind]
        for j in range(num_ovlps[iv_ref]):
            iv_ovlp = ids_ovlp[iv_ref,j]
            if (Vol_ovlp_frac[iv_ref,j] > frac_threshold):
                ii = 0
                while (id_out[ii] != iv_ovlp) & (ii < Ntot):
                    ii += 1
                id_out[ii:-1] = id_out[ii+1:]
                Ntot -= 1
        ind += 1
    return id_out[:Ntot]



@jit(nopython=True)
def select_overlaps_no_if(frac_threshold,id_selected, ids_ovlp, Vol_ovlp_frac, num_ovlps):
    Ntot = id_selected.shape[0]
    id_out = np.copy(id_selected)
    ind = 0
    while ind < Ntot:
        iv_ref = id_out[ind]
        for iv_ovlp in ids_ovlp[iv_ref,:num_ovlps[iv_ref]][Vol_ovlp_frac[iv_ref,:num_ovlps[iv_ref]] > frac_threshold]:
            ii = 0
            while (id_out[ii] != iv_ovlp) & (ii < Ntot):
                ii += 1
            id_out[ii:-1] = id_out[ii+1:]
            Ntot -= 1
        ind += 1
    return id_out[:Ntot]