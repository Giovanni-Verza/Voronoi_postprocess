import numpy as np
import os
import time
from numba.core import types
from numba.typed import Dict
from numba import jit, prange, set_num_threads, get_num_threads, get_thread_id
from . order_functions import order_ids_tracers_in_voxels, order_ids_tracers_selected_in_voxels, order_coord_tracers_in_voxels_ids_rev_copy


bool_array = types.boolean[::1]


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
    
    ijk_in_sphere = np.empty((max_num_vox_for_sphere,3),dtype=np.int64)
    vox_dist_from_xyz = np.empty(3,dtype=np.float64)
    xyz_vox_unit = XYZ_ref / voxel_side
    ijk_vox_void_center = xyz_vox_unit.astype(np.int64)
    
    i_in = max(int(xyz_vox_unit[0] - half_n_vox_side),0)
    i_out = min(int(xyz_vox_unit[0] + half_n_vox_side + 1),ngrid)
    j_in = max(int(xyz_vox_unit[1] - half_n_vox_side),0)
    j_out = min(int(xyz_vox_unit[1] + half_n_vox_side + 1),ngrid)
    k_in = max(int(xyz_vox_unit[2] - half_n_vox_side),0)
    k_out = min(int(xyz_vox_unit[2] + half_n_vox_side + 1),ngrid)

    ids_to_expore = np.zeros(max_num_tracers,dtype=np.int64)
    #ids_ovlp = np.zeros(max_num_tracers,dtype=np.int64)
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



        #mask_in_arr[:] = False
        #for a in Ids_voro_dict[id_void][:Ncells_ref]:
        #    mask_in_arr |= (Ids_voro_dict[id_trs][:Ncells_loop] == a)
        #Vol_ovlp[id_ord,progr_ovlp] = np.sum(voro_vol[(Ids_voro_dict[id_trs][:Ncells_loop])[mask_in_arr[:Ncells_loop]]])


        Vol_ovlp[id_ord,progr_ovlp] = np.sum(voro_vol[(Ids_voro_dict[id_trs][:Ncells_loop])[mask_in_arr]])
        Vol_ovlp_frac[id_ord,progr_ovlp] = Vol_ovlp[id_ord,progr_ovlp] / np.sum(voro_vol[Ids_voro_dict[id_trs][:Ncells_loop]])
        progr_ovlp += int(Vol_ovlp[id_ord,progr_ovlp] > 0.)

    #return ids_ovlp, Vol_ovlp, Vol_ovlp_frac, progr_ovlp
    return progr_ovlp
        

@jit(nopython=True)
def overlapping_fraction_core_pbc(
    id_loop, id_ord, ids_ovlp, Vol_ovlp, Vol_ovlp_frac, IDS_voids, XYZ_voids, VolVoids, Ncells, 
    max_dist_vds, R_max, Ids_voro_dict, voro_vol, ind_vox, ngrid, Lbox, voxel_side, max_num_tracers):
    
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
    
    ijk_in_sphere = np.empty((max_num_vox_for_sphere,3),dtype=np.int64)
    vox_dist_from_xyz = np.empty(3,dtype=np.float64)
    pbc_ijk_vox = np.empty(3,dtype=np.int8)
    xyz_vox_unit = XYZ_ref / voxel_side
    ijk_vox_void_center = xyz_vox_unit.astype(np.int64)
    
    i_in = int(np.floor(xyz_vox_unit[0] - half_n_vox_side))
    i_in = max(i_in,-ngrid+1)
    i_out = int(np.floor(xyz_vox_unit[0] + half_n_vox_side + 1))
    i_out = min(i_out,i_in+ngrid)
    j_in = int(np.floor(xyz_vox_unit[1] - half_n_vox_side))
    j_in = max(j_in,-ngrid+1)
    j_out = int(np.floor(xyz_vox_unit[1] + half_n_vox_side + 1))
    j_out = min(j_out,j_in+ngrid)
    k_in = int(np.floor(xyz_vox_unit[2] - half_n_vox_side))
    k_in = max(k_in,-ngrid+1)
    k_out = int(np.floor(xyz_vox_unit[2] + half_n_vox_side + 1))
    k_out = min(k_out,k_in+ngrid)

    ids_to_expore = np.zeros(max_num_tracers,dtype=np.int64)
    #ids_ovlp = np.zeros(max_num_tracers,dtype=np.int64)
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
                #// if i < xyz_vox_unit[0] (voxels at left) compare the distance wrt i+1, i.e. right vertex of voxel instead of the left one
                vox_dist_from_xyz[0] = i + int(i < xyz_vox_unit[0]) - xyz_vox_unit[0]
                vox_dist_from_xyz[1] = j + int(j < xyz_vox_unit[1]) - xyz_vox_unit[1]
                vox_dist_from_xyz[2] = k + int(k < xyz_vox_unit[2]) - xyz_vox_unit[2]

                #// Compute the square of the distance. The boolean condition is to an exact computation 
                #// of the minimum distance between the center and the voxel:
                #// if , e.g. i == ijk_vox_void_center[0] the minimum distance occurs at the side, not at vertex,
                #// i.e. the minimum radius is has the same x coord of the center, therefore the x projection
                #// of the radius is 0, i.e. (i != ijk_vox_void_center[0])=0.
                #// This algorithm automatically select the voxel to which the center belong, independently on the radius and voxel size
                vox_dist_from_xyz[0] *= vox_dist_from_xyz[0] * int(i != ijk_vox_void_center[0])
                vox_dist_from_xyz[1] *= vox_dist_from_xyz[1] * int(j != ijk_vox_void_center[1])
                vox_dist_from_xyz[2] *= vox_dist_from_xyz[2] * int(k != ijk_vox_void_center[2])

                progr += (vox_dist_from_xyz[0] + vox_dist_from_xyz[1] + vox_dist_from_xyz[2]) < r2_vox_unit
                
                
    progr_sphere = 0
    #// select voxels intersecting with the sphere centered in the void center
    for id_vox in range(progr):

        pbc_ijk_vox[0] = int(ijk_in_sphere[id_vox,0] < 0) - int(ijk_in_sphere[id_vox,0] >= ngrid)
        pbc_ijk_vox[1] = int(ijk_in_sphere[id_vox,1] < 0) - int(ijk_in_sphere[id_vox,1] >= ngrid)
        pbc_ijk_vox[2] = int(ijk_in_sphere[id_vox,2] < 0) - int(ijk_in_sphere[id_vox,2] >= ngrid)

        i_tmp = ijk_in_sphere[id_vox,0] + pbc_ijk_vox[0] * ngrid
        j_tmp = ijk_in_sphere[id_vox,1] + pbc_ijk_vox[1] * ngrid
        k_tmp = ijk_in_sphere[id_vox,2] + pbc_ijk_vox[2] * ngrid
        
        id_vox_tmp = i_tmp * ngrid * ngrid + j_tmp * ngrid + k_tmp
        #if (i_tmp == 0) & (j_tmp == 2) & (k_tmp == 1):
        #    print(id_vox_tmp)
                            
        for id_ptr in range(ind_vox[id_vox_tmp],ind_vox[id_vox_tmp+1]):
            id_trs = IDS_voids[id_ptr]
            ids_to_expore[progr_sphere] = id_trs
            dist2 = np.sum(np.square(XYZ_voids[id_trs,:] - pbc_ijk_vox * Lbox - XYZ_ref))
            progr_sphere += int((dist2 <= dist2_max) & (VolVoids[id_trs] <= VolVoids[id_void]) & (id_trs != id_void))

    progr_ovlp = 0
    for id_trs in ids_to_expore[:progr_sphere]:
        ids_ovlp[id_ord,progr_ovlp] = id_trs
        Ncells_loop = int(Ncells[id_trs]) + int(round(Ncells[id_trs]%1))
        mask_in_arr = is_in_arr(Ids_voro_dict[id_trs][:Ncells_loop],Ids_voro_dict[id_void][:Ncells_ref])



        #mask_in_arr[:] = False
        #for a in Ids_voro_dict[id_void][:Ncells_ref]:
        #    mask_in_arr |= (Ids_voro_dict[id_trs][:Ncells_loop] == a)
        #Vol_ovlp[id_ord,progr_ovlp] = np.sum(voro_vol[(Ids_voro_dict[id_trs][:Ncells_loop])[mask_in_arr[:Ncells_loop]]])


        Vol_ovlp[id_ord,progr_ovlp] = np.sum(voro_vol[(Ids_voro_dict[id_trs][:Ncells_loop])[mask_in_arr]])
        Vol_ovlp_frac[id_ord,progr_ovlp] = Vol_ovlp[id_ord,progr_ovlp] / np.sum(voro_vol[Ids_voro_dict[id_trs][:Ncells_loop]])
        progr_ovlp += int(Vol_ovlp[id_ord,progr_ovlp] > 0.)

    #return ids_ovlp, Vol_ovlp, Vol_ovlp_frac, progr_ovlp
    return progr_ovlp
        

@jit(nopython=True,parallel=True)
def overlapping_fraction_loop(
    IDS_voids, id_sorted, XYZ_voids, VolVoids, Ncells, max_dist_vds, Ids_voro_dict, voro_vol, ind_vox, ngrid, Lbox, voxel_side):
    Rmax = np.max(max_dist_vds)
    Nvoids = IDS_voids.shape[0]

    N_poisson_in_sphere=1.2
    #max_num_tracers = int(4 * 3.1416 / 3 * (4 * R2max) ** 1.5 * XYZ_voids.shape[0] / (Lbox * Lbox * Lbox) * N_poisson_in_sphere) 
    max_num_tracers = min(Nvoids,int(4 * 3.1416 / 3 * (2*Rmax) ** 3 *  np.max(ind_vox[1:] - ind_vox[:-1]) * (ngrid / Lbox) ** 3 * N_poisson_in_sphere))

    ids_ovlp = np.zeros((Nvoids,max_num_tracers),dtype=np.int64)
    Vol_ovlp = np.zeros((Nvoids,max_num_tracers))
    Vol_ovlp_frac = np.zeros((Nvoids,max_num_tracers))
    num_ovlps = np.zeros(Nvoids,dtype=np.int64)

    #ids_to_explore = IDS_voids[Ncells[IDS_voids] > 1.]
    #for id_loop in prange(ids_to_explore.shape[0]):
        #num_ovlps[ids_to_explore[id_loop]] = overlapping_fraction_core(
        #    ids_to_explore[id_loop], ids_ovlp[ids_to_explore[id_loop],:], Vol_ovlp[ids_to_explore[id_loop],:], Vol_ovlp_frac[ids_to_explore[id_loop],:], 
        #    IDS_voids, XYZ_voids, VolVoids, Ncells, R2vds, R2max, Ids_voro_dict, voro_vol, ind_vox, ngrid, voxel_side, max_num_tracers)
    for id_loop in prange(Nvoids):
        
        #print(id_loop,'/',IDS_voids.shape[0])
        id_ord = id_sorted[id_loop]
        num_ovlps[id_ord] = overlapping_fraction_core(
            id_loop, id_ord, ids_ovlp, Vol_ovlp, Vol_ovlp_frac, 
            IDS_voids, XYZ_voids, VolVoids, Ncells, max_dist_vds, Rmax, Ids_voro_dict, voro_vol, ind_vox, ngrid, voxel_side, max_num_tracers)
    return ids_ovlp, Vol_ovlp, Vol_ovlp_frac, num_ovlps


@jit(nopython=True,parallel=True)
def overlapping_fraction_loop_pbc(
    IDS_voids, id_sorted, XYZ_voids, VolVoids, Ncells, max_dist_vds, Ids_voro_dict, voro_vol, ind_vox, ngrid, Lbox, voxel_side):
    Rmax = np.max(max_dist_vds)
    Nvoids = IDS_voids.shape[0]

    N_poisson_in_sphere=1.2
    #max_num_tracers = int(4 * 3.1416 / 3 * (4 * R2max) ** 1.5 * XYZ_voids.shape[0] / (Lbox * Lbox * Lbox) * N_poisson_in_sphere) 
    max_num_tracers = min(Nvoids,int(4 * 3.1416 / 3 * (2*Rmax) ** 3 *  np.max(ind_vox[1:] - ind_vox[:-1]) * (ngrid / Lbox) ** 3 * N_poisson_in_sphere))

    ids_ovlp = np.zeros((Nvoids,max_num_tracers),dtype=np.int64)
    Vol_ovlp = np.zeros((Nvoids,max_num_tracers))
    Vol_ovlp_frac = np.zeros((Nvoids,max_num_tracers))
    num_ovlps = np.zeros(Nvoids,dtype=np.int64)

    #ids_to_explore = IDS_voids[Ncells[IDS_voids] > 1.]
    #for id_loop in prange(ids_to_explore.shape[0]):
        #num_ovlps[ids_to_explore[id_loop]] = overlapping_fraction_core(
        #    ids_to_explore[id_loop], ids_ovlp[ids_to_explore[id_loop],:], Vol_ovlp[ids_to_explore[id_loop],:], Vol_ovlp_frac[ids_to_explore[id_loop],:], 
        #    IDS_voids, XYZ_voids, VolVoids, Ncells, R2vds, R2max, Ids_voro_dict, voro_vol, ind_vox, ngrid, voxel_side, max_num_tracers)
    for id_loop in prange(Nvoids):
        
        #print(id_loop,'/',IDS_voids.shape[0])
        id_ord = id_sorted[id_loop]
        num_ovlps[id_ord] = overlapping_fraction_core_pbc(
            id_loop, id_ord, ids_ovlp, Vol_ovlp, Vol_ovlp_frac, 
            IDS_voids, XYZ_voids, VolVoids, Ncells, max_dist_vds, Rmax, Ids_voro_dict, voro_vol, ind_vox, ngrid, Lbox, voxel_side, max_num_tracers)
    return ids_ovlp, Vol_ovlp, Vol_ovlp_frac, num_ovlps


@jit(nopython=True)
def overlapping_fraction_core_TEST(
    MASK1,id_loop, id_ord, ids_ovlp, Vol_ovlp, Vol_ovlp_frac, IDS_voids, XYZ_voids, VolVoids, Ncells, 
    max_dist_vds, R_max, Ids_voro_dict, voro_vol, ind_vox, ngrid, voxel_side, max_num_tracers):
    
    id_vd_1 = IDS_voids[id_loop]
    Ncells_ref = int(Ncells[id_vd_1]) + int(round(Ncells[id_vd_1]%1))

    MASK1[:] = False
        
    XYZ_ref = XYZ_voids[id_vd_1,:]
    #Vol = VolVoids[id_loop]


    dist_max = max_dist_vds[id_vd_1] + R_max
    dist2_max = dist_max * dist_max
    # initialize arrays:
    max_num_vox_for_sphere = int(4 * 3.1416 / 3 * (int((dist_max-1) / voxel_side) + 1 + 0.5 * np.sqrt(3))**3) + 1
    max_num_vox_for_sphere = max(max_num_vox_for_sphere, 27)


    half_n_vox_side = int((dist_max-1) / voxel_side) + 1
    
    ijk_in_sphere = np.empty((max_num_vox_for_sphere,3),dtype=np.int64)
    vox_dist_from_xyz = np.empty(3,dtype=np.float64)
    xyz_vox_unit = XYZ_ref / voxel_side
    ijk_vox_void_center = xyz_vox_unit.astype(np.int64)
    
    i_in = max(int(xyz_vox_unit[0] - half_n_vox_side),0)
    i_out = min(int(xyz_vox_unit[0] + half_n_vox_side + 1),ngrid)
    j_in = max(int(xyz_vox_unit[1] - half_n_vox_side),0)
    j_out = min(int(xyz_vox_unit[1] + half_n_vox_side + 1),ngrid)
    k_in = max(int(xyz_vox_unit[2] - half_n_vox_side),0)
    k_out = min(int(xyz_vox_unit[2] + half_n_vox_side + 1),ngrid)

    ids_to_expore = np.zeros(max_num_tracers,dtype=np.int64)
    #ids_ovlp = np.zeros(max_num_tracers,dtype=np.int64)
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
            id_vd_2 = IDS_voids[id_ptr]
            ids_to_expore[progr_sphere] = id_vd_2
            dist2 = np.sum(np.square(XYZ_voids[id_vd_2,:] - XYZ_ref))
            progr_sphere += int((dist2 <= dist2_max) & (VolVoids[id_vd_2] <= VolVoids[id_vd_1]) & (id_vd_2 != id_vd_1))

    progr_ovlp = 0
    for id_vd_2 in ids_to_expore[:progr_sphere]:
        #MASK2[MASK2] = False
        MASK1[Ids_voro_dict[id_vd_1][:Ncells_ref]] = True
        ids_ovlp[id_ord,progr_ovlp] = id_vd_2
        Ncells_loop = int(Ncells[id_vd_2]) + int(round(Ncells[id_vd_2]%1))
        MASK1[Ids_voro_dict[id_vd_2][:Ncells_loop]] &= True

        #mask_in_arr = MASK1 & MASK2 #is_in_arr(Ids_voro_dict[id_vd_2][:Ncells_loop],Ids_voro_dict[id_vd_1][:Ncells_ref])
        Vol_ovlp[id_ord,progr_ovlp] = np.sum(voro_vol[MASK1])
        Vol_ovlp_frac[id_ord,progr_ovlp] = Vol_ovlp[id_ord,progr_ovlp] / np.sum(voro_vol[Ids_voro_dict[id_vd_2][:Ncells_loop]]) #VolVoids[id_vd_2] #np.sum(voro_vol[Ids_voro_dict[id_vd_2][:Ncells_loop]])
        progr_ovlp += int(Vol_ovlp[id_ord,progr_ovlp] > 0.)

    #return ids_ovlp, Vol_ovlp, Vol_ovlp_frac, progr_ovlp
    return progr_ovlp



@jit(nopython=True,parallel=True)
def overlapping_fraction_loop_TEST(
    IDS_voids, id_sorted, XYZ_voids, VolVoids, Ncells, max_dist_vds, Ids_voro_dict, voro_vol, ind_vox, ngrid, Lbox, voxel_side):
    Rmax = np.max(max_dist_vds)
    Nvoids = IDS_voids.shape[0]

    N_poisson_in_sphere=1.2
    #max_num_tracers = int(4 * 3.1416 / 3 * (4 * R2max) ** 1.5 * XYZ_voids.shape[0] / (Lbox * Lbox * Lbox) * N_poisson_in_sphere) 
    max_num_tracers = min(Nvoids,int(4 * 3.1416 / 3 * (2*Rmax) ** 3 *  np.max(ind_vox[1:] - ind_vox[:-1]) * (ngrid / Lbox) ** 3 * N_poisson_in_sphere))

    ids_ovlp = np.zeros((Nvoids,max_num_tracers),dtype=np.int64)
    Vol_ovlp = np.zeros((Nvoids,max_num_tracers))
    Vol_ovlp_frac = np.zeros((Nvoids,max_num_tracers))
    num_ovlps = np.zeros(Nvoids,dtype=np.int64)

    #ids_to_explore = IDS_voids[Ncells[IDS_voids] > 1.]
    #for id_loop in prange(ids_to_explore.shape[0]):
        #num_ovlps[ids_to_explore[id_loop]] = overlapping_fraction_core(
        #    ids_to_explore[id_loop], ids_ovlp[ids_to_explore[id_loop],:], Vol_ovlp[ids_to_explore[id_loop],:], Vol_ovlp_frac[ids_to_explore[id_loop],:], 
        #    IDS_voids, XYZ_voids, VolVoids, Ncells, R2vds, R2max, Ids_voro_dict, voro_vol, ind_vox, ngrid, voxel_side, max_num_tracers)
    
    MASK1 = Dict.empty(
        key_type=types.int64,
        value_type=bool_array)
    #MASK2 = Dict.empty(
    #    key_type=types.int64,
    #    value_type=bool_array)


    for icpu in range(get_num_threads()):
        MASK1[icpu] = np.empty(voro_vol.shape[0],dtype=np.bool_)
        #MASK2[icpu] = np.empty(voro_vol.shape[0],dtype=np.bool_)

    for id_loop in prange(Nvoids):
        #print(id_loop,'/',IDS_voids.shape[0])
        id_ord = id_sorted[id_loop]
        num_ovlps[id_ord] = overlapping_fraction_core_TEST(
            MASK1[get_thread_id()], id_loop, id_ord, ids_ovlp, Vol_ovlp, Vol_ovlp_frac, 
            IDS_voids, XYZ_voids, VolVoids, Ncells, max_dist_vds, Rmax, Ids_voro_dict, voro_vol, ind_vox, ngrid, voxel_side, max_num_tracers)
    return ids_ovlp, Vol_ovlp, Vol_ovlp_frac, num_ovlps




@jit(nopython=True)
def nearest_cell_core(
    XYZ_ref, XYZ_voro_ord, ind_vox, ngrid, voxel_side, max_iterations):

    max_num_vox_for_sphere = int(4 * 3.1416 / 3 * (max_iterations + 0.5 * np.sqrt(3))**3) + 1
    #max_num_vox_for_sphere = max(max_num_vox_for_sphere, 27)

    #half_n_vox_side = int((dist_max-1) / voxel_side) + 1
    
    ijk_in_sphere = np.empty((max_num_vox_for_sphere,3),dtype=np.int64)
    vox_dist_from_xyz = np.empty(3,dtype=np.float64)
    xyz_vox_unit = XYZ_ref / voxel_side
    ijk_vox_void_center = xyz_vox_unit.astype(np.int64)
    
    #i_in = max(int(xyz_vox_unit[0] - half_n_vox_side),0)
    #i_out = min(int(xyz_vox_unit[0] + half_n_vox_side + 1),ngrid)
    #j_in = max(int(xyz_vox_unit[1] - half_n_vox_side),0)
    #j_out = min(int(xyz_vox_unit[1] + half_n_vox_side + 1),ngrid)
    #k_in = max(int(xyz_vox_unit[2] - half_n_vox_side),0)
    #k_out = min(int(xyz_vox_unit[2] + half_n_vox_side + 1),ngrid)
    
    i_in = max(int(xyz_vox_unit[0]) - 1,0)
    i_out = min(int(xyz_vox_unit[0]) + 2,ngrid)
    j_in = max(int(xyz_vox_unit[1]) - 1,0)
    j_out = min(int(xyz_vox_unit[1]) + 2,ngrid)
    k_in = max(int(xyz_vox_unit[2]) - 1,0)
    k_out = min(int(xyz_vox_unit[2]) + 2,ngrid)

    #ids_to_expore = np.zeros(max_num_tracers,dtype=np.int64)
    ################################################################33


    #dist_2 = np.empty(max_points_sphere,dtype=np.float64)
    #ids_ordered = np.empty(max_points_sphere,dtype=np.int64)
    r2_vox_unit = 1. #dist2_max / (voxel_side * voxel_side)
    min_dist_2 = 0.
    dist2_max = 0.
    iteration = 0
    while (min_dist_2 == dist2_max) | (iteration >= max_iterations):
        progr = 0
        dist2_max = r2_vox_unit * voxel_side * voxel_side
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
                    
        #progr_sphere = 0
        #// select voxels intersecting with the sphere centered in the void center
        min_dist_2 = dist2_max
        ids_ordered = ind_vox[0]
        for id_vox in range(progr):
            
            id_vox_tmp = ijk_in_sphere[id_vox,0] * ngrid * ngrid + ijk_in_sphere[id_vox,1] * ngrid + ijk_in_sphere[id_vox,2]

            ###################################################################
            ############## begin method 1 #####################################
            ###################################################################

            #ids_ordered[id_vox] = ind_vox[id_vox_tmp] + np.argmax(np.sum(np.square(XYZ_voro_ord[ind_vox[id_vox_tmp]:ind_vox[id_vox_tmp+1],:] - XYZ_voids),axis=1))
            #dist_2[id_vox] = np.sum(np.square(XYZ_voro_ord[ids_ordered[id_vox],:] - XYZ_ref))


            ###################################################################
            ############## end method 1 #####################################
            ###################################################################

            ###################################################################
            ############## begin method 2 #####################################
            ###################################################################

            for id_trs in range(ind_vox[id_vox_tmp],ind_vox[id_vox_tmp+1]):
                dist_2 = np.sum(np.square(XYZ_voro_ord[id_trs,:] - XYZ_ref))
                is_smaller = (min_dist_2 > dist_2) & (dist_2 <= dist2_max)
                ids_ordered = ids_ordered * int(not is_smaller) + id_trs * int(is_smaller)
                min_dist_2 = min_dist_2 * int(not is_smaller) + dist_2 * int(is_smaller)

            ###################################################################
            ############## end method 2 #####################################
            ###################################################################

                        
        i_in = max(i_in - 1,0)
        i_out = min(i_out + 1,ngrid)
        j_in = max(j_in - 1,0)
        j_out = min(j_out + 1,ngrid)
        k_in = max(k_in - 1,0)
        k_out = min(k_out + 1,ngrid)
        iteration += 1

    #ids = np.argpartition(dist_2[:progr_sphere],kth)[:kth]
    #ids = np.argmax(dist_2[:progr_sphere])
    
    #return ids_ordered[ids[np.argsort(dist_2[ids])]]
    
    return ids_ordered #[ids]


@jit(nopython=True)
def nearest_cell_core_pbc(
    XYZ_ref, XYZ_voro_ord, ind_vox, ngrid, Lbox, voxel_side, max_iterations):

    max_num_vox_for_sphere = int(4 * 3.1416 / 3 * (max_iterations + 0.5 * np.sqrt(3))**3) + 1
    
    ijk_in_sphere = np.empty((max_num_vox_for_sphere,3),dtype=np.int64)
    vox_dist_from_xyz = np.empty(3,dtype=np.float64)
    pbc_ijk_vox = np.empty(3,dtype=np.int8)
    xyz_vox_unit = XYZ_ref / voxel_side
    ijk_vox_void_center = xyz_vox_unit.astype(np.int64)
    
    
    i_in = max(int(np.floor(xyz_vox_unit[0] - 1)),-ngrid+1)
    i_out = min(int(np.floor(xyz_vox_unit[0] + 2)),i_in+ngrid)
    j_in = max(int(np.floor(xyz_vox_unit[1] - 1)),-ngrid+1)
    j_out = min(int(np.floor(xyz_vox_unit[1] + 2)),j_in+ngrid)
    k_in = max(int(np.floor(xyz_vox_unit[2] - 1)),-ngrid+1)
    k_out = min(int(np.floor(xyz_vox_unit[2] + 2)),k_in+ngrid)
    k_out = min(k_out,k_in+ngrid)

    #ids_to_expore = np.zeros(max_num_tracers,dtype=np.int64)
    ################################################################33


    #dist_2 = np.empty(max_points_sphere,dtype=np.float64)
    #ids_ordered = np.empty(max_points_sphere,dtype=np.int64)
    r2_vox_unit = 1. #dist2_max / (voxel_side * voxel_side)
    min_dist_2 = 0.
    dist2_max = 0.
    iteration = 0
    while (min_dist_2 == dist2_max) | (iteration >= max_iterations):
        progr = 0
        dist2_max = r2_vox_unit * voxel_side * voxel_side
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
                    
        #progr_sphere = 0
        #// select voxels intersecting with the sphere centered in the void center
        min_dist_2 = dist2_max
        ids_ordered = ind_vox[0]
        for id_vox in range(progr):
            
            pbc_ijk_vox[0] = int(ijk_in_sphere[id_vox,0] < 0) - int(ijk_in_sphere[id_vox,0] >= ngrid)
            pbc_ijk_vox[1] = int(ijk_in_sphere[id_vox,1] < 0) - int(ijk_in_sphere[id_vox,1] >= ngrid)
            pbc_ijk_vox[2] = int(ijk_in_sphere[id_vox,2] < 0) - int(ijk_in_sphere[id_vox,2] >= ngrid)

            i_tmp = ijk_in_sphere[id_vox,0] + pbc_ijk_vox[0] * ngrid
            j_tmp = ijk_in_sphere[id_vox,1] + pbc_ijk_vox[1] * ngrid
            k_tmp = ijk_in_sphere[id_vox,2] + pbc_ijk_vox[2] * ngrid

            id_vox_tmp = i_tmp * ngrid * ngrid + j_tmp * ngrid + k_tmp

            for id_trs in range(ind_vox[id_vox_tmp],ind_vox[id_vox_tmp+1]):
                dist_2 = np.sum(np.square(XYZ_voro_ord[id_trs,:] - pbc_ijk_vox * Lbox - XYZ_ref))
                is_smaller = (min_dist_2 > dist_2) & (dist_2 <= dist2_max)
                ids_ordered = ids_ordered * int(not is_smaller) + id_trs * int(is_smaller)
                min_dist_2 = min_dist_2 * int(not is_smaller) + dist_2 * int(is_smaller)

                        
        i_in = max(i_in - 1,-ngrid+1)
        i_out = min(i_out + 1,i_in+ngrid)
        j_in = max(j_in - 1,-ngrid+1)
        j_out = min(j_out + 1,j_in+ngrid)
        k_in = max(k_in - 1,-ngrid+1)
        k_out = min(k_out + 1,k_out+ngrid)
        iteration += 1

    #ids = np.argpartition(dist_2[:progr_sphere],kth)[:kth]
    #ids = np.argmax(dist_2[:progr_sphere])
    
    #return ids_ordered[ids[np.argsort(dist_2[ids])]]
    
    return ids_ordered #[ids]


@jit(nopython=True,parallel=True)
def nearest_cell_loop(
    IDS_voids, XYZ_voids, XYZ_voro_ord, ids_reverse, ind_vox, ngrid, voxel_side, max_iterations):

    Nvoids = IDS_voids.shape[0]
    #ids_closest = np.empty(Nvoids, dtype=np.int64)
    ids_closest = np.full(XYZ_voids.shape[0],-1, dtype=np.int64)

    for id_loop in prange(Nvoids):
        ids_closest[IDS_voids[id_loop]] = ids_reverse[
            nearest_cell_core(XYZ_voids[IDS_voids[id_loop],:], 
                              XYZ_voro_ord, ind_vox, ngrid, voxel_side,max_iterations)]
    return ids_closest



@jit(nopython=True,parallel=True)
def nearest_cell_loop_pbc(
    IDS_voids, XYZ_voids, XYZ_voro_ord, ids_reverse, ind_vox, ngrid, Lbox, voxel_side, max_iterations):

    Nvoids = IDS_voids.shape[0]
    #ids_closest = np.empty(Nvoids, dtype=np.int64)
    ids_closest = np.full(XYZ_voids.shape[0],-1, dtype=np.int64)

    for id_loop in prange(Nvoids):
        ids_closest[IDS_voids[id_loop]] = ids_reverse[
            nearest_cell_core_pbc(XYZ_voids[IDS_voids[id_loop],:], 
                              XYZ_voro_ord, ind_vox, ngrid, Lbox, voxel_side,max_iterations)]
    return ids_closest




@jit(nopython=True)
def center_in_void_core(
    id_loop, ids_voro_center, IDS_voids, XYZ_voids, VolVoids, Ncells, 
    max_dist_vds, R_max, Ids_voro_dict, ind_vox, ngrid, voxel_side):
    
    id_void = IDS_voids[id_loop]
    id_voro = ids_voro_center[id_void]
    XYZ_ref = XYZ_voids[id_void,:]

    dist_max = max_dist_vds[id_void] + R_max
    dist2_max = dist_max * dist_max
    # initialize arrays:
    max_num_vox_for_sphere = int(4 * 3.1416 / 3 * (int((dist_max-1) / voxel_side) + 1 + 0.5 * np.sqrt(3))**3) + 1
    max_num_vox_for_sphere = max(max_num_vox_for_sphere, 27)


    half_n_vox_side = int((dist_max-1) / voxel_side) + 1
    

    ijk_in_sphere = np.empty((max_num_vox_for_sphere,3),dtype=np.int64)
    vox_dist_from_xyz = np.empty(3,dtype=np.float64)
    xyz_vox_unit = XYZ_ref / voxel_side
    ijk_vox_void_center = xyz_vox_unit.astype(np.int64)
    
    i_in = max(int(xyz_vox_unit[0] - half_n_vox_side),0)
    i_out = min(int(xyz_vox_unit[0] + half_n_vox_side + 1),ngrid)
    j_in = max(int(xyz_vox_unit[1] - half_n_vox_side),0)
    j_out = min(int(xyz_vox_unit[1] + half_n_vox_side + 1),ngrid)
    k_in = max(int(xyz_vox_unit[2] - half_n_vox_side),0)
    k_out = min(int(xyz_vox_unit[2] + half_n_vox_side + 1),ngrid)

    #ids_ovlp = np.zeros(max_num_tracers,dtype=np.int64)
    #Vol_ovlp = np.zeros(max_num_tracers)
    #Vol_ovlp_frac = np.zeros(max_num_tracers)

    progr = 0
    r2_vox_unit = dist2_max / (voxel_side * voxel_side)
    max_num_tracers = 0
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

                vox_in_sphere = (vox_dist_from_xyz[0] + vox_dist_from_xyz[1] + vox_dist_from_xyz[2]) < r2_vox_unit

                id_vox_tmp = i * ngrid * ngrid + j * ngrid + k
                max_num_tracers += (ind_vox[id_vox_tmp+1]-ind_vox[id_vox_tmp]) * int(vox_in_sphere)

                progr += int(vox_in_sphere)
                
    progr_sphere = 0
    max_num_tracers += 1

    ids_to_expore = np.zeros(max_num_tracers,dtype=np.int64)
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
            #progr_sphere += int((dist2 <= dist2_max) & (VolVoids[id_trs] > VolVoids[id_void]) & (id_trs != id_void))
            progr_sphere += int((dist2 <= dist2_max) & (                        # <--\
                (VolVoids[id_trs] > VolVoids[id_void]) & (id_trs != id_void) |  # <--- standard condition
                (VolVoids[id_trs] == VolVoids[id_void]) &                       # <--\
                (ids_voro_center[id_trs] == ids_voro_center[id_void]) &         # <--- for identical voids started from a different cell
                (id_trs < id_void)))                                            # <--/

    max_vol = -1.
    id_overlap = -1
    for id_trs in ids_to_expore[:progr_sphere]:
        Ncells_loop = int(Ncells[id_trs]) + int(round(Ncells[id_trs]%1))


        overlap = (id_voro in Ids_voro_dict[id_trs][:Ncells_loop]) & (VolVoids[id_trs] > max_vol)

        id_overlap = id_overlap * int(not overlap) + id_trs * int(overlap)
        max_vol = max_vol * int(not overlap) + VolVoids[id_trs] * int(overlap)

    #return ids_ovlp, Vol_ovlp, Vol_ovlp_frac, progr_ovlp
    return id_overlap



@jit(nopython=True)
def center_in_void_core_pbc(
    id_loop, ids_voro_center, IDS_voids, XYZ_voids, VolVoids, Ncells, 
    max_dist_vds, R_max, Ids_voro_dict, ind_vox, ngrid, Lbox, voxel_side):
    
    id_void = IDS_voids[id_loop]
    id_voro = ids_voro_center[id_void]
    XYZ_ref = XYZ_voids[id_void,:]

    dist_max = max_dist_vds[id_void] + R_max
    dist2_max = dist_max * dist_max
    # initialize arrays:
    max_num_vox_for_sphere = int(4 * 3.1416 / 3 * (int((dist_max-1) / voxel_side) + 1 + 0.5 * np.sqrt(3))**3) + 1
    max_num_vox_for_sphere = max(max_num_vox_for_sphere, 27)


    half_n_vox_side = int((dist_max-1) / voxel_side) + 1
    

    ijk_in_sphere = np.empty((max_num_vox_for_sphere,3),dtype=np.int64)
    vox_dist_from_xyz = np.empty(3,dtype=np.float64)
    pbc_ijk_vox = np.empty(3,dtype=np.int8)
    xyz_vox_unit = XYZ_ref / voxel_side
    ijk_vox_void_center = xyz_vox_unit.astype(np.int64)
    
    
    i_in = max(int(np.floor(xyz_vox_unit[0] - half_n_vox_side)),-ngrid+1)
    i_out = min(int(np.floor(xyz_vox_unit[0] + half_n_vox_side + 1)),i_in+ngrid)
    j_in = max(int(np.floor(xyz_vox_unit[1] - half_n_vox_side)),-ngrid+1)
    j_out = min(int(np.floor(xyz_vox_unit[1] + half_n_vox_side + 1)),j_in+ngrid)
    k_in = max(int(np.floor(xyz_vox_unit[2] - half_n_vox_side)),-ngrid+1)
    k_out = min(int(np.floor(xyz_vox_unit[2] + half_n_vox_side + 1)),k_in+ngrid)

    progr = 0
    r2_vox_unit = dist2_max / (voxel_side * voxel_side)
    max_num_tracers = 0
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

                vox_in_sphere = (vox_dist_from_xyz[0] + vox_dist_from_xyz[1] + vox_dist_from_xyz[2]) < r2_vox_unit

                #id_vox_tmp = i * ngrid * ngrid + j * ngrid + k

                i_tmp = i + (int(i < 0) - int(i >= ngrid)) * ngrid
                j_tmp = j + (int(j < 0) - int(j >= ngrid)) * ngrid
                k_tmp = k + (int(k < 0) - int(k >= ngrid)) * ngrid

                id_vox_tmp = i_tmp * ngrid * ngrid + j_tmp * ngrid + k_tmp

                #print('i:',i,'->',i_tmp,'j:',j,'->',j_tmp,'k:',k,'->',k_tmp,'ngrid:',ngrid)

                max_num_tracers += (ind_vox[id_vox_tmp+1]-ind_vox[id_vox_tmp]) * int(vox_in_sphere)

                progr += int(vox_in_sphere)
                
    progr_sphere = 0
    max_num_tracers += 1

    ids_to_expore = np.zeros(max_num_tracers,dtype=np.int64)
    #// select voxels intersecting with the sphere centered in the void center
    for id_vox in range(progr):
        pbc_ijk_vox[0] = int(ijk_in_sphere[id_vox,0] < 0) - int(ijk_in_sphere[id_vox,0] >= ngrid)
        pbc_ijk_vox[1] = int(ijk_in_sphere[id_vox,1] < 0) - int(ijk_in_sphere[id_vox,1] >= ngrid)
        pbc_ijk_vox[2] = int(ijk_in_sphere[id_vox,2] < 0) - int(ijk_in_sphere[id_vox,2] >= ngrid)

        i_tmp = ijk_in_sphere[id_vox,0] + pbc_ijk_vox[0] * ngrid
        j_tmp = ijk_in_sphere[id_vox,1] + pbc_ijk_vox[1] * ngrid
        k_tmp = ijk_in_sphere[id_vox,2] + pbc_ijk_vox[2] * ngrid

        id_vox_tmp = i_tmp * ngrid * ngrid + j_tmp * ngrid + k_tmp
        #if (i_tmp == 0) & (j_tmp == 2) & (k_tmp == 1):
        #    print(id_vox_tmp)
                            
        for id_ptr in range(ind_vox[id_vox_tmp],ind_vox[id_vox_tmp+1]):
            id_trs = IDS_voids[id_ptr]
            ids_to_expore[progr_sphere] = id_trs
            dist2 = np.sum(np.square(XYZ_voids[id_trs,:] - pbc_ijk_vox * Lbox - XYZ_ref))
            #progr_sphere += int((dist2 <= dist2_max) & (VolVoids[id_trs] > VolVoids[id_void]) & (id_trs != id_void))

            progr_sphere += int((dist2 <= dist2_max) & (                        # <--\
                (VolVoids[id_trs] > VolVoids[id_void]) & (id_trs != id_void) |  # <--- standard condition
                (VolVoids[id_trs] == VolVoids[id_void]) &                       # <--\
                (ids_voro_center[id_trs] == ids_voro_center[id_void]) &         # <--- for identical voids started from a different cell
                (id_trs < id_void)))                                            # <--/

    max_vol = -1.
    id_overlap = -1
    for id_trs in ids_to_expore[:progr_sphere]:
        Ncells_loop = int(Ncells[id_trs]) + int(round(Ncells[id_trs]%1))


        overlap = (id_voro in Ids_voro_dict[id_trs][:Ncells_loop]) & (VolVoids[id_trs] > max_vol)

        id_overlap = id_overlap * int(not overlap) + id_trs * int(overlap)
        max_vol = max_vol * int(not overlap) + VolVoids[id_trs] * int(overlap)

    #return ids_ovlp, Vol_ovlp, Vol_ovlp_frac, progr_ovlp
    return id_overlap


@jit(nopython=True,parallel=True)
def center_in_void_loop(
    IDS_voids, id_sorted, XYZ_voids, VolVoids, Ncells, ids_voro_center, 
    max_dist_vds, Ids_voro_dict, ind_vox, ngrid, voxel_side):
    Rmax = np.max(max_dist_vds)
    Nvoids = IDS_voids.shape[0]
    ids_in_void = np.zeros(Nvoids,dtype=np.int64)

    for id_loop in prange(Nvoids):
        #print(id_loop,'/',IDS_voids.shape[0])
        id_ord = id_sorted[id_loop]

        ids_in_void[id_ord] = center_in_void_core(
            id_loop, ids_voro_center, IDS_voids, XYZ_voids, VolVoids, Ncells, 
            max_dist_vds, Rmax, Ids_voro_dict, ind_vox, ngrid, voxel_side)
    return ids_in_void

@jit(nopython=True,parallel=True)
def center_in_void_loop_pbc(
    IDS_voids, id_sorted, XYZ_voids, VolVoids, Ncells, ids_voro_center, 
    max_dist_vds, Ids_voro_dict, ind_vox, ngrid, Lbox, voxel_side):
    Rmax = np.max(max_dist_vds)
    Nvoids = IDS_voids.shape[0]
    ids_in_void = np.zeros(Nvoids,dtype=np.int64)

    print('ids_voro_center:',ids_voro_center.shape,'IDS_voids:',IDS_voids.shape,'max(IDS_voids):',np.max(IDS_voids))
    for id_loop in prange(Nvoids):
        #print(id_loop,'/',IDS_voids.shape[0])
        id_ord = id_sorted[id_loop]

        ids_in_void[id_ord] = center_in_void_core_pbc(
            id_loop, ids_voro_center, IDS_voids, XYZ_voids, VolVoids, Ncells, 
            max_dist_vds, Rmax, Ids_voro_dict, ind_vox, ngrid, Lbox, voxel_side)
    return ids_in_void


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
def compute_max_dist2_pbc(Ncells,XYZ_voids,XYZ_voro,id_selected,Ids_voro_dict,Lbox):
    Nvoids = XYZ_voids.shape[0]
    dist2_max = np.zeros(Nvoids)

    #id_out = np.arange(Nvoids)[Ncells>=1]
    for i in prange(id_selected.shape[0]):
        iv = id_selected[i]
        Ncells_loop = int(Ncells[iv]) + int((Ncells[iv]%1) > 0)
        xyz_loop = np.copy(XYZ_voro[Ids_voro_dict[iv][:Ncells_loop],:])
        sgn_pbc = np.array([1.,-1.])[(XYZ_voids[iv,:] < 0.5 * Lbox).astype(np.int64)]
        mask = np.empty(Ncells_loop,dtype=np.bool_)
        #print(i,iv,Ncells_loop,xyz_loop.shape,sgn_pbc.shape,mask.shape)
        for ijk in range(3):
            mask[:] = np.abs(xyz_loop[:,ijk] - XYZ_voids[iv,ijk]) > 0.5 * Lbox
            xyz_loop[mask,ijk] += sgn_pbc[ijk] * Lbox
            #print(iv,Ncells[iv],Ncells_loop,)
        dist2_max[iv] = np.max(np.sum(np.square(xyz_loop - XYZ_voids[iv,:]),axis=1))
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
        



def overlapping_fraction(
    xyz_vds, vol_vds, Ncells, xyz_voro, vol_voro, IDs_in_voids, Lbox=-1.,lightcone=True,
    ngrid=-1,nthreads=-1,verbose=True,Omega_rad=-1.,id_selected=None,TEST=False):
    # xyz_vds: dim (num_voids,3) numpy array containing void centers
    # vol_vds: dim (num_voids,) numpy array containing void volumes
    # Ncells: dim (num_voids,) numpy array containing the (fractional) number of voronoi cell in each void
    # xyz_voro: dim (num_tracers,3) numpy array containing tracers coordinates
    # vol_voro: dim (num_tracers,) numpy array Voronoi cell volumes
    # IDs_in_voids: dict containing the IDs of all Voronoi cells building up each void
    # Lbox: side of simulation box (lightcone=False)
    # lightcone: if False consider periodic boundary condition at 0 and Lbox
    # nthreads: number of threads to use for parallel computation. If nthreads=-1, this function automatically uses all the available CPUs
    # verbose: if True the function will print logs

    verboseprint = print if verbose else lambda *a, **k: None

    verboseprint("\noverlapping_fraction started.",flush=True)

    if id_selected is None:
        verboseprint("\nid_selected not passed.",flush=True)
        id_selected = np.arange(Ncells.shape[0])[Ncells > 1.]


    try:
        nthreads_tot = int(os.environ["OMP_NUM_THREADS"])
    except:
        nthreads_tot = get_num_threads()

    if (nthreads <= 0) | (nthreads > nthreads_tot):
        nthreads = nthreads_tot

    set_num_threads(min(nthreads,id_selected.shape[0]))

    verboseprint('\n    nthreads set to',nthreads,flush=True)

    if (not lightcone) & (Lbox > 0):
        R_max = compute_max_dist2_pbc(Ncells,xyz_vds,xyz_voro,id_selected,IDs_in_voids,Lbox)**0.5
    else:
        R_max = compute_max_dist2(Ncells,xyz_vds,xyz_voro,id_selected,IDs_in_voids)**0.5
    #print('R_max:',R_max,flush=True)
    #print('xyz_vds:',xyz_vds[id_selected],flush=True)
    #print('IDs_in_voids:',IDs_in_voids,flush=True)


    verboseprint('\n    R_max computed. Max val =',(np.max(R_max)),flush=True)


    if (Lbox < 0):
        if not lightcone:
            print("    WARNING: Lbox not passed and lightcone = False. We suggest either to pass Lbox or set lightcone = True.",
                  "\n    Lightcone authomatically set to True.",flush=True)
            verboseprint("    Computing Lbox using xyz_vds as reference:",flush=True)
            lightcone = True
        else:
            verboseprint("\n    Lbox not passed, using xyz_vds as reference:",flush=True)
    if lightcone:
        if Lbox > 0:
            verboseprint("\n    Lbox passed but lightcone = True: Lbox will be ignored.",flush=True)

        offset = np.min(xyz_vds[id_selected,:],axis=0)
        max_values = np.max(xyz_vds[id_selected,:],axis=0)
        Lbox = np.max(max_values - offset)

        offset -= Lbox*1e-4
        max_values += Lbox*1e-4
        Lbox = np.max(max_values - offset)

        
        verboseprint("    min(xyz_vds) =",*offset,flush=True)
        verboseprint("    max(xyz_vds) =",*max_values,flush=True)
        verboseprint("    Lbox =",Lbox,flush=True)
    else:
        offset = 0.
        if Lbox < 0:
            max_values = np.max(xyz_vds[id_selected,:],axis=0)
            Lbox = np.max(max_values - offset)
            print("    WARNING: Lbox not passed and lightcone = False. We suggest either to pass Lbox or set lightcone = True.",flush=True)



    if ngrid < 0:
        #ngrid = max(int(round(5 * Lbox / np.sqrt(np.max(R2_max)))),3)
        ngrid = max(int(round(Lbox / np.max(R_max) )),4)

        verboseprint("\n    ngrid not passed. Set to optimal value:",ngrid,flush=True)



    voxel_side = Lbox / ngrid
    
    verboseprint('\n    order_ids_tracers_selected_in_voxels started',flush=True)
    
    t0 = time.time()

    #IDs_vds_ordered, voxel_ptr = order_ids_tracers_in_voxels(xyz_vds, ngrid, Lbox)
    
    IDs_vds_ordered, id_sorted, voxel_ptr = order_ids_tracers_selected_in_voxels(xyz_vds - offset,id_selected, ngrid, Lbox)
    
    dt = time.time() - t0
    verboseprint("    done,",StrHminSec(dt),flush=True)

    verboseprint("\n    computation started (periodic-boundaries condition "+['on','off'][int(lightcone)]+")",flush=True)

    if lightcone:
        t0 = time.time()
        ids_ovlp, Vol_ovlp, Vol_ovlp_frac, num_ovlps = overlapping_fraction_loop(
            IDs_vds_ordered, id_sorted, xyz_vds - offset, vol_vds, Ncells, R_max, IDs_in_voids, vol_voro, voxel_ptr, ngrid, Lbox, voxel_side)
        dt = time.time() - t0
    else:
        t0 = time.time()
        ids_ovlp, Vol_ovlp, Vol_ovlp_frac, num_ovlps = overlapping_fraction_loop_pbc(
            IDs_vds_ordered, id_sorted, xyz_vds, vol_vds, Ncells, R_max, IDs_in_voids, vol_voro, voxel_ptr, ngrid, Lbox, voxel_side)
        dt = time.time() - t0
    verboseprint("    done,",StrHminSec(dt),'\n',flush=True)


    if nthreads != nthreads_tot:
        set_num_threads(nthreads_tot)

    return ids_ovlp, Vol_ovlp, Vol_ovlp_frac, num_ovlps




def closest_voro(
    xyz_vds, Ncells, xyz_voro, vol_voro, IDs_in_voids, Lbox=-1.,lightcone=True,
    ngrid=-1,max_iterations=3,nthreads=-1,verbose=True,id_selected=None):
    # xyz_vds: dim (num_voids,3) numpy array containing void centers
    # Ncells: dim (num_voids,) numpy array containing the (fractional) number of voronoi cell in each void
    # xyz_voro: dim (num_tracers,3) numpy array containing tracers coordinates
    # vol_voro: dim (num_tracers,) numpy array Voronoi cell volumes
    # IDs_in_voids: dict containing the IDs of all Voronoi cells building up each void
    # Lbox: side of simulation box (lightcone=False)
    # lightcone: if False consider periodic boundary condition at 0 and Lbox
    # nthreads: number of threads to use for parallel computation. If nthreads=-1, this function automatically uses all the available CPUs
    # verbose: if True the function will print logs

    verboseprint = print if verbose else lambda *a, **k: None

    verboseprint("\ncenter_in_void started.",flush=True)

    if id_selected is None:
        verboseprint("\nid_selected not passed, set condition Ncells > 1.",flush=True)
        id_selected = np.arange(Ncells.shape[0])[Ncells > 1.]


    try:
        nthreads_tot = int(os.environ["OMP_NUM_THREADS"])
    except:
        nthreads_tot = get_num_threads()

    if (nthreads <= 0) | (nthreads > nthreads_tot):
        nthreads = nthreads_tot

    set_num_threads(min(nthreads,id_selected.shape[0]))

    verboseprint('\n    nthreads set to',nthreads,flush=True)

    if (not lightcone) & (Lbox > 0):
        R_max = compute_max_dist2_pbc(Ncells,xyz_vds,xyz_voro,id_selected,IDs_in_voids,Lbox)**0.5
    else:
        R_max = compute_max_dist2(Ncells,xyz_vds,xyz_voro,id_selected,IDs_in_voids)**0.5
    #print('R_max:',R_max,flush=True)
    #print('xyz_vds:',xyz_vds[id_selected],flush=True)
    #print('IDs_in_voids:',IDs_in_voids,flush=True)
    verboseprint('\n    R_max computed. Max val =',(np.max(R_max)),flush=True)


    if (Lbox < 0):
        if not lightcone:
            print("    WARNING: Lbox not passed and lightcone = False. We suggest either to pass Lbox or set lightcone = True.",
                  "\n    Lightcone authomatically set to True.",flush=True)
            verboseprint("    Computing Lbox using xyz_voro as reference:",flush=True)
            lightcone = True
        else:
            verboseprint("\n    Lbox not passed, using xyz_voro as reference:",flush=True)
    if lightcone:
        if Lbox > 0:
            verboseprint("\n    Lbox passed but lightcone = True: Lbox will be ignored.",flush=True)

        offset = np.min(xyz_voro,axis=0)
        max_values = np.max(xyz_voro,axis=0)
        Lbox = np.max(max_values - offset)

        offset -= Lbox*1e-4
        max_values += Lbox*1e-4
        Lbox = np.max(max_values - offset)

        
        verboseprint("    min(xyz_voro) =",*offset,flush=True)
        verboseprint("    max(xyz_voro) =",*max_values,flush=True)
        verboseprint("    Lbox =",Lbox,flush=True)
    else:
        offset = 0.
        if Lbox < 0:
            max_values = np.max(xyz_voro,axis=0)
            Lbox = np.max(max_values - offset)
            print("    WARNING: Lbox not passed and lightcone = False. We suggest either to pass Lbox or set lightcone = True.",flush=True)



    if ngrid < 0:
        #ngrid = max(int(round(5 * Lbox / np.sqrt(np.max(R2_max)))),3)
        ngrid = max(int(round(Lbox / np.max(vol_voro)**(1./3.) )),4)

        verboseprint("\n    ngrid not passed. Set to optimal value:",ngrid,flush=True)


    voxel_side_voro = Lbox / ngrid

    xyz_trs_out, ids_reverse_voro, ind_vox_voro = order_coord_tracers_in_voxels_ids_rev_copy(xyz_voro[id_selected,:] - offset, ngrid, Lbox)
    
    verboseprint('\n    order_ids_tracers_selected_in_voxels started',flush=True)
    
    t0 = time.time()

    if lightcone:
        t0 = time.time()
        ids_closest_voro = nearest_cell_loop(
            id_selected, xyz_vds-offset, xyz_trs_out, ids_reverse_voro, ind_vox_voro, ngrid, voxel_side_voro, max_iterations)
        dt = time.time() - t0
    else:
        t0 = time.time()
        ids_closest_voro = nearest_cell_loop_pbc(
            id_selected, xyz_vds-offset, xyz_trs_out, ids_reverse_voro, ind_vox_voro, ngrid, Lbox, voxel_side_voro, max_iterations)
        dt = time.time() - t0
        
    
    dt = time.time() - t0
    verboseprint("    done,",StrHminSec(dt),flush=True)
    return ids_closest_voro



def center_in_void(
    xyz_vds, vol_vds, Ncells, xyz_voro, IDs_in_voids, ids_closest_voro, Lbox=-1.,lightcone=True,
    ngrid=-1,nthreads=-1,verbose=True,id_selected=None):
    # xyz_vds: dim (num_voids,3) numpy array containing void centers
    # vol_vds: dim (num_voids,) numpy array containing void volumes
    # Ncells: dim (num_voids,) numpy array containing the (fractional) number of voronoi cell in each void
    # xyz_voro: dim (num_tracers,3) numpy array containing tracers coordinates
    # vol_voro: dim (num_tracers,) numpy array Voronoi cell volumes
    # IDs_in_voids: dict containing the IDs of all Voronoi cells building up each void
    # Lbox: side of simulation box (lightcone=False)
    # lightcone: if False consider periodic boundary condition at 0 and Lbox
    # nthreads: number of threads to use for parallel computation. If nthreads=-1, this function automatically uses all the available CPUs
    # verbose: if True the function will print logs

    verboseprint = print if verbose else lambda *a, **k: None

    verboseprint("\ncenter_in_void started.",flush=True)

    if id_selected is None:
        verboseprint("\nid_selected not passed, set condition Ncells > 1.",flush=True)
        id_selected = np.arange(Ncells.shape[0])[Ncells > 1.]

    try:
        nthreads_tot = int(os.environ["OMP_NUM_THREADS"])
    except:
        nthreads_tot = get_num_threads()

    if (nthreads <= 0) | (nthreads > nthreads_tot):
        nthreads = nthreads_tot

    set_num_threads(min(nthreads,id_selected.shape[0]))

    verboseprint('\n    nthreads set to',nthreads,flush=True)

    if (not lightcone) & (Lbox > 0):
        R_max = compute_max_dist2_pbc(Ncells,xyz_vds,xyz_voro,id_selected,IDs_in_voids,Lbox)**0.5
    else:
        R_max = compute_max_dist2(Ncells,xyz_vds,xyz_voro,id_selected,IDs_in_voids)**0.5
    #print('R_max:',R_max,flush=True)
    #print('xyz_vds:',xyz_vds[id_selected],flush=True)
    #print('IDs_in_voids:',IDs_in_voids,flush=True)
    verboseprint('\n    R_max computed. Max val =',(np.max(R_max)),flush=True)


    if (Lbox < 0):
        if not lightcone:
            print("    WARNING: Lbox not passed and lightcone = False. We suggest either to pass Lbox or set lightcone = True.",
                  "\n    Lightcone authomatically set to True.",flush=True)
            verboseprint("    Computing Lbox using xyz_voro as reference:",flush=True)
            lightcone = True
        else:
            verboseprint("\n    Lbox not passed, using xyz_voro as reference:",flush=True)
    if lightcone:
        if Lbox > 0:
            verboseprint("\n    Lbox passed but lightcone = True: Lbox will be ignored.",flush=True)

        offset = np.min(xyz_voro,axis=0)
        max_values = np.max(xyz_voro,axis=0)
        Lbox = np.max(max_values - offset)

        offset -= Lbox*1e-4
        max_values += Lbox*1e-4
        Lbox = np.max(max_values - offset)

        
        verboseprint("    min(xyz_voro) =",*offset,flush=True)
        verboseprint("    max(xyz_voro) =",*max_values,flush=True)
        verboseprint("    Lbox =",Lbox,flush=True)
    else:
        offset = 0.
        if Lbox < 0:
            max_values = np.max(xyz_voro,axis=0)
            Lbox = np.max(max_values - offset)
            print("    WARNING: Lbox not passed and lightcone = False. We suggest either to pass Lbox or set lightcone = True.",flush=True)

    if ngrid < 0:
        ngrid = max(int(round(Lbox / np.max(R_max) )),4)

        verboseprint("\n    ngrid not passed. Set to optimal value:",ngrid,flush=True)

    verboseprint('\n    order_ids_tracers_selected_in_voxels started',flush=True)
    t0 = time.time()
    #IDs_vds_ordered, voxel_ptr = order_ids_tracers_in_voxels(xyz_vds, ngrid, Lbox)
    
    IDs_vds_ordered, id_sorted, ind_vox_voids = order_ids_tracers_selected_in_voxels(xyz_vds - offset, id_selected, ngrid , Lbox)
    
    dt = time.time() - t0
    verboseprint("    done,",StrHminSec(dt),flush=True)

    verboseprint("\n    computation started (periodic-boundaries condition "+['on','off'][int(lightcone)]+")",flush=True)
    voxel_side_voids = Lbox / ngrid
    if lightcone:
        t0 = time.time()
        ids_in_void = center_in_void_loop(
            IDs_vds_ordered, id_sorted, xyz_vds - offset, vol_vds, Ncells, 
            ids_closest_voro, R_max, IDs_in_voids, ind_vox_voids, ngrid, voxel_side_voids)
        dt = time.time() - t0
    else:
        t0 = time.time()
        ids_in_void = center_in_void_loop_pbc(
            IDs_vds_ordered, id_sorted, xyz_vds - offset, vol_vds, Ncells, 
            ids_closest_voro, R_max, IDs_in_voids, ind_vox_voids, ngrid, Lbox, voxel_side_voids)
        dt = time.time() - t0
    verboseprint("    done,",StrHminSec(dt),'\n',flush=True)


    if nthreads != nthreads_tot:
        set_num_threads(nthreads_tot)

    # select overlapping pairs:
    mask_overlapping = ids_in_void > -1
    overlapping_pairs = np.empty((np.sum(mask_overlapping),2),dtype=np.int64)
    overlapping_pairs[:,0] = id_selected[mask_overlapping]
    overlapping_pairs[:,1] = ids_in_void[mask_overlapping]

    return overlapping_pairs


def center_in_void_and_closest_voro(
    xyz_vds, vol_vds, Ncells, xyz_voro, vol_voro, IDs_in_voids, Lbox=-1.,lightcone=True,
    ngrid_voro=-1,ngrid_voids=-1,max_iterations=3,nthreads=-1,verbose=True,Omega_rad=-1.,id_selected=None,TEST=False):
    # xyz_vds: dim (num_voids,3) numpy array containing void centers
    # vol_vds: dim (num_voids,) numpy array containing void volumes
    # Ncells: dim (num_voids,) numpy array containing the (fractional) number of voronoi cell in each void
    # xyz_voro: dim (num_tracers,3) numpy array containing tracers coordinates
    # vol_voro: dim (num_tracers,) numpy array Voronoi cell volumes
    # IDs_in_voids: dict containing the IDs of all Voronoi cells building up each void
    # Lbox: side of simulation box (lightcone=False)
    # lightcone: if False consider periodic boundary condition at 0 and Lbox
    # nthreads: number of threads to use for parallel computation. If nthreads=-1, this function automatically uses all the available CPUs
    # verbose: if True the function will print logs

    verboseprint = print if verbose else lambda *a, **k: None

    verboseprint("\ncenter_in_void started.",flush=True)

    if id_selected is None:
        verboseprint("\nid_selected not passed, set condition Ncells > 1.",flush=True)
        id_selected = np.arange(Ncells.shape[0])[Ncells > 1.]


    try:
        nthreads_tot = int(os.environ["OMP_NUM_THREADS"])
    except:
        nthreads_tot = get_num_threads()

    if (nthreads <= 0) | (nthreads > nthreads_tot):
        nthreads = nthreads_tot

    set_num_threads(min(nthreads,id_selected.shape[0]))

    verboseprint('\n    nthreads set to',nthreads,flush=True)

    if (not lightcone) & (Lbox > 0):
        R_max = compute_max_dist2_pbc(Ncells,xyz_vds,xyz_voro,id_selected,IDs_in_voids,Lbox)**0.5
    else:
        R_max = compute_max_dist2(Ncells,xyz_vds,xyz_voro,id_selected,IDs_in_voids)**0.5
    #print('R_max:',R_max,flush=True)
    #print('xyz_vds:',xyz_vds[id_selected],flush=True)
    #print('IDs_in_voids:',IDs_in_voids,flush=True)
    verboseprint('\n    R_max computed. Max val =',(np.max(R_max)),flush=True)


    if (Lbox < 0):
        if not lightcone:
            print("    WARNING: Lbox not passed and lightcone = False. We suggest either to pass Lbox or set lightcone = True.",
                  "\n    Lightcone authomatically set to True.",flush=True)
            verboseprint("    Computing Lbox using xyz_voro as reference:",flush=True)
            lightcone = True
        else:
            verboseprint("\n    Lbox not passed, using xyz_voro as reference:",flush=True)
    if lightcone:
        if Lbox > 0:
            verboseprint("\n    Lbox passed but lightcone = True: Lbox will be ignored.",flush=True)

        offset = np.min(xyz_voro,axis=0)
        max_values = np.max(xyz_voro,axis=0)
        Lbox = np.max(max_values - offset)

        offset -= Lbox*1e-4
        max_values += Lbox*1e-4
        Lbox = np.max(max_values - offset)

        
        verboseprint("    min(xyz_voro) =",*offset,flush=True)
        verboseprint("    max(xyz_voro) =",*max_values,flush=True)
        verboseprint("    Lbox =",Lbox,flush=True)
    else:
        offset = 0.
        if Lbox < 0:
            max_values = np.max(xyz_voro,axis=0)
            Lbox = np.max(max_values - offset)
            print("    WARNING: Lbox not passed and lightcone = False. We suggest either to pass Lbox or set lightcone = True.",flush=True)



    if ngrid_voro < 0:
        #ngrid = max(int(round(5 * Lbox / np.sqrt(np.max(R2_max)))),3)
        ngrid_voro = max(int(round(Lbox / np.max(vol_voro)**(1./3.) )),4)

        verboseprint("\n    ngrid_voro not passed. Set to optimal value:",ngrid_voro,flush=True)

    if ngrid_voids < 0:
        ngrid_voids = max(int(round(Lbox / np.max(R_max) )),4)

        verboseprint("\n    ngrid_voids not passed. Set to optimal value:",ngrid_voids,flush=True)




    voxel_side_voro = Lbox / ngrid_voro

    xyz_trs_out, ids_reverse_voro, ind_vox_voro = order_coord_tracers_in_voxels_ids_rev_copy(xyz_voro - offset, ngrid_voro, Lbox)
    
    verboseprint('\n    order_ids_tracers_selected_in_voxels started',flush=True)
    
    t0 = time.time()

    if lightcone:
        t0 = time.time()
        ids_closest_voro = nearest_cell_loop(
            id_selected, xyz_vds-offset, xyz_trs_out, ids_reverse_voro, ind_vox_voro, ngrid_voro, voxel_side_voro, max_iterations)
        dt = time.time() - t0
    else:
        t0 = time.time()
        ids_closest_voro = nearest_cell_loop_pbc(
            id_selected, xyz_vds-offset, xyz_trs_out, ids_reverse_voro, ind_vox_voro, ngrid_voro, Lbox, voxel_side_voro, max_iterations)
        dt = time.time() - t0
    
    dt = time.time() - t0
    verboseprint("    done,",StrHminSec(dt),flush=True)

    
    verboseprint('\n    order_ids_tracers_selected_in_voxels started',flush=True)
    #IDs_vds_ordered, voxel_ptr = order_ids_tracers_in_voxels(xyz_vds, ngrid, Lbox)
    
    IDs_vds_ordered, id_sorted, ind_vox_voids = order_ids_tracers_selected_in_voxels(xyz_vds - offset, id_selected, ngrid_voids , Lbox)
    
    dt = time.time() - t0
    verboseprint("    done,",StrHminSec(dt),flush=True)

    verboseprint("\n    computation started (periodic-boundaries condition "+['on','off'][int(lightcone)]+")",flush=True)
    voxel_side_voids = Lbox / ngrid_voids
    if lightcone:
        t0 = time.time()
        ids_in_void = center_in_void_loop(
            IDs_vds_ordered, id_sorted, xyz_vds - offset, vol_vds, Ncells, 
            ids_closest_voro, R_max, IDs_in_voids, ind_vox_voids, ngrid_voids, voxel_side_voids)
        dt = time.time() - t0
    else:
        t0 = time.time()
        ids_in_void = center_in_void_loop_pbc(
            IDs_vds_ordered, id_sorted, xyz_vds - offset, vol_vds, Ncells, 
            ids_closest_voro, R_max, IDs_in_voids, ind_vox_voids, ngrid_voids, Lbox, voxel_side_voids)
        dt = time.time() - t0
    verboseprint("    done,",StrHminSec(dt),'\n',flush=True)


    if nthreads != nthreads_tot:
        set_num_threads(nthreads_tot)

    # select overlapping pairs:
    mask_overlapping = ids_in_void > -1
    overlapping_pairs = np.empty((np.sum(mask_overlapping),2),dtype=np.int64)
    overlapping_pairs[:,0] = id_selected[mask_overlapping]
    overlapping_pairs[:,1] = ids_in_void[mask_overlapping]

    return overlapping_pairs, ids_closest_voro


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
def select_overlaps_center_in_void(overlapping_pairs,VolArr):
    Ntot = overlapping_pairs.shape[0]
    #id_sort = (np.argsort(VolArr[th_voids.ids_ovlp_center[ith][:,0]])[::-1]).astype(dtype=np.int64, order='C')
    id_sort = np.empty(Ntot,dtype=np.int64)
    id_sort[:] = np.argsort(VolArr[overlapping_pairs[:,0]])[::-1]

    overlapping_pairs_out = np.copy(overlapping_pairs[id_sort,:])
    ind = 0
    while ind < Ntot:
        if overlapping_pairs_out[ind,0] in overlapping_pairs_out[:Ntot,1]:
            overlapping_pairs_out[ind:-1,:] = overlapping_pairs_out[ind+1:,:]
            Ntot -= 1
        ind += 1
        
    return overlapping_pairs_out[:Ntot,:]


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