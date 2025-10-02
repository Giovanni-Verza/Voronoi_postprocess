import numpy as np
from numba import jit, prange


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
    id_sorted = np.arange(num_tracers)
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
        id_sorted[id_new] = i_sel
        #id_trs_orig[i_sel] = id_new

        n_trs_vox[id_grid] -= 1
        
    return id_trs_out, id_sorted, ind_vox


@jit(nopython=True)
def order_coord_tracers_in_voxels_ids_rev_copy(xyz_trs, ngrid, Lbox):

    num_tracers = xyz_trs.shape[0]
    n_trs_vox = np.zeros(ngrid*ngrid*ngrid,dtype=np.int64) # array containing the number of tracer for each voxel
    xyz_trs_out = np.empty(xyz_trs.shape,dtype=xyz_trs.dtype)
    ids_reverse = np.empty(xyz_trs.shape[0],dtype=np.int64)

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
        xyz_trs_out[id_new,:] = xyz_trs[i_tr,:]
        ids_reverse[id_new] = i_tr #attr[i_tr]

        n_trs_vox[id_grid] -= 1
        
    return xyz_trs_out, ids_reverse, ind_vox


