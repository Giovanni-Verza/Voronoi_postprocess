import numpy as np
from numba import jit, prange


@jit(nopython=True)
def order_coord_tracers_in_voxels_permutation(
    xyz_trs, ngrid, Lbox):

    num_tracers = xyz_trs.shape[0]
    n_trs_vox = np.zeros(ngrid*ngrid*ngrid,dtype=np.int64) # array containing the number of tracer for each voxel
    permutation_ind = np.empty(num_tracers,dtype=np.int64)

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
        permutation_ind[i_tr] = ind_vox[id_grid]  + n_trs_vox[id_grid] - 1

        n_trs_vox[id_grid] -= 1
    
    
    for i in range(num_tracers):
        id_next = i
        #temp = 0
        while (permutation_ind[id_next] >= 0):
 
            #// Swap the current element according
            #// to the permutation in P
            xyz_trs[i,0], xyz_trs[permutation_ind[id_next],0] = xyz_trs[permutation_ind[id_next],0], xyz_trs[i,0]
            xyz_trs[i,1], xyz_trs[permutation_ind[id_next],1] = xyz_trs[permutation_ind[id_next],1], xyz_trs[i,1]
            xyz_trs[i,2], xyz_trs[permutation_ind[id_next],2] = xyz_trs[permutation_ind[id_next],2], xyz_trs[i,2]
            #xyz_trs[i,:], xyz_trs[permutation_ind[id_next],:] = xyz_trs[permutation_ind[id_next],:], xyz_trs[i,:]
            
            temp = permutation_ind[id_next]
            
 
 
            #// Subtract n from an entry in P
            #// to make it negative which indicates
            #// the corresponding move
            #// has been performed
            permutation_ind[id_next] -= num_tracers
            id_next = temp
        
    return ind_vox
            


@jit(nopython=True)
def order_coord_tracers_in_voxels_copy(
    xyz_trs, ngrid, Lbox):

    num_tracers = xyz_trs.shape[0]
    n_trs_vox = np.zeros(ngrid*ngrid*ngrid,dtype=np.int64) # array containing the number of tracer for each voxel
    xyz_trs_out = np.empty(xyz_trs.shape,dtype=xyz_trs.dtype)

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

        n_trs_vox[id_grid] -= 1
        
    return xyz_trs_out, ind_vox


@jit(nopython=True)
def order_coord_ids_tracers_in_voxels_permutation(
    xyz_trs, id_trs, ngrid, Lbox):

    num_tracers = xyz_trs.shape[0]
    n_trs_vox = np.zeros(ngrid*ngrid*ngrid,dtype=np.int64) # array containing the number of tracer for each voxel
    permutation_ind = np.empty(num_tracers,dtype=np.int64)

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
        permutation_ind[i_tr] = ind_vox[id_grid]  + n_trs_vox[id_grid] - 1

        n_trs_vox[id_grid] -= 1
    
    
    for i in range(num_tracers):
        id_next = i
        #temp = 0
        while (permutation_ind[id_next] >= 0):
 
            #// Swap the current element according
            #// to the permutation in P
            xyz_trs[i,0], xyz_trs[permutation_ind[id_next],0] = xyz_trs[permutation_ind[id_next],0], xyz_trs[i,0]
            xyz_trs[i,1], xyz_trs[permutation_ind[id_next],1] = xyz_trs[permutation_ind[id_next],1], xyz_trs[i,1]
            xyz_trs[i,2], xyz_trs[permutation_ind[id_next],2] = xyz_trs[permutation_ind[id_next],2], xyz_trs[i,2]
            #xyz_trs[i,:], xyz_trs[permutation_ind[id_next],:] = xyz_trs[permutation_ind[id_next],:], xyz_trs[i,:]
            
            id_trs[i], id_trs[permutation_ind[id_next]] = id_trs[permutation_ind[id_next]], id_trs[i]

            
            temp = permutation_ind[id_next]
            
 
 
            #// Subtract n from an entry in P
            #// to make it negative which indicates
            #// the corresponding move
            #// has been performed
            permutation_ind[id_next] -= num_tracers
            id_next = temp
        
    return ind_vox



@jit(nopython=True)
def order_coord_ids_tracers_in_voxels_copy(
    xyz_trs, id_trs, ngrid, Lbox):

    num_tracers = xyz_trs.shape[0]
    n_trs_vox = np.zeros(ngrid*ngrid*ngrid,dtype=np.int64) # array containing the number of tracer for each voxel
    xyz_trs_out = np.empty(xyz_trs.shape,dtype=xyz_trs.dtype)
    id_trs_out = np.empty(num_tracers,dtype=id_trs.dtype)

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
        id_trs_out[id_new] = id_trs[i_tr]

        n_trs_vox[id_grid] -= 1
        
    return xyz_trs_out, id_trs_out, ind_vox



@jit(nopython=True)
def order_ids_tracers_in_voxels_copy(
    xyz_trs, id_trs, ngrid, Lbox):

    num_tracers = xyz_trs.shape[0]
    n_trs_vox = np.zeros(ngrid*ngrid*ngrid,dtype=np.int64) # array containing the number of tracer for each voxel
    
    #xyz_trs_out = np.empty(xyz_trs.shape,dtype=xyz_trs.dtype)
    id_trs_out = np.empty(num_tracers,dtype=id_trs.dtype)

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
        id_trs_out[id_new] = id_trs[i_tr]

        n_trs_vox[id_grid] -= 1
        
    return id_trs_out, ind_vox
