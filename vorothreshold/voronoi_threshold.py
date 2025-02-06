import numpy as np
import os
import time
from numba.core import types
from numba.typed import Dict
from numba import jit, prange, set_num_threads, get_num_threads, get_thread_id
from . utilities import StrHminSec

int_array = types.int64[::1]


@jit(nopython=True)
def is_not_in_arr(ar1,ar2):
    mask = np.ones(len(ar1), dtype=np.bool_)
    for a in ar2:
        mask &= (ar1 != a)
    return mask


@jit(nopython=True)
def is_in_arr(ar1,ar2):
    mask = np.zeros(len(ar1), dtype=np.bool_)
    for a in ar2:
        mask |= (ar1 == a)
    return mask


@jit(nopython=True)
def cluster_accretion(IDthresholds,ID_core,numPart,neighbor_ptr,neighbor_ids,Nthresholds,threshold,VoroXYZ,VoroVol,zDens):
    #print(i_progr,flush=True)
    
    IDnext = ID_core
    IDanchor = IDnext
    

    #IDthresholds = np.zeros(numPart,dtype=np.int_)
    ID_to_explore = np.zeros(numPart,dtype=np.int_)
    ID_cluster = np.zeros(numPart,dtype=np.int_)
    Ncells_in_void = np.zeros(Nthresholds)
    Vol_interp = np.zeros(Nthresholds)
    Xcm_interp = np.zeros((Nthresholds,3))
    eigenvalues = np.zeros((Nthresholds,3))
    eigenvectors = np.zeros((Nthresholds,3,3))
    shape_matrix = np.zeros((3,3))


    Ncells = 0
    ID_to_explore[0] = IDanchor
    ID_cluster[0] = IDnext

    IDthresholds[Ncells] = IDnext
    Ncells = 1
    Nneighbors = 1
    VolTot = VoroVol[IDnext] 
    
    numerator_dens = 1. / zDens[IDnext]
    Dens = numerator_dens / VolTot
    Xcm = VoroXYZ[IDnext,:] * VoroVol[IDnext] / zDens[IDnext] #np.zeros(3)
    Norm_cm = VoroVol[IDnext] / zDens[IDnext] #np.zeros(3)

    for ith in range(Nthresholds):
        Condition = (Dens <= threshold[ith]) & (Ncells < numPart) #(Ncells < numPart-1)
        #print(ith,Ncells,Condition,numPart-1)
        while Condition:
            ## Add neighbor particles and update ID_to_explore:
            #VtoCluster = np.isin(neighbor_ids[neighbor_ptr[IDanchor]:neighbor_ptr[IDanchor+1]],ID_to_explore[:Nneighbors],assume_unique=True,invert=True)
            #VtoCluster &= np.isin(neighbor_ids[neighbor_ptr[IDanchor]:neighbor_ptr[IDanchor+1]],IDthresholds[:Ncells+1],assume_unique=True,invert=True)
            VtoCluster = is_not_in_arr(neighbor_ids[neighbor_ptr[IDanchor]:neighbor_ptr[IDanchor+1]],ID_to_explore[:Nneighbors])
            VtoCluster &= is_not_in_arr(neighbor_ids[neighbor_ptr[IDanchor]:neighbor_ptr[IDanchor+1]],IDthresholds[:Ncells+1])
            NtoAdd = np.sum(VtoCluster)
            ID_to_explore[Nneighbors:Nneighbors+NtoAdd] = neighbor_ids[neighbor_ptr[IDanchor]:neighbor_ptr[IDanchor+1]][VtoCluster]
            #for ii in ID_to_explore[Nneighbors:Nneighbors+NtoAdd]:
            #    ax.plot(VoroXYZ[[IDanchor,ii],0],VoroXYZ[[IDanchor,ii],1],VoroXYZ[[IDanchor,ii],2],lw=1,c=CC)

            Cond_innert = True
            ID_to_explore[Nneighbors:Nneighbors+NtoAdd] = ID_to_explore[Nneighbors:Nneighbors+NtoAdd][
                (np.argsort(VoroVol[ID_to_explore[Nneighbors:Nneighbors+NtoAdd]] * zDens[ID_to_explore[Nneighbors:Nneighbors+NtoAdd]])[::-1])]
            inner_progr = 0
            while Cond_innert:
                IDnext = ID_to_explore[Nneighbors+inner_progr]
                #ax.scatter(VoroXYZ[IDnext,0],VoroXYZ[IDnext,1],VoroXYZ[IDnext,2],c=CC)
                IDthresholds[Ncells] = IDnext
                Ncells += 1
                numerator_dens += 1. / zDens[IDnext]
                VolTot += VoroVol[IDnext]
                #NormVolTot += VoroVol[IDnext] * zDens[IDnext]
                #Dens = Ncells / VolTot / zDens[IDnext]
                #Dens = Ncells / VolTot / zDens_centr
                #Dens = Ncells / NormVolTot
                Dens = numerator_dens / VolTot
                Xcm += VoroXYZ[IDnext,:] * VoroVol[IDnext] / zDens[IDnext]
                Norm_cm += VoroVol[IDnext] / zDens[IDnext]

                #print(ith,Ncells,inner_progr,Dens,IDnext,Nneighbors,VoroVol[IDnext] * zDens[IDnext],NormVolTot,VoroVol[IDnext],VolTot)
                #print(Ncells,Dens, Ncells / VolTot / zDens[IDnext])
                inner_progr += 1
                Cond_innert = (Dens <= threshold[ith]) & (Ncells < numPart) & (inner_progr < NtoAdd)


            Nneighbors += NtoAdd

            Ichange = np.argwhere(ID_to_explore[:Nneighbors] == IDanchor)[0,0]
            ID_to_explore[Ichange:Nneighbors-1] = ID_to_explore[Ichange+1:Nneighbors] 
            Nneighbors -= 1

            ## find new IDnext as the lowest norm density among neighbor cells
            IDanchor = ID_to_explore[:Nneighbors][np.argmax(VoroVol[ID_to_explore[:Nneighbors]]* zDens[ID_to_explore[:Nneighbors]])]
    
            Condition = (Dens <= threshold[ith]) & (Ncells < numPart)
            #out_progr += 1

        if Ncells < 2:        
            return Xcm_interp, Vol_interp, Ncells_in_void, eigenvalues, eigenvectors



        #if (Dens >= threshold[ith]): 
        VolPrevious = VolTot-VoroVol[IDthresholds[Ncells-1]]
        numerator_dens_previous = numerator_dens - 1./zDens[IDthresholds[Ncells-1]]
        frac = (threshold[ith] * VolPrevious - numerator_dens_previous) / (1. / zDens[IDthresholds[Ncells-1]] - threshold[ith] * VoroVol[IDthresholds[Ncells-1]])
        #if frac == 0:
        #    print(ID_core,'Ncells:',Ncells,'VolPrevious:',VolPrevious,'VolTot:',VolTot,'VolLast:',VoroVol[IDthresholds[Ncells-1]],
        #          'numerator_dens_previous:',numerator_dens_previous,'numerator_dens:',numerator_dens,'numerator_dens_last:',1./zDens[IDthresholds[Ncells-1]],flush=True)
        Vol_interp[ith] = VolPrevious + frac * VoroVol[IDthresholds[Ncells-1]]
        Ncells_in_void[ith] = Ncells-1+frac

        Coord_norm = np.sum(1. / (VoroVol[IDthresholds[:Ncells-1]] * zDens[IDthresholds[:Ncells-1]])) +  1. / (frac * VoroVol[IDthresholds[Ncells-1]] * zDens[IDthresholds[Ncells-1]])
        Xcm_interp[ith,0] = (np.sum(VoroXYZ[IDthresholds[:Ncells-1],0] / (VoroVol[IDthresholds[:Ncells-1]] * zDens[IDthresholds[:Ncells-1]])) + 
                            VoroXYZ[IDthresholds[Ncells-1],0] / (frac * VoroVol[IDthresholds[Ncells-1]] * zDens[IDthresholds[Ncells-1]])) / Coord_norm
        Xcm_interp[ith,1] = (np.sum(VoroXYZ[IDthresholds[:Ncells-1],1] / (VoroVol[IDthresholds[:Ncells-1]] * zDens[IDthresholds[:Ncells-1]])) + 
                            VoroXYZ[IDthresholds[Ncells-1],1] / (frac * VoroVol[IDthresholds[Ncells-1]] * zDens[IDthresholds[Ncells-1]])) / Coord_norm
        Xcm_interp[ith,2] = (np.sum(VoroXYZ[IDthresholds[:Ncells-1],2] / (VoroVol[IDthresholds[:Ncells-1]] * zDens[IDthresholds[:Ncells-1]])) + 
                            VoroXYZ[IDthresholds[Ncells-1],2] / (frac * VoroVol[IDthresholds[Ncells-1]] * zDens[IDthresholds[Ncells-1]])) / Coord_norm
        
        # geometrical shape
        # density field
        for i in range(3):
            shape_matrix[i,i] = (np.sum((VoroXYZ[IDthresholds[:Ncells-1],(i+1) % 3]**2 + VoroXYZ[IDthresholds[:Ncells-1],(i+2) % 3]**2) / zDens[IDthresholds[:Ncells-1]]) + 
                                    (VoroXYZ[IDthresholds[Ncells-1],(i+1) % 3]**2 + VoroXYZ[IDthresholds[Ncells-1],(i+2) % 3]**2)  * frac / zDens[IDthresholds[Ncells-1]])
            for j in range(i):
                shape_matrix[i,j] = -(np.sum(VoroXYZ[IDthresholds[:Ncells-1],i] * VoroXYZ[IDthresholds[:Ncells-1],j] / zDens[IDthresholds[:Ncells-1]]) + 
                                        VoroXYZ[IDthresholds[Ncells-1],i] * VoroXYZ[IDthresholds[Ncells-1],j] * frac / zDens[IDthresholds[Ncells-1]])
                shape_matrix[j,i] = shape_matrix[i,j]

        eigenvalues[ith,:], eigenvectors[ith,:,:] = np.linalg.eig(shape_matrix)
    #IDthresholds[:Ncells]
    return Xcm_interp, Vol_interp, Ncells_in_void, eigenvalues, eigenvectors



@jit(nopython=True)
def cluster_accretion(
    Xcm_interp, Vol_interp, Ncells_in_void, eigenvalues, eigenvectors,IDvoro_in_void, # vectors/dicts to update
    ID_core,numPart,neighbor_ptr,neighbor_ids,Nthresholds,threshold,VoroXYZ,VoroVol,zDens):
    #print(i_progr,flush=True)
    
    IDnext = ID_core
    IDanchor = IDnext
    
    #IDvoro_in_void = np.zeros(numPart,dtype=np.int_)
    ID_to_explore = np.zeros(numPart,dtype=np.int_)
    ID_cluster = np.zeros(numPart,dtype=np.int_)
    #Ncells_in_void = np.zeros(Nthresholds)
    #Vol_interp = np.zeros(Nthresholds)
    #Xcm_interp = np.zeros((Nthresholds,3))
    #eigenvalues = np.zeros((Nthresholds,3))
    #eigenvectors = np.zeros((Nthresholds,3,3))
    shape_matrix = np.zeros((3,3))


    Ncells = 0
    ID_to_explore[0] = IDanchor
    ID_cluster[0] = IDnext

    IDvoro_in_void[Ncells] = IDnext
    Ncells = 1
    Nneighbors = 1
    VolTot = VoroVol[IDnext] 
    
    numerator_dens = 1. / zDens[IDnext]
    Dens = numerator_dens / VolTot
    Xcm = VoroXYZ[IDnext,:] * VoroVol[IDnext] / zDens[IDnext] #np.zeros(3)
    Norm_cm = VoroVol[IDnext] / zDens[IDnext] #np.zeros(3)

    for ith in range(Nthresholds):
        Condition = (Dens <= threshold[ith]) & (Ncells < numPart) #(Ncells < numPart-1)
        #print(ith,Ncells,Condition,numPart-1)
        while Condition:
            ## Add neighbor particles and update ID_to_explore:
            VtoCluster = is_not_in_arr(neighbor_ids[neighbor_ptr[IDanchor]:neighbor_ptr[IDanchor+1]],ID_to_explore[:Nneighbors])
            VtoCluster &= is_not_in_arr(neighbor_ids[neighbor_ptr[IDanchor]:neighbor_ptr[IDanchor+1]],IDvoro_in_void[:Ncells+1])
            NtoAdd = np.sum(VtoCluster)
            ID_to_explore[Nneighbors:Nneighbors+NtoAdd] = neighbor_ids[neighbor_ptr[IDanchor]:neighbor_ptr[IDanchor+1]][VtoCluster]

            Cond_innert = True
            ID_to_explore[Nneighbors:Nneighbors+NtoAdd] = ID_to_explore[Nneighbors:Nneighbors+NtoAdd][
                (np.argsort(VoroVol[ID_to_explore[Nneighbors:Nneighbors+NtoAdd]] * zDens[ID_to_explore[Nneighbors:Nneighbors+NtoAdd]])[::-1])]
            inner_progr = 0
            while Cond_innert:
                IDnext = ID_to_explore[Nneighbors+inner_progr]

                IDvoro_in_void[Ncells] = IDnext
                Ncells += 1
                numerator_dens += 1. / zDens[IDnext]
                VolTot += VoroVol[IDnext]
                
                Dens = numerator_dens / VolTot
                Xcm += VoroXYZ[IDnext,:] * VoroVol[IDnext] / zDens[IDnext]
                Norm_cm += VoroVol[IDnext] / zDens[IDnext]

                inner_progr += 1
                Cond_innert = (Dens <= threshold[ith]) & (Ncells < numPart) & (inner_progr < NtoAdd)


            Nneighbors += NtoAdd

            Ichange = np.argwhere(ID_to_explore[:Nneighbors] == IDanchor)[0,0]
            ID_to_explore[Ichange:Nneighbors-1] = ID_to_explore[Ichange+1:Nneighbors] 
            Nneighbors -= 1

            ## find new IDnext as the lowest norm density among neighbor cells
            IDanchor = ID_to_explore[:Nneighbors][np.argmax(VoroVol[ID_to_explore[:Nneighbors]]* zDens[ID_to_explore[:Nneighbors]])]
    
            Condition = (Dens <= threshold[ith]) & (Ncells < numPart)
            #out_progr += 1

        if Ncells < 2:     
            continue   
            #return Xcm_interp, Vol_interp, Ncells_in_void, eigenvalues, eigenvectors



        #if (Dens >= threshold[ith]): 
        VolPrevious = VolTot-VoroVol[IDvoro_in_void[Ncells-1]]
        numerator_dens_previous = numerator_dens - 1./zDens[IDvoro_in_void[Ncells-1]]
        frac = (threshold[ith] * VolPrevious - numerator_dens_previous) / (1. / zDens[IDvoro_in_void[Ncells-1]] - threshold[ith] * VoroVol[IDvoro_in_void[Ncells-1]])
        #if frac == 0:
        #    print(ID_core,'Ncells:',Ncells,'VolPrevious:',VolPrevious,'VolTot:',VolTot,'VolLast:',VoroVol[IDvoro_in_void[Ncells-1]],
        #          'numerator_dens_previous:',numerator_dens_previous,'numerator_dens:',numerator_dens,'numerator_dens_last:',1./zDens[IDvoro_in_void[Ncells-1]],flush=True)
        Vol_interp[ith] = VolPrevious + frac * VoroVol[IDvoro_in_void[Ncells-1]]
        Ncells_in_void[ith] = Ncells-1+frac

        Coord_norm = np.sum(1. / (VoroVol[IDvoro_in_void[:Ncells-1]] * zDens[IDvoro_in_void[:Ncells-1]])) +  1. / (frac * VoroVol[IDvoro_in_void[Ncells-1]] * zDens[IDvoro_in_void[Ncells-1]])
        Xcm_interp[ith,0] = (np.sum(VoroXYZ[IDvoro_in_void[:Ncells-1],0] / (VoroVol[IDvoro_in_void[:Ncells-1]] * zDens[IDvoro_in_void[:Ncells-1]])) + 
                            VoroXYZ[IDvoro_in_void[Ncells-1],0] / (frac * VoroVol[IDvoro_in_void[Ncells-1]] * zDens[IDvoro_in_void[Ncells-1]])) / Coord_norm
        Xcm_interp[ith,1] = (np.sum(VoroXYZ[IDvoro_in_void[:Ncells-1],1] / (VoroVol[IDvoro_in_void[:Ncells-1]] * zDens[IDvoro_in_void[:Ncells-1]])) + 
                            VoroXYZ[IDvoro_in_void[Ncells-1],1] / (frac * VoroVol[IDvoro_in_void[Ncells-1]] * zDens[IDvoro_in_void[Ncells-1]])) / Coord_norm
        Xcm_interp[ith,2] = (np.sum(VoroXYZ[IDvoro_in_void[:Ncells-1],2] / (VoroVol[IDvoro_in_void[:Ncells-1]] * zDens[IDvoro_in_void[:Ncells-1]])) + 
                            VoroXYZ[IDvoro_in_void[Ncells-1],2] / (frac * VoroVol[IDvoro_in_void[Ncells-1]] * zDens[IDvoro_in_void[Ncells-1]])) / Coord_norm
        
        # geometrical shape
        for i in range(3):
            shape_matrix[i,i] = (np.sum((VoroXYZ[IDvoro_in_void[:Ncells-1],(i+1) % 3]**2 + VoroXYZ[IDvoro_in_void[:Ncells-1],(i+2) % 3]**2) / zDens[IDvoro_in_void[:Ncells-1]]) + 
                                    (VoroXYZ[IDvoro_in_void[Ncells-1],(i+1) % 3]**2 + VoroXYZ[IDvoro_in_void[Ncells-1],(i+2) % 3]**2)  * frac / zDens[IDvoro_in_void[Ncells-1]])
            for j in range(i):
                shape_matrix[i,j] = -(np.sum(VoroXYZ[IDvoro_in_void[:Ncells-1],i] * VoroXYZ[IDvoro_in_void[:Ncells-1],j] / zDens[IDvoro_in_void[:Ncells-1]]) + 
                                        VoroXYZ[IDvoro_in_void[Ncells-1],i] * VoroXYZ[IDvoro_in_void[Ncells-1],j] * frac / zDens[IDvoro_in_void[Ncells-1]])
                shape_matrix[j,i] = shape_matrix[i,j]

        eigenvalues[ith,:], eigenvectors[ith,:,:] = np.linalg.eig(shape_matrix)
    #IDvoro_in_void[:Ncells]




@jit(nopython=True)
def cluster_accretion_loops_sequential(ID_voro_dict, threshold_arr,ID_core_arr,neighbor_ptr,neighbor_ids,VoroXYZ,VoroVol,zDens,max_num_part):
    Nthresholds = threshold_arr.shape[0]
    Num_vds = ID_core_arr.shape[0]
    #numPart = VoroXYZ.shape[0]
    
    Xcm = np.zeros((Num_vds,Nthresholds,3))
    Vol_interp = np.zeros((Num_vds,Nthresholds))
    Ncells_in_void = np.zeros((Num_vds,Nthresholds))
    ell_eigenvalues = np.zeros((Num_vds,Nthresholds,3))
    ell_eigenvectors = np.zeros((Num_vds,Nthresholds,3,3))

    for iv in range(Num_vds):
        ID_voro_dict[iv] = np.zeros(max_num_part,dtype=np.int_)
        cluster_accretion(Xcm[iv,:,:], Vol_interp[iv,:], Ncells_in_void[iv,:], ell_eigenvalues[iv,:,:], ell_eigenvectors[iv,:,:,:],
                          ID_voro_dict[iv],ID_core_arr[iv],max_num_part,neighbor_ptr,neighbor_ids,Nthresholds,threshold_arr,VoroXYZ,VoroVol,zDens)
        Ncells_loop = int(Ncells_in_void[iv,-1]) + int((Ncells_in_void[iv,-1]%1) > 0)
        ID_voro_dict[iv] = ID_voro_dict[iv][:Ncells_loop]
    return Xcm, Vol_interp, Ncells_in_void, ell_eigenvalues, ell_eigenvectors


@jit(nopython=True,parallel=True)
def cluster_accretion_loops_parallel(ID_voro_dict, threshold_arr,ID_core_arr,neighbor_ptr,neighbor_ids,VoroXYZ,VoroVol,zDens,max_num_part):
    Nthresholds = threshold_arr.shape[0]
    Num_vds = ID_core_arr.shape[0]
    #numPart = VoroXYZ.shape[0]
    
    Xcm = np.zeros((Num_vds,Nthresholds,3))
    Vol_interp = np.zeros((Num_vds,Nthresholds))
    Ncells_in_void = np.zeros((Num_vds,Nthresholds))
    ell_eigenvalues = np.zeros((Num_vds,Nthresholds,3))
    ell_eigenvectors = np.zeros((Num_vds,Nthresholds,3,3))

    for iv in prange(Num_vds):
        cluster_accretion(Xcm[iv,:,:], Vol_interp[iv,:], Ncells_in_void[iv,:], ell_eigenvalues[iv,:,:], ell_eigenvectors[iv,:,:,:],
                          ID_voro_dict[iv],ID_core_arr[iv],max_num_part,neighbor_ptr,neighbor_ids,Nthresholds,threshold_arr,VoroXYZ,VoroVol,zDens)
        #Xcm[iv,:,:], Vol_interp[iv,:], Ncells_in_void[iv,:], ell_eigenvalues[iv,:,:], ell_eigenvectors[iv,:,:,:] = \
        #    cluster_accretion(ID_voro_dict[iv],ID_core_arr[iv],max_num_part,neighbor_ptr,neighbor_ids,Nthresholds,threshold_arr,VoroXYZ,VoroVol,zDens)
    return Xcm, Vol_interp, Ncells_in_void, ell_eigenvalues, ell_eigenvectors




def voronoi_threshold(threshold,ID_core_arr,neighbor_ptr,neighbor_ids,VoroXYZ,VoroVol,zDens,nthreads=-1,verbose=True,max_num_part=-1):

    verboseprint = print if verbose else lambda *a, **k: None
    verboseprint('\n    voronoi_threshold started',flush=True)
    try:
        nthreads_tot = int(os.environ["OMP_NUM_THREADS"])
    except:
        nthreads_tot = get_num_threads()

    if (nthreads <= 0) | (nthreads > nthreads_tot):
        nthreads = nthreads_tot

    Num_vds = ID_core_arr.shape[0]
    set_num_threads(min(nthreads,Num_vds))

    verboseprint('\n    nthreads set to',nthreads,flush=True)


    if np.isscalar(threshold):
        threshold_arr = np.array([threshold])
    else:
        threshold_arr = np.array(threshold)
    threshold_arr = np.sort(threshold_arr)


    
    # maximum possible number of particles
    if max_num_part < 1:
        max_num_part = int(VoroXYZ.shape[0]*np.max(threshold_arr))
    verboseprint('\n    max_num_part set to',max_num_part,flush=True)

    Num_vds = ID_core_arr.shape[0]
    input_mask = np.arange(Num_vds)[(1. / (VoroVol[ID_core_arr] * zDens[ID_core_arr])) <= np.max(threshold_arr)]
    Num_selection = input_mask.shape[0]

    ID_voro_dict = Dict.empty(
        key_type=types.int64,
        value_type=int_array)
    
    t0 = time.time()
    verboseprint('\n    computation started',flush=True)
    if nthreads > 1:
        for i_sel in range(Num_selection):
            ID_voro_dict[i_sel] = np.zeros(max_num_part,dtype=np.int_)
        Xcm, Vol_interp, Ncells_in_void, ell_eigenvalues, ell_eigenvectors = cluster_accretion_loops_parallel(
            ID_voro_dict, threshold_arr,ID_core_arr[input_mask],neighbor_ptr,neighbor_ids,VoroXYZ,VoroVol,zDens,max_num_part)
    else:
        Xcm, Vol_interp, Ncells_in_void, ell_eigenvalues, ell_eigenvectors = cluster_accretion_loops_sequential(
            ID_voro_dict, threshold_arr,ID_core_arr[input_mask],neighbor_ptr,neighbor_ids,VoroXYZ,VoroVol,zDens,max_num_part)

    dt = time.time() - t0
    verboseprint("    done,",StrHminSec(dt),'\n',flush=True)

    if nthreads_tot != get_num_threads():
        set_num_threads(nthreads_tot)
        
    return input_mask, ID_voro_dict, Xcm, Vol_interp, Ncells_in_void, ell_eigenvalues, ell_eigenvectors

