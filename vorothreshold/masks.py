import numpy as np
import healpy as hp
from numba import  jit, prange, set_num_threads, get_num_threads
import os
from . overlaps import compute_max_dist2



def borders_mask_bruteforce(RAvoro,DECvoro,Ncells,ID_voro_dict,mask_pix):

    #npix = hp.nside2npix(nside)
    #mask_pix = np.zeros((npix))
    #phi = np.pi/180. * RAvoro
    #theta = np.pi/2. - DECvoro*np.pi/180.
    #pix = hp.ang2pix(nside, theta, phi)
    #mask_pix[pix] = 1.


    nside = hp.get_nside(mask_pix)
    npix = hp.nside2npix(nside)

    id_selected = np.arange(Ncells.shape[0])[Ncells >= 1]
    mask_vds = np.ones(id_selected.shape[0],dtype=np.bool_)
    #mask_vds[id_selected] = True
    for i in range(id_selected.shape[0]):
        iv = id_selected[i]
        Ncells_loop = int(Ncells[iv]) + int((Ncells[iv]%1) > 0)
        phi_voro = np.pi/180. * RAvoro[ID_voro_dict[iv][:Ncells_loop]]
        theta_voro = np.pi/2. - DECvoro[ID_voro_dict[iv][:Ncells_loop]] * np.pi/180.
        pix = hp.ang2pix(nside, theta_voro, phi_voro)
        for ii in pix:
            i_neigh = hp.get_all_neighbours(nside,ii)
            if np.sum(mask_pix[i_neigh]) < i_neigh.shape[0]:
                mask_vds[i] = False
                break
    return id_selected[mask_vds]



def ids_pix_noborder(mask_pix_bool,theta_voro,phi_voro,padding_npix):
    nside = hp.get_nside(mask_pix_bool)
    npix = hp.nside2npix(nside)
    #mask_pix_border = np.zeros(npix,dtype=np.bool_)
    mask_pix_border = np.copy(mask_pix_bool).astype(np.bool_)
    #for ipix in np.arange(npix)[mask_pix_bool.astype(np.bool_)]:
    #    i_neigh = hp.get_all_neighbours(nside,ipix)
    #    mask_pix_border[ipix] = (np.sum(mask_pix_bool[i_neigh]) == i_neigh.shape[0])
    for progr in range(padding_npix):
        mask_pix_border_old = np.copy(mask_pix_border)
        for ipix in np.arange(npix)[mask_pix_border_old]:
            i_neigh = hp.get_all_neighbours(nside,ipix)
            mask_pix_border[ipix] = (np.sum(mask_pix_border_old[i_neigh]) == i_neigh.shape[0])


    return mask_pix_border[hp.ang2pix(nside, theta_voro, phi_voro)], mask_pix_border


def ids_pix_noborder_RADEC(mask_pix_bool,RAvoro,DECvoro,padding_npix):
    phi_voro = np.pi/180. * RAvoro
    theta_voro = np.pi/2. - DECvoro*np.pi/180.
    return ids_pix_noborder(mask_pix_bool,theta_voro,phi_voro,padding_npix)


@jit(nopython=True,parallel=True)
def borders_mask_inner(mask_voro,ID_voro_dict,Ncells):
    # Remove non-voids
    id_selected = np.arange(Ncells.shape[0])[Ncells >= 1]
    mask_vds = np.ones(id_selected.shape[0],dtype=np.bool_)

    for i in prange(id_selected.shape[0]):
        iv = id_selected[i]
        Ncells_loop = int(Ncells[iv]) + int((Ncells[iv]%1) > 0)
        mask_vds[i] = np.all(mask_voro[ID_voro_dict[iv][:Ncells_loop]])
    return id_selected[mask_vds]


def borders_mask(mask_pix_bool,RAvoro,DECvoro,ID_voro_dict,Ncells,padding_npix,nthreads=-1):

    try:
        nthreads_tot = int(os.environ["OMP_NUM_THREADS"])
    except:
        nthreads_tot = get_num_threads()

    if (nthreads <= 0) | (nthreads > nthreads_tot):
        nthreads = nthreads_tot
    else:
        set_num_threads(nthreads)

    mask_voro, mask_pix_border = ids_pix_noborder_RADEC(mask_pix_bool,RAvoro,DECvoro,padding_npix)
    id_selected = borders_mask_inner(mask_voro,ID_voro_dict,Ncells)

    if nthreads != nthreads_tot:
        set_num_threads(nthreads_tot)

    return id_selected, mask_voro, mask_pix_border


def borders_mask_OLD(xyz_cm,max_ang_dist,RAvoro,DECvoro,Ncells,ID_voro_dict,nside):

    npix = hp.nside2npix(nside)
    mask_pix = np.zeros((npix))
    phi = np.pi/180. * RAvoro
    theta = np.pi/2. - DECvoro*np.pi/180.
    pix = hp.ang2pix(nside, theta, phi)
    mask_pix[pix] = 1.

    dist_cm = np.sum(np.square(xyz_cm),axis=1)
    

    mask_vds = np.zeros(xyz_cm.shape[0],dtype=np.bool_)
    id_selected = np.arange(Ncells.shape[0])[Ncells >= 1]
    mask_vds[id_selected] = True
    for i in range(id_selected.shape[0]):
        #print(i)

        iv = id_selected[i]
        
        pix_outer = hp.query_disc(nside, xyz_cm[iv,[1,0,2]] / dist_cm[iv], max_ang_dist[iv], inclusive=True, fact=4, nest=False, buff=None)
        if 0 in mask_pix[pix_outer]:
            Ncells_loop = int(Ncells[iv]) + int((Ncells[iv]%1) > 0)
            phi_voro = np.pi/180. * RAvoro[ID_voro_dict[iv][:Ncells_loop]]
            theta_voro = np.pi/2. - DECvoro[ID_voro_dict[iv][:Ncells_loop]] * np.pi/180.
            pix = hp.ang2pix(nside, theta_voro, phi_voro)
            for ii in pix:
                i_neigh = hp.get_all_neighbours(nside,ii)
                #if np.sum(mask_pix[i_neigh]) < i_neigh.shape[0]:
                if 0 in i_neigh:
                    mask_vds[iv] = False
                    break
    return mask_vds



@jit(nopython=True)
def dist_limit_mask(id_selected,xyz_cm,dist_min,dist_max,xyz_voro,Ncells,ID_voro_dict):

    max_dist_from_cm = compute_max_dist2(Ncells,xyz_cm,xyz_voro,id_selected,ID_voro_dict) ** 0.5

    dist_cm = np.sqrt(np.sum(np.square(xyz_cm[id_selected,:]),axis=1))
    dist_max2 = dist_max * dist_max
    dist_min2 = dist_min * dist_min
    mask_out = np.ones(id_selected.shape[0],dtype=np.bool_)
    
    for i in range(id_selected.shape[0]):
        iv = id_selected[i]
        if (dist_cm[i] + max_dist_from_cm[iv]) > dist_max:
            Ncells_loop = int(Ncells[iv]) + int((Ncells[iv]%1) > 0)
            ii = 0
            while (ii < Ncells_loop) & mask_out[i]:
                mask_out[i] = np.sum(np.square(xyz_voro[ID_voro_dict[iv][ii],:])) < dist_max2
                ii += 1
                #if np.sum(np.square(xyz_voro[ID_voro_dict[iv][ii],:])) > dist_max2:
                #    mask_out[i] = False

        if (dist_cm[i] - max_dist_from_cm[iv]) < dist_min:
            Ncells_loop = int(Ncells[iv]) + int((Ncells[iv]%1) > 0)
            ii = 0
            while (ii < Ncells_loop) & mask_out[i]:
                mask_out[i] = np.sum(np.square(xyz_voro[ID_voro_dict[iv][ii],:])) > dist_min2
                ii += 1
            #for ii in range(Ncells_loop):
            #    if np.sum(np.square(xyz_voro[ID_voro_dict[iv][ii],:])) < dist_min2:
            #        mask_out[i] = False
        #else:
        #    print((dist_cm[i] - max_dist_from_cm[iv]), dist_min)

    return id_selected[mask_out]




def compute_max_dist_deg(Ncells,XYZ_voids,XYZ_voro,Ids_voro_dict):

    dist_vds = np.sqrt(np.sum(np.square(XYZ_voids),axis=1))
    Nvoids = XYZ_voids.shape[0]
    dist_ang_max = np.zeros(Nvoids)
    #id_out = np.arange(Nvoids)[Ncells>=1]
    id_selected = np.arange(Ncells.shape[0])[Ncells >= 1]
    for iv in id_selected:
        #iv = id_selected[iv]

        Ncells_loop = int(Ncells[iv]) + int((Ncells[iv]%1) > 0)
        

        proj = np.sum(XYZ_voro[Ids_voro_dict[iv][:Ncells_loop],:] * XYZ_voids[iv],axis=1)/(
            np.sqrt(np.sum(XYZ_voro[Ids_voro_dict[iv][:Ncells_loop],:]**2,axis=1)) * dist_vds[iv])
        #print(iv,proj)

        dist_ang_max[iv] = np.arccos(np.min(proj))
    return dist_ang_max
