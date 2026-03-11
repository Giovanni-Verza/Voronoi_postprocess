import glob
import numpy as np
import os
import time
import healpy as hp
import joblib

from numba.core import types
from numba.typed import Dict
from numba import jit, prange, set_num_threads, get_num_threads, get_thread_id

from . read_funcs import read_adjfile, read_voronoi_vide, load_pickle_safe
from . masks import borders_mask_bruteforce, dist_limit_mask, borders_mask, borders_mask_inner
from . overlaps import select_overlaps, overlapping_fraction, center_in_void, closest_voro, select_overlaps_center_in_void
from . utilities import from_XYZ_to_rRAdec, from_rRAdec_to_XYZ, ComovingDistanceOverh, RedshiftFromComovingDistanceOverh, StrHminSec
from . voronoi_threshold import is_in_arr, voronoi_threshold


int_array = types.int64[::1]
int_array_2d = types.int64[:,::1]
float_array = types.float64[::1]
float_array_2d = types.float64[:,::1]




@jit(nopython=True,parallel=True)
def compute_overlaps_all_parallel_compiled(
    Nthresholds, Nfrac, frac_ovlp_arr, ids_selected, sor_by_vol, ids_ovlp, Vol_ovlp_frac, num_ovlps):

    Ncombo = Nthresholds * Nfrac


    len_ids_out = np.zeros(Ncombo,dtype=np.int64)
    
    id_out_dict = Dict.empty(
        key_type=types.int64,
        value_type=int_array)

    for ith in range(Nthresholds):
        for ifrac in range(Nfrac):
            id_out_dict[ith * Nfrac + ifrac] = np.empty(sor_by_vol[ith].shape[0],dtype=np.int64)

    for ii in prange(Ncombo):
        ith = int(ii / Nfrac)
        ifrac = ii - ith * Nfrac

        id_out_tmp = select_overlaps(frac_ovlp_arr[ifrac],ids_selected[ith],sor_by_vol[ith], ids_ovlp[ith], Vol_ovlp_frac[ith], num_ovlps[ith])

        len_ids_out[ii] = id_out_tmp.shape[0]
        id_out_dict[ii][:len_ids_out[ii]] = id_out_tmp
    
    return id_out_dict, len_ids_out


def compute_overlaps_all_parallel(
    ids_threshold,frac_ovlp_arr, Xcm, Vol_interp, Ncells_in_void, VoroXYZ, VoroVol, ID_voro_dict,Lbox,lightcone,ids_selected,nthreads,verbose):

    Nthresholds = ids_threshold.shape[0]
    Nfrac = frac_ovlp_arr.shape[0]

    Ncombo = Nthresholds * Nfrac

    ids_ovlp = Dict.empty(
        key_type=types.int64,
        value_type=int_array_2d)
    
    Vol_ovlp_frac = Dict.empty(
        key_type=types.int64,
        value_type=float_array_2d)
    
    num_ovlps = Dict.empty(
        key_type=types.int64,
        value_type=int_array)
    
    sor_by_vol = Dict.empty(
        key_type=types.int64,
        value_type=int_array)
        
    for ith in range(Nthresholds):
        #print(ith,flush=True)


        if len(ids_threshold[ith]) == 0:
            sor_by_vol[ith] = np.zeros(0,dtype=np.int64)
            #self.id_out[ith][frac_ovlp] = np.zeros(0,dtype=np.int64)

            ids_ovlp[ith] = np.zeros((0,0),dtype=np.int64)
            Vol_ovlp = np.zeros((0,0))
            Vol_ovlp_frac[ith] = np.zeros((0,0))
            num_ovlps[ith] = np.zeros(0,dtype=np.int64)

            continue


        ids_ovlp[ith], Vol_ovlp, Vol_ovlp_frac[ith], num_ovlps[ith] = overlapping_fraction(
            Xcm[:,ids_threshold[ith],:], Vol_interp[:,ids_threshold[ith]], Ncells_in_void[:,ids_threshold[ith]], VoroXYZ, VoroVol, ID_voro_dict,
            Lbox=Lbox,lightcone=lightcone,id_selected=ids_selected[ids_threshold[ith]],nthreads=nthreads,verbose=verbose)
        sor_by_vol[ith] = (np.argsort(Vol_interp[ids_selected[ith],ith])[::-1]).astype(dtype=np.int64, order='C')


    id_out_dict, len_ids_out = compute_overlaps_all_parallel_compiled(
        Nthresholds, Nfrac, frac_ovlp_arr, ids_selected, sor_by_vol, ids_ovlp, Vol_ovlp_frac, num_ovlps)
    
    return id_out_dict, len_ids_out
    
class voronoi_threshold_finder:
    def __init__(self,threshold,lightcone=True,Lbox=-1.,
                 ID_core=None,neighbor_ptr=None,neighbor_ids=None,VoroVol=None,
                 VoroXYZ=None,RAvoro=None,DECvoro=None,redshift_voro=None,dist_voro=None,
                 tracer_dens=None,npadding_ang=None,ang_paddig_rad=None,comov_range=None,z_range=None,
                 vide_path=None,OmegaM=None,w0=-1.,wa=0.,nthreads=-1,verbose=True,max_num_part=-1,
                 healpix_maskpath=None,healpix_mask=None,voro_border_mask=None,nside=128):
        
        #if verbose:
        #    verbose = True
        self.verbose = verbose
        self.z_from_dist = None
        self.__Lbox = Lbox
        self.__lightcone = lightcone

        self.VoroVol = VoroVol
        self.VoroXYZ = VoroXYZ
        self.RAvoro = RAvoro
        self.DECvoro = DECvoro
        self.vide_path = vide_path
        self.comov_range = comov_range
        self.z_range = z_range
        self.OmegaM = OmegaM
        self.w0 = w0
        self.wa = wa
        
        verboseprint = print if verbose else lambda *a, **k: None
        
        if nthreads <= 0:
            try:
                nthreads  = int(os.environ["OMP_NUM_THREADS"])
            except:
                nthreads  = get_num_threads()
        try:
            if nthreads > int(os.environ["OMP_NUM_THREADS"]):
                nthreads = int(os.environ["OMP_NUM_THREADS"])
        except:
            if nthreads  > get_num_threads():
                nthreads  = get_num_threads()

        self.nthreads = nthreads
 

        #if not lightcone:
        #    raise ValueError('Simulation box option has not been developed yet. This class currently works with lightcone=True option only.')
        #else:
        if lightcone & (tracer_dens is None):
            raise ValueError('tracer_dens not passed. When lightcone=True the number density of each tracer is required.')

        self.threshold = np.array(threshold).reshape(-1)

        if (vide_path is None):
            if ID_core is None:
                raise ValueError('ID_core not passed.')
            if self.VoroVol is None:
                raise ValueError('VoroVol not passed.')
            if (neighbor_ptr is None) | (neighbor_ids is None):
                raise ValueError('neighbor_ptr or neighbor_ids not passed.')
            if max_num_part <= 0:
                max_num_part = len(self.VoroVol) // len(ID_core)
                verboseprint('    max_num_part < 0: authomatically set to len(VoroVol) // len(ID_core):',max_num_part,flush=True)
                

            if lightcone:
                if ((self.RAvoro is None) | (self.DECvoro is None)): 
                    if (self.VoroXYZ is None):
                        raise ValueError('Voronoi coordinates not passed, pass either VoroXYZ, or RAvoro, DECvoro and dist_voro or redhsift voro, or both.')
                    print('RAvoro and DECvoro not passed, computed from VoroXYZ.',flush=True)
                    self.RAvoro, self.DECvoro, dist_voro = from_XYZ_to_rRAdec(VoroXYZ[:,0],VoroXYZ[:,1],VoroXYZ[:,2])
                if (tracer_dens is None):
                    raise ValueError('tracer_dens not passed.')
                if (self.VoroXYZ is None):

                    if dist_voro is None:
                        if redshift_voro is None:
                            raise ValueError('Voronoi coordinates not passed, pass either VoroXYZ, or RAvoro, DECvoro and dist_voro or redhsift_voro, or both.')
                        
                        if OmegaM is None:
                            print('OmegaM not passed. I will convert redhsift to comoving distance as r=c*z [km].',flush=True)
                            dist_voro = c_kms * redshift_voro
                        else:
                            dist_z = ComovingDistanceOverh(OmegaM,w0,wa)
                            dist_voro = dist_z.get_dist(redshift_voro)
                    self.VoroXYZ = np.array(from_rRAdec_to_XYZ(dist_voro,self.RAvoro,self.DECvoro)).T
            else:
                if (self.VoroXYZ is None):
                    raise ValueError('Voronoi coordinates not passed.')
                
                if (Lbox is None):
                    Lbox = np.max(VoroXYZ) - np.min(VoroXYZ)
                    raise Warning('Lbox not passed, set to max(VoroXYZ) - min(VoroXYZ)')
                self.__Lbox = Lbox
                verboseprint('    Lbox from VIDE:',self.__Lbox,flush=True)
                if (tracer_dens is None):
                    tracer_dens = np.full(self.VoroVol.shape[0],self.VoroVol.shape[0]/self.__Lbox**3) #self.__Lbox**3/self.VoroVol.shape[0]/self.VoroVol
                else:
                    raise Warning('tracer_dens passed differs from None, tracer_dens will not be computed from Vide output.')
                
            
        else:
            # load VTFE scheme from adjfile
            verboseprint('    Loading VIDE data.',flush=True)
            t0 = time.time()

            adjfile = glob.glob(vide_path+'/adj_*')[0] #vide_path + '/adj_' + vide_out_name + '.dat'
            neighbor_ptr, neighbor_ids = read_adjfile(adjfile)

            # recover vide_out_name
            vide_out_name = adjfile.split('adj_')[1].split('.dat')[0]
            #if ID_core is None:
            # Load ids of cells belonging to minima
            ID_core = np.loadtxt(vide_path+'/untrimmed_voidDesc_all_'+vide_out_name+'.out', comments='#', skiprows=2)[:,2].astype(np.int64)
            # Load Voronoi cells volume, ids and tracers position
            

            if lightcone:
                ids_voro, self.VoroVol, self.VoroXYZ, self.RAvoro, self.DECvoro, redshift_voro = read_voronoi_vide(vide_path,vide_out_name)

                # regularize VoroVol:
                self.VoroVol[self.VoroVol == 0] = 1e-25
                self.VoroVol[np.isnan(self.VoroVol)] = 1e-25
                self.VoroVol[np.isinf(self.VoroVol)] = 1e-25

                if OmegaM is None:
                    OmegaM = load_pickle_safe(vide_path+'/sample_info.dat')['omegaM']
                else:
                    OmegaM_VIDE = load_pickle_safe(vide_path+'/sample_info.dat')['omegaM']
                    if OmegaM != OmegaM_VIDE:
                        raise Warning('OmegaM passed differs from the value used for Vide. Passed: '+str(OmegaM)+', Vide:'+str(OmegaM_VIDE))
    
                
                if w0 != -1.:
                    raise Warning('w0 passed differs from the value used for Vide. Passed: '+str(w0)+', Vide: -1.0')
                if wa != 0.:
                    raise Warning('wa passed differs from the value used for Vide. Passed: '+str(wa)+', Vide: 0.0')

                dist_z = ComovingDistanceOverh(OmegaM,w0,wa)

                dist_voro = dist_z.get_dist(redshift_voro)
                self.VoroXYZ[:,:] = np.array(from_rRAdec_to_XYZ(dist_voro,self.RAvoro,self.DECvoro)).T

                del ids_voro, redshift_voro

                if healpix_mask is None:
                    if healpix_maskpath is None:
                        healpix_maskpath = vide_path + '/mask_map.fits'
                    else:
                        raise Warning('Healpix mask at '+str(healpix_maskpath)+' will be used instead of the vide mask.')
                else:
                    raise Warning('Healpix mask passed. It  will be used instead of the vide mask.')
                
                
            else:
                ids_voro, self.VoroVol, self.VoroXYZ, RAvoro, DECvoro, redshift_voro = read_voronoi_vide(vide_path,vide_out_name)
                del ids_voro, RAvoro, DECvoro, redshift_voro

                self.__Lbox = load_pickle_safe(vide_path+'/sample_info.dat')['boxLen']
                verboseprint('    Lbox from VIDE:',self.__Lbox,flush=True)
                if (tracer_dens is None):
                    tracer_dens = np.full(self.VoroVol.shape[0],self.VoroVol.shape[0]/self.__Lbox**3) #self.__Lbox**3/self.VoroVol.shape[0]/self.VoroVol
                else:
                    raise Warning('tracer_dens passed differs from None, tracer_dens will not be computed from Vide output.')
                
            


            verboseprint('        done:',StrHminSec(time.time()-t0),flush=True)

            if max_num_part <= 0:

                max_num_part = int(5 * np.max(np.loadtxt(vide_path+'/untrimmed_centers_all_'+vide_out_name+'.out', comments="#")[:,9]))
                verboseprint('    max_num_part < 0: authomatically set to 5 * max(num_part):',max_num_part,flush=True)

            if (neighbor_ptr is None) or (neighbor_ids is None):
                raise ValueError('VTF not passed. Either pass neighbor_ptr and neighbor_ids or vide_path.')


        # check for isolated voronoi cells:
        mask_ID = neighbor_ptr[ID_core] != neighbor_ptr[ID_core+1]
        self.ID_core = ID_core[mask_ID]
        if np.sum(mask_ID) != mask_ID.shape[0]:
            verboseprint('        found',mask_ID.shape[0]-mask_ID.shape[0],'IDcore corresponding to isolated Voronoi cells. They will be removed',flush=True)
        del mask_ID
        #verboseprint('    voronoi_threshold started, nthreads =',nthreads,flush=True)
        t0 = time.time()
        # Get threshold void properties for all the threshold values passed
        self.void_selected, self.ID_voro_dict, self.Xcm, self.Vol_interp, self.Ncells_in_void, self.ell_eigenvalues, self.ell_eigenvectors = voronoi_threshold(
            self.threshold,self.ID_core,neighbor_ptr,neighbor_ids,self.VoroXYZ,self.VoroVol,tracer_dens,Lbox=self.__Lbox,nthreads=self.nthreads,verbose=verbose,max_num_part=max_num_part)
        verboseprint('        main computation done:',StrHminSec(time.time()-t0),flush=True)
        

        self.ids_selected = dict()
        
        if not lightcone:
            for ith in range(len(threshold)):
                self.ids_selected[ith] = np.arange(self.Ncells_in_void.shape[0])[self.Ncells_in_void[:,ith] > 1.]
        else:
            if voro_border_mask is None:
                if (not (comov_range is None)) & (not (z_range is None)) & (lightcone):
                    raise Warning('both comov_range and z_range are passed, only comov_range will be considered.')
                if (comov_range is None) & (z_range is None) & (lightcone):
                    i_min = np.argmin(dist_voro)
                    i_max = np.argmax(dist_voro)
                    comov_range = [dist_voro[i_min] + 3.5 * (tracer_dens[i_min] ** (-1./3.)),
                                dist_voro[i_max] - 3.5 * (tracer_dens[i_max] ** (-1./3.))]
                    raise ValueError('comov_range and z_range are both None. One of them is required when lightcone = True.')
                if (comov_range is None):
                    comov_range = dist_z.get_dist(np.array(z_range))
            
                comov_range = np.array(comov_range)
                if len(comov_range.shape) == 1:
                    self.comov_range = np.empty((len(threshold),2))
                    self.comov_range[:,0] = min(comov_range)
                    self.comov_range[:,1] = max(comov_range)
                elif comov_range.shape[0] < len(threshold):
                    self.comov_range = np.empty((len(threshold),2))
                    self.comov_range[:,0] = min(comov_range)
                    self.comov_range[:,1] = max(comov_range)
                    raise Warning('comov_range shape do not match threshold lenght. Only min and max of comov_range will be considered')


            verboseprint('    angular and radial mask started.',flush=True)
            t0 = time.time()
           
            self.RA = dict()
            self.DEC = dict()
            self.comov_dist = dict()
            self.redshift = dict()
            self.z_from_dist = RedshiftFromComovingDistanceOverh(OmegaM,w0,wa)

            self.healpix_mask = dict()
            for ith in range(len(threshold)):
                if voro_border_mask is None:
                    if npadding_ang is None:
                        if ang_paddig_rad is None:
                            trs_mask = (dist_voro >= self.comov_range[ith,0]) & (dist_voro <= self.comov_range[ith,1])
                            ang_paddig_rad = 2. * np.max(tracer_dens[trs_mask]**(-1./3.) / dist_voro[trs_mask])
                    if (ith == 0):
                        if healpix_mask is None:
                            try:
                                mask_pix = hp.read_map(healpix_maskpath)
                                nside = hp.get_nside(mask_pix)
                                verboseprint('    angular mask loaded, nside =',nside,flush=True)
                            except:
                                if healpix_maskpath != None:
                                    print('healpix_maskpath not found in path, computing with nside =',nside,flush=True)
                                npix = hp.nside2npix(nside)
                                mask_pix = np.zeros(npix,dtype=np.float32)
                                pix = hp.ang2pix(nside, np.pi/2. - self.DECvoro*np.pi/180., np.pi/180.*self.RAvoro)
                                for ii in np.arange(npix)[~mask_pix.astype(np.bool_)]:
                                    if np.sum(mask_pix[hp.get_all_neighbours(nside,ii)]) >= 6:
                                        mask_pix[ii] = 1.
                                verboseprint('    angular mask not in path, builded with nside =',nside,flush=True)
                        else:
                            mask_pix = healpix_mask.astype(np.float32)
                            nside = hp.get_nside(mask_pix)
                            verboseprint('    angular mask passe, nside =',nside,'converted to np.float32',flush=True)
                            #npadding_ang = int((ang_paddig_rad + hp.nside2resol(nside)) / hp.nside2resol(nside))

                        if npadding_ang is None:
                            npadding_ang = int(round(ang_paddig_rad / hp.nside2resol(nside)))
                            verboseprint('    ang_paddig_rad =',ang_paddig_rad,'npadding_ang =',npadding_ang,flush=True)
                        else:
                            npadding_ang = int(npadding_ang)
                            verboseprint('    npadding_ang =',npadding_ang,flush=True)

                    #mask_ids = borders_mask_bruteforce(self.RAvoro, self.DECvoro, self.Ncells_in_void[:,ith], self.ID_voro_dict,nside)
                    #if voro_border_mask is None:
                    mask_ids, mask_voro, self.healpix_mask[ith] = borders_mask(mask_pix,self.RAvoro,self.DECvoro,self.ID_voro_dict,self.Ncells_in_void[:,ith],npadding_ang,nthreads=nthreads)
                    if len(mask_ids) == 0:
                        self.ids_selected[ith] = np.zeros(0,dtype=np.int64)
                    else:
                        self.ids_selected[ith] = dist_limit_mask(mask_ids,self.Xcm[:,ith,:],self.comov_range[ith,0],self.comov_range[ith,1],
                                                self.VoroXYZ,self.Ncells_in_void[:,ith],self.ID_voro_dict) 
                    self.z_range = self.z_from_dist.get_redshift(self.comov_range.reshape(-1)).reshape(self.comov_range.shape)
                else:
                    self.ids_selected[ith] = borders_mask_inner(voro_border_mask,self.ID_voro_dict,self.Ncells_in_void[:,ith])
                     
            verboseprint('        done:',StrHminSec(time.time()-t0),flush=True)

        
        self.OmegaM = OmegaM
        self.w0 = w0
        self.wa = wa
        self.max_num_part = max_num_part

        self.ids_ovlp = dict()
        self.Vol_ovlp_frac = dict()
        self.num_ovlps = dict()
        self.sor_by_vol = dict()

        self.ids_closest_voro = dict()
        self.ids_ovlp_center = dict()

        self.id_out = dict() 
        for ith in range(len(threshold)):
            self.id_out[ith] = dict() 

    def compute_overlaps(self,frac_ovlp,thresholds=None,ids_threshold=None,verbose=None):
        if not (verbose is None):
            self.verbose = verbose
        if (ids_threshold is None):
            if thresholds is None:
                thresholds = self.threshold
                ids_threshold = np.arange(len(self.threshold))
        elif thresholds is None:
            if np.isscalar(ids_threshold):
                ids_threshold = np.array([ids_threshold])
        else:
            if np.isscalar(thresholds):
                ids_threshold = np.array([ids_threshold])
            ids_threshold = np.arange(len(self.threshold))[is_in_arr(self.threshold,thresholds)]
        if np.isscalar(frac_ovlp):
            for ith in ids_threshold:
                if not (ith in self.Vol_ovlp_frac.keys()):

                    if len(self.ids_selected[ith]) == 0:
                        self.sor_by_vol[ith] = np.zeros(0,dtype=np.int64)
                        #self.id_out[ith][frac_ovlp] = np.zeros(0,dtype=np.int64)

                        self.ids_ovlp[ith] = np.zeros((0,0),dtype=np.int64)
                        Vol_ovlp = np.zeros((0,0))
                        self.Vol_ovlp_frac[ith] = np.zeros((0,0))
                        self.num_ovlps[ith] = np.zeros(0,dtype=np.int64)

                        continue

                    self.ids_ovlp[ith], Vol_ovlp, self.Vol_ovlp_frac[ith], self.num_ovlps[ith] = overlapping_fraction(
                        self.Xcm[:,ith,:], self.Vol_interp[:,ith], self.Ncells_in_void[:,ith], self.VoroXYZ, self.VoroVol, self.ID_voro_dict,
                        Lbox=self.__Lbox,lightcone=self.__lightcone,id_selected=self.ids_selected[ith],nthreads=self.nthreads,verbose=self.verbose)
                    self.sor_by_vol[ith] = np.argsort(self.Vol_interp[self.ids_selected[ith],ith])[::-1].astype(dtype=np.int64,order='C')

                self.id_out[ith][frac_ovlp] = select_overlaps(
                    frac_ovlp,self.ids_selected[ith],self.sor_by_vol[ith], self.ids_ovlp[ith], self.Vol_ovlp_frac[ith], self.num_ovlps[ith])
                #print('ith',ith,'ids_ovlp:',ids_ovlp.dtype, 'Vol_ovlp:',Vol_ovlp.dtype, 'Vol_ovlp_frac:',Vol_ovlp_frac.dtype, 'num_ovlps:',
                #      num_ovlps.dtype,'sor_by_vol:',sor_by_vol.dtype,'self.id_out[ith][frac_ovlp]:',self.id_out[ith][frac_ovlp].dtype,flush=True)
        else:

            Nthresholds = ids_threshold.shape[0]

            frac_ovlp_arr = np.array(frac_ovlp)

            Nfrac = frac_ovlp_arr.shape[0]

            Ncombo = Nthresholds * Nfrac

            ids_selected_numba = Dict.empty(
                key_type=types.int64,
                value_type=int_array)
            
            for kk in self.ids_selected.keys():
                ids_selected_numba[kk] = self.ids_selected[kk]
    
            id_out_dict, len_ids_out = compute_overlaps_all_parallel(
                ids_threshold,frac_ovlp_arr, self.Xcm, self.Vol_interp, self.Ncells_in_void, self.VoroXYZ, self.VoroVol, 
                self.ID_voro_dict, self.__Lbox, self.__lightcone, ids_selected_numba, nthreads=self.nthreads, verbose=verbose)
            
            for ii in range(Ncombo):
                ith = int(ii / Nfrac)
                ifrac = ii - ith * Nfrac
                self.id_out[ids_threshold[ith]][frac_ovlp_arr[ifrac]] = id_out_dict[ii][:len_ids_out[ii]]
            

    def compute_overlaps_all_test(self,threshold,frac_ovlp):

        threshold_arr = np.sort(np.array(threshold))
        Nthresholds = threshold_arr.shape[0]

        frac_ovlp_arr = np.sort(np.array(frac_ovlp))
        Nfrac = frac_ovlp_arr.shape[0]

        Ncombo = Nthresholds * Nfrac


        print('threshold_arr',threshold_arr,Nthresholds)
        print('frac_ovlp_arr',frac_ovlp_arr,Nfrac)

        ids_ovlp = Dict.empty(
            key_type=types.int64,
            value_type=int_array_2d)
        
        Vol_ovlp_frac = Dict.empty(
            key_type=types.int64,
            value_type=float_array_2d)
        
        num_ovlps = Dict.empty(
            key_type=types.int64,
            value_type=int_array)
        
        sor_by_vol = Dict.empty(
            key_type=types.int64,
            value_type=int_array)
        
        id_out_dict = Dict.empty(
            key_type=types.int64,
            value_type=int_array)
        
        len_ids_out = np.zeros(Ncombo,dtype=np.int64)
        
        for ith in range(Nthresholds):
            if len(self.ids_selected[ith]) == 0:
                sor_by_vol[ith] = np.zeros(0,dtype=np.int64)

                ids_ovlp[ith] = np.zeros((0,0),dtype=np.int64)
                Vol_ovlp = np.zeros((0,0))
                Vol_ovlp_frac[ith] = np.zeros((0,0))
                num_ovlps[ith] = np.zeros(0,dtype=np.int64)

                continue

            ids_ovlp[ith], Vol_ovlp, Vol_ovlp_frac[ith],num_ovlps[ith]  = overlapping_fraction(
                self.Xcm[:,ith,:], self.Vol_interp[:,ith], self.Ncells_in_void[:,ith], self.VoroXYZ, self.VoroVol, self.ID_voro_dict,
                Lbox=self.__Lbox,lightcone=self.__lightcone,id_selected=self.ids_selected[ith],nthreads=self.nthreads,verbose=self.verbose)
            
            sor_by_vol[ith] = (np.argsort(self.Vol_interp[self.ids_selected[ith],ith])[::-1]).astype(dtype=np.int64, order='C')

            for ifrac in range(Nfrac):
                id_out_dict[ith * Nfrac + ifrac] = np.empty(sor_by_vol[ith].shape[0],dtype=np.int64)
        
        for ii in prange(Ncombo):
            ith = int(ii / Nfrac)
            ifrac = ii - ith * Nfrac

            print(ii,ith,ifrac,frac_ovlp_arr[ifrac],flush=True) #.shape[0],id_out_tmp.dtype)
            id_out_tmp = select_overlaps(frac_ovlp_arr[ifrac],self.ids_selected[ith],sor_by_vol[ith], ids_ovlp[ith], Vol_ovlp_frac[ith], num_ovlps[ith])
            print('   ',id_out_dict[ii].shape[0],id_out_tmp.shape[0],id_out_tmp.dtype,flush=True)

            len_ids_out[ii] = id_out_tmp.shape[0]
            id_out_dict[ii][:len_ids_out[ii]] = id_out_tmp
    

        for ii in range(Ncombo):
            ith = int(ii / Nfrac)
            ifrac = ii - ith * Nfrac
            self.id_out[ith][frac_ovlp_arr[ifrac]] = id_out_dict[ii][:len_ids_out[ii]]
        

    # return values
    def get_values(self,threshold,key,frac_ovlp=1,center_ovlp=False,verbose=None):
        if not (verbose is None):
            self.verbose = verbose
            self.verbose = verbose
        
        ith = (np.arange(len(self.threshold))[self.threshold == threshold])[0]
        if center_ovlp:
            if not ('center_ovlp' in self.id_out[ith].keys()):
                verboseprint = print if self.verbose else lambda *a, **k: None
                verboseprint('        select center_ovlp, ith=',ith,flush=True)
                t0 = time.time()
                if not (ith in self.ids_closest_voro.keys()):
                    if len(self.ids_selected[ith]) == 0:
                        self.ids_closest_voro[ith] = np.zeros(0,dtype=np.int64)
                    else:
                        self.ids_closest_voro[ith] = closest_voro(
                            self.Xcm[:,ith,:], self.Ncells_in_void[:,ith], self.VoroXYZ, self.VoroVol, self.ID_voro_dict,
                            id_selected=self.ids_selected[ith],Lbox=self.__Lbox,lightcone=self.__lightcone,
                            ngrid=-1,max_iterations=3,nthreads=self.nthreads,verbose=self.verbose)
                    
                if not (ith in self.ids_ovlp_center.keys()):
                    if len(self.ids_selected[ith]) == 0:
                        self.ids_ovlp_center[ith] = np.zeros(0,dtype=np.int64)
                    else:
                        self.ids_ovlp_center[ith] = center_in_void(
                            self.Xcm[:,ith,:], self.Vol_interp[:,ith], self.Ncells_in_void[:,ith], self.VoroXYZ, self.ID_voro_dict, 
                            self.ids_closest_voro[ith],
                            id_selected=self.ids_selected[ith],Lbox=self.__Lbox,lightcone=self.__lightcone,
                            ngrid=-1,nthreads=self.nthreads,verbose=self.verbose)
                if len(self.ids_selected[ith]) == 0:
                    self.id_out[ith]['center_ovlp'] = np.zeros(0,dtype=np.int64)
                else:
                    ids_ovlp_center_out = select_overlaps_center_in_void(self.ids_ovlp_center[ith],self.Vol_interp[self.ids_selected[ith],ith])

                    #mask_ovlp = np.zeros(self.ids_selected[ith].shape[0],dtype=np.bool_)
                    #mask_ovlp[self.ids_selected[ith]] = True
                    mask_ovlp = np.ones(self.ids_selected[ith].shape[0],dtype=np.bool_)
                    mask_ovlp[ids_ovlp_center_out[:,0]] = False

                    #self.id_out[ith]['center_ovlp'] = np.arange(self.Xcm.shape[0])[mask_ovlp]
                    self.id_out[ith]['center_ovlp'] = np.arange(self.ids_selected[ith].shape[0])[mask_ovlp]
                    del mask_ovlp, ids_ovlp_center_out
                
                verboseprint('        done:',StrHminSec(time.time()-t0),flush=True)
            
            id_ovlp_out = self.ids_selected[ith][self.id_out[ith]['center_ovlp']]
            #id_ovlp_out = self.id_out[ith]['center_ovlp']

        elif frac_ovlp < 1:
            if not (frac_ovlp in self.id_out[ith].keys()):
                verboseprint = print if self.verbose else lambda *a, **k: None
                verboseprint('        select overlaps, ith=',ith,'frac_ovlp =',frac_ovlp,flush=True)
                t0 = time.time()
                if not (ith in self.Vol_ovlp_frac.keys()):


                    if len(self.ids_selected[ith]) == 0:
                        self.sor_by_vol[ith] = np.zeros(0,dtype=np.int64)
                        #self.id_out[ith][frac_ovlp] = np.zeros(0,dtype=np.int64)

                        self.ids_ovlp[ith] = np.zeros((0,0),dtype=np.int64)
                        Vol_ovlp = np.zeros((0,0))
                        self.Vol_ovlp_frac[ith] = np.zeros((0,0))
                        self.num_ovlps[ith] = np.zeros(0,dtype=np.int64)

                    else:
                        self.ids_ovlp[ith], Vol_ovlp, self.Vol_ovlp_frac[ith], self.num_ovlps[ith] = overlapping_fraction(
                            self.Xcm[:,ith,:], self.Vol_interp[:,ith], self.Ncells_in_void[:,ith], self.VoroXYZ, self.VoroVol, self.ID_voro_dict,
                            Lbox=self.__Lbox,lightcone=self.__lightcone,id_selected=self.ids_selected[ith],nthreads=self.nthreads,verbose=self.verbose)
                        self.sor_by_vol[ith] = np.argsort(self.Vol_interp[self.ids_selected[ith],ith])[::-1].astype(dtype=np.int64,order='C')

                self.id_out[ith][frac_ovlp] = select_overlaps(frac_ovlp,self.ids_selected[ith],self.sor_by_vol[ith], self.ids_ovlp[ith], self.Vol_ovlp_frac[ith], self.num_ovlps[ith])
                verboseprint('        done:',StrHminSec(time.time()-t0),flush=True)
                #verboseprint('            keys:',self.id_out[ith].keys(),flush=True)
            
            id_ovlp_out = self.ids_selected[ith][self.id_out[ith][frac_ovlp]]
   
        elif frac_ovlp < 0:
            raise ValueError('frac_ovlp cannot be negative, value passed: '+str(frac_ovlp))
        else:
            id_ovlp_out = self.ids_selected[ith]


        if key == 'Ncells':
            # number of voronoi cells contained in each void. Thei are flat as the only a fraction of the last voronoi volume is considered.
            return self.Ncells_in_void[id_ovlp_out,ith]
        
        elif key == 'ID_original_sample':
            # ID of voids of the original VIDE catalog that have been thresholded
            return self.void_selected[id_ovlp_out]
        
        elif key == 'id_selected':
            # IDs of voids of the entire voronoi_threshold output that reach the threshold value and satifty the overlaps condition.
            return id_ovlp_out
        

        elif key == 'id_wrt_all':
            # IDs of voids of the entire voronoi_threshold output that reach the threshold value and satifty the overlaps condition.
            if center_ovlp:
                return self.id_out[ith]['center_ovlp']
            elif frac_ovlp < 1:
                return self.id_out[ith][frac_ovlp]
            else:
                return np.arange(self.ids_selected[ith].shape[0])
        
        elif key == 'closest_voro':
            # ID of voronoi cell containing the volume weighted baricenter
            if not ('center_ovlp' in self.id_out[ith].keys()):
                if len(self.ids_selected[ith]) == 0:
                    self.ids_closest_voro[ith] = np.zeros(0,dtype=np.int64)
                    return self.ids_closest_voro[ith]
                self.ids_closest_voro[ith] = closest_voro(
                    self.Xcm[:,ith,:], self.Ncells_in_void[:,ith], self.VoroXYZ, self.VoroVol, self.ID_voro_dict,
                    id_selected=self.ids_selected[ith],Lbox=self.__Lbox,lightcone=self.__lightcone,
                    ngrid=-1,max_iterations=3,nthreads=self.nthreads,verbose=self.verbose)
            return self.ids_closest_voro[ith]
        
        elif key == 'xyz':
            # Comoving coordinates of the volume weighted baricenter
            return self.Xcm[id_ovlp_out,ith,:]
        
        elif key == 'RA':
            # Right ascension of the volume weighted baricenter
            if not self.__lightcone:
                raise ValueError('\''+key+'\' not available when lightcone=False')
            if not (ith in self.RA.keys()):
                self.comov_dist[ith], self.RA[ith], self.DEC[ith] = from_XYZ_to_rRAdec(self.Xcm[self.ids_selected[ith],ith,0],
                                                                                       self.Xcm[self.ids_selected[ith],ith,1],
                                                                                       self.Xcm[self.ids_selected[ith],ith,2])
            if frac_ovlp < 1:
                return self.RA[ith][self.id_out[ith][frac_ovlp]]
            else:
                return self.RA[ith]
        
        elif key == 'DEC':
            # Declination of the volume weighted baricenter
            if not self.__lightcone:
                raise ValueError('\''+key+'\' not available when lightcone=False')
            if not (ith in self.DEC.keys()):
                self.comov_dist[ith], self.RA[ith], self.DEC[ith] = from_XYZ_to_rRAdec(self.Xcm[self.ids_selected[ith],ith,0],
                                                                                       self.Xcm[self.ids_selected[ith],ith,1],
                                                                                       self.Xcm[self.ids_selected[ith],ith,2])
            if frac_ovlp < 1:
                return self.DEC[ith][self.id_out[ith][frac_ovlp]]
            else:
                return self.DEC[ith]
        
        elif key == 'comov_dist':
            # Comoving distance of the volume weighted baricenter
            if not self.__lightcone:
                raise ValueError('\''+key+'\' not available when lightcone=False')
            if not (ith in self.comov_dist.keys()):
                self.comov_dist[ith], self.RA[ith], self.DEC[ith] = from_XYZ_to_rRAdec(self.Xcm[self.ids_selected[ith],ith,0],
                                                                                       self.Xcm[self.ids_selected[ith],ith,1],
                                                                                       self.Xcm[self.ids_selected[ith],ith,2])
            if frac_ovlp < 1:
                return self.comov_dist[ith][self.id_out[ith][frac_ovlp]]
            else:
                return self.comov_dist[ith]
        
        elif key == 'redshift':
            # Redshift of the volume weighted baricenter
            if not self.__lightcone:
                raise ValueError('\''+key+'\' not available when lightcone=False')
            if not (ith in self.redshift.keys()):
                if not (ith in self.comov_dist.keys()):
                    self.comov_dist[ith], self.RA[ith], self.DEC[ith] = from_XYZ_to_rRAdec(self.Xcm[self.ids_selected[ith],ith,0],
                                                                                           self.Xcm[self.ids_selected[ith],ith,1],
                                                                                           self.Xcm[self.ids_selected[ith],ith,2])
                self.redshift[ith] = self.z_from_dist.get_redshift(self.comov_dist[ith])
            if frac_ovlp < 1:
                return self.redshift[ith][self.id_out[ith][frac_ovlp]]
            else:
                return self.redshift[ith]
        
        elif key == 'volume':
            # Void volumes
            return self.Vol_interp[id_ovlp_out,ith]
        
        elif key == 'radius':
            # Void effective radius
            return (self.Vol_interp[id_ovlp_out,ith]* 3. / (4. * np.pi)) ** (1./3.)
        
        elif key == 'ell_eigenvalues':
            # eigenvalues of the inertial tensor
            return self.ell_eigenvalues[id_ovlp_out,ith,:]

        elif key == 'ell_eigenvectors':
            # eigenvectors of the inertial tensor
            return self.ell_eigenvectors[id_ovlp_out,ith,:,:]

        elif key == 'angular_mask':
            # heapix mask for Voronoi cells
            if not self.__lightcone:
                raise ValueError('\''+key+'\' not available when lightcone=False')
            return self.healpix_mask[ith].astype(np.float32)

        elif key == 'comov_range':
            # comoving range for Voronoi cells
            if not self.__lightcone:
                raise ValueError('\''+key+'\' not available when lightcone=False')
            return self.comov_range[ith,:]

        elif key == 'z_range':
            # redshift range for Voronoi cells
            if not self.__lightcone:
                raise ValueError('\''+key+'\' not available when lightcone=False')
            return self.z_range[ith,:]

        else:
            all_keys = ['Ncells','ID_original_sample','id_selected','id_wrt_all','closest_voro','xyz','RA','DEC','redshift','volume','comov_dist',
                        'radius','ell_eigenvalues','ell_eigenvectors','angular_mask','comov_range','z_range']
            
            all_k_str = ''
            for k_ok in all_keys:
                all_k_str += k_ok+', '
            
            raise ValueError(key + ' key unknown. Available keys: '+all_k_str[:-2])

    
