import glob
import numpy as np
import os
import time

from numba.core import types
from numba.typed import Dict
from numba import jit, prange, set_num_threads, get_num_threads, get_thread_id

from . read_funcs import read_adjfile, read_voronoi_vide
from . masks import borders_mask_bruteforce, dist_limit_mask
from . overlaps import select_overlaps, overlapping_fraction
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


    len_ids_out = np.zeros(Ncombo,dtype=np.int_)
    
    id_out_dict = Dict.empty(
        key_type=types.int64,
        value_type=int_array)

    for ith in range(Nthresholds):
        for ifrac in range(Nfrac):
            id_out_dict[ith * Nfrac + ifrac] = np.empty(sor_by_vol[ith].shape[0],dtype=np.int_)

    for ii in prange(Ncombo):
        ith = int(ii / Nfrac)
        ifrac = ii - ith * Nfrac

        #print(ii,ith,ifrac)

        id_out_tmp = select_overlaps(frac_ovlp_arr[ifrac],ids_selected[ith],sor_by_vol[ith], ids_ovlp[ith], Vol_ovlp_frac[ith], num_ovlps[ith])

        len_ids_out[ii] = id_out_tmp.shape[0]
        id_out_dict[ii][:len_ids_out[ii]] = id_out_tmp
    
    return id_out_dict, len_ids_out


def compute_overlaps_all_parallel(
    ids_threshold,frac_ovlp_arr, Xcm, Vol_interp, Ncells_in_void, VoroXYZ, VoroVol, ID_voro_dict,ids_selected,nthreads,verbose):

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
        print(ith,flush=True)
        ids_ovlp[ith], Vol_ovlp, Vol_ovlp_frac[ith], num_ovlps[ith]  = overlapping_fraction(
            Xcm[:,ids_threshold[ith],:], Vol_interp[:,ids_threshold[ith]], Ncells_in_void[:,ids_threshold[ith]], VoroXYZ, VoroVol, ID_voro_dict,
            id_selected=ids_selected[ids_threshold[ith]],nthreads=nthreads,verbose=verbose)
        sor_by_vol[ith] = (np.argsort(Vol_interp[ids_selected[ith],ith])[::-1]).astype(dtype=np.int_, order='C')


    #print('parallel started',flush=True)
    id_out_dict, len_ids_out = compute_overlaps_all_parallel_compiled(
        Nthresholds, Nfrac, frac_ovlp_arr, ids_selected, sor_by_vol, ids_ovlp, Vol_ovlp_frac, num_ovlps)
    
    return id_out_dict, len_ids_out


class voronoi_threshold_finder:
    def __init__(self,threshold,lightcone=True,ID_core=None,neighbor_ptr=None,neighbor_ids=None,VoroXYZ=None,VoroVol=None,tracer_dens=None,
                 vide_path=None,comov_range=None,z_range=None,OmegaM=None,w0=-1.,wa=1.,nthreads=-1,verbose=False,verbose_essential=True,max_num_part=-1):
        
        if verbose:
            verbose_essential = True
        self.verbose = verbose
        self.verbose_essential = verbose_essential

        
        essentialprint = print if verbose_essential else lambda *a, **k: None
        
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
 

        if not lightcone:
            raise ValueError('Simulation box option has not been developed yet. This class currently works with lightcone=True option only.')
        else:
            if tracer_dens is None:
                raise ValueError('tracer_dens not passed. When lightcone=True the number density of each tracer is required.')

        if np.isscalar(threshold):
            self.threshold = np.array([threshold])
        else:
            self.threshold = np.array(threshold)

        if not (vide_path is None):
            # load VTFE scheme from adjfile
            essentialprint('    Loading VIDE data.',flush=True)
            t0 = time.time()

            adjfile = glob.glob(vide_path+'/adj_*')[0] #vide_path + '/adj_' + vide_out_name + '.dat'
            neighbor_ptr, neighbor_ids = read_adjfile(adjfile)

            # recover vide_out_name
            vide_out_name = adjfile.split('adj_')[1].split('.dat')[0]
            #if ID_core is None:
            # Load ids of cells belonging to minima
            ID_core = np.loadtxt(vide_path+'/untrimmed_voidDesc_all_'+vide_out_name+'.out', comments='#', skiprows=2)[:,2].astype(np.int_)
            

            dist_z = ComovingDistanceOverh(OmegaM,w0,wa)

            # Load Voronoi cells volume, ids and tracers position
            ids_voro, self.VoroVol, self.VoroXYZ, self.RAvoro, self.DECvoro, redshift_voro = read_voronoi_vide(vide_path,vide_out_name)
            dist_voro = dist_z.get_dist(redshift_voro)
            self.VoroXYZ[:,:] = np.array(from_rRAdec_to_XYZ(dist_voro,self.RAvoro,self.DECvoro)).T
            del ids_voro, redshift_voro

            essentialprint('        done:',StrHminSec(time.time()-t0),flush=True)

            if max_num_part < 0:

                max_num_part = int(5 * np.max(np.loadtxt(vide_path+'/untrimmed_centers_all_'+vide_out_name+'.out', comments="#")[:,9]))
                if verbose:
                    print('max_num_part < 0: authomatically set to 5 * max(num_part):',max_num_part,flush=True)

            if (neighbor_ptr is None) or (neighbor_ids is None):
                raise ValueError('VTF not passed. Either pass neighbor_ptr and neighbor_ids or vide_path.')

        if (comov_range is None) & (z_range is None) & (lightcone):
            raise ValueError('comov_range and z_range are both None. One of them is required when lightcone = True.')
        if (not (comov_range is None)) & (not (z_range is None)) & (lightcone):
            raise Warning('both comov_range and z_range are passed, only comov_range will be considered.')
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



        essentialprint('    voronoi_threshold started, nthreads =',nthreads,flush=True)
        t0 = time.time()
        # Get threshold void properties for all the threshold values passed
        self.void_selected, self.ID_voro_dict, self.Xcm, self.Vol_interp, self.Ncells_in_void, self.ell_eigenvalues, self.ell_eigenvectors = voronoi_threshold(
            self.threshold,ID_core,neighbor_ptr,neighbor_ids,self.VoroXYZ,self.VoroVol,tracer_dens,nthreads=nthreads,verbose=verbose,max_num_part=max_num_part)
        essentialprint('        main computation done:',StrHminSec(time.time()-t0),flush=True)
        

        essentialprint('    angular and radial mask started.',flush=True)
        t0 = time.time()

        self.ids_selected = dict()
        for ith in range(len(threshold)):
            nside = 128
            mask_ids = borders_mask_bruteforce(self.RAvoro, self.DECvoro, self.Ncells_in_void[:,ith], self.ID_voro_dict,nside)
            self.ids_selected[ith] = dist_limit_mask(mask_ids,self.Xcm[:,ith,:],self.comov_range[ith,0],self.comov_range[ith,1],
                                        self.VoroXYZ,self.Ncells_in_void[:,ith],self.ID_voro_dict) 
        essentialprint('        done:',StrHminSec(time.time()-t0),flush=True)

        self.id_out = dict() 
        for ith in range(len(threshold)):
            self.id_out[ith] = dict() 
        
        self.OmegaM=OmegaM
        self.w0=w0
        self.wa=wa
        self.z_from_dist = None
        self.max_num_part = max_num_part

    def compute_overlaps(self,frac_ovlp,thresholds=None,ids_threshold=None):
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
                ids_ovlp, Vol_ovlp, Vol_ovlp_frac, num_ovlps = overlapping_fraction(
                    self.Xcm[:,ith,:], self.Vol_interp[:,ith], self.Ncells_in_void[:,ith], self.VoroXYZ, self.VoroVol, self.ID_voro_dict,
                    id_selected=self.ids_selected[ith],nthreads=self.nthreads,verbose=self.verbose)
                sor_by_vol = np.argsort(self.Vol_interp[self.ids_selected[ith],ith])[::-1]
                self.id_out[ith][frac_ovlp] = select_overlaps(frac_ovlp,self.ids_selected[ith],sor_by_vol, ids_ovlp, Vol_ovlp_frac, num_ovlps)
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
                self.ID_voro_dict, ids_selected_numba, nthreads=self.nthreads, verbose=self.verbose)
            
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
        
        len_ids_out = np.zeros(Ncombo,dtype=np.int_)
        
        for ith in range(Nthresholds):
            ids_ovlp[ith], Vol_ovlp, Vol_ovlp_frac[ith],num_ovlps[ith]  = overlapping_fraction(
                self.Xcm[:,ith,:], self.Vol_interp[:,ith], self.Ncells_in_void[:,ith], self.VoroXYZ, self.VoroVol, self.ID_voro_dict,
                id_selected=self.ids_selected[ith],nthreads=self.nthreads,verbose=self.verbose)
            #sor_by_vol[ith] = np.argsort(self.Vol_interp[self.ids_selected[ith],ith])[::-1]
            #print('ids_ovlp:',out_tmp[0].dtype,out_tmp[0].shape)
            #ids_ovlp[ith] = out_tmp[0]
            #print('done\n')
            #print('Vol_ovlp:',out_tmp[1].dtype,out_tmp[1].shape)
            #Vol_ovlp = out_tmp[1]
            #print('done\n')
            #print('Vol_ovlp_frac:',out_tmp[2].dtype,out_tmp[2].shape)
            #Vol_ovlp_frac[ith] = out_tmp[2]
            #print('done\n')
            #print('num_ovlps:',out_tmp[3].dtype,out_tmp[3].shape)
            #num_ovlps[ith] = out_tmp[3]
            #print('done\n')
            #print('sor_by_vol:')
            #sort_reordered = (np.argsort(self.Vol_interp[self.ids_selected[ith],ith])[::-1]).astype(dtype=np.int_, order='C')
            #print('sor_by_vol:',sort_reordered.dtype,sort_reordered.shape,sort_reordered.flags['C_CONTIGUOUS'])
            sor_by_vol[ith] = (np.argsort(self.Vol_interp[self.ids_selected[ith],ith])[::-1]).astype(dtype=np.int_, order='C')
            #print('done\n')
            #print('check:',np.argsort(self.Vol_interp[self.ids_selected[ith],ith])[::-1]-sor_by_vol[ith])

            for ifrac in range(Nfrac):
                id_out_dict[ith * Nfrac + ifrac] = np.empty(sor_by_vol[ith].shape[0],dtype=np.int_)
        
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
    def get_values(self,threshold,key,frac_ovlp):
        
        all_keys = ['Ncells','ID_original_sample','id_selected','xyz','RA','DEC','redshift','volume','comov_dist',
                    'radius','ell_eigenvalues','ell_eigenvectors','central_dens','id_wrt_all']
        
        if not (key in all_keys):
            all_k_str = ''
            for k_ok in all_keys:
                all_k_str += k_ok+', '
            
            raise ValueError(key + ' key unknown. Available keys: '+all_k_str[:-2])
        
        ith = (np.arange(len(self.threshold))[self.threshold == threshold])[0]

        if frac_ovlp < 1:
            if not (frac_ovlp in self.id_out[ith].keys()):
                essentialprint = print if self.verbose_essential else lambda *a, **k: None
                essentialprint('        select overlaps, ith=',ith,'frac_ovlp =',frac_ovlp,flush=True)
                t0 = time.time()

                ids_ovlp, Vol_ovlp, Vol_ovlp_frac, num_ovlps = overlapping_fraction(
                    self.Xcm[:,ith,:], self.Vol_interp[:,ith], self.Ncells_in_void[:,ith], self.VoroXYZ, self.VoroVol, self.ID_voro_dict,
                    id_selected=self.ids_selected[ith],nthreads=self.nthreads,verbose=self.verbose)
                sor_by_vol = np.argsort(self.Vol_interp[self.ids_selected[ith],ith])[::-1]

                self.id_out[ith][frac_ovlp] = select_overlaps(frac_ovlp,self.ids_selected[ith],sor_by_vol, ids_ovlp, Vol_ovlp_frac, num_ovlps)
                essentialprint('        done:',StrHminSec(time.time()-t0),flush=True)
                #essentialprint('            keys:',self.id_out[ith].keys(),flush=True)
            
            id_ovlp_out = self.ids_selected[ith][self.id_out[ith][frac_ovlp]]
        
        elif frac_ovlp < 0:
            raise ValueError('frac_ovlp cannot be negative, value passed: '+str(frac_ovlp))
        else:
            id_ovlp_out = self.ids_selected[ith]


        if key == 'Ncells':
            # number of voronoi cells contained in each void. Thei are flat as the only a fraction of the last voronoi volume is considered.
            return self.Ncells_in_void[id_ovlp_out,ith]
        
        if key == 'ID_original_sample':
            # ID of voids of the original VIDE catalog that have been thresholded
            return self.void_selected[id_ovlp_out]
        
        if key == 'id_selected':
            # IDs of voids of the entire voronoi_threshold output that reach the threshold value and satifty the overlaps condition.
            return id_ovlp_out
        

        if key == 'id_wrt_all':
            # IDs of voids of the entire voronoi_threshold output that reach the threshold value and satifty the overlaps condition.
            if frac_ovlp < 1:
                return self.id_out[ith][frac_ovlp]
            else:
                return np.arange(self.ids_selected[ith].shape[0])
        
        if key == 'xyz':
            # Comoving coordinates of the volume weighted baricenter
            return self.Xcm[id_ovlp_out,ith,:]
        
        if key == 'RA':
            # Right ascension of the volume weighted baricenter
            return from_XYZ_to_rRAdec(self.Xcm[id_ovlp_out,ith,:])[:,1]
        
        if key == 'DEC':
            # Declination of the volume weighted baricenter
            return from_XYZ_to_rRAdec(self.Xcm[id_ovlp_out,ith,:])[:,2]
        
        if key == 'comov_dist':
            # Comoving distance of the volume weighted baricenter
            return from_XYZ_to_rRAdec(self.Xcm[id_ovlp_out,ith,:])[:,0]
        
        if key == 'redshift':
            # Redshift of the volume weighted baricenter
            if self.z_from_dist is None:
                self.z_from_dist = RedshiftFromComovingDistanceOverh(self.OmegaM,self.w0,self.wa)
            return self.z_from_dist.get_redshift(from_XYZ_to_rRAdec(self.Xcm[id_ovlp_out,ith,:])[:,0])
        
        if key == 'volume':
            # Void volumes
            return self.Vol_interp[id_ovlp_out,ith]
        
        if key == 'radius':
            # Void effective radius
            return (self.Vol_interp[id_ovlp_out,ith]* 3. / (4. * np.pi)) ** (1./3.)
        
        if key == 'ell_eigenvalues':
            # eigenvalues of the inertial tensor
            return self.ell_eigenvalues[id_ovlp_out,ith,:]

        if key == 'ell_eigenvectors':
            # eigenvectors of the inertial tensor
            return self.ell_eigenvectors[id_ovlp_out,ith,:,:]

    
