
import numpy as np
import struct
from netCDF4 import Dataset
from numba import jit
#from typing import List


__all__ = ['read_voronoi_vide','read_voronoi_vide', 'voro_in_vide_voids', 'vide_voids_cat']

def read_adjfile_slow(adjfile):
    with open(adjfile, "rb") as adj:
        Npart = struct.unpack('i', adj.read(4))[0]  # Read number of particles
        
        # Pointer to neighboring vertices of vertices:
        neighbor_ptr = np.zeros(Npart+1,dtype=np.int_)
        neighbor_counter = np.zeros(Npart,dtype=np.int_)

        # Read adjacency data
        for i in range(Npart):
            nadj = struct.unpack('i', adj.read(4))[0]
            neighbor_ptr[i+1] = neighbor_ptr[i] + nadj
            
        # Neighboring vertices of vertices - Delaunay scheme:
        neighbor_ids = np.empty(neighbor_ptr[-1],dtype=np.int_)

        # Fill neighbor_ids
        for i in range(Npart):
            nin = struct.unpack('i', adj.read(4))[0]
            #if nin > 0:
            for _ in range(nin):
                j = struct.unpack('i', adj.read(4))[0]
                neighbor_ids[neighbor_ptr[i]+neighbor_counter[i]] = j 
                neighbor_ids[neighbor_ptr[j]+neighbor_counter[j]] = i
                neighbor_counter[i] += 1
                neighbor_counter[j] += 1
                
    # Ids of vertices adjacent to vertex i: neighbor_ids[neighbor_ptr[i]:neighbor_ptr[i+1]]
    return neighbor_ptr, neighbor_ids

@jit(nopython=True)
def read_adjfile_inner_loop(Npart,neighbor_ptr,neighbor_ids,raw_data):
    #neighbor_ids = np.empty(neighbor_ptr[-1], dtype=np.int_)

    neighbor_counter = np.zeros(Npart, dtype=np.int_)
    index = 0
    for i in range(Npart):
        num_neighbors = raw_data[index]
        index += 1
        for _ in range(num_neighbors):
            j = raw_data[index]
            neighbor_ids[neighbor_ptr[i] + neighbor_counter[i]] = j
            neighbor_ids[neighbor_ptr[j] + neighbor_counter[j]] = i
            neighbor_counter[i] += 1
            neighbor_counter[j] += 1
            index += 1
    

def read_adjfile(adjfile):
    with open(adjfile, "rb") as adj:
        # Read the total number of particles
        Npart = struct.unpack('i', adj.read(4))[0]
        
        # Read all adjacency sizes in one go
        adj_sizes = np.frombuffer(adj.read(4 * Npart), dtype=np.int32)
        
        # Compute neighbor_ptr
        neighbor_ptr = np.zeros(Npart + 1, dtype=np.int_)
        np.cumsum(adj_sizes, out=neighbor_ptr[1:])
        
        # Total number of neighbors
        total_neighbors = neighbor_ptr[-1]
        
        # Pre-allocate neighbor_ids
        neighbor_ids = np.empty(total_neighbors, dtype=np.int_)

        data = adj.read(total_neighbors * 4)
        
        # Read all neighbors' IDs in bulk
        raw_data = np.frombuffer(data, dtype=np.int32)

        read_adjfile_inner_loop(Npart,neighbor_ptr,neighbor_ids,raw_data)
    return neighbor_ptr, neighbor_ids
        


def read_voronoi_vide(vide_out,fullName):


    # load box and part info
    infoFile = vide_out+"/zobov_slice_"+fullName+".par"
    File = Dataset(infoFile, 'r')
    ranges = np.zeros((3,2))
    ranges[0][0] = getattr(File, 'range_x_min')
    ranges[0][1] = getattr(File, 'range_x_max')
    ranges[1][0] = getattr(File, 'range_y_min')
    ranges[1][1] = getattr(File, 'range_y_max')
    ranges[2][0] = getattr(File, 'range_z_min')
    ranges[2][1] = getattr(File, 'range_z_max')
    isObservation = getattr(File, 'is_observation')
    numPartTot = getattr(File, 'mask_index')
    File.close()
    boxLen = ranges[:,1] - ranges[:,0]



    # load Voronoi volume (unnormalized)
    volFile = vide_out+"/vol_"+fullName+".dat"
    with open(volFile, mode="rb") as File:
        chk = np.fromfile(File, dtype=np.int32,count=1)
        vols = np.fromfile(File, dtype=np.float32,count=numPartTot)

    # load Voronoi coords
    partFile = vide_out+"/zobov_slice_"+fullName
    with open(partFile, mode="rb") as File:
        chk = np.fromfile(File, dtype=np.int32,count=1)
        # Np from zobov_slice_ e' diverso da vu.loadPart(vide_out)
        Np = np.fromfile(File, dtype=np.int32,count=1)[0]
        chk = np.fromfile(File, dtype=np.int32,count=1)

        chk = np.fromfile(File, dtype=np.int32,count=1)
        x = np.fromfile(File, dtype=np.float32,count=Np)
        x *= boxLen[0]
        #if isObservation != 1:
        #x += ranges[0][0]
        chk = np.fromfile(File, dtype=np.int32,count=1)

        chk = np.fromfile(File, dtype=np.int32,count=1)
        y = np.fromfile(File, dtype=np.float32,count=Np)
        y *= boxLen[1]
        #if isObservation != 1:
        #y += ranges[1][0]
        chk = np.fromfile(File, dtype=np.int32,count=1)

        chk = np.fromfile(File, dtype=np.int32,count=1)
        z = np.fromfile(File, dtype=np.float32,count=Np)
        z *= boxLen[2]
        #if isObservation != 1:
        #z += ranges[2][0]
        chk = np.fromfile(File, dtype=np.int32,count=1)

        chk = np.fromfile(File, dtype=np.int32,count=1)
        RA = np.fromfile(File, dtype=np.float32,count=Np)
        chk = np.fromfile(File, dtype=np.int32,count=1)

        chk = np.fromfile(File, dtype=np.int32,count=1)
        Dec = np.fromfile(File, dtype=np.float32,count=Np)
        chk = np.fromfile(File, dtype=np.int32,count=1)

        chk = np.fromfile(File, dtype=np.int32,count=1)
        redshift = np.fromfile(File, dtype=np.float32,count=Np)
        chk = np.fromfile(File, dtype=np.int32,count=1)

        chk = np.fromfile(File, dtype=np.int32,count=1)
        uniqueID = np.fromfile(File, dtype=np.int64,count=Np)
        chk = np.fromfile(File, dtype=np.int32,count=1)

    # compute Voronoi volume normalization
    videnorm = Np / np.prod(boxLen)


    # compute Voronoi quantities
    c_kms = 299792.458

    uniqueID = uniqueID[:numPartTot]
    #VoroVol = vols / videnorm
    VoroXYZ = np.array([x[:numPartTot],y[:numPartTot],z[:numPartTot]]).T
    RA = RA[:numPartTot]
    Dec = Dec[:numPartTot]
    redshift = redshift[:numPartTot]/c_kms

    return uniqueID, vols / videnorm, VoroXYZ, RA, Dec, redshift




class update_dict:
    def __init__(self, **kwds):
        self.__dict__.update(kwds)




class voro_in_vide_voids:
    def __init__(self,vide_out,fullName,dataPortion="all",untrimmed=True):
        zoneFile = vide_out+"/voidZone_"+fullName+".dat"
        void2Zones = []
        with open(zoneFile, mode="rb") as File:
            numZonesTot = np.fromfile(File, dtype=np.int32,count=1)[0]
            numZonesTot = numZonesTot
            for iZ in range(numZonesTot):
                numZones = np.fromfile(File, dtype=np.int32,count=1)[0]
                void2Zones.append(update_dict(numZones = numZones,zoneIDs = []))

                for p in range(numZones):
                    zoneID = np.fromfile(File, dtype=np.int32,count=1)[0]
                    void2Zones[iZ].zoneIDs.append(zoneID)
        self.void2Zones = void2Zones


        #print("Loading particle-zone membership info...")
        zonePartFile = vide_out+"/voidPart_"+fullName+".dat"
        zones2Parts = []
        with open(zonePartFile) as File:
            chk = np.fromfile(File, dtype=np.int32,count=1)
            numZonesTot = np.fromfile(File, dtype=np.int32,count=1)[0]
            for iZ in range(numZonesTot):
                numPart = np.fromfile(File, dtype=np.int32,count=1)[0]
                zones2Parts.append(update_dict(numPart = numPart, partIDs = []))

                for p in range(numPart):
                    partID = np.fromfile(File, dtype=np.int32,count=1)[0]
                    zones2Parts[iZ].partIDs.append(partID)
        self.zones2Parts = zones2Parts


        if untrimmed:
            prefix = "untrimmed_"
        else:
            prefix = ""

        self.voidID = np.loadtxt(vide_out+"/"+prefix+"voidDesc_"+dataPortion+"_"+fullName+".out", comments="#", skiprows=2)[:,1].astype(np.int_)

    def get_voro_from_uniqueID(self,voidID):

        #partOut = np.zeros(0,np.int_)
        partOut = []
        for iZ in range(self.void2Zones[voidID].numZones):
            zoneID = self.void2Zones[voidID].zoneIDs[iZ]
            #partOut = np.concatenate(partOut,zones2Parts[zoneID].partIDs)
            partOut.append(self.zones2Parts[zoneID].partIDs)

        return np.array(partOut).reshape(-1)
    

    def get_voro_from_ID(self,ivd):

        #partOut = np.zeros(0,np.int_)
        partOut = []
        for iZ in range(self.void2Zones[self.voidID[ivd]].numZones):
            zoneID = self.void2Zones[self.voidID[ivd]].zoneIDs[iZ]
            #partOut = np.concatenate(partOut,zones2Parts[zoneID].partIDs)
            partOut.append(self.zones2Parts[zoneID].partIDs)

        return np.array(partOut).reshape(-1)




def vide_voids_cat(vide_out_dir,fullName,dataPortion='all',untrimmed=True,as_dict=True,values_out=None):
    if untrimmed:
        prefix = "untrimmed_"
    else:
        prefix = ""
    keys_center = ['barycenter','volume_norm','radius','redshift','volume','voidID','dens_contr','num_part',
                'parent_ID','tree_level','num_children','central_dens']
    keys_sky = ['RA','DEC']
    keys_desc = ['file_void','core_ID','core_dens','zone_vol','zone_part', 'void_zone', 'void_prob']
    keys_core = ['core_pos','RAcore','DECcore','redshift_core']
    keys_shape = ['ellip','eigenvalues','eigenvec1','eigenvec2','eigenvec3']
    keys_info = ['num_part_tot']
    # void ID, ellip, eig(1), eig(2), eig(3), eigv(1)-x, eiv(1)-y, eigv(1)-z, eigv(2)-x, eigv(2)-y, eigv(2)-z, eigv(3)-x, eigv(3)-y, eigv(3)-z
    keys_all = keys_center + keys_sky + keys_desc + keys_core + keys_shape + keys_info
    #keys_all = ['barycenter','volume_norm','radius','redshift','volume','voidID','dens_contr','num_part',
    #            'parent_ID','tree_level','num_children','central_dens','RA','DEC',
    #            'coreID','core_dens','core_pos','RAcore','DECcore','redshift_core']
    
    do_center = False
    do_sky = False
    do_desc = False
    do_core = False
    do_shape = False
    do_info = False
    if values_out is None:
        selected_output = False
        do_center = True
        do_sky = True
        do_desc = True
        do_core = True
        do_shape = True
        do_info = True
    else:
        selected_output = True
        do_core = False
        if np.isscalar(values_out):
            values_out = [values_out]
        for kk in values_out:
            if kk not in keys_all:
                all_k_str = ''
                for k_ok in values_out:
                    all_k_str += k_ok+', '
                
                raise ValueError(kk + ' key unknown. available keys: '+all_k_str[:-2])
            
        for kk in keys_center:
            if kk in values_out:
                do_center = True
                break
        for kk in keys_sky:
            if kk in values_out:
                do_sky = True
                break
        for kk in keys_desc:
            if kk in values_out:
                do_desc = True
                break
        for kk in keys_core:
            if kk in values_out:
                do_core = True
                break
        for kk in keys_shape:
            if kk in values_out:
                do_shape = True
                break
        for kk in keys_info:
            if kk in values_out:
                do_info = True
                break

    dict_out = dict()
    if do_center:
        catData = np.loadtxt(vide_out_dir+"/"+prefix+"centers_"+dataPortion+"_"+fullName+".out", comments="#")
        # center x,y,z (Mpc/h), volume (normalized), radius (Mpc/h), redshift, volume (Mpc/h^3), void ID, density contrast, num part, parent ID, tree level, number of children, central density

        dict_out['barycenter'] = catData[:,:3]
        dict_out['volume_norm'] = catData[:,3]
        dict_out['radius'] = catData[:,4]
        dict_out['redshift'] = catData[:,5]
        dict_out['volume'] = catData[:,6]
        dict_out['voidID'] = catData[:,7].astype(np.int_)
        dict_out['dens_contr'] = catData[:,8]
        dict_out['num_part'] = catData[:,9].astype(np.int_)
        dict_out['parent_ID'] = catData[:,10].astype(np.int_)
        dict_out['tree_level'] = catData[:,11].astype(np.int_)
        dict_out['num_children'] = catData[:,12].astype(np.int_)
        dict_out['central_dens'] = catData[:,13]

    if do_sky:
        catData = np.loadtxt(vide_out_dir+"/"+prefix+"sky_positions_"+dataPortion+"_"+fullName+".out")
        dict_out['RA'] = catData[:,0]
        dict_out['DEC'] = catData[:,1]



    if do_desc:
        #'file_void','core_ID','core_dens','zone_vol','zone_part', 'void_prob'
        #ID FileVoid# CoreParticle CoreDens ZoneVol Zone#Part Void#Zones VoidVol Void#Part VoidDensContrast VoidProb

        catData = np.loadtxt(vide_out_dir+"/"+prefix+"voidDesc_"+dataPortion+"_"+fullName+".out", comments="#", skiprows=2)

        dict_out['file_void'] = catData[:,1].astype(np.int_)
        dict_out['core_ID'] = catData[:,2].astype(np.int_)
        dict_out['core_dens'] = catData[:,3]
        dict_out['zone_vol'] = catData[:,4]
        dict_out['zone_part'] = catData[:,5].astype(np.int_)
        dict_out['void_zone'] = catData[:,6].astype(np.int_)
        dict_out['void_prob'] = catData[:,10]

        del catData
        

    if do_core:
        voro_id, VolCell, VoroXYZ, RAvoro, DECvoro, redshift_voro = read_voronoi_vide(vide_out_dir,fullName)
        del voro_id, VolCell

        dict_out['core_pos'] = VoroXYZ[dict_out['core_ID'],:]
        dict_out['RAcore'] = RAvoro[dict_out['core_ID']]
        dict_out['DECcore'] = DECvoro[dict_out['core_ID']]
        dict_out['redshift_core'] = redshift_voro[dict_out['core_ID']]

        del VoroXYZ, RAvoro, DECvoro, redshift_voro


    if do_shape:
        fileName = vide_out_dir+"/"+prefix+"shapes_"+dataPortion+"_"+fullName+".out"

        ellipticity = np.loadtxt(fileName, comments="#")[:,1:14]
        dict_out['ellip'] = ellipticity[:,0]
        dict_out['eigenvalues'] = ellipticity[:,1:4]
        dict_out['eigenvec1'] = ellipticity[:,4:7]
        dict_out['eigenvec2'] = ellipticity[:,7:10]
        dict_out['eigenvec3'] = ellipticity[:,10:13]
        del ellipticity
    if do_info:
        infoFile = vide_out_dir+"/zobov_slice_"+fullName+".par"

        File = Dataset(infoFile, 'r')
        dict_out['num_part_tot'] = getattr(File, 'mask_index')
        File.close()


    if as_dict:
        if selected_output:
            new_dict = dict()
            for kk in values_out:
                new_dict[kk] = dict_out[kk]
            return new_dict
        else:
            return dict_out
    
    elif selected_output:
        return (dict_out[kk] for kk in values_out)
    
    else:
        return (dict_out[kk] for kk in keys_all)
        