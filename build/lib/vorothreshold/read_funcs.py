
import numpy as np
import struct
from netCDF4 import Dataset
#from typing import List


__all__ = ['read_voronoi_vide','read_voronoi_vide', 'voro_in_vide_voids', 'vide_voids_cat']
'''
class Particle:
    def __init__(self):
        self.dens = 0.0
        self.nadj = 0
        self.ncnt = 0
        self.adj = []

def read_adjfile_old(adjfile):
    with open(adjfile, "rb") as adj:
        np = struct.unpack('i', adj.read(4))[0]  # Read number of particles
        mockIndex = np 
        print(f"adj: {np} particles")

        particles = [Particle() for _ in range(np)]

        # Read adjacency data
        for i in range(np):
            particles[i].nadj = struct.unpack('i', adj.read(4))[0]
            particles[i].adj = [0] * particles[i].nadj if particles[i].nadj > 0 else []
            particles[i].ncnt = 0  # Initialize adjacency counter

        for i in range(np):
            nin = struct.unpack('i', adj.read(4))[0]
            if nin > 0:
                for _ in range(nin):
                    j = struct.unpack('i', adj.read(4))[0]
                    if j < np:
                        assert i < j
                        if particles[i].ncnt == particles[i].nadj:
                            print(f"OVERFLOW for particle {i} (pending {j}). List of accepted:")
                            for q in range(particles[i].nadj):
                                print(f"  {particles[i].adj[q]}")
                        if particles[j].ncnt == particles[j].nadj:
                            print(f"OVERFLOW for particle {j} (pending {i}). List of accepted:")
                            for q in range(particles[j].nadj):
                                print(f"  {particles[j].adj[q]}")
                        particles[i].adj[particles[i].ncnt] = j
                        particles[j].adj[particles[j].ncnt] = i
                        particles[i].ncnt += 1
                        particles[j].ncnt += 1
                    else:
                        print(f"{i}: adj = {j}")

        # Verify adjacency pairs
        #for i in range(np):
        #    if particles[i].ncnt != particles[i].nadj and i < mockIndex:
        #        particles[i].nadj = particles[i].ncnt
        #        print(f"We didn't get all of {i}'s adj's; {nin} != {particles[i].nadj}.")
    return particles

'''


def read_adjfile(adjfile):
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




def vide_voids_cat(vide_out_dir,fullName,dataPortion='all',untrimmed=True,as_dict=False,values_out=None):
    if untrimmed:
        prefix = "untrimmed_"
    else:
        prefix = ""
    keys_all = ['barycenter','volume_norm','radius','redshift','volume','voidID','dens_contr','num_part',
                'parent_ID','tree_level','num_children','central_dens','RA','DEC',
                'coreID','core_dens','core_pos','RAcore','DECcore','redshift_core']
    
    if values_out is None:
        selected_output = False
        do_core = True
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
            
        for kk in ['core_pos','RAcore','DECcore','redshift_core']:
            if kk in values_out:
                do_core = True
                break

    catData = np.loadtxt(vide_out_dir+"/"+prefix+"sky_positions_"+dataPortion+"_"+fullName+".out")
    RA_bary = catData[:,0]
    DEC_bary = catData[:,1]


    catData = np.loadtxt(vide_out_dir+"/"+prefix+"centers_"+dataPortion+"_"+fullName+".out", comments="#")
    # center x,y,z (Mpc/h), volume (normalized), radius (Mpc/h), redshift, volume (Mpc/h^3), void ID, density contrast, num part, parent ID, tree level, number of children, central density

    barycenter = catData[:,:3]
    volume_norm = catData[:,3]
    radius = catData[:,4]
    redshift = catData[:,5]
    volume_phys = catData[:,6]
    voidID = catData[:,7].astype(np.int_)
    dens_contr = catData[:,8]
    num_part = catData[:,9].astype(np.int_)
    parent_ID = catData[:,10].astype(np.int_)
    tree_level = catData[:,11].astype(np.int_)
    num_children = catData[:,12].astype(np.int_)
    central_dens = catData[:,13]


    catData = np.loadtxt(vide_out_dir+"/"+prefix+"voidDesc_"+dataPortion+"_"+fullName+".out", comments="#", skiprows=2)

    coreParticle = catData[:,2].astype(np.int_)
    coreDens = catData[:,3]

    del catData


    #fileName = vide_out_dir+"/"+prefix+"shapes_"+dataPortion+"_"+fullName+".out"

    #ellipticity = np.loadtxt(fileName, comments="#")[:,1:14]

    if do_core:
        voro_id, VolCell, VoroXYZ, RAvoro, DECvoro, redshift_voro = read_voronoi_vide(vide_out_dir,fullName)
        del voro_id, VolCell

        core_pos = VoroXYZ[coreParticle,:]
        RAcore = RAvoro[coreParticle]
        DECcore = DECvoro[coreParticle]
        redshift_core = redshift_voro[coreParticle]

        del VoroXYZ, RAvoro, DECvoro, redshift_voro

    if as_dict | selected_output:
        dict_out =  dict({'barycenter':barycenter,
                          'volume_norm':volume_norm,
                          'radius':radius,
                          'redshift':redshift,
                          'volume':volume_phys,
                          'voidID':voidID,
                          'dens_contr':dens_contr,
                          'num_part':num_part,
                          'parent_ID':parent_ID,
                          'tree_level':tree_level,
                          'num_children':num_children,
                          'central_dens':central_dens,
                          'RA':RA_bary,
                          'DEC':DEC_bary,
                          'coreID':coreParticle,
                          'core_dens':coreDens})
        if do_core:
            dict_out['core_pos'] = core_pos
            dict_out['RAcore'] = RAcore
            dict_out['DECcore'] = DECcore
            dict_out['redshift_core'] = redshift_core
        
        if as_dict & selected_output:
            new_dict = dict()
            for kk in values_out:
                new_dict[kk] = dict_out[kk]
            return new_dict
        
        elif selected_output:
            return (dict_out[kk] for kk in values_out)
        
    return (barycenter, volume_norm, radius, redshift, volume_phys, voidID, dens_contr, 
            num_part, parent_ID, tree_level, num_children, central_dens, RA_bary, DEC_bary, 
            coreParticle, coreDens, core_pos, RAcore, DECcore, redshift_core)
    
