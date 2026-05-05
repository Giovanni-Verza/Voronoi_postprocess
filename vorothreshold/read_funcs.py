import glob
import numpy as np
import struct
from numba import jit
import joblib
import subprocess
import tempfile
import os
import h5py
import shutil
#from typing import List


__all__ = ['read_voronoi_vide', 'voro_in_vide_voids', 'vide_voids_cat', 'load_pickle_safe',
           'read_adjfile', 'read_adjfile_safe',
           'is_vide_voro', 'read_void_database', 'find_adjfile',
           'find_void_zone_file', 'find_zone_part_file']


# ---------------------------------------------------------------------------
# VIDE format detection and helpers (v1 "master" vs v2 "voro" branch)
# ---------------------------------------------------------------------------
# v2 (voro) consolidates per-tracer data in tracers.dat (HDF5) and the void
# catalog in void_database.out, and renames a few binary files. v1 (master)
# wrote zobov_slice_<sample>.par + vol_<sample>.dat + binary zobov_slice_<sample>
# plus untrimmed_*_*.out catalog files. These helpers route to the right one.

def is_vide_voro(vide_out):
    return os.path.exists(os.path.join(os.path.expanduser(vide_out), 'tracers.dat'))


def find_adjfile(vide_out):
    voro = os.path.join(os.path.expanduser(vide_out), 'adjacencies.dat')
    if os.path.exists(voro):
        return voro
    matches = glob.glob(os.path.join(os.path.expanduser(vide_out), 'adj_*'))
    if not matches:
        raise FileNotFoundError('No adjacency file found in '+str(vide_out))
    return matches[0]


def find_void_zone_file(vide_out):
    voro = os.path.join(os.path.expanduser(vide_out), 'zobov_void_zone_members.dat')
    if os.path.exists(voro):
        return voro
    matches = glob.glob(os.path.join(os.path.expanduser(vide_out), 'voidZone_*'))
    if not matches:
        raise FileNotFoundError('No void-zone file found in '+str(vide_out))
    return matches[0]


def find_zone_part_file(vide_out, sample_name=None):
    voro = os.path.join(os.path.expanduser(vide_out), 'zobov_zone_part_members.dat')
    if os.path.exists(voro):
        return voro
    if sample_name is None:
        matches = glob.glob(os.path.join(os.path.expanduser(vide_out), 'voidPart_*'))
        if not matches:
            raise FileNotFoundError('No zone-part file found in '+str(vide_out))
        return matches[0]
    return os.path.join(os.path.expanduser(vide_out), 'voidPart_'+sample_name+'.dat')


_VOID_DATABASE_COLS = {
    'voidID':       (0,  np.int32),
    'type':         (1,  None),  # string column
    'cx':           (2,  np.float64),
    'cy':           (3,  np.float64),
    'cz':           (4,  np.float64),
    'volume_norm':  (5,  np.float64),
    'volume':       (6,  np.float64),
    'radius':       (7,  np.float64),
    'redshift':     (8,  np.float64),
    'RA':           (9,  np.float64),
    'DEC':          (10, np.float64),
    'dens_contr':   (11, np.float64),
    'max_extent':   (12, np.float64),
    'nearest_edge': (13, np.float64),
    'num_part':     (14, np.int32),
    'parent_ID':    (15, np.int32),
    'tree_level':   (16, np.int32),
    'num_children': (17, np.int32),
    'central_dens': (18, np.float64),
    'core_ID':      (19, np.int64),
    'core_dens':    (20, np.float64),
    'zone_vol':     (21, np.float64),
    'zone_part':    (22, np.int32),
    'void_zone':    (23, np.int32),
    'void_prob':    (24, np.float64),
    'ellip':        (25, np.float64),
}


def read_void_database(vide_out, dataPortion='all', untrimmed=True):
    """Parse void_database.out (VIDE voro branch).

    Returns a dict with named columns. dataPortion / untrimmed reproduce the
    pre-trim filename semantics from VIDE 1.0:
        untrimmed=True                  -> all rows (default)
        dataPortion='all',  untrimmed=False  -> exclude type == 'edge'
        dataPortion='central'                -> keep only type == 'central'
    """
    path = os.path.join(os.path.expanduser(vide_out), 'void_database.out')
    if not os.path.exists(path):
        raise FileNotFoundError(path)

    arr = np.genfromtxt(path, comments='#', dtype=str)
    if arr.size == 0 or arr.shape[0] == 0:
        out = {kk: np.empty(0, dtype=tt if tt is not None else 'U16')
               for kk, (_, tt) in _VOID_DATABASE_COLS.items()}
        out['barycenter']   = np.empty((0, 3), dtype=np.float64)
        out['eigenvalues']  = np.empty((0, 3), dtype=np.float64)
        out['eigenvec1']    = np.empty((0, 3), dtype=np.float64)
        out['eigenvec2']    = np.empty((0, 3), dtype=np.float64)
        out['eigenvec3']    = np.empty((0, 3), dtype=np.float64)
        return out
    if arr.ndim == 1:
        arr = arr[None, :]

    if not untrimmed:
        if dataPortion == 'central':
            mask = arr[:, 1] == 'central'
        else:
            mask = arr[:, 1] != 'edge'
        arr = arr[mask]

    out = dict()
    for name, (col, tt) in _VOID_DATABASE_COLS.items():
        if tt is None:
            out[name] = arr[:, col]
        else:
            out[name] = arr[:, col].astype(tt)

    out['barycenter']  = np.stack([out.pop('cx'), out.pop('cy'), out.pop('cz')], axis=1)
    out['eigenvalues'] = arr[:, 26:29].astype(np.float64)
    out['eigenvec1']   = arr[:, 29:32].astype(np.float64)
    out['eigenvec2']   = arr[:, 32:35].astype(np.float64)
    out['eigenvec3']   = arr[:, 35:38].astype(np.float64)

    return out


def load_pickle_safe(pkl_file):
    # Loads a joblib object in an isolated Python process and returns it safely.
    
    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
        temp_filename = temp_file.name
    
    # Prepare the loop_string to execute in subprocess
    loop_string = f"""
import joblib
import h5py

# Load the joblib object
obj = joblib.load('{pkl_file}')

# Save the object's attributes to HDF5
with h5py.File('{temp_filename}', 'w') as ff:
    for kk, vv in vars(obj).items():
        dt = h5py.special_dtype(vlen=str) 
        if isinstance(vv, str):
            ff.create_dataset(kk, data=vv, dtype=h5py.special_dtype(vlen=str) )
        elif vv is None:
            pass
        else:
            ff.create_dataset(kk, data=vv)
"""
    
    # Run a separate Python process to load the object and re-save it in HDF5
    subprocess.run([
        shutil.which("python"), "-c", loop_string
    ], check=True)

    # Load the object back safely from the HDF5 file
    obj = {}
    with h5py.File(temp_filename, 'r') as ff:
        for kk in ff.keys():
            obj[kk] = ff[kk][()]
            if type(obj[kk]) is bytes:
                obj[kk] = obj[kk].decode("utf-8")

    # Cleanup the temporary file
    os.remove(temp_filename)
    
    return obj

def read_adjfile_slow(adjfile):
    with open(adjfile, "rb") as adj:
        Npart = struct.unpack('i', adj.read(4))[0]  # Read number of particles
        
        # Pointer to neighboring vertices of vertices:
        neighbor_ptr = np.zeros(Npart+1,dtype=np.int32)
        neighbor_counter = np.zeros(Npart,dtype=np.int32)

        # Read adjacency data
        for i in range(Npart):
            nadj = struct.unpack('i', adj.read(4))[0]
            neighbor_ptr[i+1] = neighbor_ptr[i] + nadj
            
        # Neighboring vertices of vertices - Delaunay scheme:
        neighbor_ids = -np.ones(neighbor_ptr[-1],dtype=np.int32)

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
    #neighbor_ids = np.empty(neighbor_ptr[-1], dtype=np.int32)

    neighbor_counter = np.zeros(Npart, dtype=np.int32)
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
        neighbor_ptr = np.zeros(Npart + 1, dtype=np.int32)
        np.cumsum(adj_sizes, out=neighbor_ptr[1:])
        
        # Total number of neighbors
        total_neighbors = neighbor_ptr[-1]
        
        # Pre-allocate neighbor_ids
        neighbor_ids = np.empty(total_neighbors, dtype=np.int32)

        #data = adj.read(total_neighbors * 4)
        data = adj.read((total_neighbors + Npart) * 4)
        
        # Read all neighbors' IDs in bulk
        raw_data = np.frombuffer(data, dtype=np.int32)

        read_adjfile_inner_loop(Npart,neighbor_ptr,neighbor_ids,raw_data)
    return neighbor_ptr, neighbor_ids
        


@jit(nopython=True)
def read_adjfile_counts(Npart,raw_data):
    #neighbor_ids = np.empty(neighbor_ptr[-1], dtype=np.int32)

    neighbor_counter = np.zeros(Npart, dtype=np.int32)
    index = 0
    for i in range(Npart):
        num_neighbors = raw_data[index]
        index += 1
        for _ in range(num_neighbors):
            j = raw_data[index]
            neighbor_counter[i] += 1
            neighbor_counter[j] += 1
            index += 1
    return neighbor_counter

def read_adjfile_safe(adjfile):
    with open(adjfile, "rb") as adj:
        # Read the total number of particles
        Npart = struct.unpack('i', adj.read(4))[0]

        adj_sizes_unsafe = np.frombuffer(adj.read(4 * Npart), dtype=np.int32)

        #data = adj.read(np.sum(adj_sizes_unsafe) * 4)
        data = adj.read((np.sum(adj_sizes_unsafe) + Npart) * 4)
        
        # Read all neighbors' IDs in bulk
        raw_data = np.frombuffer(data, dtype=np.int32)
        
        # Read all adjacency sizes in one go
        adj_sizes = read_adjfile_counts(Npart,raw_data) #np.frombuffer(adj.read(4 * Npart), dtype=np.int32)
    
    with open(adjfile, "rb") as adj:
        # Read the total number of particles
        Npart = struct.unpack('i', adj.read(4))[0]

        adj_sizes_unsafe = np.frombuffer(adj.read(4 * Npart), dtype=np.int32)

        # Compute neighbor_ptr
        neighbor_ptr = np.zeros(Npart + 1, dtype=np.int32)
        np.cumsum(adj_sizes, out=neighbor_ptr[1:])
        
        # Total number of neighbors
        total_neighbors = neighbor_ptr[-1]
        
        # Pre-allocate neighbor_ids
        neighbor_ids = np.empty(total_neighbors, dtype=np.int32)

        data = adj.read((total_neighbors + Npart) * 4)
        
        # Read all neighbors' IDs in bulk
        raw_data = np.frombuffer(data, dtype=np.int32)

        read_adjfile_inner_loop(Npart,neighbor_ptr,neighbor_ids,raw_data)
    return neighbor_ptr, neighbor_ids
        



def _read_voronoi_vide_voro(vide_out):
    """Reader for the VIDE 'voro' branch: tracers.dat (HDF5) holds everything."""
    tracers_file = os.path.join(os.path.expanduser(vide_out), 'tracers.dat')
    with h5py.File(tracers_file, 'r') as ff:
        x_min = float(ff.attrs['range_x_min']);  x_max = float(ff.attrs['range_x_max'])
        y_min = float(ff.attrs['range_y_min']);  y_max = float(ff.attrs['range_y_max'])
        z_min = float(ff.attrs['range_z_min']);  z_max = float(ff.attrs['range_z_max'])
        Np = int(ff.attrs['num_tracers'])

        x = ff['x'][:].astype(np.float32) - np.float32(x_min)
        y = ff['y'][:].astype(np.float32) - np.float32(y_min)
        z = ff['z'][:].astype(np.float32) - np.float32(z_min)

        if 'RA' in ff:
            RA = ff['RA'][:].astype(np.float32)
        else:
            RA = np.zeros(Np, dtype=np.float32)
        if 'Dec' in ff:
            Dec = ff['Dec'][:].astype(np.float32)
        else:
            Dec = np.zeros(Np, dtype=np.float32)
        if 'redshifts' in ff:
            redshift = ff['redshifts'][:].astype(np.float32)
        else:
            redshift = np.zeros(Np, dtype=np.float32)

        uniqueID = ff['unique_ids'][:]

        # Raw voro++ output. In observation mode, edge-flagged cells have
        # inflated tails; the watershed sees the walled form (1e-27). To
        # reproduce VIDE 1.0's vol_*.dat (which was already the walled form
        # with edge cells effectively excluded), apply the same wall here.
        vols = ff['volumes'][:].astype(np.float32)
        if 'edge_flags' in ff:
            vols[ff['edge_flags'][:] > 0] = np.float32(1e-27)

    boxLen = np.array([x_max - x_min, y_max - y_min, z_max - z_min])
    videnorm = Np / np.prod(boxLen)

    VoroXYZ = np.stack([x, y, z], axis=1)

    return uniqueID, vols / videnorm, VoroXYZ, RA, Dec, redshift


def _read_voronoi_vide_master(vide_out, sample_name=None):
    """Reader for the legacy VIDE 'master' branch (zobov_slice_<name>.par + vol_<name>.dat)."""
    from netCDF4 import Dataset  # only required for v1; deferred import keeps v2 dependency-free

    if sample_name is None:
        infoFile = glob.glob(os.path.expanduser(vide_out)+'/zobov_slice_*.par')[0]
        sample_name = infoFile.split('zobov_slice_')[-1].replace('.par','')
    else:
        infoFile = os.path.expanduser(vide_out)+"/zobov_slice_"+sample_name+".par"

    # load box and part info

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
    volFile = os.path.expanduser(vide_out)+"/vol_"+sample_name+".dat"
    with open(volFile, mode="rb") as File:
        chk = np.fromfile(File, dtype=np.int32,count=1)
        vols = np.fromfile(File, dtype=np.float32,count=numPartTot)

    # load Voronoi coords
    partFile = os.path.expanduser(vide_out)+"/zobov_slice_"+sample_name
    with open(partFile, mode="rb") as File:
        chk = np.fromfile(File, dtype=np.int32,count=1)
        # Np from zobov_slice_ e' diverso da vu.loadPart(vide_out)
        Np = np.fromfile(File, dtype=np.int32,count=1)[0]
        chk = np.fromfile(File, dtype=np.int32,count=1)

        chk = np.fromfile(File, dtype=np.int32,count=1)
        x = np.fromfile(File, dtype=np.float32,count=Np)
        x *= boxLen[0]
        chk = np.fromfile(File, dtype=np.int32,count=1)

        chk = np.fromfile(File, dtype=np.int32,count=1)
        y = np.fromfile(File, dtype=np.float32,count=Np)
        y *= boxLen[1]
        chk = np.fromfile(File, dtype=np.int32,count=1)

        chk = np.fromfile(File, dtype=np.int32,count=1)
        z = np.fromfile(File, dtype=np.float32,count=Np)
        z *= boxLen[2]
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

    videnorm = Np / np.prod(boxLen)
    c_kms = 299792.458

    uniqueID = uniqueID[:numPartTot]
    VoroXYZ = np.array([x[:numPartTot],y[:numPartTot],z[:numPartTot]]).T
    RA = RA[:numPartTot]
    Dec = Dec[:numPartTot]
    redshift = redshift[:numPartTot]/c_kms

    return uniqueID, vols / videnorm, VoroXYZ, RA, Dec, redshift


def read_voronoi_vide(vide_out, sample_name=None):
    """Load per-tracer info from a VIDE sample directory.

    Auto-detects between the new 'voro' branch (tracers.dat HDF5) and the
    legacy 'master' branch (zobov_slice_<sample>.par + vol_<sample>.dat).
    """
    if is_vide_voro(vide_out):
        return _read_voronoi_vide_voro(vide_out)
    return _read_voronoi_vide_master(vide_out, sample_name=sample_name)




class update_dict:
    def __init__(self, **kwds):
        self.__dict__.update(kwds)




class OLD_voro_in_vide_voids:
    def __init__(self,vide_out,sample_name=None,dataPortion="all",untrimmed=True):
        zoneFile = find_void_zone_file(vide_out)
        if sample_name is None:
            try:
                sample_name = zoneFile.replace(os.path.expanduser(vide_out)+'/voidZone_','').replace('.dat','')
            except Exception:
                sample_name = ''
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
        zonePartFile = find_zone_part_file(vide_out, sample_name=sample_name)
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


        if is_vide_voro(vide_out):
            db = read_void_database(vide_out, dataPortion=dataPortion, untrimmed=untrimmed)
            self.voidID = db['voidID']
        else:
            prefix = "untrimmed_" if untrimmed else ""
            self.voidID = np.loadtxt(os.path.expanduser(vide_out)+"/"+prefix+"voidDesc_"+dataPortion+"_"+sample_name+".out", comments="#", skiprows=2)[:,1].astype(np.int32)

    def get_voro_from_uniqueID(self,voidID):

        #partOut = np.zeros(0,np.int32)
        partOut = []
        for iZ in range(self.void2Zones[voidID].numZones):
            zoneID = self.void2Zones[voidID].zoneIDs[iZ]
            #partOut = np.concatenate(partOut,zones2Parts[zoneID].partIDs)
            partOut.append(self.zones2Parts[zoneID].partIDs)

        return np.array(partOut).reshape(-1)
    

    def get_voro_from_ID(self,ivd):

        #partOut = np.zeros(0,np.int32)
        partOut = []
        for iZ in range(self.void2Zones[self.voidID[ivd]].numZones):
            zoneID = self.void2Zones[self.voidID[ivd]].zoneIDs[iZ]
            #partOut = np.concatenate(partOut,zones2Parts[zoneID].partIDs)
            partOut.append(self.zones2Parts[zoneID].partIDs)

        return np.array(partOut).reshape(-1)




def load_void_zones_inner(raw_data):
    numZonesTot = raw_data[0]
    iprogr = 0
    index = 1
    numZones = np.empty(numZonesTot,dtype=np.int32)
    zoneIDs = dict()
    while iprogr < numZonesTot:
        numZones[iprogr] = raw_data[index]
        zoneIDs[iprogr] = raw_data[index+1:index+numZones[iprogr]+1]
        
        index += numZones[iprogr] + 1
        iprogr += 1
    return numZones, zoneIDs

def load_void_zone_part(zonePartFile):
    with open(zonePartFile, mode="rb") as File:
        num_bytes = len(File.read())
    print(num_bytes)
    with open(zonePartFile, mode="rb") as File:

        data = File.read(num_bytes)
    
        # Read all neighbors' IDs in bulk
        raw_data = np.frombuffer(data, dtype=np.int32)

    return raw_data

def load_partzone(vide_out,sample_name=None,dataPortion="all",untrimmed=True):
    #print("Loading particle-zone membership info...")
    zonePartFile = find_zone_part_file(vide_out, sample_name=sample_name)
    zones2Parts = []
    with open(zonePartFile, mode="rb") as File:
        num_bytes = len(File.read())
        print(num_bytes)
    with open(zonePartFile, mode="rb") as File:

        data = File.read(num_bytes)

        # Read all neighbors' IDs in bulk
        raw_data = np.frombuffer(data, dtype=np.int32)

    return raw_data

def load_partzone_inner(raw_data):
    chk = raw_data[0]
    numZonesTot = raw_data[1]
    iprogr = 0
    index = 2
    numPart = np.empty(numZonesTot,dtype=np.int32)
    partID = dict()
    while iprogr < numZonesTot:
        numPart[iprogr] = raw_data[index]
        partID[iprogr] = raw_data[index+1:index+numPart[iprogr]+1]
        
        index += numPart[iprogr] + 1
        iprogr += 1
    return numPart, partID



class voro_in_vide_voids:
    def __init__(self,vide_out,sample_name=None,dataPortion="all",untrimmed=True):
        zoneFile = find_void_zone_file(vide_out)
        self.numZones, self.zoneIDs = load_void_zones_inner(load_void_zone_part(zoneFile))

        zonePartFile = find_zone_part_file(vide_out, sample_name=sample_name)
        self.numPart, self.partID = load_partzone_inner(load_void_zone_part(zonePartFile))

        try:
            if is_vide_voro(vide_out):
                db = read_void_database(vide_out, dataPortion=dataPortion, untrimmed=untrimmed)
                self.voidID = db['voidID']
            else:
                prefix = "untrimmed_" if untrimmed else ""
                self.voidID = np.loadtxt(os.path.expanduser(vide_out)+"/"+prefix+"voidDesc_"+dataPortion+"_"+sample_name+".out", comments="#", skiprows=2)[:,1].astype(np.int32)
        except Exception:
            self.voidID = np.empty(0,dtype=np.int32)

    def get_voro_from_uniqueID(self,voidID):
        #partOut = np.zeros(0,np.int32)
        partOut = np.zeros(0,dtype=np.int32)
        for iZ in range(self.numZones[voidID]):
            partOut = np.append(partOut,self.partID[self.zoneIDs[voidID][iZ]])

        return partOut
    

    def get_voro_from_ID(self,ivd):
        #partOut = np.zeros(0,np.int32)
        partOut = np.zeros(0,dtype=np.int32)
        for iZ in range(self.numZones[self.voidID[ivd]]):
            partOut = np.append(partOut,self.partID[self.zoneIDs[self.voidID[ivd]][iZ]])

        return partOut
    
    


_VIDE_CAT_KEYS_CENTER = ['barycenter','volume_norm','radius','redshift','volume','voidID',
                         'dens_contr','num_part','parent_ID','tree_level','num_children','central_dens']
_VIDE_CAT_KEYS_SKY    = ['RA','DEC']
_VIDE_CAT_KEYS_DESC   = ['file_void','core_ID','core_dens','zone_vol','zone_part','void_zone','void_prob']
_VIDE_CAT_KEYS_CORE   = ['core_pos','RAcore','DECcore','redshift_core']
_VIDE_CAT_KEYS_SHAPE  = ['ellip','eigenvalues','eigenvec1','eigenvec2','eigenvec3']
_VIDE_CAT_KEYS_INFO   = ['num_part_tot']
_VIDE_CAT_KEYS_ALL    = (_VIDE_CAT_KEYS_CENTER + _VIDE_CAT_KEYS_SKY + _VIDE_CAT_KEYS_DESC
                         + _VIDE_CAT_KEYS_CORE + _VIDE_CAT_KEYS_SHAPE + _VIDE_CAT_KEYS_INFO)


def _vide_voids_cat_voro(vide_out_dir, dataPortion, untrimmed, want_core, want_info):
    """Read a VIDE 'voro' branch sample directory into the legacy dict layout."""
    db = read_void_database(vide_out_dir, dataPortion=dataPortion, untrimmed=untrimmed)

    out = dict()
    out['barycenter']   = db['barycenter'].astype(np.float64)
    out['volume_norm']  = db['volume_norm']
    out['radius']       = db['radius']
    out['redshift']     = db['redshift']
    out['volume']       = db['volume']
    out['voidID']       = db['voidID']
    out['dens_contr']   = db['dens_contr']
    out['num_part']     = db['num_part']
    out['parent_ID']    = db['parent_ID']
    out['tree_level']   = db['tree_level']
    out['num_children'] = db['num_children']
    out['central_dens'] = db['central_dens']

    out['RA']  = db['RA']
    out['DEC'] = db['DEC']

    # voidDesc fields. v1's "file_void" was the watershed void ID — same as
    # voidID in v2. void_prob is now in the same row.
    out['file_void'] = db['voidID']
    out['core_ID']   = db['core_ID']
    out['core_dens'] = db['core_dens']
    out['zone_vol']  = db['zone_vol']
    out['zone_part'] = db['zone_part']
    out['void_zone'] = db['void_zone']
    out['void_prob'] = db['void_prob']

    out['ellip']        = db['ellip']
    out['eigenvalues']  = db['eigenvalues']
    out['eigenvec1']    = db['eigenvec1']
    out['eigenvec2']    = db['eigenvec2']
    out['eigenvec3']    = db['eigenvec3']

    if want_core:
        _, _, VoroXYZ, RAvoro, DECvoro, redshift_voro = read_voronoi_vide(vide_out_dir)
        core = out['core_ID']
        out['core_pos']      = VoroXYZ[core, :]
        out['RAcore']        = RAvoro[core]
        out['DECcore']       = DECvoro[core]
        out['redshift_core'] = redshift_voro[core]

    if want_info:
        with h5py.File(os.path.join(os.path.expanduser(vide_out_dir), 'tracers.dat'), 'r') as ff:
            out['num_part_tot'] = int(ff.attrs['num_tracers'])

    return out


def _vide_voids_cat_master(vide_out_dir, sample_name, dataPortion, untrimmed,
                           do_center, do_sky, do_desc, do_core, do_shape, do_info):
    """Legacy reader: reproduces the old multi-file VIDE 1.0 catalog layout."""
    from netCDF4 import Dataset

    prefix = "untrimmed_" if untrimmed else ""

    if sample_name is None:
        path_test = os.path.expanduser(vide_out_dir)+"/"+prefix+"centers_"+dataPortion+"_"
        center_file = glob.glob(path_test+'*')[0]
        sample_name = center_file.replace(path_test,'').replace('.out','')

    out = dict()

    if do_center:
        catData = np.loadtxt(os.path.expanduser(vide_out_dir)+"/"+prefix+"centers_"+dataPortion+"_"+sample_name+".out", comments="#")
        if catData.shape == (0,):
            catData = np.empty((0,14),dtype=np.float32)
        out['barycenter']   = catData[:,:3]
        out['volume_norm']  = catData[:,3]
        out['radius']       = catData[:,4]
        out['redshift']     = catData[:,5]
        out['volume']       = catData[:,6]
        out['voidID']       = catData[:,7].astype(np.int32)
        out['dens_contr']   = catData[:,8]
        out['num_part']     = catData[:,9].astype(np.int32)
        out['parent_ID']    = catData[:,10].astype(np.int32)
        out['tree_level']   = catData[:,11].astype(np.int32)
        out['num_children'] = catData[:,12].astype(np.int32)
        out['central_dens'] = catData[:,13]

    if do_sky:
        catData = np.loadtxt(os.path.expanduser(vide_out_dir)+"/"+prefix+"sky_positions_"+dataPortion+"_"+sample_name+".out")
        if catData.shape == (0,):
            catData = np.empty((0,5),dtype=np.float32)
        out['RA']  = catData[:,0]
        out['DEC'] = catData[:,1]

    if do_desc | do_core:
        catData = np.loadtxt(os.path.expanduser(vide_out_dir)+"/"+prefix+"voidDesc_"+dataPortion+"_"+sample_name+".out", comments="#", skiprows=2)
        if catData.shape == (0,):
            catData = np.empty((0,11),dtype=np.float32)
        out['file_void'] = catData[:,1].astype(np.int32)
        out['core_ID']   = catData[:,2].astype(np.int32)
        out['core_dens'] = catData[:,3]
        out['zone_vol']  = catData[:,4]
        out['zone_part'] = catData[:,5].astype(np.int32)
        out['void_zone'] = catData[:,6].astype(np.int32)
        out['void_prob'] = catData[:,10]
        del catData

    if do_core:
        _, _, VoroXYZ, RAvoro, DECvoro, redshift_voro = read_voronoi_vide(os.path.expanduser(vide_out_dir), sample_name)
        core = out['core_ID']
        out['core_pos']      = VoroXYZ[core, :]
        out['RAcore']        = RAvoro[core]
        out['DECcore']       = DECvoro[core]
        out['redshift_core'] = redshift_voro[core]

    if do_shape:
        fileName = os.path.expanduser(vide_out_dir)+"/"+prefix+"shapes_"+dataPortion+"_"+sample_name+".out"
        try:
            ellipticity = np.loadtxt(fileName, comments="#")[:,1:14]
        except Exception:
            if np.loadtxt(fileName, comments="#").shape == (0,):
                ellipticity = np.empty((0,14),dtype=np.float32)
        out['ellip']       = ellipticity[:,0]
        out['eigenvalues'] = ellipticity[:,1:4]
        out['eigenvec1']   = ellipticity[:,4:7]
        out['eigenvec2']   = ellipticity[:,7:10]
        out['eigenvec3']   = ellipticity[:,10:13]

    if do_info:
        infoFile = os.path.expanduser(vide_out_dir)+"/zobov_slice_"+sample_name+".par"
        File = Dataset(infoFile, 'r')
        out['num_part_tot'] = getattr(File, 'mask_index')
        File.close()

    return out


def vide_voids_cat(vide_out_dir, sample_name=None, dataPortion='all', untrimmed=True,
                   as_dict=True, values_out=None):
    """Read a VIDE void catalog into a dict of named arrays.

    Auto-routes between the new 'voro' branch (single void_database.out +
    tracers.dat) and the legacy 'master' branch (untrimmed_*_*.out family).
    """
    if values_out is None:
        selected_output = False
        do_center = do_sky = do_desc = do_core = do_shape = do_info = True
    else:
        selected_output = True
        if np.isscalar(values_out):
            values_out = [values_out]
        for kk in values_out:
            if kk not in _VIDE_CAT_KEYS_ALL:
                raise ValueError(kk + ' key unknown. available keys: '+', '.join(_VIDE_CAT_KEYS_ALL))
        do_center = any(kk in _VIDE_CAT_KEYS_CENTER for kk in values_out)
        do_sky    = any(kk in _VIDE_CAT_KEYS_SKY    for kk in values_out)
        do_desc   = any(kk in _VIDE_CAT_KEYS_DESC   for kk in values_out)
        do_core   = any(kk in _VIDE_CAT_KEYS_CORE   for kk in values_out)
        do_shape  = any(kk in _VIDE_CAT_KEYS_SHAPE  for kk in values_out)
        do_info   = any(kk in _VIDE_CAT_KEYS_INFO   for kk in values_out)

    if is_vide_voro(vide_out_dir):
        dict_out = _vide_voids_cat_voro(vide_out_dir, dataPortion, untrimmed,
                                        want_core=do_core, want_info=do_info)
    else:
        dict_out = _vide_voids_cat_master(vide_out_dir, sample_name, dataPortion, untrimmed,
                                          do_center, do_sky, do_desc, do_core, do_shape, do_info)

    if as_dict:
        if selected_output:
            return {kk: dict_out[kk] for kk in values_out}
        return dict_out

    if selected_output:
        return (dict_out[kk] for kk in values_out)
    return (dict_out[kk] for kk in _VIDE_CAT_KEYS_ALL)
