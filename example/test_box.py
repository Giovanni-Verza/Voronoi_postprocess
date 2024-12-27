import numpy as np
import os
import sys
import matplotlib.pyplot as plt
from scipy.spatial import Delaunay 
import time
sys.path.append(os.path.dirname(os.path.expanduser('../threshold_voids')))
try:
    import vide.voidUtil as vu
except:
    #analysis_functions_dir = '/home/giovanni/Desktop/VSF_forcasts/Roman/vfs_analysis_func'
    #sys.path.append(os.path.dirname(os.path.expanduser(analysis_functions_dir)))
    import threshold_voids.vide_postproc_mod as vu
import threshold_voids.voronoi_threshold_box as tv
#to reactivate interctive matplotlib mode after having imported vide utils:
plt.rcParams.update({'backend':'QtAgg',
                     'backend_fallback':True})
%matplotlib qt
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman"]})

path_to_test= 'data/box/examples/example_simulation/sim_ss1.0/sample_sim_ss1.0_z0.00_d00'
catalog = vu.loadVoidCatalog(path_to_test, 
                            dataPortion="all", 
                            loadParticles=True,
                            untrimmed=True)
print('box info:','\n    nvoids:',catalog.numVoids,'\n    num part:',catalog.numPartTot,
      '\n    box len',catalog.boxLen,'\n    vol norm:',catalog.volNorm,'\n    fake density:',catalog.sampleInfo.fakeDensity)

Reff = vu.getArray(catalog.voids,'radius')
imax = np.argmax(Reff)

#info largest void:
numPart = catalog.voids[imax].numPart
catalog.voids[imax].coreParticle
catalog.voids[imax].coreDens
catalog.voids[imax].centralDen
catalog.voids[imax].voidID
catalog.voids[imax].voidVol
catalog.voids[imax].volume 
catalog.voids[imax].zoneNumPart # uguale a numPart
catalog.voids[imax].zoneVol # uguale a voidVol 

Reff[imax]**3*4*np.pi/3*catalog.volNorm - catalog.voids[imax].voidVol
Reff[imax]**3*4*np.pi/3/catalog.voids[imax].volume-1
(catalog.voids[imax].volume * 3. / (4.*np.pi)) ** (1/3)

voidPart = vu.getVoidPart(catalog, catalog.voids[imax].voidID)
VoroVol = vu.getArray(voidPart,'volume')
VoroID = vu.getArray(voidPart,'uniqueID').astype(np.int_)
VoroXYZ = np.array([vu.getArray(voidPart,'x'),vu.getArray(voidPart,'y'),vu.getArray(voidPart,'z')]).T
IDsorted = np.argsort(VoroVol)[::-1]

MeanBoxDens = catalog.volNorm
XYZ_PBC = VoroXYZ - ((VoroXYZ - VoroXYZ[IDsorted[0],:]) * 2 / catalog.boxLen).astype(np.int_) * catalog.boxLen
tri = Delaunay(XYZ_PBC)
neighbor = tri.vertex_neighbor_vertices
Nmax_neighbors = np.max(neighbor[0][1:] - neighbor[0][:-1])

IDthresholds = np.zeros(numPart,dtype=np.int_)
ID_to_explore = np.zeros(numPart,dtype=np.int_)



Condition = True
IDnext = np.argmax(VoroVol)
ID_to_explore[0] = IDnext
Nneighbors = 1
Dens = 0.
VolTot = 0.
Ncells = 1
threshold = 0.3
while Condition:
    #Dens += 1. / VoroVol[IDnext] / MeanBoxDens
    IDthresholds[Ncells-1] = IDnext
    VolTot += VoroVol[IDnext]
    Dens = Ncells / VolTot / MeanBoxDens
    MMtoAdd = ~np.isin(neighbor[1][neighbor[0][IDnext]:neighbor[0][IDnext+1]],ID_to_explore[:Nneighbors])
    MMtoAdd &= ~np.isin(neighbor[1][neighbor[0][IDnext]:neighbor[0][IDnext+1]],IDthresholds[:Ncells+1])
    NtoAdd = np.sum(MMtoAdd)
    ID_to_explore[Nneighbors:Nneighbors+NtoAdd] = neighbor[1][neighbor[0][IDnext]:neighbor[0][IDnext+1]][MMtoAdd]
    Nneighbors += NtoAdd

    Ichange = np.argwhere(ID_to_explore[:Nneighbors] == IDnext)[0,0]
    ID_to_explore[Ichange:Nneighbors-1] = ID_to_explore[Ichange+1:Nneighbors] 
    Nneighbors -= 1

    print(Ncells,1. / VoroVol[IDnext] / MeanBoxDens,Ncells,Dens,IDnext,Nneighbors)
    IDnext = ID_to_explore[:Nneighbors][np.argmax(VoroVol[ID_to_explore[:Nneighbors]])]
    

    Ncells += 1
    Condition = (Dens <= threshold) & (Ncells < numPart)

if Ncells < numPart:
    delta_Vol = VoroVol[IDthresholds[Ncells-2]]
    delta_dens = Dens - (Ncells-2) / (VolTot-delta_Vol) / MeanBoxDens
    Vinterp = delta_Vol / delta_dens * (threshold - Dens) + VolTot
    Rinterp = (Vinterp * 3. / (4.*np.pi)) ** (1/3)
else:
    Rinterp = -1.
    Vinterp = 0.




dist_from_min = np.sum((XYZ_PBC-XYZ_PBC[0,:])**2,axis=1)**0.5
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
#ax.scatter(VoroXYZ[:,0],VoroXYZ[:,1],VoroXYZ[:,2],color=plt.cm.jet(np.arange(VoroXYZ.shape[0])/VoroXYZ.shape[0]))
#ax.scatter(VoroXYZ[:,0],VoroXYZ[:,1],VoroXYZ[:,2],color=plt.cm.jet(np.argsort(VoroVol)/VoroXYZ.shape[0]))
CC =np.argsort(np.sum((XYZ_PBC-XYZ_PBC[0,:])**2,axis=1))/VoroXYZ.shape[0]
CC = (dist_from_min - np.min(dist_from_min))/(np.max(dist_from_min) - np.min(dist_from_min))
log_vol = VoroVol #np.log(VoroVol)
#CC = (log_vol - np.min(log_vol))/(np.max(log_vol) - np.min(log_vol))
ax.scatter(XYZ_PBC[:,0],XYZ_PBC[:,1],XYZ_PBC[:,2],color= plt.cm.coolwarm_r(CC))
ax.scatter(XYZ_PBC[0,0],XYZ_PBC[0,1],XYZ_PBC[0,2],color='k')

for ii in range(*neighbor[0][:2]):
    ax.plot(XYZ_PBC[[0,neighbor[1][ii]],0],XYZ_PBC[[0,neighbor[1][ii]],1],XYZ_PBC[[0,neighbor[1][ii]],2],color='k',lw=1,ls=':')
for ii in IDthresholds[:Ncells]:
    ax.plot(XYZ_PBC[[0,ii],0],XYZ_PBC[[0,ii],1],XYZ_PBC[[0,ii],2]) #,color='r')
    



Rres, Vres, Ncells, Dens_out, Rout, Vout, Dens_in, Rin, Vin = tv.voronoi_threhosld_box_videcat(threshold,path_to_test,dataPortion="all",untrimmed=True)


analysis_functions_dir = '/home/giovanni/Desktop/VSF_forcasts/Roman/vfs_analysis_func'
sys.path.append(os.path.dirname(os.path.expanduser(analysis_functions_dir)))
from vfs_analysis_func.functions_for_data import *
from vfs_analysis_func.DensityReconstructionFunctions import *
from vfs_analysis_func.rescale_and_overlapping import *


XYZ_vds = vu.getArray(catalog.voids,'macrocenter')
XYZ_trs = catalog.partPos
print('\nComputation Started', flush=True)
time1 = time.time()
OutMeanDensInSphere, OutDensInShell, OutNtracersInSphere, R_measureNorm = \
    DensityRecBox_PBC(XYZ_vds,XYZ_trs,catalog.boxLen[0],np.ones(XYZ_vds.shape[0]),18,100.,101)
DeltaT = time.time() - time1
print('\nComputation finished',time.time()-time1, flush=True)
Rsphere = TresholdRadii_linear(OutMeanDensInSphere,R_measureNorm,threshold)

Rbins = np.logspace(1,2,21)
Nvds_voronoi = np.histogram(Rres,bins=Rbins)[0]
Nvds_sphere = np.histogram(Rsphere,bins=Rbins)[0]
Rmean = 0.5 * (Rbins[1:] + Rbins[:-1])
delta_R = Rbins[1:] - Rbins[:-1]

mps = MeanBoxDens ** (-1/3)
plt.errorbar(Rmean,Nvds_voronoi/delta_R/np.prod(catalog.boxLen),yerr=Nvds_voronoi**0.5/delta_R/np.prod(catalog.boxLen),
             capsize=4,ls='-',marker='o',label='Voronoi')
plt.errorbar(Rmean,Nvds_sphere/delta_R/np.prod(catalog.boxLen),yerr=Nvds_sphere**0.5/delta_R/np.prod(catalog.boxLen),
             capsize=4,ls='-',marker='o',label='Spherical')
progr = 0
for nn in [1.,2.,3.]:
    plt.axvline(nn*mps,ls=['--','-.',':'][progr],c='k',lw=1,label=str(nn)+'mps')
    progr += 1
plt.legend()
plt.loglog()
