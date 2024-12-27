import numpy as np
import os
import sys
import matplotlib.pyplot as plt
from scipy.spatial import Delaunay 
import healpy as hp
import time
sys.path.append(os.path.dirname(os.path.expanduser('../threshold_voids')))
try:
    import vide.voidUtil as vu
except:
    import threshold_voids.vide_postproc_mod as vu
import threshold_voids.voronoi_threshold_lightcone as tv
#to reactivate interctive matplotlib mode after having imported vide utils:
plt.rcParams.update({'backend':'QtAgg',
                     'backend_fallback':True})
%matplotlib qt
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman"]})

path_to_test= 'data/lightcone/examples/example_observation/sample_example_observation'
catalog = vu.loadVoidCatalog(path_to_test, 
                            dataPortion="all", 
                            loadParticles=True,
                            untrimmed=True)
print('box info:','\n    nvoids:',catalog.numVoids,'\n    num part:',catalog.numPartTot,
      '\n    box len',catalog.boxLen,'\n    vol norm:',catalog.volNorm,'\n    fake density:',catalog.sampleInfo.fakeDensity)


NumDensBox = catalog.numPartTot/np.prod(catalog.boxLen)
VoroNormVol = NumDensBox / catalog.volNorm
c_kms = 299792.458

VolAll = vu.getArray(catalog.part,'volume') * VoroNormVol
Zall = vu.getArray(catalog.part,'redshift') / c_kms
VolTotVoro = np.sum(VolAll[(Zall >= catalog.sampleInfo.zRange[0]) & (Zall <= catalog.sampleInfo.zRange[1])])

Om = catalog.sampleInfo.omegaM
z_for_spline = np.linspace(0.,0.5,301)
import excursion_set_functions as es
analysis_functions_dir = '/home/giovanni/Desktop/VSF_forcasts/Roman/vfs_analysis_func'
sys.path.append(os.path.dirname(os.path.expanduser(analysis_functions_dir)))
from vfs_analysis_func.functions_for_data import *
inverse_cHoverH0 = es.spline.cubic_spline(z_for_spline,Inverse_HubbleFactorNomalized(z_for_spline,Om,-1.,0.)*2997.92458)

NpartZrange = np.sum((Zall >= catalog.sampleInfo.zRange[0]) & (Zall <= catalog.sampleInfo.zRange[1]))
part_comov_dist = inverse_cHoverH0.get_integral(0.,Zall)
healpy_mask = hp.read_map(catalog.sampleInfo.maskFile)
sky_fraction = 1.0*len(healpy_mask[healpy_mask>0])/len(healpy_mask)
#hp.mollview(healpy_mask)
SolidAng = 4*np.pi*sky_fraction
ComovRange = inverse_cHoverH0.get_integral(0.,np.array(catalog.sampleInfo.zRange))
lightcone_volume = SolidAng / 3. * (ComovRange[1]**3 - ComovRange[0]**3)
NumDensCone = NpartZrange / lightcone_volume
zbins_hist = np.linspace(catalog.sampleInfo.zRange[0],catalog.sampleInfo.zRange[1],21)
Comovbins_hist = inverse_cHoverH0.get_integral(0.,zbins_hist)
BinsVol = SolidAng / 3. * (Comovbins_hist[1:]**3 - Comovbins_hist[:-1]**3)
NumDensZ = np.histogram(Zall,bins = zbins_hist)[0] / BinsVol


threshold = [0.3,0.5,0.6]
OUT1_interp, OUT1, OUT1_prev, IDs_part_out1, Ncells_out1, Ncells_tot1 = tv.voronoi_threhosld_lightcone_videcat(
    threshold,path_to_test,dataPortion="all",untrimmed=True,MeanConeDens=NumDensCone,DensZ=None,distance_func_z=None,voro_in_basin=True,cluster_method=True)


OUT2_interp, OUT2, OUT2_prev, IDs_part_out2, Ncells_out2, Ncells_tot2 = tv.voronoi_threhosld_lightcone_videcat(
    threshold,path_to_test,dataPortion="all",untrimmed=True,MeanConeDens=NumDensCone,DensZ=None,distance_func_z=None,voro_in_basin=True,cluster_method=False)

OUT2_interp, OUT2, OUT2_prev, IDs_part_out2, Ncells_out2, Ncells_tot2 = tv.voronoi_threhosld_lightcone_videcat(
    threshold,path_to_test,dataPortion="all",untrimmed=True,MeanConeDens=NumDensCone,DensZ=None,distance_func_z=None,voro_in_basin=False,nCPU=1)

OUT3_interp, OUT3, OUT3_prev, IDs_part_out3, Ncells_out3, Ncells_tot3 = tv.voronoi_threhosld_lightcone_videcat(
    threshold,path_to_test,dataPortion="all",untrimmed=True,MeanConeDens=NumDensCone,DensZ=None,distance_func_z=None,voro_in_basin=False)





#info largest void:
Reff = vu.getArray(catalog.voids,'radius')
imax = np.argmax(Reff)

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
VoroVol = vu.getArray(voidPart,'volume') * VoroNormVol
#check normalization:
print('check volume normalization, relative error:',np.sum(VoroVol) / catalog.voids[imax].volume-1)
VoroID = vu.getArray(voidPart,'uniqueID').astype(np.int_)
VoroXYZ = np.array([vu.getArray(voidPart,'x'),vu.getArray(voidPart,'y'),vu.getArray(voidPart,'z')]).T
VoroZ = vu.getArray(voidPart,'redshift') / c_kms
IDsorted = np.argsort(VoroVol)[::-1]






part_redshift = vu.getArray(voidPart,'redshift') / c_kms
part_RA = vu.getArray(voidPart,'ra')
part_DEC = vu.getArray(voidPart,'dec')
#check metric
#@jit(nopython=True)
def from_rRAdec_to_XYZ(r,RA,dec):
    x = r * np.cos(dec * np.pi / 180.) * np.sin(RA * np.pi / 180.)
    y = r * np.cos(dec * np.pi / 180.) * np.cos(RA * np.pi / 180.)
    z = r * np.sin(dec * np.pi / 180.)
    return x, y, z
XYZ_rec = np.array(from_rRAdec_to_XYZ(inverse_cHoverH0.get_integral(0.,part_redshift),part_RA,part_DEC)).T


dist_rec = np.sum((XYZ_rec - XYZ_rec[0,:])**2,axis=1)**0.5
dist_vide = np.sum((VoroXYZ - VoroXYZ[0,:])**2,axis=1)**0.5
print('rel. err. dist:',np.max(dist_rec[1:] / dist_vide[1:]-1),np.min(dist_rec[1:] / dist_vide[1:]-1))



MeanDens = catalog.volNorm
tri = Delaunay(VoroXYZ)
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
    #Dens += 1. / VoroVol[IDnext] / MeanDens
    IDthresholds[Ncells-1] = IDnext
    VolTot += VoroVol[IDnext]
    Dens = Ncells / VolTot / NumDensCone #MeanDens
    MMtoAdd = ~np.isin(neighbor[1][neighbor[0][IDnext]:neighbor[0][IDnext+1]],ID_to_explore[:Nneighbors])
    MMtoAdd &= ~np.isin(neighbor[1][neighbor[0][IDnext]:neighbor[0][IDnext+1]],IDthresholds[:Ncells+1])
    NtoAdd = np.sum(MMtoAdd)
    ID_to_explore[Nneighbors:Nneighbors+NtoAdd] = neighbor[1][neighbor[0][IDnext]:neighbor[0][IDnext+1]][MMtoAdd]
    Nneighbors += NtoAdd

    Ichange = np.argwhere(ID_to_explore[:Nneighbors] == IDnext)[0,0]
    ID_to_explore[Ichange:Nneighbors-1] = ID_to_explore[Ichange+1:Nneighbors] 
    Nneighbors -= 1

    print(Ncells,1. / VoroVol[IDnext] / NumDensCone,Ncells,Dens,IDnext,Nneighbors) #MeanDens
    IDnext = ID_to_explore[:Nneighbors][np.argmax(VoroVol[ID_to_explore[:Nneighbors]])]
    

    Ncells += 1
    Condition = (Dens <= threshold) & (Ncells < numPart)

if Ncells < numPart:
    delta_Vol = VoroVol[IDthresholds[Ncells-2]]
    delta_dens = Dens - (Ncells-2) / (VolTot-delta_Vol) / NumDensCone #MeanDens
    Vinterp = delta_Vol / delta_dens * (threshold - Dens) + VolTot
    Rinterp = (Vinterp * 3. / (4.*np.pi)) ** (1/3)
else:
    Rinterp = -1.
    Vinterp = 0.




dist_from_min = np.sum((VoroXYZ-VoroXYZ[0,:])**2,axis=1)**0.5
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
#ax.scatter(VoroXYZ[:,0],VoroXYZ[:,1],VoroXYZ[:,2],color=plt.cm.jet(np.arange(VoroXYZ.shape[0])/VoroXYZ.shape[0]))
#ax.scatter(VoroXYZ[:,0],VoroXYZ[:,1],VoroXYZ[:,2],color=plt.cm.jet(np.argsort(VoroVol)/VoroXYZ.shape[0]))
CC =np.argsort(np.sum((VoroXYZ-VoroXYZ[0,:])**2,axis=1))/VoroXYZ.shape[0]
CC = (dist_from_min - np.min(dist_from_min))/(np.max(dist_from_min) - np.min(dist_from_min))
log_vol = VoroVol #np.log(VoroVol)
#CC = (log_vol - np.min(log_vol))/(np.max(log_vol) - np.min(log_vol))
ax.scatter(VoroXYZ[:,0],VoroXYZ[:,1],VoroXYZ[:,2],color= plt.cm.coolwarm_r(CC))
ax.scatter(VoroXYZ[0,0],VoroXYZ[0,1],VoroXYZ[0,2],color='k')

#for ii in range(*neighbor[0][:2]):
#    ax.plot(VoroXYZ[[0,neighbor[1][ii]],0],VoroXYZ[[0,neighbor[1][ii]],1],VoroXYZ[[0,neighbor[1][ii]],2],color='k',lw=1,ls=':')
#for ii in IDthresholds[:Ncells]:
#    ax.plot(VoroXYZ[[0,ii],0],VoroXYZ[[0,ii],1],VoroXYZ[[0,ii],2]) #,color='r')
for ii in range(Ncells-1):
    ax.plot(VoroXYZ[IDthresholds[ii:ii+2],0],VoroXYZ[IDthresholds[ii:ii+2],1],VoroXYZ[IDthresholds[ii:ii+2],2]) #,color='r')
    


from scipy import optimize
z_mean_bins = 0.5 * (zbins_hist[1:] + zbins_hist[:-1])


def density_poly_fit_O3(x,Params):
    return Params[0] + Params[1] * x + Params[2] * x ** 2 + Params[3] * x ** 3
def poly_fit_O3(z,p0,p1,p2,p3):
    return density_poly_fit_O3(z,np.array([p0,p1,p2,p3]))
PolyParams_z, pcov = optimize.curve_fit(poly_fit_O3,z_mean_bins,NumDensZ)

def interp_numdens(z):
    return poly_fit_O3(z,*PolyParams_z)

#Rinterp, Vinterp, Ncells, Dens_out, Rout, Vout, Dens_in, Rin, Vin = tv.voronoi_threhosld_lightcone_videcat(threshold,path_to_test,dataPortion="all",untrimmed=True,DensZ=NumDensZ)
OUTinterp,OUT,OUT_prev, Ncells = tv.voronoi_threhosld_lightcone_videcat(threshold,path_to_test,dataPortion="all",untrimmed=True,DensZ=interp_numdens)


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

mps = MeanDens ** (-1/3)
plt.errorbar(Rmean,Nvds_voronoi/delta_R/np.prod(catalog.boxLen),yerr=Nvds_voronoi**0.5/delta_R/np.prod(catalog.boxLen),
             capsize=4,ls='-',marker='o',label='Voronoi')
plt.errorbar(Rmean,Nvds_sphere/delta_R/np.prod(catalog.boxLen),yerr=Nvds_sphere**0.5/delta_R/np.prod(catalog.boxLen),
             capsize=4,ls='-',marker='o',label='Spherical')
progr = 0
for nn in [1.,2.,3.]:
    plt.axvline(nn*mps,ls=['--','-.',':'][progr],c='k',lw=1,label=str(nn)+'mps')
    progr += 1
plt.legend()

