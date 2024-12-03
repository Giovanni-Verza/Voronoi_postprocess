import numpy as np
import os
import sys
from scipy.spatial import Delaunay 
try:
    import vide.voidUtil as vu
except:
    import vide_postproc_mod as vu

#to reactivate interctive matplotlib mode after having imported vide utils:



def single_void_computation(threshold,VoroXYZ,VoroVol,BoxLen,MeanBoxDens):
    IDnext = np.argmax(VoroVol)

    XYZ_PBC = VoroXYZ - ((VoroXYZ - VoroXYZ[IDnext,:]) * 2 / BoxLen).astype(np.int_) * BoxLen
    numPart = VoroXYZ.shape[0]

    if numPart > 5:
        tri = Delaunay(XYZ_PBC)
        neighbor = tri.vertex_neighbor_vertices

        IDthresholds = np.zeros(numPart,dtype=np.int_)
        ID_to_explore = np.zeros(numPart,dtype=np.int_)

        Condition = True
        ID_to_explore[0] = IDnext
        Nneighbors = 1
        Dens = 0.
        VolTot = 0.
        Ncells = 1
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

            #print(Ncells,1. / VoroVol[IDnext] / MeanBoxDens,Ncells,Dens,IDnext,Nneighbors)
            IDnext = ID_to_explore[:Nneighbors][np.argmax(VoroVol[ID_to_explore[:Nneighbors]])]
            

            Ncells += 1
            Condition = (Dens <= threshold) & (Ncells < numPart)
    else:

        Condition = True
        IDthresholds = np.zeros(numPart,dtype=np.int_)
        ID_to_explore = np.argsort(VoroVol)[::-1]
        Nneighbors = 1
        Dens = 0.
        VolTot = 0.
        Ncells = 1
        while Condition:
            #Dens += 1. / VoroVol[IDnext] / MeanBoxDens
            IDthresholds[Ncells-1] = IDnext
            VolTot += VoroVol[IDnext]
            Dens = Ncells / VolTot / MeanBoxDens
            
            IDnext = ID_to_explore[Ncells]

            Ncells += 1
            Condition = (Dens <= threshold) & (Ncells < numPart)
    Ncells -= 1
    if Ncells <= 1:
        Rinterp = -1.
        Vinterp = 0. 
        Dens_previous = -1.
        VolPrevious = 0.
    else: #  Ncells < numPart:
        delta_Vol = VoroVol[IDthresholds[Ncells-1]]
        VolPrevious = VolTot-delta_Vol
        Dens_previous = (Ncells-1) / VolPrevious / MeanBoxDens
        delta_dens = Dens - Dens_previous
        #delta_dens = Dens - (Ncells-1) / (VolTot-delta_Vol) / MeanBoxDens
        Vinterp = delta_Vol / delta_dens * (threshold - Dens) + VolTot
        Rinterp = (Vinterp * 3. / (4.*np.pi)) ** (1/3)

    return Rinterp, Vinterp, Ncells, Dens, (VolTot * 3. / (4.*np.pi)) ** (1/3), VolTot, Dens_previous, (VolPrevious * 3. / (4.*np.pi)) ** (1/3), VolPrevious




def voronoi_threhosld_box_videcat(threshold,path_to_test,dataPortion="all",untrimmed=True):
    catalog = vu.loadVoidCatalog(path_to_test, 
                                dataPortion=dataPortion, 
                                loadParticles=True,
                                untrimmed=untrimmed)
    print('box info:','\n    nvoids:',catalog.numVoids,'\n    num part:',catalog.numPartTot,
        '\n    box len',catalog.boxLen,'\n    vol norm:',catalog.volNorm,'\n    fake density:',catalog.sampleInfo.fakeDensity)

    MeanBoxDens = catalog.volNorm
    BoxLen = catalog.boxLen
    Rinterp = np.empty(catalog.numVoids)
    Vinterp = np.empty(catalog.numVoids)
    Ncells = np.empty(catalog.numVoids,dtype=np.int_)
    Dens_out = np.empty(catalog.numVoids)
    Rout = np.empty(catalog.numVoids)
    Vout = np.empty(catalog.numVoids)
    Dens_in = np.empty(catalog.numVoids)
    Rin = np.empty(catalog.numVoids)
    Vin = np.empty(catalog.numVoids)
    for ivd in range(catalog.numVoids):
        print(ivd+1,'/',catalog.numVoids)
        voidPart = vu.getVoidPart(catalog, catalog.voids[ivd].voidID)
        VoroVol = vu.getArray(voidPart,'volume')
        VoroXYZ = np.array([vu.getArray(voidPart,'x'),vu.getArray(voidPart,'y'),vu.getArray(voidPart,'z')]).T

        Rinterp[ivd], Vinterp[ivd], Ncells[ivd], Dens_out[ivd], Rout[ivd], Vout[ivd], Dens_in[ivd], Rin[ivd], Vin[ivd] = single_void_computation(threshold,VoroXYZ,VoroVol,BoxLen,MeanBoxDens)
    return Rinterp, Vinterp, Ncells, Dens_out, Rout, Vout, Dens_in, Rin, Vin



