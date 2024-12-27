import numpy as np
import os
import sys
from scipy.spatial import Delaunay 
try:
    import vide.voidUtil as vu
except:
    from . import vide_postproc_mod as vu
import time
from multiprocessing import Process, Pool
import multiprocessing.managers
from functools import partial


def single_void_computation_constmeandens(threshold,VoroXYZ,VoroVol,MeanConeDens):
    IDnext = np.argmax(VoroVol)

    numPart = VoroXYZ.shape[0]

    if numPart < 2:
        return -1., 0.
    if numPart > 5:
        tri = Delaunay(VoroXYZ)
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
            Dens = Ncells / VolTot / MeanConeDens
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
            Dens = Ncells / VolTot / MeanConeDens
            
            IDnext = ID_to_explore[Ncells]

            Ncells += 1
            Condition = (Dens <= threshold) & (Ncells < numPart)
    Ncells -= 1
    if Ncells <= 1:
        Rinterp = -1.
        Vinterp = 0. 
        Dens_previous = -1.
        VolPrevious = 0.

        
    else: #Ncells < numPart:
        delta_Vol = VoroVol[IDthresholds[Ncells-1]]
        VolPrevious = VolTot-delta_Vol
        Dens_previous = (Ncells-1) / VolPrevious / MeanConeDens
        delta_dens = Dens - Dens_previous

        Vinterp = delta_Vol / delta_dens * (threshold - Dens) + VolTot
        Rinterp = (Vinterp * 3. / (4.*np.pi)) ** (1/3)

    #else:
    #    Rinterp = -1.
    #    Vinterp = 0. 
    return Rinterp, Vinterp, Ncells, Dens, (VolTot * 3. / (4.*np.pi)) ** (1/3), VolTot, Dens_previous, (VolPrevious * 3. / (4.*np.pi)) ** (1/3), VolPrevious


#@jit(nopython=True)
def single_void_computation_main(numPart,neighbor,IDnext,i_progr,Nthresholds,threshold,VoroXYZ,VoroVol,zDens,OUT_interp,OUT,OUT_prev,Ncells_out):
    IDthresholds = np.zeros(numPart,dtype=np.int_)
    ID_to_explore = np.zeros(numPart,dtype=np.int_)

    Ncells = 1
    ID_to_explore[0] = IDnext
    IDthresholds[Ncells-1] = IDnext
    Nneighbors = 1
    VolTot = VoroVol[IDnext] #0.
    NormVolTot = VoroVol[IDnext] * zDens[IDnext] #0.
    Dens = Ncells / NormVolTot #0.
    Xcm = VoroXYZ[IDnext,:] * VoroVol[IDnext] / zDens[IDnext] #np.zeros(3)
    Norm_cm = VoroVol[IDnext] / zDens[IDnext] #np.zeros(3)

    #print(0,Ncells,Dens,IDnext,Nneighbors,VoroVol[IDnext] * zDens[IDnext],NormVolTot,VoroVol[IDnext],VolTot)
    #zDens_centr = zDens[IDnext]
    for ith in range(Nthresholds):
        Condition = (Dens <= threshold[ith]) & (Ncells < numPart) #(Ncells < numPart-1)
        #print(ith,Ncells,Condition,numPart-1)
        while Condition:
            ## Add neighbor particles and update ID_to_explore:
            MMtoAdd = ~np.isin(neighbor[1][neighbor[0][IDnext]:neighbor[0][IDnext+1]],ID_to_explore[:Nneighbors])
            MMtoAdd &= ~np.isin(neighbor[1][neighbor[0][IDnext]:neighbor[0][IDnext+1]],IDthresholds[:Ncells+1])
            NtoAdd = np.sum(MMtoAdd)
            ID_to_explore[Nneighbors:Nneighbors+NtoAdd] = neighbor[1][neighbor[0][IDnext]:neighbor[0][IDnext+1]][MMtoAdd]
            Nneighbors += NtoAdd

            Ichange = np.argwhere(ID_to_explore[:Nneighbors] == IDnext)[0,0]
            ID_to_explore[Ichange:Nneighbors-1] = ID_to_explore[Ichange+1:Nneighbors] 
            Nneighbors -= 1

            ## find new IDnext as the lowest norm density among neighbor cells
            #IDnext = ID_to_explore[:Nneighbors][np.argmax(VoroVol[ID_to_explore[:Nneighbors]])]
            IDnext = ID_to_explore[:Nneighbors][np.argmax(VoroVol[ID_to_explore[:Nneighbors]]* zDens[ID_to_explore[:Nneighbors]])]
    
            ## compute quantities
            #Dens += 1. / VoroVol[IDnext] / MeanBoxDens
            Ncells += 1
            IDthresholds[Ncells-1] = IDnext
            VolTot += VoroVol[IDnext]
            NormVolTot += VoroVol[IDnext] * zDens[IDnext]
            #Dens = Ncells / VolTot / zDens[IDnext]
            #Dens = Ncells / VolTot / zDens_centr
            Dens = Ncells / NormVolTot
            Xcm += VoroXYZ[IDnext,:] * VoroVol[IDnext] / zDens[IDnext]
            Norm_cm += VoroVol[IDnext] / zDens[IDnext]

            #print(ith,Ncells,Dens,IDnext,Nneighbors,VoroVol[IDnext] * zDens[IDnext],NormVolTot,VoroVol[IDnext],VolTot)

            Condition = (Dens <= threshold[ith]) & (Ncells < numPart)
            
        OUT[ith][i_progr] = ((VolTot * 3. / (4.*np.pi)) ** (1/3), VolTot, Xcm / Norm_cm, Dens)
        #Ncells -= 1
        if Ncells <= 1:

            VolPrevious = 0.
            NormVolPrevious = 0.
            Dens_previous = 0.
            delta_dens = Dens
            Xcm_prev = Xcm *1.
            Norm_cm_prev = Norm_cm

            OUT_prev[ith][i_progr] = (0., VolPrevious, Xcm/Norm_cm, Dens_previous)
            OUT_interp[ith][i_progr] = (-1., 0., Xcm/Norm_cm)

            
        else: #Ncells < numPart:
            #delta_Vol = VoroVol[IDthresholds[Ncells-1]]
            VolPrevious = VolTot-VoroVol[IDthresholds[Ncells-1]]
            NormVolPrevious = NormVolTot - VoroVol[IDthresholds[Ncells-1]] * zDens[IDthresholds[Ncells-1]]
            Dens_previous = (Ncells-1) / NormVolPrevious #VolPrevious / zDens[IDthresholds[Ncells-1]]
            delta_dens = Dens - Dens_previous
            Xcm_prev = Xcm - VoroXYZ[IDthresholds[Ncells-1],:] * VoroVol[IDthresholds[Ncells-1]] / zDens[IDthresholds[Ncells-1]]
            Norm_cm_prev = Norm_cm - VoroVol[IDthresholds[Ncells-1]] / zDens[IDthresholds[Ncells-1]]

            OUT_prev[ith][i_progr] = ((VolPrevious * 3. / (4.*np.pi)) ** (1/3), VolPrevious, Xcm_prev/Norm_cm_prev, Dens_previous)
            if (Dens >= threshold[ith]): # (Ncells < numPart):
                #Vinterp = VoroVol[IDthresholds[Ncells-1]] / delta_dens * (threshold - Dens) + VolTot
                Vinterp = VoroVol[IDthresholds[Ncells-1]] / delta_dens * (threshold[ith] - Dens_previous) + VolPrevious
                Rinterp = (Vinterp * 3. / (4.*np.pi)) ** (1/3)
                Xcm_interp = Xcm_prev + VoroXYZ[IDthresholds[Ncells-1],:] * VoroVol[IDthresholds[Ncells-1]] / delta_dens * (threshold[ith] - Dens_previous) / zDens[IDthresholds[Ncells-1]]
                Norm_cm_interp = Norm_cm_prev + VoroVol[IDthresholds[Ncells-1]] / delta_dens * (threshold[ith] - Dens_previous) / zDens[IDthresholds[Ncells-1]]
                
                OUT_interp[ith][i_progr] = (Rinterp, Vinterp, Xcm_interp/Norm_cm_interp)
            else:
                OUT_interp[ith][i_progr] = (-1., VolTot, Xcm/Norm_cm)
        Ncells_out[ith][i_progr] = Ncells
        
    return IDthresholds[:Ncells]



def single_void_computation_Zdens(i_progr,Nthresholds,threshold,VoroXYZ,VoroVol,zDens,OUT_interp,OUT,OUT_prev,Ncells_out,ID_core=-1):
    numPart = VoroXYZ.shape[0]
    if ID_core >= 0:
        IDnext = ID_core
    else:
        IDnext = np.argmax(VoroVol)
    #print(numPart)

    if numPart < 5:
        #print(numPart)
        IDthresholds = np.zeros(numPart,dtype=np.int_)
        ID_to_explore = np.argsort(VoroVol)[::-1]


        Ncells = 1
        Nneighbors = 1
        VolTot = VoroVol[IDnext] #0.
        NormVolTot = VoroVol[IDnext] * zDens[IDnext] #0.
        Dens = Ncells / NormVolTot #0.
        Xcm = VoroXYZ[IDnext,:] * VoroVol[IDnext] / zDens[IDnext]
        Norm_cm = VoroVol[IDnext] / zDens[IDnext]
        #print(0,Ncells,Dens,IDnext,Nneighbors,NormVolTot,VolTot)
        for ith in range(Nthresholds):
            Condition = (Dens <= threshold[ith]) & (Ncells < numPart)
            while Condition:
                IDnext = ID_to_explore[Ncells]
                Ncells += 1
                IDthresholds[Ncells-1] = IDnext

                VolTot += VoroVol[IDnext]
                NormVolTot += VoroVol[IDnext] * zDens[IDnext]
                #Dens = Ncells / VolTot / zDens[IDnext]
                #Dens = Ncells / VolTot / zDens_centr
                Dens = Ncells / NormVolTot
                Xcm += VoroXYZ[IDnext,:] * VoroVol[IDnext] / zDens[IDnext]
                Norm_cm += VoroVol[IDnext] / zDens[IDnext]
                #print(ith,Ncells,Dens,IDnext,Nneighbors,NormVolTot,VolTot)
                Condition = (Dens <= threshold[ith]) & (Ncells < numPart)
            #######################
            
            OUT[ith][i_progr] = ((VolTot * 3. / (4.*np.pi)) ** (1/3), VolTot, Xcm / Norm_cm, Dens)
            #Ncells -= 1
            if Ncells <= 1:

                VolPrevious = 0.
                NormVolPrevious = 0.
                Dens_previous = 0.
                delta_dens = Dens
                Xcm_prev = Xcm *1.
                Norm_cm_prev = Norm_cm

                OUT_prev[ith][i_progr] = (0., VolPrevious, Xcm/Norm_cm, Dens_previous)
                OUT_interp[ith][i_progr] = (-1., 0., Xcm/Norm_cm,Ncells)

                
            else: #Ncells < numPart:
                #delta_Vol = VoroVol[IDthresholds[Ncells-1]]
                VolPrevious = VolTot-VoroVol[IDthresholds[Ncells-1]]
                NormVolPrevious = NormVolTot - VoroVol[IDthresholds[Ncells-1]] * zDens[IDthresholds[Ncells-1]]
                Dens_previous = (Ncells-1) / NormVolPrevious #VolPrevious / zDens[IDthresholds[Ncells-1]]
                delta_dens = Dens - Dens_previous
                Xcm_prev = Xcm - VoroXYZ[IDthresholds[Ncells-1],:] * VoroVol[IDthresholds[Ncells-1]] / zDens[IDthresholds[Ncells-1]]
                Norm_cm_prev = Norm_cm - VoroVol[IDthresholds[Ncells-1]] / zDens[IDthresholds[Ncells-1]]

                OUT_prev[ith][i_progr] = ((VolPrevious * 3. / (4.*np.pi)) ** (1/3), VolPrevious, Xcm_prev/Norm_cm_prev, Dens_previous)
                if (Dens >= threshold[ith]):
                    frac = (threshold[ith] * NormVolPrevious + 1 - Ncells) / (1. - threshold[ith] * VoroVol[IDthresholds[Ncells-1]] * zDens[IDthresholds[Ncells-1]])
                    Vinterp = VolPrevious  + frac * VoroVol[IDthresholds[Ncells-1]]
                    Rinterp = (Vinterp * 3. / (4.*np.pi)) ** (1/3)
                    Xcm_interp = Xcm_prev + frac * VoroXYZ[IDthresholds[Ncells-1],:] * VoroVol[IDthresholds[Ncells-1]] / zDens[IDthresholds[Ncells-1]]
                    Norm_cm_interp = Norm_cm_prev + frac * VoroVol[IDthresholds[Ncells-1]] / zDens[IDthresholds[Ncells-1]]

                    #Vinterp = VoroVol[IDthresholds[Ncells-1]] / delta_dens * (threshold[ith] - Dens_previous) + VolPrevious
                    #Rinterp = (Vinterp * 3. / (4.*np.pi)) ** (1/3)
                    #Xcm_interp = Xcm_prev + VoroXYZ[IDthresholds[Ncells-1],:] * VoroVol[IDthresholds[Ncells-1]] / delta_dens * (threshold[ith] - Dens_previous) / zDens[IDthresholds[Ncells-1]]
                    #Norm_cm_interp = Norm_cm_prev + VoroVol[IDthresholds[Ncells-1]] / delta_dens * (threshold[ith] - Dens_previous) / zDens[IDthresholds[Ncells-1]]
                    
                    OUT_interp[ith][i_progr] = (Rinterp, Vinterp, Xcm_interp/Norm_cm_interp,Ncells-1+frac)
                else:
                    OUT_interp[ith][i_progr] = (-1., VolTot, Xcm/Norm_cm,Ncells)
            Ncells_out[ith][i_progr] = Ncells
                
            '''
            OUT[ith][i_progr] = ((VolTot * 3. / (4.*np.pi)) ** (1/3), VolTot, Xcm / Norm_cm, Dens)
            Ncells -= 1
            if Ncells <= 1:
                Rinterp = -1.
                Vinterp = 0. 
                Dens_previous = -1.
                VolPrevious = 0.

                OUT_prev[ith][i_progr] = (0., VolPrevious, Xcm/Norm_cm, Dens_previous)

                
            else: #Ncells < numPart:
                #delta_Vol = VoroVol[IDthresholds[Ncells-1]]
                VolPrevious = VolTot-VoroVol[IDthresholds[Ncells-1]]
                Dens_previous = (Ncells-1) / VolPrevious / zDens[Ncells-1]
                delta_dens = Dens - Dens_previous
                Xcm_prev = Xcm - VoroXYZ[IDthresholds[Ncells-1],:] * VoroVol[IDthresholds[Ncells-1]] / zDens[IDthresholds[Ncells-1]]
                Norm_cm_prev = Norm_cm - VoroVol[IDthresholds[Ncells-1]] / zDens[IDthresholds[Ncells-1]]

                OUT_prev[ith][i_progr] = ((VolPrevious * 3. / (4.*np.pi)) ** (1/3), VolPrevious, Xcm_prev/Norm_cm_prev, Dens_previous)
                if (Ncells < numPart):
                    #Vinterp = VoroVol[IDthresholds[Ncells-1]] / delta_dens * (threshold - Dens) + VolTot
                    Vinterp = VoroVol[IDthresholds[Ncells-1]] / delta_dens * (threshold[ith] - Dens_previous) + VolPrevious
                    Rinterp = (Vinterp * 3. / (4.*np.pi)) ** (1/3)
                    Xcm_interp = Xcm_prev + VoroXYZ[IDthresholds[Ncells-1],:] * VoroVol[IDthresholds[Ncells-1]] / delta_dens * (threshold[ith] - Dens_previous) / zDens[IDthresholds[Ncells-1]]
                    Norm_cm_interp = Norm_cm_prev + VoroVol[IDthresholds[Ncells-1]] / delta_dens * (threshold[ith] - Dens_previous) / zDens[IDthresholds[Ncells-1]]
                    
                    OUT_interp[ith][i_progr] = (Rinterp, Vinterp, Xcm_interp/Norm_cm_interp)
                else:
                    OUT_interp[ith][i_progr] = (-1., -1., Xcm/Norm_cm)
            Ncells_out[ith][i_progr] = Ncells
            '''

        #for ith in range(Nthresholds):
        #    OUT_interp[ith][i_progr] = (0., 0., np.mean(VoroXYZ,axis=0))
        #    OUT[ith][i_progr] = (0., 0., np.mean(VoroXYZ,axis=0),-1.)
        #    OUT_prev[ith][i_progr] = (0., 0., np.mean(VoroXYZ,axis=0),-1.)
        return IDthresholds[:Ncells] #OUT_interp, OUT, OUT_prev, numPart
    #else: #if numPart > 5:

    #fig = plt.figure()
    #ax = fig.add_subplot(projection='3d')
    #ax.scatter(VoroXYZ[:,0],VoroXYZ[:,1],VoroXYZ[:,2],s=1,alpha=0.25,c='k')
    
    tri = Delaunay(VoroXYZ)
    neighbor = tri.vertex_neighbor_vertices

    IDthresholds = np.zeros(numPart,dtype=np.int_)
    ID_to_explore = np.zeros(numPart,dtype=np.int_)

    Ncells = 1
    ID_to_explore[0] = IDnext
    IDthresholds[Ncells-1] = IDnext
    Nneighbors = 1
    VolTot = VoroVol[IDnext] #0.
    NormVolTot = VoroVol[IDnext] * zDens[IDnext] #0.
    Dens = Ncells / NormVolTot #0.
    Xcm = VoroXYZ[IDnext,:] * VoroVol[IDnext] / zDens[IDnext] #np.zeros(3)
    Norm_cm = VoroVol[IDnext] / zDens[IDnext] #np.zeros(3)

    #print(0,Ncells,Dens,IDnext,Nneighbors,VoroVol[IDnext] * zDens[IDnext],NormVolTot,VoroVol[IDnext],VolTot)
    #zDens_centr = zDens[IDnext]
    for ith in range(Nthresholds):
        Condition = (Dens <= threshold[ith]) & (Ncells < numPart) #(Ncells < numPart-1)
        #print(ith,Ncells,Condition,numPart-1)
        while Condition:
            #ax.scatter(VoroXYZ[IDnext,0],VoroXYZ[IDnext,1],VoroXYZ[IDnext,2])
            ## Add neighbor particles and update ID_to_explore:
            MMtoAdd = ~np.isin(neighbor[1][neighbor[0][IDnext]:neighbor[0][IDnext+1]],ID_to_explore[:Nneighbors])
            MMtoAdd &= ~np.isin(neighbor[1][neighbor[0][IDnext]:neighbor[0][IDnext+1]],IDthresholds[:Ncells+1])
            NtoAdd = np.sum(MMtoAdd)
            ID_to_explore[Nneighbors:Nneighbors+NtoAdd] = neighbor[1][neighbor[0][IDnext]:neighbor[0][IDnext+1]][MMtoAdd]
            #for ii in ID_to_explore[Nneighbors:Nneighbors+NtoAdd]:
            #    ax.plot(VoroXYZ[[IDnext,ii],0],VoroXYZ[[IDnext,ii],1],VoroXYZ[[IDnext,ii],2],lw=1,c=plt.cm.tab10((Ncells-1)%10))
            Nneighbors += NtoAdd

            Ichange = np.argwhere(ID_to_explore[:Nneighbors] == IDnext)[0,0]
            ID_to_explore[Ichange:Nneighbors-1] = ID_to_explore[Ichange+1:Nneighbors] 
            Nneighbors -= 1

            ## find new IDnext as the lowest norm density among neighbor cells
            #IDnext = ID_to_explore[:Nneighbors][np.argmax(VoroVol[ID_to_explore[:Nneighbors]])]
            IDnext = ID_to_explore[:Nneighbors][np.argmax(VoroVol[ID_to_explore[:Nneighbors]]* zDens[ID_to_explore[:Nneighbors]])]
    
            ## compute quantities
            #Dens += 1. / VoroVol[IDnext] / MeanBoxDens
            Ncells += 1
            IDthresholds[Ncells-1] = IDnext
            VolTot += VoroVol[IDnext]
            NormVolTot += VoroVol[IDnext] * zDens[IDnext]
            #Dens = Ncells / VolTot / zDens[IDnext]
            #Dens = Ncells / VolTot / zDens_centr
            Dens = Ncells / NormVolTot
            Xcm += VoroXYZ[IDnext,:] * VoroVol[IDnext] / zDens[IDnext]
            Norm_cm += VoroVol[IDnext] / zDens[IDnext]

            #print(ith,Ncells,Dens,IDnext,Nneighbors,VoroVol[IDnext] * zDens[IDnext],NormVolTot,VoroVol[IDnext],VolTot)

            Condition = (Dens <= threshold[ith]) & (Ncells < numPart)

        '''
        while Condition:
            #Dens += 1. / VoroVol[IDnext] / MeanBoxDens
            IDthresholds[Ncells-1] = IDnext
            VolTot += VoroVol[IDnext]
            NormVolTot += VoroVol[IDnext] * zDens[IDnext]
            #Dens = Ncells / VolTot / zDens[IDnext]
            #Dens = Ncells / VolTot / zDens_centr
            Dens = Ncells / NormVolTot
            Xcm += VoroXYZ[IDnext,:] * VoroVol[IDnext] / zDens[IDnext]
            Norm_cm += VoroVol[IDnext] / zDens[IDnext]

            MMtoAdd = ~np.isin(neighbor[1][neighbor[0][IDnext]:neighbor[0][IDnext+1]],ID_to_explore[:Nneighbors])
            if Ncells < numPart-1:
                MMtoAdd &= ~np.isin(neighbor[1][neighbor[0][IDnext]:neighbor[0][IDnext+1]],IDthresholds[:Ncells+1])
                NtoAdd = np.sum(MMtoAdd)
                ID_to_explore[Nneighbors:Nneighbors+NtoAdd] = neighbor[1][neighbor[0][IDnext]:neighbor[0][IDnext+1]][MMtoAdd]
                Nneighbors += NtoAdd

                Ichange = np.argwhere(ID_to_explore[:Nneighbors] == IDnext)[0,0]
                ID_to_explore[Ichange:Nneighbors-1] = ID_to_explore[Ichange+1:Nneighbors] 
                Nneighbors -= 1

                #print(ith,Ncells,Ncells,Dens,Ncells / VolTot / zDens_centr,IDnext,Nneighbors,NormVolTot,VolTot)
                #IDnext = ID_to_explore[:Nneighbors][np.argmax(VoroVol[ID_to_explore[:Nneighbors]])]
                IDnext = ID_to_explore[:Nneighbors][np.argmax(VoroVol[ID_to_explore[:Nneighbors]]* zDens[ID_to_explore[:Nneighbors]])]
        

            Ncells += 1
            Condition = (Dens <= threshold[ith]) & (Ncells < numPart)
        '''
        OUT[ith][i_progr] = ((VolTot * 3. / (4.*np.pi)) ** (1/3), VolTot, Xcm / Norm_cm, Dens)
        #Ncells -= 1
        if Ncells <= 1:

            VolPrevious = 0.
            NormVolPrevious = 0.
            Dens_previous = 0.
            delta_dens = Dens
            Xcm_prev = Xcm *1.
            Norm_cm_prev = Norm_cm

            OUT_prev[ith][i_progr] = (0., VolPrevious, Xcm/Norm_cm, Dens_previous)
            OUT_interp[ith][i_progr] = (-1., 0., Xcm/Norm_cm, Ncells)

            
        else: #Ncells < numPart:
            #delta_Vol = VoroVol[IDthresholds[Ncells-1]]
            VolPrevious = VolTot-VoroVol[IDthresholds[Ncells-1]]
            NormVolPrevious = NormVolTot - VoroVol[IDthresholds[Ncells-1]] * zDens[IDthresholds[Ncells-1]]
            Dens_previous = (Ncells-1) / NormVolPrevious #VolPrevious / zDens[IDthresholds[Ncells-1]]
            delta_dens = Dens - Dens_previous
            Xcm_prev = Xcm - VoroXYZ[IDthresholds[Ncells-1],:] * VoroVol[IDthresholds[Ncells-1]] / zDens[IDthresholds[Ncells-1]]
            Norm_cm_prev = Norm_cm - VoroVol[IDthresholds[Ncells-1]] / zDens[IDthresholds[Ncells-1]]

            OUT_prev[ith][i_progr] = ((VolPrevious * 3. / (4.*np.pi)) ** (1/3), VolPrevious, Xcm_prev/Norm_cm_prev, Dens_previous)
            if (Dens >= threshold[ith]): # (Ncells < numPart):
                frac = (threshold[ith] * NormVolPrevious + 1 - Ncells) / (1. - threshold[ith] * VoroVol[IDthresholds[Ncells-1]] * zDens[IDthresholds[Ncells-1]])
                Vinterp = VolPrevious  + frac * VoroVol[IDthresholds[Ncells-1]]
                Rinterp = (Vinterp * 3. / (4.*np.pi)) ** (1/3)
                Xcm_interp = Xcm_prev + frac * VoroXYZ[IDthresholds[Ncells-1],:] * VoroVol[IDthresholds[Ncells-1]] / zDens[IDthresholds[Ncells-1]]
                Norm_cm_interp = Norm_cm_prev + frac * VoroVol[IDthresholds[Ncells-1]] / zDens[IDthresholds[Ncells-1]]

                #Vinterp = VoroVol[IDthresholds[Ncells-1]] / delta_dens * (threshold[ith] - Dens_previous) + VolPrevious
                #Rinterp = (Vinterp * 3. / (4.*np.pi)) ** (1/3)
                #Xcm_interp = Xcm_prev + VoroXYZ[IDthresholds[Ncells-1],:] * VoroVol[IDthresholds[Ncells-1]] / delta_dens * (threshold[ith] - Dens_previous) / zDens[IDthresholds[Ncells-1]]
                #Norm_cm_interp = Norm_cm_prev + VoroVol[IDthresholds[Ncells-1]] / delta_dens * (threshold[ith] - Dens_previous) / zDens[IDthresholds[Ncells-1]]
                
                OUT_interp[ith][i_progr] = (Rinterp, Vinterp, Xcm_interp/Norm_cm_interp, Ncells-1+frac)
            else:
                OUT_interp[ith][i_progr] = (-1., VolTot, Xcm/Norm_cm, Ncells)
        Ncells_out[ith][i_progr] = Ncells
        #Ncells+=1

    #else:
    #    Rinterp = -1.
    #    Vinterp = 0. 
    return IDthresholds[:Ncells]




def cluster_accretion(i_progr,Nthresholds,threshold,VoroXYZ,VoroVol,zDens,OUT_interp,OUT,OUT_prev,Ncells_out,ID_core=-1):
    print(i_progr,flush=True)
    numPart = VoroXYZ.shape[0]
    if ID_core >= 0:
        IDnext = ID_core
    else:
        IDnext = np.argmax(VoroVol)
    IDanchor = IDnext
    #print(numPart)

    if numPart < 5:
        #print(numPart)
        IDthresholds = np.zeros(numPart,dtype=np.int_)
        ID_to_explore = np.argsort(VoroVol)[::-1]


        Ncells = 1
        Nneighbors = 1
        VolTot = VoroVol[IDnext] #0.
        NormVolTot = VoroVol[IDnext] * zDens[IDnext] #0.
        Dens = Ncells / NormVolTot #0.
        Xcm = VoroXYZ[IDnext,:] * VoroVol[IDnext] / zDens[IDnext]
        Norm_cm = VoroVol[IDnext] / zDens[IDnext]
        #print(0,Ncells,Dens,IDnext,Nneighbors,NormVolTot,VolTot)
        for ith in range(Nthresholds):
            Condition = (Dens <= threshold[ith]) & (Ncells < numPart)
            while Condition:
                IDnext = ID_to_explore[Ncells]
                Ncells += 1
                IDthresholds[Ncells-1] = IDnext

                VolTot += VoroVol[IDnext]
                NormVolTot += VoroVol[IDnext] * zDens[IDnext]
                #Dens = Ncells / VolTot / zDens[IDnext]
                #Dens = Ncells / VolTot / zDens_centr
                Dens = Ncells / NormVolTot
                Xcm += VoroXYZ[IDnext,:] * VoroVol[IDnext] / zDens[IDnext]
                Norm_cm += VoroVol[IDnext] / zDens[IDnext]
                #print(ith,Ncells,Dens,IDnext,Nneighbors,NormVolTot,VolTot)
                Condition = (Dens <= threshold[ith]) & (Ncells < numPart)
            #######################
            
            OUT[ith][i_progr] = ((VolTot * 3. / (4.*np.pi)) ** (1/3), VolTot, Xcm / Norm_cm, Dens)
            #Ncells -= 1
            if Ncells <= 1:

                VolPrevious = 0.
                NormVolPrevious = 0.
                Dens_previous = 0.
                Xcm_prev = Xcm *1.
                Norm_cm_prev = Norm_cm

                OUT_prev[ith][i_progr] = (0., VolPrevious, Xcm/Norm_cm, Dens_previous)
                OUT_interp[ith][i_progr] = (-1., 0., Xcm/Norm_cm,Ncells)

                
            else: #Ncells < numPart:
                #delta_Vol = VoroVol[IDthresholds[Ncells-1]]
                VolPrevious = VolTot-VoroVol[IDthresholds[Ncells-1]]
                NormVolPrevious = NormVolTot - VoroVol[IDthresholds[Ncells-1]] * zDens[IDthresholds[Ncells-1]]
                Dens_previous = (Ncells-1) / NormVolPrevious #VolPrevious / zDens[IDthresholds[Ncells-1]]
                Xcm_prev = Xcm - VoroXYZ[IDthresholds[Ncells-1],:] * VoroVol[IDthresholds[Ncells-1]] / zDens[IDthresholds[Ncells-1]]
                Norm_cm_prev = Norm_cm - VoroVol[IDthresholds[Ncells-1]] / zDens[IDthresholds[Ncells-1]]

                OUT_prev[ith][i_progr] = ((VolPrevious * 3. / (4.*np.pi)) ** (1/3), VolPrevious, Xcm_prev/Norm_cm_prev, Dens_previous)
                if (Dens >= threshold[ith]):
                    frac = (threshold[ith] * NormVolPrevious + 1 - Ncells) / (1. - threshold[ith] * VoroVol[IDthresholds[Ncells-1]] * zDens[IDthresholds[Ncells-1]])
                    Vinterp = VolPrevious  + frac * VoroVol[IDthresholds[Ncells-1]]
                    Rinterp = (Vinterp * 3. / (4.*np.pi)) ** (1/3)
                    Xcm_interp = Xcm_prev + frac * VoroXYZ[IDthresholds[Ncells-1],:] * VoroVol[IDthresholds[Ncells-1]] / zDens[IDthresholds[Ncells-1]]
                    Norm_cm_interp = Norm_cm_prev + frac * VoroVol[IDthresholds[Ncells-1]] / zDens[IDthresholds[Ncells-1]]


                    OUT_interp[ith][i_progr] = (Rinterp, Vinterp, Xcm_interp/Norm_cm_interp,Ncells-1+frac)
                else:
                    OUT_interp[ith][i_progr] = (-1., VolTot, Xcm/Norm_cm,Ncells)
            Ncells_out[ith][i_progr] = Ncells


        return IDthresholds[:Ncells] 
    

    #fig = plt.figure()
    #ax = fig.add_subplot(projection='3d')
    #ax.scatter(VoroXYZ[:,0],VoroXYZ[:,1],VoroXYZ[:,2],s=1,alpha=0.25,c='k')
    
    tri = Delaunay(VoroXYZ)
    neighbor = tri.vertex_neighbor_vertices

    IDthresholds = np.zeros(numPart,dtype=np.int_)
    ID_to_explore = np.zeros(numPart,dtype=np.int_)
    ID_cluster = np.zeros(numPart,dtype=np.int_)

    Ncells = 1
    ID_to_explore[0] = IDanchor
    ID_cluster[0] = IDnext

    IDthresholds[Ncells-1] = IDnext
    Nneighbors = 1
    VolTot = VoroVol[IDnext] #0.
    NormVolTot = VoroVol[IDnext] * zDens[IDnext] #0.
    Dens = Ncells / NormVolTot #0.
    Xcm = VoroXYZ[IDnext,:] * VoroVol[IDnext] / zDens[IDnext] #np.zeros(3)
    Norm_cm = VoroVol[IDnext] / zDens[IDnext] #np.zeros(3)

    #print(0,Ncells,Dens,IDnext,Nneighbors,VoroVol[IDnext] * zDens[IDnext],NormVolTot,VoroVol[IDnext],VolTot)
    #zDens_centr = zDens[IDnext]
    #out_progr = 0
    for ith in range(Nthresholds):
        Condition = (Dens <= threshold[ith]) & (Ncells < numPart) #(Ncells < numPart-1)
        #print(ith,Ncells,Condition,numPart-1)
        while Condition:
            #CC = plt.cm.tab10(out_progr%10)
            #ax.scatter(VoroXYZ[IDanchor,0],VoroXYZ[IDanchor,1],VoroXYZ[IDanchor,2],c=CC)
            ## Add neighbor particles and update ID_to_explore:
            VtoCluster = ~np.isin(neighbor[1][neighbor[0][IDanchor]:neighbor[0][IDanchor+1]],ID_to_explore[:Nneighbors])
            VtoCluster &= ~np.isin(neighbor[1][neighbor[0][IDanchor]:neighbor[0][IDanchor+1]],IDthresholds[:Ncells+1])
            NtoAdd = np.sum(VtoCluster)
            ID_to_explore[Nneighbors:Nneighbors+NtoAdd] = neighbor[1][neighbor[0][IDanchor]:neighbor[0][IDanchor+1]][VtoCluster]
            #for ii in ID_to_explore[Nneighbors:Nneighbors+NtoAdd]:
            #    ax.plot(VoroXYZ[[IDanchor,ii],0],VoroXYZ[[IDanchor,ii],1],VoroXYZ[[IDanchor,ii],2],lw=1,c=CC)

            Cond_innert = True
            ID_to_explore[Nneighbors:Nneighbors+NtoAdd] = ID_to_explore[Nneighbors:Nneighbors+NtoAdd][
                (np.argsort(VoroVol[ID_to_explore[Nneighbors:Nneighbors+NtoAdd]] * zDens[ID_to_explore[Nneighbors:Nneighbors+NtoAdd]])[::-1])]
            inner_progr = 0
            while Cond_innert:
                Ncells += 1
                IDnext = ID_to_explore[Nneighbors+inner_progr]
                #ax.scatter(VoroXYZ[IDnext,0],VoroXYZ[IDnext,1],VoroXYZ[IDnext,2],c=CC)
                IDthresholds[Ncells-1] = IDnext
                VolTot += VoroVol[IDnext]
                NormVolTot += VoroVol[IDnext] * zDens[IDnext]
                #Dens = Ncells / VolTot / zDens[IDnext]
                #Dens = Ncells / VolTot / zDens_centr
                Dens = Ncells / NormVolTot
                Xcm += VoroXYZ[IDnext,:] * VoroVol[IDnext] / zDens[IDnext]
                Norm_cm += VoroVol[IDnext] / zDens[IDnext]

                #print(ith,Ncells,inner_progr,Dens,IDnext,Nneighbors,VoroVol[IDnext] * zDens[IDnext],NormVolTot,VoroVol[IDnext],VolTot)

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
            
        OUT[ith][i_progr] = ((VolTot * 3. / (4.*np.pi)) ** (1/3), VolTot, Xcm / Norm_cm, Dens)
        #Ncells -= 1
        if Ncells <= 1:

            VolPrevious = 0.
            NormVolPrevious = 0.
            Dens_previous = 0.
            Xcm_prev = Xcm *1.
            Norm_cm_prev = Norm_cm

            OUT_prev[ith][i_progr] = (0., VolPrevious, Xcm/Norm_cm, Dens_previous)
            OUT_interp[ith][i_progr] = (-1., 0., Xcm/Norm_cm, Ncells)

            
        else: 
            VolPrevious = VolTot-VoroVol[IDthresholds[Ncells-1]]
            NormVolPrevious = NormVolTot - VoroVol[IDthresholds[Ncells-1]] * zDens[IDthresholds[Ncells-1]]
            Dens_previous = (Ncells-1) / NormVolPrevious 
            Xcm_prev = Xcm - VoroXYZ[IDthresholds[Ncells-1],:] * VoroVol[IDthresholds[Ncells-1]] / zDens[IDthresholds[Ncells-1]]
            Norm_cm_prev = Norm_cm - VoroVol[IDthresholds[Ncells-1]] / zDens[IDthresholds[Ncells-1]]

            OUT_prev[ith][i_progr] = ((VolPrevious * 3. / (4.*np.pi)) ** (1/3), VolPrevious, Xcm_prev/Norm_cm_prev, Dens_previous)
            if (Dens >= threshold[ith]): 
                frac = (threshold[ith] * NormVolPrevious + 1 - Ncells) / (1. - threshold[ith] * VoroVol[IDthresholds[Ncells-1]] * zDens[IDthresholds[Ncells-1]])
                Vinterp = VolPrevious  + frac * VoroVol[IDthresholds[Ncells-1]]
                Rinterp = (Vinterp * 3. / (4.*np.pi)) ** (1/3)
                Xcm_interp = Xcm_prev + frac * VoroXYZ[IDthresholds[Ncells-1],:] * VoroVol[IDthresholds[Ncells-1]] / zDens[IDthresholds[Ncells-1]]
                Norm_cm_interp = Norm_cm_prev + frac * VoroVol[IDthresholds[Ncells-1]] / zDens[IDthresholds[Ncells-1]]

                OUT_interp[ith][i_progr] = (Rinterp, Vinterp, Xcm_interp/Norm_cm_interp, Ncells-1+frac)
            else:
                OUT_interp[ith][i_progr] = (-1., VolTot, Xcm/Norm_cm, Ncells)
        Ncells_out[ith][i_progr] = Ncells

    return IDthresholds[:Ncells]







@jit(nopython=True)
def is_not_in_arr(ar1,ar2):
    mask = np.ones(len(ar1), dtype=np.bool_)
    for a in ar2:
        mask &= (ar1 != a)
    return mask

@jit(nopython=True)
def cluster_accretion(ID_core,neighbor_ptr,neighbor_ids,Nthresholds,threshold,VoroXYZ,VoroVol,zDens):
    #print(i_progr,flush=True)
    numPart = VoroXYZ.shape[0]
    #if ID_core >= 0:
    #    IDnext = ID_core
    #else:
    #    IDnext = np.argmax(VoroVol)
    #IDnext = np.argmax(VoroVol)
    IDnext = ID_core
    IDanchor = IDnext
    #print(numPart)


    IDthresholds = np.zeros(numPart,dtype=np.int_)
    ID_to_explore = np.zeros(numPart,dtype=np.int_)
    ID_cluster = np.zeros(numPart,dtype=np.int_)

    Ncells = 0
    ID_to_explore[0] = IDanchor
    ID_cluster[0] = IDnext

    IDthresholds[Ncells] = IDnext
    Ncells = 1
    Nneighbors = 1
    VolTot = VoroVol[IDnext] #0.
    #NormVolTot = VoroVol[IDnext] * zDens[IDnext] #0.
    #Dens = Ncells / NormVolTot #0.
    numerator_dens = 1. / zDens[IDnext]
    Dens = numerator_dens / VolTot
    Xcm = VoroXYZ[IDnext,:] * VoroVol[IDnext] / zDens[IDnext] #np.zeros(3)
    Norm_cm = VoroVol[IDnext] / zDens[IDnext] #np.zeros(3)

    #print(0,Ncells,Dens,IDnext,Nneighbors,VoroVol[IDnext] * zDens[IDnext],NormVolTot,VoroVol[IDnext],VolTot)
    #zDens_centr = zDens[IDnext]
    #out_progr = 0
    for ith in range(Nthresholds):
        Condition = (Dens <= threshold[ith]) & (Ncells < numPart) #(Ncells < numPart-1)
        #print(ith,Ncells,Condition,numPart-1)
        while Condition:
            #CC = plt.cm.tab10(out_progr%10)
            #ax.scatter(VoroXYZ[IDanchor,0],VoroXYZ[IDanchor,1],VoroXYZ[IDanchor,2],c=CC)
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




        #if (Dens >= threshold[ith]): 
        VolPrevious = VolTot-VoroVol[IDthresholds[Ncells-1]]
        numerator_dens_previous = numerator_dens - 1./zDens[IDthresholds[Ncells-1]]
        frac = (threshold[ith] * VolPrevious - numerator_dens_previous) / (1. / zDens[IDthresholds[Ncells-1]] - threshold[ith] * VoroVol[IDthresholds[Ncells-1]])

        Vol_interp = VolPrevious + frac * VoroVol[IDthresholds[Ncells-1]]
        Ncells_in_void = Ncells-1+frac

        Xcm_interp = np.zeros(3)
        Coord_norm = np.sum(1. / (VoroVol[IDthresholds[:Ncells-1]] * zDens[IDthresholds[:Ncells-1]])) +  1. / (frac * VoroVol[IDthresholds[Ncells-1]] * zDens[IDthresholds[Ncells-1]])
        Xcm_interp[0] = (np.sum(VoroXYZ[IDthresholds[:Ncells-1],0] / (VoroVol[IDthresholds[:Ncells-1]] * zDens[IDthresholds[:Ncells-1]])) + 
                            VoroXYZ[IDthresholds[Ncells-1],0] / (frac * VoroVol[IDthresholds[Ncells-1]] * zDens[IDthresholds[Ncells-1]])) / Coord_norm
        Xcm_interp[1] = (np.sum(VoroXYZ[IDthresholds[:Ncells-1],1] / (VoroVol[IDthresholds[:Ncells-1]] * zDens[IDthresholds[:Ncells-1]])) + 
                            VoroXYZ[IDthresholds[Ncells-1],1] / (frac * VoroVol[IDthresholds[Ncells-1]] * zDens[IDthresholds[Ncells-1]])) / Coord_norm
        Xcm_interp[2] = (np.sum(VoroXYZ[IDthresholds[:Ncells-1],2] / (VoroVol[IDthresholds[:Ncells-1]] * zDens[IDthresholds[:Ncells-1]])) + 
                            VoroXYZ[IDthresholds[Ncells-1],2] / (frac * VoroVol[IDthresholds[Ncells-1]] * zDens[IDthresholds[Ncells-1]])) / Coord_norm
        
        # geometrical shape
        shape_matrix = np.zeros((3,3))
        # density field
        for i in range(3):
            shape_matrix[i,i] = (np.sum((VoroXYZ[IDthresholds[:Ncells-1],(i+1) % 3]**2 + VoroXYZ[IDthresholds[:Ncells-1],(i+2) % 3]**2) / zDens[IDthresholds[:Ncells-1]]) + 
                                    (VoroXYZ[IDthresholds[Ncells-1],(i+1) % 3]**2 + VoroXYZ[IDthresholds[Ncells-1],(i+2) % 3]**2)  * frac / zDens[IDthresholds[Ncells-1]])
            for j in range(i):
                shape_matrix[i,j] = -(np.sum(VoroXYZ[IDthresholds[:Ncells-1],i] * VoroXYZ[IDthresholds[:Ncells-1],j] / zDens[IDthresholds[:Ncells-1]]) + 
                                        VoroXYZ[IDthresholds[Ncells-1],i] * VoroXYZ[IDthresholds[Ncells-1],j] * frac / zDens[IDthresholds[Ncells-1]])
                shape_matrix[j,i] = shape_matrix[i,j]

        eigenvalues, eigenvectors = np.linalg.eig(shape_matrix)
        #ell_shape = 1. - (eigenvalues[0] / (eigenvalues[2] + int(eigenvalues[2] == 0.))) ** (0.25)
        
        #frac = (threshold[ith] * NormVolPrevious + 1 - Ncells) / (1. - threshold[ith] * VoroVol[IDthresholds[Ncells-1]] * zDens[IDthresholds[Ncells-1]])
        #Vinterp = VolPrevious  + frac * VoroVol[IDthresholds[Ncells-1]]
        #Rinterp = (Vinterp * 3. / (4.*np.pi)) ** (1/3)
        #Xcm_interp = Xcm_prev + frac * VoroXYZ[IDthresholds[Ncells-1],:] * VoroVol[IDthresholds[Ncells-1]] / zDens[IDthresholds[Ncells-1]]
        #Norm_cm_interp = Norm_cm_prev + frac * VoroVol[IDthresholds[Ncells-1]] / zDens[IDthresholds[Ncells-1]]

        #OUT_interp[ith][i_progr] = (Rinterp, Vinterp, Xcm_interp/Norm_cm_interp, Ncells-1+frac)
        #else:
        #    OUT_interp[ith][i_progr] = (-1., VolTot, Xcm/Norm_cm, Ncells)
            
    return IDthresholds[:Ncells], Xcm_interp, Vol_interp, Ncells_in_void, eigenvalues, eigenvectors



def XYZ_from_rRAdec(r,RA,dec):
    out = np.empty((len(r),3))
    out[:,0] = r * np.cos(dec * np.pi / 180.) * np.sin(RA * np.pi / 180.)
    out[:,1] = r * np.cos(dec * np.pi / 180.) * np.cos(RA * np.pi / 180.)
    out[:,2] = r * np.sin(dec * np.pi / 180.)
    return out


def select_IDs_general(ID_core, XYZ_particles, Rmax):
    mask = np.ones(XYZ_particles.shape[0],dtype=np.bool_)
    mask[ID_core] = False
    for i in range(XYZ_particles.shape[1]):
        mask[:] &= np.abs(XYZ_particles[:,i] - XYZ_particles[ID_core,i]) < Rmax
    IDs_out = np.empty(np.sum(mask)+1,dtype=np.int_)
    IDs_out[0] = ID_core
    IDs_out[1:] = np.arange(XYZ_particles.shape[0])[mask]
    return IDs_out



def parallel_launch_standard(ivd,IDs_core,Rmax,part_xyz,part_vol,density_z,Ncells_tot,Nthresholds,threshold_arr,OUT_interp,OUT,OUT_prev,Ncells_out):
    IDs_select = select_IDs_general(IDs_core[ivd], part_xyz, Rmax)
    VoroVol = part_vol[IDs_select]
    VoroDensZ = density_z[IDs_select]
    Ncells_tot[ivd] = len(VoroDensZ)
    
    VoroXYZ = part_xyz[IDs_select,:]

    IDs_particles = single_void_computation_Zdens(ivd,Nthresholds,threshold_arr,VoroXYZ,VoroVol,VoroDensZ,OUT_interp,OUT,OUT_prev,Ncells_out,ID_core=0)
    return IDs_select[IDs_particles], ivd



def parallel_launch_cluster(ivd,IDs_core,Rmax,part_xyz,part_vol,density_z,Ncells_tot,Nthresholds,threshold_arr,OUT_interp,OUT,OUT_prev,Ncells_out):
    IDs_select = select_IDs_general(IDs_core[ivd], part_xyz, Rmax)
    VoroVol = part_vol[IDs_select]
    VoroDensZ = density_z[IDs_select]
    Ncells_tot[ivd] = len(VoroDensZ)
    
    VoroXYZ = part_xyz[IDs_select,:]

    IDs_particles = cluster_accretion(ivd,Nthresholds,threshold_arr,VoroXYZ,VoroVol,VoroDensZ,OUT_interp,OUT,OUT_prev,Ncells_out,ID_core=0)
    return IDs_select[IDs_particles], ivd

def voronoi_threhosld_lightcone_videcat(threshold,path_to_test,dataPortion="all",untrimmed=True,voro_in_basin=False,
                                        MeanConeDens=-1.,DensZ=None,distance_func_z=None,Rmax=-1.,nCPU=-1,cluster_method=True):
    

    c_kms = 299792.458

    print('Load catalog...', flush=True)
    time1 = time.time()
    catalog = vu.loadVoidCatalog(path_to_test, 
                                dataPortion=dataPortion, 
                                loadParticles=True,
                                untrimmed=untrimmed)
    DeltaT = time.time() - time1
    hh = int(DeltaT / 3600)
    mm = int(DeltaT / 60) - hh * 60
    sec = DeltaT % 60
    print('    done. ' + str(hh) + ' h, ' + str(mm) + ' m, ' + str(sec) + ' sec.\n', flush=True)
    print('box info:','\n    nvoids:',catalog.numVoids,'\n    num part:',catalog.numPartTot,
        '\n    box len',catalog.boxLen,'\n    vol norm:',catalog.volNorm,'\n    fake density:',catalog.sampleInfo.fakeDensity)

    if np.isscalar(threshold):
        threshold_arr = [threshold]
    else:
        threshold_arr = threshold
    Nthresholds = len(threshold_arr)

    NumDensBox = catalog.numPartTot/np.prod(catalog.boxLen)
    VoroNormVol = NumDensBox / catalog.volNorm

    if DensZ == None:
        '''
        for ivd in range(catalog.numVoids):
            print(ivd+1,'/',catalog.numVoids)
            voidPart = vu.getVoidPart(catalog, catalog.voids[ivd].voidID)
            VoroVol = vu.getArray(voidPart,'volume') * VoroNormVol
            VoroXYZ = np.array([vu.getArray(voidPart,'x'),vu.getArray(voidPart,'y'),vu.getArray(voidPart,'z')]).T

            Rinterp[ivd], Vinterp[ivd], Ncells[ivd], Dens_out[ivd], Rout[ivd], Vout[ivd], Dens_in[ivd], Rin[ivd], Vin[ivd] = \
                single_void_computation_constmeandens(threshold,VoroXYZ,VoroVol,MeanConeDens)
        '''
        if MeanConeDens < 0.:
            DensZ = lambda x: np.full(len(x), catalog.volNorm)
        else:
            DensZ = lambda x: np.full(len(x), MeanConeDens)
    if distance_func_z == None:
        XYZ_func = lambda voidPart: np.array([vu.getArray(voidPart,'x'),vu.getArray(voidPart,'y'),vu.getArray(voidPart,'z')]).T
    else:
        XYZ_func = lambda voidPart: XYZ_from_rRAdec(distance_func_z(vu.getArray(voidPart,'redshift')/c_kms),vu.getArray(voidPart,'ra'),vu.getArray(voidPart,'dec'))


    if cluster_method:
        func_to_use = cluster_accretion
        print('Method: cluster accretion',flush=True)
    else:
        func_to_use = single_void_computation_Zdens
        print('Method: sequential lowest dense neighbor',flush=True)

    if (nCPU <= 0) | (nCPU > multiprocessing.cpu_count()):
        nCPU = multiprocessing.cpu_count()

    if (nCPU > 1) & (not voro_in_basin):


        if cluster_method:
            func_to_use = parallel_launch_cluster
        else:
            func_to_use = parallel_launch_standard


        class MyManager(multiprocessing.managers.BaseManager):
            pass
        MyManager.register('np_zeros', np.zeros, multiprocessing.managers.ArrayProxy)
        
        mp_np = MyManager()
        mp_np.start()

        OUT_interp = []
        for ith in range(Nthresholds):
            OUT_interp.append(mp_np.np_zeros(catalog.numVoids,dtype=[('R', np.float_), ('Vol', np.float_), ('Xcm', np.float_, 3), ('Ninterp', np.float_)]))
        OUT = []
        for ith in range(Nthresholds):
            OUT.append(mp_np.np_zeros(catalog.numVoids,dtype=[('R', np.float_), ('Vol', np.float_), ('Xcm', np.float_, 3), ('delta', np.float_)]))
        OUT_prev = []
        for ith in range(Nthresholds):
            OUT_prev.append(mp_np.np_zeros(catalog.numVoids,dtype=[('R', np.float_), ('Vol', np.float_), ('Xcm', np.float_, 3), ('delta', np.float_)]))
        Ncells_out = []
        for ith in range(Nthresholds):
            Ncells_out.append(mp_np.np_zeros(catalog.numVoids,dtype=np.int_))
        Ncells_tot = mp_np.np_zeros(catalog.numVoids,dtype=np.int_)
    else:

        if cluster_method:
            func_to_use = cluster_accretion
        else:
            func_to_use = single_void_computation_Zdens

        OUT_interp = []
        for ith in range(Nthresholds):
            OUT_interp.append(np.zeros(catalog.numVoids,dtype=[('R', np.float_), ('Vol', np.float_), ('Xcm', np.float_, 3), ('Ninterp', np.float_)]))
        OUT = []
        for ith in range(Nthresholds):
            OUT.append(np.zeros(catalog.numVoids,dtype=[('R', np.float_), ('Vol', np.float_), ('Xcm', np.float_, 3), ('delta', np.float_)]))
        OUT_prev = []
        for ith in range(Nthresholds):
            OUT_prev.append(np.zeros(catalog.numVoids,dtype=[('R', np.float_), ('Vol', np.float_), ('Xcm', np.float_, 3), ('delta', np.float_)]))
        Ncells_out = []
        for ith in range(Nthresholds):
            Ncells_out.append(np.zeros(catalog.numVoids,dtype=np.int_))
        Ncells_tot = np.zeros(catalog.numVoids,dtype=np.int_)


    print('\nthresholding started', flush=True)
    IDs_part_out = []
    time2 = time.time()
    if not voro_in_basin:
        IDs_core = vu.getArray(catalog.voids,'coreParticle').astype(np.int_)
        uniqueID = vu.getArray(catalog.part,'uniqueID').astype(np.int_)
        #check continuity of uniqueID 
        #if np.max(uniqueID[1:] - uniqueID[:-1]) > 1:
        #    print('check uniqueID, np.max(uniqueID[1:] - uniqueID[:-1]) =',np.max(uniqueID[1:] - uniqueID[:-1]),flush=True)
        #part_dec = vu.getArray(catalog.part,'dec')
        #part_ra = vu.getArray(catalog.part,'ra')
        part_vol = vu.getArray(catalog.part,'volume') * VoroNormVol
        #part_redshift = vu.getArray(catalog.part,'redshift') / c_kms
        density_z = DensZ(vu.getArray(catalog.part,'redshift') / c_kms)
        if distance_func_z == None:
            part_xyz = np.array([vu.getArray(catalog.part,'x'),vu.getArray(catalog.part,'y'),vu.getArray(catalog.part,'z')]).T
        else:
            part_xyz = XYZ_from_rRAdec(distance_func_z(vu.getArray(catalog.part,'redshift') / c_kms),vu.getArray(catalog.part,'ra'),vu.getArray(catalog.part,'dec'))
        if Rmax < 0:
            Rmax = (np.sum(part_vol) / 10) ** (1./3.)
            print('passed Rmax < 0, set to',Rmax,flush=True)

        if nCPU > 1:
            print('parallel execution, nCPU:',nCPU)
            #def parallel_launch(ivd):
            #    IDs_select = select_IDs_general(IDs_core[ivd], part_xyz, Rmax)
            #    VoroVol = part_vol[IDs_select]
            #    VoroDensZ = density_z[IDs_select]
            #    Ncells_tot[ivd] = len(VoroDensZ)
            #    
            #    VoroXYZ = part_xyz[IDs_select,:]
            
            #    IDs_particles = single_void_computation_Zdens(ivd,Nthresholds,threshold_arr,VoroXYZ,VoroVol,VoroDensZ,OUT_interp,OUT,OUT_prev,Ncells_out,ID_core=0)
            #    return IDs_select[IDs_particles], ivd
            

            #t0 = time.time()
            parallel_launch_loop = partial(func_to_use, IDs_core=IDs_core,Rmax=Rmax,part_xyz=part_xyz,part_vol=part_vol,density_z=density_z,
                                           Ncells_tot=Ncells_tot,Nthresholds=Nthresholds,threshold_arr=threshold_arr,
                                           OUT_interp=OUT_interp,OUT=OUT,OUT_prev=OUT_prev,Ncells_out=Ncells_out)
            with multiprocessing.Pool(nCPU) as pool:
                result = pool.map(parallel_launch_loop, range(0,catalog.numVoids))
                #result.get()
            #print('\nFinished',StrHminSec(time.time()-t0))
            IDs_part_out = []
            #ID_sort = np.empty(catalog.numVoids,dtype=np.int_)
            for ivd in range(catalog.numVoids):
                IDs_part_out.append(uniqueID[result[ivd][0]])
                #ID_sort[ivd] = result[ivd][1]


            for ith in range(Nthresholds):
                OUT_interp[ith] = np.array(OUT_interp[ith])
                OUT[ith] = np.array(OUT[ith])
                OUT_prev[ith] = np.array(OUT_prev[ith])
                Ncells_out[ith] = np.array(Ncells_out[ith])
                Ncells_tot[ith] = np.array(Ncells_tot[ith])
        else:
            print('sequential execution, nCPU:',nCPU)
            for ivd in range(catalog.numVoids):
                #print(ivd+1,'/',catalog.numVoids)
                #ID_core = catalog.voids[ivd].coreParticle
                IDs_select = select_IDs_general(IDs_core[ivd], part_xyz, Rmax)
                VoroVol = part_vol[IDs_select]
                VoroDensZ = density_z[IDs_select]
                Ncells_tot[ivd] = len(VoroDensZ)
                
                VoroXYZ = part_xyz[IDs_select,:]

                IDs_particles = func_to_use(ivd,Nthresholds,threshold_arr,VoroXYZ,VoroVol,VoroDensZ,OUT_interp,OUT,OUT_prev,Ncells_out,ID_core=0)
                IDs_part_out.append(uniqueID[IDs_select[IDs_particles]])
    else:
        for ivd in range(catalog.numVoids):
            #print(ivd+1,'/',catalog.numVoids)
            voidPart = vu.getVoidPart(catalog, catalog.voids[ivd].voidID)
            VoroVol = vu.getArray(voidPart,'volume') * VoroNormVol
            VoroDensZ = DensZ(vu.getArray(voidPart,'redshift') / c_kms)
            IDs_select = vu.getArray(voidPart,'uniqueID').astype(np.int_)
            Ncells_tot[ivd] = len(VoroDensZ)
            
            VoroXYZ = XYZ_func(voidPart) ##np.array([vu.getArray(voidPart,'x'),vu.getArray(voidPart,'y'),vu.getArray(voidPart,'z')]).T

            #Rinterp[ivd], Vinterp[ivd], Ncells[ivd], Dens_out[ivd], Rout[ivd], Vout[ivd], Dens_in[ivd], Rin[ivd], Vin[ivd] = \
            #    single_void_computation_Zdens(threshold,VoroXYZ,VoroVol,DensZ(VoroZ))
            #OUT_interp, OUT, OUT_prev, Ncells[ivd] = 
            IDs_particles = func_to_use(ivd,Nthresholds,threshold_arr,VoroXYZ,VoroVol,VoroDensZ,OUT_interp,OUT,OUT_prev,Ncells_out)
            IDs_part_out.append(IDs_select[IDs_particles])
            #Rinterp[ivd], Vinterp[ivd], Xcm_interp[ivd] = OUT_interp
            #Rout[ivd], Vout[ivd], Xcm_out[ivd], Dens_out[ivd] = OUT
            #Rin[ivd], Vin[ivd], Xcm_in[ivd], Dens_in[ivd] = OUT_prev
    DeltaT = time.time() - time2
    hh = int(DeltaT / 3600)
    mm = int(DeltaT / 60) - hh * 60
    sec = DeltaT % 60
    print('    done. ' + str(hh) + ' h, ' + str(mm) + ' m, ' + str(sec) + ' sec.\n', flush=True)
    DeltaT = time.time() - time1
    hh = int(DeltaT / 3600)
    mm = int(DeltaT / 60) - hh * 60
    sec = DeltaT % 60
    print('Total time:' + str(hh) + ' h, ' + str(mm) + ' m, ' + str(sec) + ' sec.\n', flush=True)
    
    return OUT_interp, OUT, OUT_prev, IDs_part_out, Ncells_out, Ncells_tot #(Rinterp, Vinterp, Xcm_interp), (Rout, Vout, Xcm_out, Dens_out), (Rin, Vin, Xcm_in, Dens_in), Ncells
    
    #return Rinterp, Vinterp, Ncells, Dens_out, Rout, Vout, Dens_in, Rin, Vin





