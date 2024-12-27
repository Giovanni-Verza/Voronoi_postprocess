#+
#   VIDE -- Void IDentification and Examination -- ./python_tools/vide/voidUtil/plotUtil.py
#   Copyright (C) 2010-2014 Guilhem Lavaux
#   Copyright (C) 2011-2014 P. M. Sutter
#
#   This program is free software; you can redistribute it and/or modify
#   it under the terms of the GNU General Public License as published by
#   the Free Software Foundation; version 2 of the License.
# 
#
#   This program is distributed in the hope that it will be useful,
#   but WITHOUT ANY WARRANTY; without even the implied warranty of
#   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#   GNU General Public License for more details.
#
#   You should have received a copy of the GNU General Public License along
#   with this program; if not, write to the Free Software Foundation, Inc.,
#   51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
#+
__all__=['plotNumberFunction','plotEllipDist','plotVoidCells']

from vide.backend.classes import *
from .plotDefs import *
import numpy as np
import os
import pylab as plt
import vide.apTools as vp
from vide.voidUtil import getArray, shiftPart, getVoidPart

def fill_between(x, y1, y2=0, ax=None, **kwargs):
    """Plot filled region between `y1` and `y2`.

    This function works exactly the same as matplotlib's fill_between, except
    that it also plots a proxy artist (specifically, a rectangle of 0 size)
    so that it can be added it appears on a legend.
    """
    ax = ax if ax is not None else plt.gca()
    ax.fill_between(x, y1, y2, interpolate=True, **kwargs)
    p = plt.Rectangle((0, 0), 0, 0, **kwargs)
    ax.add_patch(p)

# -----------------------------------------------------------------------------
def plotNumberFunction(catalogList, 
                       figDir="./", 
                       plotName="numberfunc",
                       cumulative=True,
                       binWidth=1):

# plots a cumulative number function
#   catalogList: list of void catalogs to plot
#   figDir: output directory for figures
#   plotName: name to prefix to all outputs
#   cumulative: if True, plots cumulative number function
#   binWidth: width of histogram bins in Mpc/h
# returns:
#   ellipDistList: array of len(catalogList), 
#                   each element has array of size bins, number, +/- 1 sigma

  print("Plotting number function")
  
  catalogList = np.atleast_1d(catalogList)

  plt.clf()
  plt.xlabel("$R_{eff}$ [$h^{-1}Mpc$]", fontsize=14)
  
  if cumulative:
    plt.ylabel(r"log ($n$ (> R) [$h^3$ Gpc$^{-3}$])", fontsize=14)
  else:
    plt.ylabel(r"log ($dn/dR$ [$h^3$ Gpc$^{-3}$])", fontsize=14)
 
  ellipDistList = [] 

  for (iSample,catalog) in enumerate(catalogList):
    sample = catalog.sampleInfo
    data = getArray(catalog.voids, 'radius')
  
    if sample.dataType == "observation":
      maskFile = sample.maskFile
  
      boxVol = vp.getSurveyProps(maskFile,
                                 sample.zBoundary[0], sample.zBoundary[1],
                                 sample.zRange[0], sample.zRange[1], "all",
                                 selectionFuncFile=sample.selFunFile)[0]
    else:
      boxVol = sample.boxLen*sample.boxLen*(sample.zBoundaryMpc[1] -
                                            sample.zBoundaryMpc[0])
  
    boxVol *= 1.e-9 # Mpc->Gpc
  
    bins = 100./binWidth
    hist, binEdges = np.histogram(data, bins=bins, range=(0., 100.))
    binCenters = 0.5*(binEdges[1:] + binEdges[:-1])

    if cumulative:
      foundStart = False
      for iBin in range(len(hist)):
        if not foundStart and hist[iBin] == 0:
          continue
        foundStart = True
        hist[iBin] = np.sum(hist[iBin:])
  
    nvoids = len(data)
    var = hist * (1. - hist/nvoids)
    sig = np.sqrt(var)
  
    lowerbound = hist - sig
    upperbound = hist + sig
  
    mean = np.log10(hist/boxVol)
    lowerbound = np.log10(lowerbound/boxVol)
    upperbound = np.log10(upperbound/boxVol)
  
    lineColor = colorList[iSample]
    lineTitle = sample.fullName
    
    trim = (lowerbound > .01)
    mean = mean[trim]
    binCentersToUse = binCenters[trim]
    lower = lowerbound[trim]
    upper = upperbound[trim]
  
    alpha = 0.55
    fill_between(binCentersToUse, lower, upper,
               label=lineTitle, color=lineColor,
               alpha=alpha,
               )
  
    lineStyle = '-'
    plt.plot(binCentersToUse, mean, lineStyle,
                color=lineColor,
                linewidth=3)
    
    ellipDistList.append((binCentersToUse, mean, lower, upper))
  
  plt.legend(loc = "upper right", fancybox=True, prop={'size':14})
  
  plt.savefig(figDir+"/fig_"+plotName+".pdf", bbox_inches="tight")
  plt.savefig(figDir+"/fig_"+plotName+".eps", bbox_inches="tight")
  plt.savefig(figDir+"/fig_"+plotName+".png", bbox_inches="tight")

  return ellipDistList 


# -----------------------------------------------------------------------------
def plotEllipDist(catalogList, 
                  figDir="./", 
                  plotName="ellipdist"):

# plots ellipticity distributions
#   catalogList: list of void catalogs to plot
#   figDir: output directory for figures
#   plotName: name to prefix to all outputs
# returns:
#   ellipDistList: array of len(catalogList), 
#                  each element has array of ellipticity distributions

  print("Plotting ellipticity distributions")
  
  plt.clf()
  plt.xlabel(r"Ellipticity $\epsilon$", fontsize=14)
  plt.ylabel(r"P($\epsilon$)", fontsize=14)
  
  ellipDistList = [] 

  catalogList = np.atleast_1d(catalogList)

  for (iSample,catalog) in enumerate(catalogList):
    sample = catalog.sampleInfo
    data = getArray(catalog.voids, 'ellipticity')
  
    dataWeights = np.ones_like(data)/len(data)
    dataHist, dataBins = np.histogram(data, bins=10, weights=dataWeights,
                                         range=(0.0,0.35)) 
    binCenters = 0.5*(dataBins[1:] + dataBins[:-1])
  
    plt.plot(binCenters, dataHist, label=sample.fullName, 
             color=colorList[iSample])
  
    ellipDistList.append((dataBins, dataHist,))
    
  plt.legend(loc = "upper right", fancybox=True, prop={'size':14})
  
  plt.savefig(figDir+"/fig_"+plotName+".pdf", bbox_inches="tight")
  plt.savefig(figDir+"/fig_"+plotName+".eps", bbox_inches="tight")
  plt.savefig(figDir+"/fig_"+plotName+".png", bbox_inches="tight")
  
  return


# -----------------------------------------------------------------------------
def plotVoidCells(catalog,
              voidID,
              figDir="./",
              plotName="cells",
              plotDensity=True,
              sliceWidth=250):

# plots the particles belonging to a void
#   catalog: void catalog
#   voidID: ID of void to plot
#   figDir: output directory for figures
#   plotName: name to prefix to all outputs
#   sliceWidth: width of slice in Mpc/h

  sample = catalog.sampleInfo
  periodicLine = getPeriodic(sample)

  plt.clf()

  iVoid = -1
  for i in range(len(catalog.voids)):
    if catalog.voids[i].voidID == voidID:
      iVoid = i
      break

  if iVoid == -1:
    print("Void ID %d not found!" % voidID)
    return
  
  sliceCenter = catalog.voids[iVoid].macrocenter

  xwidth = sliceWidth
  ywidth = sliceWidth
  zwidth = max(sliceWidth/4., 50)

  xmin = -xwidth/2.
  xmax = xwidth/2.
  ymin = -ywidth/2.
  ymax = ywidth/2.
  zmin = -zwidth/2.
  zmax = zwidth/2.

  #Slice Particles
  if plotDensity:
    part = catalog.partPos
    part = shiftPart(part, sliceCenter, periodicLine, catalog.ranges) 

    filter = (part[:,0] > xmin) & (part[:,0] < xmax) & \
             (part[:,1] > ymin) & (part[:,1] < ymax) & \
             (part[:,2] > zmin) & (part[:,2] < zmax)
    part = part[filter]

    extent = [xmin, xmax, ymin, ymax]
    hist, xedges, yedges = np.histogram2d(part[:,0], part[:,1], normed=False,
                                          bins=64)

    hist = np.log10(hist+1)
    plt.imshow(hist,
               aspect='equal',
               extent=extent,
               interpolation='gaussian',
               cmap='YlGnBu_r')


  # overlay voids as circles
  fig = plt.gcf()

  voidPart = getVoidPart(catalog, voidID)

  newpart = np.zeros((len(voidPart),3))
  volume=np.zeros(len(voidPart))
  radius=np.zeros(len(voidPart))

  newpart[:,0] = getArray(voidPart, 'x')
  newpart[:,1] = getArray(voidPart, 'y')
  newpart[:,2] = getArray(voidPart, 'z')

  volume = getArray(voidPart, 'volume')
  radius = (3.*volume/(4.*np.pi))**(1./3.)

  shiftedPartVoid =shiftPart(newpart,sliceCenter, periodicLine, catalog.ranges)

  #Limiting plotted cells to cells into the slice
  #Possibility to only plot bigger cells (through cellsradiuslim)
  cellsMinlimz = zmin
  cellsMaxlimz = zmax
  cellsradiuslim = 0.0

  for p in range(len(volume)):
    if (shiftedPartVoid[p,2]>(cellsMinlimz) and \
        shiftedPartVoid[p,2]<(cellsMaxlimz) and \
        radius[p]>cellsradiuslim):
      color = 'blue'
      circle = plt.Circle((shiftedPartVoid[p,0], \
                           shiftedPartVoid[p,1]), \
                           radius[p],
                           alpha =.2, fc=color,edgecolor=None,linewidth=1)
      fig.gca().add_artist(circle)

  title="cells"+str(voidID)
  plt.title(title, fontsize=20)
  plotName="cells"+str(voidID)

  plt.savefig(figDir+"/fig_"+plotName+".pdf", bbox_inches="tight")
  plt.savefig(figDir+"/fig_"+plotName+".eps", bbox_inches="tight")
  plt.savefig(figDir+"/fig_"+plotName+".png", bbox_inches="tight")

  return
