#+
#   VIDE -- Void IDentification and Examination -- ./python_tools/vide/voidUtil/catalogUtil.py
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

# Various utilities for loading and modifying particle datasets

import numpy as np
from netCDF4 import Dataset
import sys
from vide.backend import *
#import vide.apTools as vp
import pickle
from .periodic_kdtree import PeriodicCKDTree
import os

NetCDFFile = Dataset
ncFloat = 'f8'

# -----------------------------------------------------------------------------
def loadPart(sampleDir):
    print("    Loading particle data...")
    sys.stdout.flush()

    with open(sampleDir+"/sample_info.dat", 'rb') as input:
      sample = pickle.load(input)

    infoFile = sampleDir+"/zobov_slice_"+sample.fullName+".par"
    File = NetCDFFile(infoFile, 'r')
    ranges = np.zeros((3,2))
    ranges[0][0] = getattr(File, 'range_x_min')
    ranges[0][1] = getattr(File, 'range_x_max')
    ranges[1][0] = getattr(File, 'range_y_min')
    ranges[1][1] = getattr(File, 'range_y_max')
    ranges[2][0] = getattr(File, 'range_z_min')
    ranges[2][1] = getattr(File, 'range_z_max')
    isObservation = getattr(File, 'is_observation')
    maskIndex = getattr(File, 'mask_index')
    File.close()
    mul = np.zeros((3))
    mul[:] = ranges[:,1] - ranges[:,0]

    partFile = sampleDir+"/zobov_slice_"+sample.fullName
    iLine = 0
    partData = []
    part = np.zeros((3))
    with open(partFile, mode="rb") as File:
      chk = np.fromfile(File, dtype=np.int32,count=1)
      Np = np.fromfile(File, dtype=np.int32,count=1)[0]
      chk = np.fromfile(File, dtype=np.int32,count=1)

      chk = np.fromfile(File, dtype=np.int32,count=1)
      x = np.fromfile(File, dtype=np.float32,count=Np)
      x *= mul[0]
      if isObservation != 1:
        x += ranges[0][0]
      chk = np.fromfile(File, dtype=np.int32,count=1)

      chk = np.fromfile(File, dtype=np.int32,count=1)
      y = np.fromfile(File, dtype=np.float32,count=Np)
      y *= mul[1]
      if isObservation != 1:
        y += ranges[1][0]
      chk = np.fromfile(File, dtype=np.int32,count=1)

      chk = np.fromfile(File, dtype=np.int32,count=1)
      z = np.fromfile(File, dtype=np.float32,count=Np)
      z *= mul[2]
      if isObservation != 1:
        z += ranges[2][0]
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

    if isObservation == 1:
      x = x[0:maskIndex]# * 100/300000
      y = y[0:maskIndex]# * 100/300000
      z = z[0:maskIndex]# * 100/300000
      RA = RA[0:maskIndex]
      Dec = Dec[0:maskIndex]
      redshift = redshift[0:maskIndex]
      uniqueID = uniqueID[0:maskIndex]

    partData = np.column_stack((x,y,z))

    extraData = np.column_stack((RA,Dec,redshift,uniqueID))

    boxLen = mul

    #if isObservation == 1:
    #  # look for the mask file
    #  if os.access(sample.maskFile, os.F_OK):
    #    maskFile = sample.maskFile
    #  else:
    #    maskFile = sampleDir+"/"+os.path.basename(sample.maskFile)
    #    print "Using maskfile found in:", maskFile
    #  props = vp.getSurveyProps(maskFile, sample.zBoundary[0],
    #                            sample.zBoundary[1],
    #                            sample.zBoundary[0],
    #                            sample.zBoundary[1], "all",
    #                            selectionFuncFile=sample.selFunFile,
    #                            useComoving=sample.useComoving)
    #  boxVol = props[0]
    #  volNorm = maskIndex/boxVol
    #else:
    boxVol = np.prod(boxLen)
    volNorm = len(x)/boxVol

    isObservationData = isObservation == 1

    return partData, boxLen, volNorm, isObservationData, ranges, extraData

# -----------------------------------------------------------------------------
def getVolNorm(sampleDir):
    with open(sampleDir+"/sample_info.dat", 'rb') as input:
      sample = pickle.load(input)

    infoFile = sampleDir+"/zobov_slice_"+sample.fullName+".par"
    File = NetCDFFile(infoFile, 'r')
    ranges = np.zeros((3,2))
    ranges[0][0] = getattr(File, 'range_x_min')
    ranges[0][1] = getattr(File, 'range_x_max')
    ranges[1][0] = getattr(File, 'range_y_min')
    ranges[1][1] = getattr(File, 'range_y_max')
    ranges[2][0] = getattr(File, 'range_z_min')
    ranges[2][1] = getattr(File, 'range_z_max')
    isObservation = getattr(File, 'is_observation')
    maskIndex = getattr(File, 'mask_index')
    File.close()
    mul = np.zeros((3))
    mul[:] = ranges[:,1] - ranges[:,0]

    partFile = sampleDir+"/zobov_slice_"+sample.fullName
    with open(partFile, mode="rb") as File:
      chk = np.fromfile(File, dtype=np.int32,count=1)
      Np = np.fromfile(File, dtype=np.int32,count=1)[0]

    boxLen = mul

    #if isObservation == 1:
    #  # look for the mask file
    #  if os.access(sample.maskFile, os.F_OK):
    #    maskFile = sample.maskFile
    #  else:
    #    maskFile = sampleDir+"/"+os.path.basename(sample.maskFile)
    #    print "Using maskfile found in:", maskFile
    #  props = vp.getSurveyProps(maskFile, sample.zBoundary[0],
    #                            sample.zBoundary[1],
    #                            sample.zBoundary[0],
    #                            sample.zBoundary[1], "all",
    #                            selectionFuncFile=sample.selFunFile,
    #                            useComoving=sample.useComoving)
    #  boxVol = props[0]
    #  volNorm = maskIndex/boxVol
    #else:
    boxVol = np.prod(boxLen)
    volNorm = Np/boxVol

    return volNorm


# -----------------------------------------------------------------------------
def loadPartVel(sampleDir):
    #print "    Loading particle velocities..."
    sys.stdout.flush()

    with open(sampleDir+"/sample_info.dat", 'rb') as input:
      sample = pickle.load(input)

    infoFile = sampleDir+"/zobov_slice_"+sample.fullName+".par"
    File = NetCDFFile(infoFile, 'r')
    isObservation = getattr(File, 'is_observation')

    if isObservation:
      print("No velocities for observations!")
      return -1

    vx = File.variables['vel_x'][0:]
    vy = File.variables['vel_y'][0:]
    vz = File.variables['vel_z'][0:]

    File.close()

    partVel = np.column_stack((vx,vy,vz))

    return partVel

# -----------------------------------------------------------------------------
def getPartTree(catalog):

  sample = catalog.sampleInfo
  partData = catalog.partPos
  boxLen = catalog.boxLen

  periodicLine = getPeriodic(sample)

  periodic = 1.*boxLen

  if not "x" in periodicLine: periodic[0] = -1
  if not "y" in periodicLine: periodic[1] = -1
  if not "z" in periodicLine: periodic[2] = -1

  return PeriodicCKDTree(periodic, partData)

# -----------------------------------------------------------------------------
def getBall(partTree, center, radius):

  return partTree.query_ball_point(center, r=radius)


# -----------------------------------------------------------------------------
def shiftPart(inPart, center, periodicLine, ranges):

  part = inPart.copy()
  newCenter = 1.*center;

  boxLen = np.zeros((3))
  boxLen[0] = ranges[0][1] - ranges[0][0]
  boxLen[1] = ranges[1][1] - ranges[1][0]
  boxLen[2] = ranges[2][1] - ranges[2][0]

  # shift to box coordinates
  part[:,0] -= ranges[0][0]
  part[:,1] -= ranges[1][0]
  part[:,2] -= ranges[2][0]

  newCenter[:] -= ranges[:,0]

  part[:,0] -= newCenter[0]
  part[:,1] -= newCenter[1]
  part[:,2] -= newCenter[2]

  shiftUs = np.abs(part[:,0]) > boxLen[0]/2.
  if ("x" in periodicLine): part[shiftUs,0] -= \
                            np.copysign(boxLen[0], part[shiftUs,0])
  shiftUs = np.abs(part[:,1]) > boxLen[1]/2.
  if ("y" in periodicLine): part[shiftUs,1] -= \
                            np.copysign(boxLen[1], part[shiftUs,1])
  shiftUs = np.abs(part[:,2]) > boxLen[2]/2.
  if ("z" in periodicLine): part[shiftUs,2] -= \
                            np.copysign(boxLen[2], part[shiftUs,2])


  #part[:,0] += ranges[0][0]
  #part[:,1] += ranges[1][0]
  #part[:,2] += ranges[2][0]

  return part

# -----------------------------------------------------------------------------

class Bunch:
    def __init__(self, **kwds):
        self.__dict__.update(kwds)

class Catalog:
  numVoids = 0
  numPartTot = 0
  numZonesTot = 0
  volNorm = 0
  boxLen = np.zeros((3))
  ranges = np.zeros((3,2))
  part = None
  zones2Parts = None
  void2Zones = None
  voids = None
  sampleInfo = None

# -----------------------------------------------------------------------------
def loadVoidCatalog(sampleDir, dataPortion="central", loadParticles=True,
                    untrimmed=False):

# loads a void catalog
# by default, loads parent-level voids with central densities greater than 0.2*mean
#   sampleDir: path to VIDE output directory
#   dataPortion: "central" or "all"
#   loadParticles: if True, also load particle information
#   untrimmed: if True, catalog contains all voids, regardless of density or hierarchy level

  sys.stdout.flush()

  catalog = Catalog()
  with open(sampleDir+"/sample_info.dat", 'rb') as input:
    sample = pickle.load(input)
  catalog.sampleInfo = sample

  print("Loading info...")
  infoFile = sampleDir+"/zobov_slice_"+sample.fullName+".par"
  File = NetCDFFile(infoFile, 'r')
  ranges = np.zeros((3,2))
  ranges[0][0] = getattr(File, 'range_x_min')
  ranges[0][1] = getattr(File, 'range_x_max')
  ranges[1][0] = getattr(File, 'range_y_min')
  ranges[1][1] = getattr(File, 'range_y_max')
  ranges[2][0] = getattr(File, 'range_z_min')
  ranges[2][1] = getattr(File, 'range_z_max')
  catalog.boxLen[0] = ranges[0][1] - ranges[0][0]
  catalog.boxLen[1] = ranges[1][1] - ranges[1][0]
  catalog.boxLen[2] = ranges[2][1] - ranges[2][0]
  catalog.ranges = ranges
  File.close()

  volNorm = getVolNorm(sampleDir)
  catalog.volNorm = volNorm

  if untrimmed:
    prefix = "untrimmed_"
  else:
    prefix = ""

  print("Loading voids...")
  fileName = sampleDir+"/"+prefix+"voidDesc_"+dataPortion+"_"+sample.fullName+".out"
  catData = np.loadtxt(fileName, comments="#", skiprows=2)
  catalog.voids = []
  for line in catData:
    catalog.voids.append(Bunch(iVoid = int(line[0]),
                               voidID = int(line[1]),
                               coreParticle = line[2],
                               coreDens = line[3],
                               zoneVol = line[4],
                               zoneNumPart = line[5],
                               numZones = int(line[6]),
                               voidVol = line[7],
                               numPart = int(line[8]),
                               densCon = line[9],
                               voidProb = line[10],
                               radius = 0., # this is read in later
                               macrocenter = np.zeros((3)),
                               redshift = 0,
                               RA = 0,
                               Dec = 0,
                               parentID = 0,
                               treeLevel = 0,
                               numChildren = 0,
                               centralDen = 0.,
                               ellipticity = 0.,
                               eigenVals = np.zeros((3)),
                               eigenVecs = np.zeros((3,3)),
                               ))

  catalog.numVoids = len(catalog.voids)
  print("Read %d voids" % catalog.numVoids)

  print("Loading macrocenters...")
  iLine = 0
  for line in open(sampleDir+"/"+prefix+"macrocenters_"+dataPortion+"_"+sample.fullName+".out"):
    line = line.split()
    catalog.voids[iLine].macrocenter[0] = float(line[1])
    catalog.voids[iLine].macrocenter[1] = float(line[2])
    catalog.voids[iLine].macrocenter[2] = float(line[3])
    iLine += 1

  iLine = 0
  fileName = sampleDir+"/"+prefix+"sky_positions_"+dataPortion+"_"+sample.fullName+".out"
  catData = np.loadtxt(fileName, comments="#")
  for line in catData:
    catalog.voids[iLine].RA = float(line[0])
    catalog.voids[iLine].Dec = float(line[1])
    iLine += 1


  print("Loading derived void information...")
  fileName = sampleDir+"/"+prefix+"centers_"+dataPortion+"_"+sample.fullName+".out"
  catData = np.loadtxt(fileName, comments="#")
  for (iLine,line) in enumerate(catData):
    catalog.voids[iLine].volume = float(line[6])
    catalog.voids[iLine].radius = float(line[4])
    catalog.voids[iLine].redshift = float(line[5])
    catalog.voids[iLine].parentID = float(line[10])
    catalog.voids[iLine].treeLevel = float(line[11])
    catalog.voids[iLine].numChildren = float(line[12])
    catalog.voids[iLine].centralDen = float(line[13])
    iLine += 1

  fileName = sampleDir+"/"+prefix+"shapes_"+dataPortion+"_"+sample.fullName+".out"
  catData = np.loadtxt(fileName, comments="#")
  for (iLine,line) in enumerate(catData):
    catalog.voids[iLine].ellipticity = float(line[1])

    catalog.voids[iLine].eigenVals[0] = float(line[2])
    catalog.voids[iLine].eigenVals[1] = float(line[3])
    catalog.voids[iLine].eigenVals[2] = float(line[4])

    catalog.voids[iLine].eigenVecs[0][0] = float(line[5])
    catalog.voids[iLine].eigenVecs[0][1] = float(line[6])
    catalog.voids[iLine].eigenVecs[0][2] = float(line[7])

    catalog.voids[iLine].eigenVecs[1][0] = float(line[8])
    catalog.voids[iLine].eigenVecs[1][1] = float(line[9])
    catalog.voids[iLine].eigenVecs[1][2] = float(line[10])

    catalog.voids[iLine].eigenVecs[2][0] = float(line[11])
    catalog.voids[iLine].eigenVecs[2][1] = float(line[12])
    catalog.voids[iLine].eigenVecs[2][2] = float(line[13])

    iLine += 1

  if loadParticles:
    print("Loading all particles...")
    partData, boxLen, volNorm, isObservationData, ranges, extraData = loadPart(sampleDir)
    numPartTot = len(partData)
    catalog.numPartTot = numPartTot
    catalog.partPos = partData
    catalog.part = []
    for i in range(len(partData)):
      catalog.part.append(Bunch(x = partData[i][0],
                                y = partData[i][1],
                                z = partData[i][2],
                                volume = 0,
                                ra = extraData[i][0],
                                dec = extraData[i][1],
                                redshift = extraData[i][2],
                                uniqueID = extraData[i][3]))


    print("Loading volumes...")
    volFile = sampleDir+"/vol_"+sample.fullName+".dat"
    with open(volFile, mode="rb") as File:
      chk = np.fromfile(File, dtype=np.int32,count=1)
      vols = np.fromfile(File, dtype=np.float32,count=numPartTot)
    for ivol in range(len(vols)):
      catalog.part[ivol].volume = vols[ivol] / volNorm

    print("Loading zone-void membership info...")
    zoneFile = sampleDir+"/voidZone_"+sample.fullName+".dat"
    catalog.void2Zones = []
    with open(zoneFile, mode="rb") as File:
      numZonesTot = np.fromfile(File, dtype=np.int32,count=1)[0]
      catalog.numZonesTot = numZonesTot
      for iZ in range(numZonesTot):
        numZones = np.fromfile(File, dtype=np.int32,count=1)[0]
        catalog.void2Zones.append(Bunch(numZones = numZones,
                                        zoneIDs = []))

        for p in range(numZones):
          zoneID = np.fromfile(File, dtype=np.int32,count=1)[0]
          catalog.void2Zones[iZ].zoneIDs.append(zoneID)

    print("Loading particle-zone membership info...")
    zonePartFile = sampleDir+"/voidPart_"+sample.fullName+".dat"
    catalog.zones2Parts = []
    with open(zonePartFile) as File:
      chk = np.fromfile(File, dtype=np.int32,count=1)
      numZonesTot = np.fromfile(File, dtype=np.int32,count=1)[0]
      for iZ in range(numZonesTot):
        numPart = np.fromfile(File, dtype=np.int32,count=1)[0]
        catalog.zones2Parts.append(Bunch(numPart = numPart,
                                         partIDs = []))

        for p in range(numPart):
          partID = np.fromfile(File, dtype=np.int32,count=1)[0]
          catalog.zones2Parts[iZ].partIDs.append(partID)

  return catalog


# -----------------------------------------------------------------------------
def getArray(objectList, attr):

  if hasattr(objectList[0], attr):
    ndim = np.shape( np.atleast_1d( getattr(objectList[0], attr) ) )[0]
    attrArr = np.zeros(( len(objectList), ndim ))

    for idim in range(ndim):
      attrArr[:,idim] = np.fromiter((np.atleast_1d(getattr(v, attr))[idim] \
                                    for v in objectList), float )

    if ndim == 1: attrArr = attrArr[:,0]

    return attrArr
  else:
    print(" Attribute", attr, "not found!")
    return -1


# -----------------------------------------------------------------------------
def getVoidPart(catalog, voidID):

  partOut = []
  for iZ in range(catalog.void2Zones[voidID].numZones):
    zoneID = catalog.void2Zones[voidID].zoneIDs[iZ]
    for p in range(catalog.zones2Parts[zoneID].numPart):
      partID = catalog.zones2Parts[zoneID].partIDs[p]
      partOut.append(catalog.part[partID])

  return partOut

# -----------------------------------------------------------------------------
def filterVoidsOnSize(catalog, rMin):
  catalog.voids = [v for v in catalog.voids if v.radius >= rMin]
  catalog.numVoids = len(catalog.voids)
  return catalog

# -----------------------------------------------------------------------------
def filterVoidsOnTreeLevel(catalog, level):
  catalog.voids = [v for v in catalog.voids if v.treeLevel == level]
  if level == -1:
    catalog.voids = [v for v in catalog.voids if v.numChildren == 0]
  catalog.numVoids = len(catalog.voids)
  return catalog

# -----------------------------------------------------------------------------
def filterVoidsOnCentralDen(catalog, maxCentralDen):
  catalog.voids = [v for v in catalog.voids if v.centralDen <= maxCentralDen]
  catalog.numVoids = len(catalog.voids)
  return catalog

# -----------------------------------------------------------------------------
def filterVoidsOnCoreDen(catalog, maxCoreDen):
  catalog.voids = [v for v in catalog.voids if v.coreDens <= maxCoreDen]
  catalog.numVoids = len(catalog.voids)
  return catalog

# -----------------------------------------------------------------------------
def filterVoidsOnDenCon(catalog, minDenCon):
  catalog.voids = [v for v in catalog.voids if v.densCon >= minDenCon]

  return catalog



# -----------------------------------------------------------------------------
def stackVoids(catalog, stackMode = "ball"):

# builds a stack of voids from the given catalog
#   catalog: void catalog
#   stackMode:
#     "ball": spherical cut
#     "voronoi": only void member particles
#
# returns:
#   stackedPart: array of relative particle positions in the stack

  rMax = 100.
  periodicLine = getPeriodic(catalog.sampleInfo)

  if stackMode == "ball":
    partTree = getPartTree(catalog)

  stackedPart = []
  for void in catalog.voids:
    center = void.macrocenter

    if stackMode == "ball":
      localPart = catalog.partPos[ getBall(partTree, center, rMax) ]
    else:
      voidPart = getVoidPart(catalog, void.voidID)
      localPart = np.zeros((len(voidPart),3))
      localPart[:,0] = getArray(voidPart, 'x')
      localPart[:,1] = getArray(voidPart, 'y')
      localPart[:,2] = getArray(voidPart, 'z')

    shiftedPart = shiftPart(localPart, center, periodicLine, catalog.ranges)

    stackedPart.extend(shiftedPart)

  return stackedPart
