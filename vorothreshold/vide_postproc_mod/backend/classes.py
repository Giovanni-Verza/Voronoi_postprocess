#+
#   VIDE -- Void IDentification and Examination -- ./python_tools/vide/backend/classes.py
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
# classes and routines used to support scripts

import os
from numpy import abs
#import matplotlib as mpl
#mpl.use('Agg')

LIGHT_SPEED = 299792.458

class Stack:
  zMin = 0.0
  zMax = 0.1
  rMin = 5
  rMax = 15
  zMinPlot = 0.0
  zMaxPlot = 0.1
  includeInHubble = True
  partOfCombo = False
  needProfile = True
  rescaleMode = "rmax" # options: "rmax" to scale to largest void in stack
                       #          "rv"   normalize each void
  maxVoids = -1 # maximum number of voids to allow in the stack

  def __init__(self, zMin, zMax, rMin, rMax, includeInHubble, partOfCombo, 
               zMinPlot=None, needProfile=True, rescaleMode="rmax",
               maxVoids=-1, fittingMode="mcmc"):
    self.zMin = zMin
    self.zMax = zMax
    self.rMin = rMin
    self.rMax = rMax
    self.zMaxPlot = zMax
    self.includeInHubble = includeInHubble
    self.partOfCombo = partOfCombo
    self.needProfile = needProfile
    self.rescaleMode = rescaleMode
    self.maxVoids = maxVoids
    self.fittingMode = fittingMode
   
    if zMinPlot == None:
      self.zMinPlot = self.zMin
    else:
      self.zMinPlot = zMinPlot

class Sample:
  dataType = "observation"
  dataFormat = "sdss"
  dataFile = "lss.dr72dim.dat"
  dataUnit = 1
  fullName = "lss.dr72dim.dat"
  nickName = "dim"
  zobovDir = ""
  maskFile = "rast_window_512.fits"
  selFunFile = "czselfunc.all.dr72dim.dat"
  zBoundary = (0.0, 0.1)
  zBoundaryMpc = (0., 300)
  shiftSimZ = False
  zRange    = (0.0, 0.1)
  omegaM    = 0.27
  minVoidRadius = -1
  fakeDensity = 0.01
  profileBinSize = 2 # Mpc
  autoNumInStack = -1 # set to >0 to automatically generate stacks of size N
  autoPartInStack = -1 # set to >0 to automatically generate stacks with N particles
  numAPSlices = 1
  volumeLimited = True
  includeInHubble = True
  partOfCombo  = False
  isCombo = False
  
  comboList = []

  # applies to simulations only
  boxLen = 1024 # Mpc/h
  usePecVel = False
  subsample = 1.0
  numSubvolumes = 1
  mySubvolume = 1

  stacks = []

  def __init__(self, dataFile="", fullName="", dataUnit=1,
               nickName="", maskFile="", selFunFile="",
               zBoundary=(), zRange=(), zBoundaryMpc=(), shiftSimZ=False,
               minVoidRadius=-1, fakeDensity=0.01, volumeLimited=True,
               numAPSlices=1,
               includeInHubble=True, partOfCombo=False, isCombo=False, 
               comboList=(), profileBinSize=2.0,
               boxLen=1024, usePecVel=False, omegaM=0.27, 
               numSubvolumes=1, mySubvolume=1, dataFormat="sdss",
               useComoving=True,
               dataType="observation",
               subsample=1.0, useLightCone=False, autoNumInStack=-1,
               autoPartInStack=-1):
    self.dataFile = dataFile
    self.fullName = fullName
    self.nickName = nickName
    self.maskFile = maskFile
    self.selFunFile = selFunFile
    self.zBoundary = zBoundary 
    self.zBoundaryMpc = zBoundaryMpc
    self.shiftSimZ = shiftSimZ
    self.zRange = zRange
    self.minVoidRadius = minVoidRadius
    self.fakeDensity = fakeDensity
    self.volumeLimited = volumeLimited
    self.includeInHubble = includeInHubble 
    self.partOfCombo = partOfCombo 
    self.isCombo = isCombo
    self.comboList = comboList
    self.numAPSlices = numAPSlices
    self.zobovDir = None
    self.profileBinSize = profileBinSize
    self.dataType = dataType
    self.boxLen = boxLen
    self.usePecVel = usePecVel
    self.omegaM = omegaM
    self.numSubvolumes = numSubvolumes
    self.mySubvolume = mySubvolume
    self.dataFormat = dataFormat
    self.subsample = subsample
    self.useLightCone = useLightCone
    self.dataUnit = dataUnit
    self.useComoving = useComoving
    self.autoNumInStack = autoNumInStack
    self.autoPartInStack = autoPartInStack

    self.stacks = []

  def addStack(self, zMin, zMax, rMin, rMax, 
                includeInHubble, partOfCombo,zMinPlot=None, 
                needProfile=True, rescaleMode="rmax",
                maxVoids=-1, fittingMode="mcmc"):
    self.stacks.append(Stack(zMin, zMax, rMin, rMax,
                            includeInHubble, partOfCombo, 
                            zMinPlot=zMinPlot, needProfile=needProfile,
                            rescaleMode=rescaleMode,
                            maxVoids=maxVoids,
                            fittingMode=fittingMode))

  def getHubbleStacks(self):
    stacksForHubble = []
    for stack in self.stacks:
      if stack.includeInHubble:
        stacksForHubble.append(stack)
    return stacksForHubble

  def getUniqueRBins(self):
    uniqueRStacks = []
    for stack in self.stacks:
      if not stack.includeInHubble:
        continue
      alreadyHere = False
      for stackCheck in uniqueRStacks:
        if stack.rMin == stackCheck.rMin and stack.rMax == stackCheck.rMax:
          alreadyHere = True
          break
      if not alreadyHere:
        uniqueRStacks.append(stack)
    return uniqueRStacks

  def getUniqueZBins(self):
    uniqueZStacks = []
    for stack in self.stacks:
      if not stack.includeInHubble:
        continue
      alreadyHere = False
      for stackCheck in uniqueZStacks:
        if stack.zMin == stackCheck.zMin and stack.zMax == stackCheck.zMax:
          alreadyHere = True
          break
      if not alreadyHere:
        uniqueZStacks.append(stack)
    return uniqueZStacks

# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
def jobSuccessful(logFile, doneString):
  jobDone = False
  checkLine = ""
  if os.access(logFile, os.F_OK):
    #filelines = file(logFile, "r").readlines()
    #if len(filelines) >= 1:
    #  checkLine = filelines[-1]
    for line in open(logFile, 'r'):
      if doneString in line: jobDone = True
  return jobDone

def getStackSuffix(zMin, zMax, rMin, rMax, dataPortion, customLine=""):
  return "z"+str(zMin)+"-"+str(zMax)+"_"+str(rMin)+"-"+str(rMax)+\
           "Mpc"+customLine+"_"+dataPortion

def getPeriodic(sample):
  periodicLine = ""
  if sample.dataType != "observation":
    if sample.numSubvolumes == 1: periodicLine += "xy"
    if abs(sample.zBoundaryMpc[1]  - sample.zBoundaryMpc[0] - \
              sample.boxLen) <= 1.e-1:
      periodicLine += "z"
  return periodicLine

def my_import(name):
    mod = __import__(name)
    components = name.split('.')
    for comp in components[1:]:
        mod = getattr(mod, comp)
    return mod
