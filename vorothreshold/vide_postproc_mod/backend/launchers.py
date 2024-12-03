#+
#   VIDE -- Void IDentification and Examination -- ./python_tools/vide/backend/launchers.py
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
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# routines which communicate with individual data analysis portions -
#   they make the analyzeVoids.py script easier to read

import os
import glob
from . import classes
import numpy as np
import numpy.ma as ma
import os
import shutil
import glob
import subprocess
import sys
from   pylab import figure
from   netCDF4 import Dataset
from vide.backend.classes import *
import pickle
import vide.apTools as vp
import scipy.interpolate as interpolate

NetCDFFile = Dataset
ncFloat = 'f8' # Double precision

LIGHT_SPEED = 299792.458

# -----------------------------------------------------------------------------
def launchGenerate(sample, binPath, workDir=None, inputDataDir=None,
                   zobovDir=None, figDir=None, logFile=None, useComoving=False,
                   continueRun=None,regenerate=False):

  if sample.dataType == "observation":
    sampleName = sample.fullName

    if regenerate:
      inputParameterFlag = "inputParameter " + zobovDir+"/zobov_slice_"+sampleName+".par"
      outputFile = zobovDir+"/regenerated_zobov_slice_" + sampleName
    else:
      inputParameterFlag = ""
      outputFile = zobovDir+"/zobov_slice_" + sampleName

    if sample.dataFile == "":
      datafile = inputDataDir+"/"+sampleName
    else:
      datafile = inputDataDir+"/"+sample.dataFile

    maskFile = sample.maskFile

    if useComoving:
      useComovingFlag = "useComoving"
    else:
      useComovingFlag = ""

    conf="""
      catalog %s
      mask %s
      output %s
      params %s
      zMin %g
      zMax %g
      density_fake %g
      %s
      %s
      omegaM %g
      """ % (datafile, maskFile, outputFile,
             zobovDir+"/zobov_slice_"+sampleName+".par",
             sample.zBoundary[0], sample.zBoundary[1], sample.fakeDensity,
             useComovingFlag, inputParameterFlag, sample.omegaM)

    parmFile = os.getcwd()+"/generate_"+sample.fullName+".par"

    if regenerate or not (continueRun and jobSuccessful(logFile, "Done!\n")):
      with open(parmFile, mode="wt") as f:
        f.write(conf)
      arg1 = "--configFile=%s" % parmFile
      with open(logFile, 'wt') as log:
        subprocess.call([binPath, arg1], stdout=log, stderr=log)
      if jobSuccessful(logFile, "Done!\n"):
        print("done")
      else:
        print("FAILED!")
        exit(-1)

    else:
      print("already done!")

    if os.access(parmFile, os.F_OK): os.unlink(parmFile)

    if os.access("contour_map.fits", os.F_OK):
      os.system("mv %s %s" % ("contour_map.fits", zobovDir))
      os.system("mv %s %s" % ("mask_map.fits", zobovDir))

    if os.access("comoving_distance.txt", os.F_OK):
      os.system("mv %s %s" % ("comoving_distance.txt", zobovDir))

    if os.access("mask_index.txt", os.F_OK):
      os.system("mv %s %s" % ("mask_index.txt", zobovDir))
      os.system("mv %s %s" % ("total_particles.txt", zobovDir))

    if os.access("galaxies.txt", os.F_OK):
      os.system("mv %s %s" % ("galaxies.txt", zobovDir))
      os.system("mv %s %s" % ("mock_galaxies.txt", zobovDir))
      os.system("mv %s %s" % ("mock_boundary.txt", zobovDir))
      os.system("mv %s %s" % ("mock_sphere.txt", zobovDir))

  else: # simulation
    sampleName = sample.fullName

    datafile = inputDataDir+"/"+sample.dataFile

    # check if the final subsampling is done
    lastSample = sample.subsample.split(', ')[-1]
    doneLine = "Done! %5.2e\n" % float(lastSample)
    if (continueRun and jobSuccessful(logFile, doneLine)):
      print("already done!")
      return

    prevSubSample = -1
    firstSubSample = -1
    for thisSubSample in sample.subsample.split(', '):

      if prevSubSample == -1:
        inputParameterFlag = ""
        outputFile = zobovDir+"/zobov_slice_" + sampleName + "_ss" + \
                     thisSubSample
        keepFraction = float(thisSubSample)
        subSampleLine = "subsample %g" % keepFraction
        resubSampleLine = ""
        firstSubSample = keepFraction
      else:
        inputParameterFlag = "inputParameter " + zobovDir+"/zobov_slice_"+\
                             sampleName+"_ss"+prevSubSample+".par"
        outputFile = zobovDir+"/zobov_slice_" + sampleName + "_ss" + \
                     thisSubSample
        keepFraction = float(thisSubSample)/float(prevSubSample)
        subSampleLine = "subsample %s" % firstSubSample
        resubSampleLine = "resubsample %g" % keepFraction

      includePecVelString = ""
      if sample.usePecVel: includePecVelString = "peculiarVelocities"

      useLightConeString = ""
      if sample.useLightCone: useLightConeString = "cosmo"

      if sample.dataFormat == "multidark" or sample.dataFormat == "random":
        dataFileLine = "multidark " + datafile
      elif sample.dataFormat == "gadget":
        dataFileLine = "gadget " + datafile
      elif sample.dataFormat == "gadget2":
        dataFileLine = "gadget2 " + datafile
      elif sample.dataFormat == "ahf":
        dataFileLine = "gadget " + datafile
      elif sample.dataFormat == "sdf":
        dataFileLine = "sdf " + datafile
      elif sample.dataFormat == "ramses":
        ramsesId = int(os.path.split(datafile)[1][-5:len(datafile)]) # picks out the particle file (should be the output_NNNNN, then extracts the output id "NNNNN" as an integer)
        dataFileLine = "ramses " + datafile + "/"
      else:
        raise ValueError("unknown dataFormat '%s'" % sample.dataFormat)

      iX = float(sample.mySubvolume[0])
      iY = float(sample.mySubvolume[1])

      xMin = iX/sample.numSubvolumes * sample.boxLen
      yMin = iY/sample.numSubvolumes * sample.boxLen
      xMax = (iX+1)/sample.numSubvolumes * sample.boxLen
      yMax = (iY+1)/sample.numSubvolumes * sample.boxLen

      reshiftFlag = ""
      if not sample.shiftSimZ: reshiftFlag = "preReShift"

      if sample.dataFormat == "ramses":
        ramsesIdLine = "ramsesId " + str(ramsesId)
      else:
        ramsesIdLine = ""

      conf="""
      %s
      output %s
      outputParameter %s
      %s
      %s
      gadgetUnit %g
      %s
      rangeX_min %g
      rangeX_max %g
      rangeY_min %g
      rangeY_max %g
      rangeZ_min %g
      rangeZ_max %g
      %s
      %s
      %s
      %s
      """ % (dataFileLine,
             outputFile,
             outputFile+".par",
             includePecVelString,
             useLightConeString,
             sample.dataUnit,
             ramsesIdLine,
             xMin, xMax, yMin, yMax,
             sample.zBoundaryMpc[0], sample.zBoundaryMpc[1],
             subSampleLine,resubSampleLine,inputParameterFlag,reshiftFlag)

      parmFile = os.getcwd()+"/generate_"+sample.fullName+".par"

      with open(parmFile, mode="wt") as f:
        f.write(conf)

      if (prevSubSample == -1):
        cmd = "%s --configFile=%s" % (binPath,parmFile)
        log = open(logFile, 'w')
      else:
        cmd = "%s --configFile=%s" % (binPath,parmFile)
        log = open(logFile, 'a')
      arg1 = "--configFile=%s" % parmFile


      subprocess.call(cmd, stdout=log, stderr=log, shell=True)
      log.close()

      # remove intermediate files
      if (prevSubSample != -1):
       os.unlink(zobovDir+"/zobov_slice_"+sampleName+"_ss"+prevSubSample+".par")
       os.unlink(zobovDir+"/zobov_slice_"+sampleName+"_ss"+prevSubSample)

      doneLine = "Done! %5.2e\n" % keepFraction
      if not jobSuccessful(logFile, doneLine):
        print("FAILED!")   ### dies here for now
        exit(-1)

      prevSubSample = thisSubSample

    if jobSuccessful(logFile, doneLine): print("done")

    # place the final subsample
    os.system("mv %s %s" % (zobovDir+"/zobov_slice_"+sampleName+"_ss"+\
              prevSubSample, zobovDir+"/zobov_slice_"+sampleName))
    os.system("mv %s %s" % (zobovDir+"/zobov_slice_"+sampleName+"_ss"+\
              prevSubSample+".par", zobovDir+"/zobov_slice_"+sampleName+".par"))

    if os.access("comoving_distance.txt", os.F_OK):
      os.system("mv %s %s" % ("comoving_distance.txt", zobovDir))

    if os.access(parmFile, os.F_OK):
      os.unlink(parmFile)

    if os.access("mask_index.txt", os.F_OK):
      os.system("mv %s %s" % ("mask_index.txt", zobovDir))
      os.system("mv %s %s" % ("total_particles.txt", zobovDir))
      #os.system("mv %s %s" % ("sample_info.txt", zobovDir))

  # add to sample info file
  if sample.dataType == "observation":
    (boxVol, nbar) = vp.getSurveyProps(sample.maskFile, sample.zRange[0],
     sample.zRange[1], sample.zRange[0], sample.zRange[1], "all",
     useComoving=useComoving)
  else:
    iX = float(sample.mySubvolume[0])
    iY = float(sample.mySubvolume[1])
    xMin = iX/sample.numSubvolumes * sample.boxLen
    yMin = iY/sample.numSubvolumes * sample.boxLen
    xMax = (iX+1)/sample.numSubvolumes * sample.boxLen
    yMax = (iY+1)/sample.numSubvolumes * sample.boxLen
    zMin = sample.zBoundaryMpc[0]
    zMax = sample.zBoundaryMpc[1]

    boxVol = (xMax-xMin)*(yMax-yMin)*(zMax-zMin)
    nbar = 1.0

  numTracers = int(open(zobovDir+"/mask_index.txt", "r").read())
  numTotal = int(open(zobovDir+"/total_particles.txt", "r").read())

  meanSep = (1.*numTracers/boxVol/nbar)**(-1/3.)

  # save this sample's information
  with open(zobovDir+"/sample_info.dat", mode='wb') as output:
    pickle.dump(sample, output, pickle.HIGHEST_PROTOCOL)

  fp = open(zobovDir+"/sample_info.txt", 'w')
  fp.write("Sample name: %s\n" % sample.fullName)
  fp.write("Sample nickname: %s\n" % sample.nickName)
  fp.write("Data type: %s\n" % sample.dataType)
  fp.write("Redshift range: %f - %f\n" %(sample.zBoundary[0],sample.zBoundary[1]))
  if (sample.dataType == "simulation"):
    fp.write("Particles placed on lightcone: %g\n" % sample.useLightCone)
    fp.write("Peculiar velocities included: %g\n" % sample.usePecVel)
    if (len(sample.subsample) == 1):
      fp.write("Additional subsampling fraction: %s\n" % sample.subsample)
    else:
      fp.write("Additional subsampling fraction: %s\n" % sample.subsample[-1])
    fp.write("Simulation box length (Mpc/h): %g\n" % sample.boxLen)
    fp.write("Simulation Omega_M: %g\n" % sample.omegaM)
    fp.write("Number of simulation subvolumes: %s\n" % sample.numSubvolumes)
    fp.write("My subvolume index: %s\n" % sample.mySubvolume)
  fp.write("Estimated volume (cubic Mpc/h): %g\n" % boxVol)
  fp.write("Number of real (non-boundary) tracers: %d\n" % numTracers)
  fp.write("Total number of tracers: %d\n" % numTotal)
  fp.write("Estimated mean tracer separation (Mpc/h): %g\n" % meanSep)
  fp.write("Minimum void size actually used (Mpc/h): %g\n" % sample.minVoidRadius)
  fp.close()

# -----------------------------------------------------------------------------
def launchZobov(sample, binPath, zobovDir=None, logDir=None, continueRun=None,
                 numZobovDivisions=None, numZobovThreads=None, mergingThreshold=0.2):

  sampleName = sample.fullName

  datafile = zobovDir+"zobov_slice_"+sampleName

  logFile = logDir+"/zobov_"+sampleName+".out"

  vozScript = "./scr_"+sampleName

  if os.access(vozScript, os.F_OK):
    os.unlink(vozScript)

  if sample.dataType == "observation":
    maskIndex = open(zobovDir+"/mask_index.txt", "r").read()
    totalPart = open(zobovDir+"/total_particles.txt", "r").read()
    maxDen = mergingThreshold*float(maskIndex)/float(totalPart)
  else:
    maskIndex = -1
    maxDen = mergingThreshold
    if numZobovDivisions == 1:
      print("  WARNING! You are using a single ZOBOV division with a simulation. Periodic boundaries will not be respected!")

  if not (continueRun and jobSuccessful(logFile, "Done!\n")):
    for fileName in glob.glob(zobovDir+"/part._"+sampleName+".*"):
      os.unlink(fileName)

    if os.access(zobovDir+"/adj_"+sampleName+".dat", os.F_OK):
      os.unlink(zobovDir+"adj_"+sampleName+".dat")

    if os.access(zobovDir+"/voidDesc_"+sampleName+".out", os.F_OK):
      os.unlink(zobovDir+"/voidDesc_"+sampleName+".out")

    cmd = [binPath+"/vozinit", datafile, "0.1", "1.0", str(numZobovDivisions), \
                      "_"+sampleName, str(numZobovThreads), \
                      binPath, zobovDir, str(maskIndex)]
    log = open(logFile, 'w')
    subprocess.call(cmd, stdout=log, stderr=log)
    log.close()

    cmd = ["./%s" % vozScript]
    log = open(logFile, 'a')
    subprocess.call(cmd, stdout=log, stderr=log)
    log.close()

    # re-weight the volumes based on selection function
    if sample.dataType == "observation" and \
       sample.selFunFile != None:

      # load volumes
      volFile = zobovDir+"/vol_"+sampleName+".dat"
      with open(volFile, mode="rb") as File:
        numPartTot = np.fromfile(File, dtype=np.int32,count=1)
        vols = np.fromfile(File, dtype=np.float32,count=numPartTot)

      # load redshifts
      partFile = zobovDir+"/zobov_slice_"+sample.fullName
      with open(partFile, mode="rb") as File:
        chk = np.fromfile(File, dtype=np.int32,count=1)
        Np = np.fromfile(File, dtype=np.int32,count=1)
        chk = np.fromfile(File, dtype=np.int32,count=1)

        # x
        chk = np.fromfile(File, dtype=np.int32,count=1)
        redshifts = np.fromfile(File, dtype=np.float32,count=Np)
        chk = np.fromfile(File, dtype=np.int32,count=1)

        # y
        chk = np.fromfile(File, dtype=np.int32,count=1)
        redshifts = np.fromfile(File, dtype=np.float32,count=Np)
        chk = np.fromfile(File, dtype=np.int32,count=1)

        # z
        chk = np.fromfile(File, dtype=np.int32,count=1)
        redshifts = np.fromfile(File, dtype=np.float32,count=Np)
        chk = np.fromfile(File, dtype=np.int32,count=1)

        # RA
        chk = np.fromfile(File, dtype=np.int32,count=1)
        redshifts = np.fromfile(File, dtype=np.float32,count=Np)
        chk = np.fromfile(File, dtype=np.int32,count=1)

        # Dec
        chk = np.fromfile(File, dtype=np.int32,count=1)
        redshifts = np.fromfile(File, dtype=np.float32,count=Np)
        chk = np.fromfile(File, dtype=np.int32,count=1)

        # z
        chk = np.fromfile(File, dtype=np.int32,count=1)
        redshifts = np.fromfile(File, dtype=np.float32,count=Np)
        chk = np.fromfile(File, dtype=np.int32,count=1)

      # build selection function interpolation
      selfuncData = np.genfromtxt(sample.selFunFile)
      selfunc = interpolate.interp1d(selfuncData[:,0], selfuncData[:,1],
                                     kind='cubic', bounds_error=False,
                                     fill_value=1.0)
      # re-weight and write
       ## TEST
      #redshifts /= 10000.
      for i in range(len(vols)):
        vols[i] *= selfunc(redshifts[i])

      volFile = zobovDir+"/vol_weighted_"+sampleName+".dat"
      with open(volFile, mode='wb') as File:
        numPartTot.astype(np.int32).tofile(File)
        vols.astype(np.float32).tofile(File)

      volFileToUse = zobovDir+"/vol_weighted_"+sampleName+".dat"
    else:
      volFileToUse = zobovDir+"/vol_"+sampleName+".dat"


    cmd = [binPath+"/jozov2", \
           zobovDir+"/adj_"+sampleName+".dat", \
           volFileToUse, \
           zobovDir+"/voidPart_"+sampleName+".dat", \
           zobovDir+"/voidZone_"+sampleName+".dat", \
           zobovDir+"/voidDesc_"+sampleName+".out", \
           str(maxDen), str(maskIndex)]
    log = open(logFile, 'a')
    subprocess.call(cmd, stdout=log, stderr=log)
    log.close()

    # don't need the subbox files
    for fileName in glob.glob(zobovDir+"/part._"+sampleName+".*"):
      os.unlink(fileName)

    if jobSuccessful(logFile, "Done!\n"):
      print("done")
    else:
      print("FAILED!")
      exit(-1)

  else:
    print("already done!")

  if os.access(vozScript, os.F_OK):
    os.unlink(vozScript)

# -----------------------------------------------------------------------------
def launchPrune(sample, binPath,
                summaryFile=None, logFile=None, zobovDir=None,
                continueRun=None, useComoving=False, mergingThreshold=0.2):

  sampleName = sample.fullName

  numVoids = sum(1 for line in \
                 open(zobovDir+"/voidDesc_"+sampleName+".out"))
  numVoids -= 2

  if sample.dataType == "observation":
    mockIndex = open(zobovDir+"/mask_index.txt", "r").read()
    totalPart = open(zobovDir+"/total_particles.txt", "r").read()
    maxDen = mergingThreshold*float(mockIndex)/float(totalPart)
    observationLine = " --isObservation"
    #periodicLine = " --periodic=''"
  else:
    mockIndex = -1
    maxDen = mergingThreshold
    observationLine = ""

  periodicLine = " --periodic='" + getPeriodic(sample) + "'"

  if useComoving:
    useComovingFlag = " --useComoving"
  else:
    useComovingFlag = ""

  if sample.minVoidRadius == -1:
    minRadius = -1
    for line in open(zobovDir+"/sample_info.txt"):
      if "Estimated mean tracer separation" in line:
        minRadius = float(line.split()[5])
        break
    if minRadius == -1:
      print("Could not grab mean tracer separation!?")
      exit(-1)
  else:
    minRadius = sample.minVoidRadius

  if not (continueRun and (jobSuccessful(logFile, "NetCDF: Not a valid ID\n") \
          or jobSuccessful(logFile, "Done!\n"))):
    cmd = binPath
    cmd += " --partFile=" + zobovDir+"/zobov_slice_"+str(sampleName)
    cmd += " --voidDesc=" + zobovDir+"/voidDesc_"+str(sampleName)+".out"
    cmd += " --void2Zone="+zobovDir+"/voidZone_"+str(sampleName)+".dat"
    cmd += " --zone2Part=" + zobovDir+"/voidPart_"+str(sampleName)+".dat"
    cmd += " --partVol=" + zobovDir+"/vol_"+str(sampleName)+".dat"
    cmd += " --partAdj=" + zobovDir+"/adj_"+str(sampleName)+".dat"
    cmd += " --extraInfo=" + zobovDir+"/zobov_slice_"+str(sampleName)+\
           ".par"
    cmd += " --tolerance=1.0"
    cmd += " --mockIndex=" + str(mockIndex)
    cmd += " --maxCentralDen=" + str(maxDen)
    cmd += " --zMin=" + str(sample.zRange[0])
    cmd += " --zMax=" + str(sample.zRange[1])
    cmd += " --rMin=" + str(minRadius)
    cmd += " --numVoids=" + str(numVoids)
    cmd += observationLine
    cmd += periodicLine
    cmd += useComovingFlag
    cmd += " --omegaM=" + str(sample.omegaM)
    cmd += " --outputDir=" + zobovDir
    cmd += " --sampleName=" + str(sampleName)
    log = open(logFile, 'w')
    log.write(f"Command is {cmd}\n")
    subprocess.call(cmd, stdout=log, stderr=log, shell=True)
    log.close()

    if jobSuccessful(logFile, "NetCDF: Not a valid ID\n") or \
       jobSuccessful(logFile, "Done!\n"):
      print("done")
    else:
      print("FAILED!")
      #exit(-1)

  else:
    print("already done!")


# -----------------------------------------------------------------------------
def launchVoidOverlap(sample1, sample2, sample1Dir, sample2Dir,
                      binPath, thisDataPortion=None,
                      logFile=None,
                      continueRun=None, outputFile=None,
                      overlapFrac=0.25,
                      matchMethod=None, strictMatch=False):

  sampleName1 = sample1.fullName
  sampleName2 = sample2.fullName

  periodicLine = " --periodic='" + getPeriodic(sample1) + "' "

  if strictMatch:
    matchPrefix = ""
  else:
    matchPrefix = "trimmed_nodencut_"

  if sample1.dataType == "observation" or sample2.dataType == "observation":
    observationLine = " --isObservation"
  else:
    observationLine = ""

  if not (continueRun and jobSuccessful(logFile, "Done!\n")):
    cmd = binPath
    cmd += " --partFile1=" + sample1Dir+"/zobov_slice_" + \
           str(sampleName1)
    cmd += " --volFile1=" + sample1Dir+"/vol_" + \
           str(sampleName1)+".dat"
    cmd += " --voidFile1=" + sample1Dir+"/voidDesc_" + \
           thisDataPortion+"_"+str(sampleName1)+".out"
    cmd += " --infoFile1=" + sample1Dir+"/zobov_slice_" + \
           str(sampleName1)+".par"
    cmd += " --centerFile1=" + sample1Dir + \
           "/macrocenters_"+thisDataPortion+"_"+str(sampleName1)+".out"
    cmd += " --shapeFile1=" + sample1Dir + \
           "/shapes_"+thisDataPortion+"_"+str(sampleName1)+".out"
    cmd += " --zoneFile1=" + sample1Dir+"/voidZone_" + \
           str(sampleName1)+".dat"
    cmd += " --zonePartFile1=" + sample1Dir+"/voidPart_" + \
           str(sampleName1)+".dat"

    cmd += " --partFile2=" + sample2Dir+"/zobov_slice_" + \
           str(sampleName2)
    cmd += " --volFile2=" + sample2Dir+"/vol_" + \
           str(sampleName2)+".dat"
    cmd += " --voidFile2=" + sample2Dir+"/"+matchPrefix+"voidDesc_" + \
           thisDataPortion+"_"+str(sampleName2)+".out"
    cmd += " --infoFile2=" + sample2Dir+"/zobov_slice_" + \
           str(sampleName2)+".par"
    cmd += " --centerFile2=" + sample2Dir + \
           "/"+matchPrefix+"macrocenters_"+thisDataPortion+"_"+str(sampleName2)+".out"
    cmd += " --shapeFile2=" + sample2Dir + \
           "/"+matchPrefix+"shapes_"+thisDataPortion+"_"+str(sampleName2)+".out"
    cmd += " --zoneFile2=" + sample2Dir+"/voidZone_" + \
           str(sampleName2)+".dat"
    cmd += " --zonePartFile2=" + sample2Dir+"/voidPart_" + \
           str(sampleName2)+".dat"

    cmd += " --overlapFrac=" + str(overlapFrac)

    cmd += observationLine

    if matchMethod == "useID": cmd += " --useID"
    cmd += periodicLine
    cmd += " --outfile=" + outputFile
    #cmd += " &> " + logFile
    #open("temp.par",'w').write(cmd)
    #os.system(cmd)
    log = open(logFile, 'w')
    subprocess.call(cmd, stdout=log, stderr=log, shell=True)
    log.close()

    #if jobSuccessful(logFile, "Done!\n"):
    print("done")
    #else:
    #  print "FAILED!"
    #  exit(-1)

  else:
    print("already done!")


# -----------------------------------------------------------------------------
def launchVelocityStack(sample, stack, binPath,
                         velField_file,
                         thisDataPortion=None, logDir=None,
                         voidDir=None, runSuffix=None,
                         zobovDir=None,
                         summaryFile=None,
                         continueRun=None, dataType=None, prefixRun=""):

  sampleName = sample.fullName

  runSuffix = getStackSuffix(stack.zMin, stack.zMax, stack.rMin,
                             stack.rMax, thisDataPortion)

  logFile = logDir+("/%svelocity_stack_"%prefixRun)+sampleName+"_"+runSuffix+".out"

  voidCenters=voidDir+"/centers.txt"
#  Rmax =

  centralRadius = stack.rMin * 0.25
  Rextracut = stack.rMax*3 + 1
  Rcircular = stack.rMax*3 + 2

  parameters="--velocityField=%s --voidCenters=%s --Rmax=%e --L0=%e --numBins=%d" % (velField_file, voidCenters, Rmax, Boxsize, numBins)

  if not (continueRun and jobSuccessful(logFile, "Done!\n")):
    #cmd = "%s %s &> %s" % (binPath,parameters,logFile)
    #os.system(cmd)
    cmd = "%s %s" % (binPath,parameters)
    log = open(logFile, 'w')
    subprocess.call(cmd, stdout=log, stderr=log, shell=True)
    log.close()
    if jobSuccessful(logFile, "Done!\n"):
      print("done")
    else:
      print("FAILED!")
      exit(-1)

  else:
    print("already done!")
