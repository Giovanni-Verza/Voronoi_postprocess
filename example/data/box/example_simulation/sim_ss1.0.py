#!/usr/bin/env/python
import os
from vide.backend.classes import *

continueRun = False # set to True to enable restarting aborted jobs
startCatalogStage = 1
endCatalogStage   = 3
               
regenerateFlag = False

mergingThreshold = 1e-9

dataSampleList = []
           
setName = "sim_ss1.0"

workDir = "/home/giovanni/Desktop/Codes/Voronoi_postprocess/test/data/box/examples/example_simulation//sim_ss1.0/"
inputDataDir = "/home/giovanni/Desktop/Codes/Voronoi_postprocess/test/data/box/examples/"
figDir = "/home/giovanni/Desktop/Codes/Voronoi_postprocess/test/data/box/figs/example_simulation//sim_ss1.0/"
logDir = "/home/giovanni/Desktop/Codes/Voronoi_postprocess/test/data/box/logs/example_simulation//sim_ss1.0/"

numZobovDivisions = 2
numZobovThreads = 2
               
newSample = Sample(dataFile = "example_simulation_z0.0.dat",
                   dataFormat = "multidark",
                   dataUnit = 1,
                   fullName = "sim_ss1.0_z0.00_d00",
                   nickName = " SS 1.0, z = 0.00",
                   dataType = "simulation",
                   zBoundary = (0.00, 0.36),
                   zRange    = (0.00, 0.36),
                   zBoundaryMpc = (0.00, 999.98),
                   shiftSimZ = False,
                   omegaM    = 0.2847979853038958,
                   minVoidRadius = 1,
                   profileBinSize = "auto",
                   includeInHubble = True,
                   partOfCombo = False,
                   boxLen = 999.983,
                   usePecVel = False,
                   numSubvolumes = 1,
                   mySubvolume = "00",
                   useLightCone = False,
                   subsample = "1.0")
dataSampleList.append(newSample)
  