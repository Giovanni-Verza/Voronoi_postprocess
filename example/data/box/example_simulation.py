#+
#   VIDE -- Void IDentification and Examination -- ./python_tools/void_pipeline/datasets/example_simulation.py
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
import os

PWD=os.getcwd()

# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# CONFIGURATION

# if True, will scan log files for last known completed state and run from there
continueRun = False

# stages:
#   1 : extract redshift slices from data
#   2 : void extraction using zobov
#   3 : removal of small voids and voids near the edge 
startCatalogStage = 1
endCatalogStage   = 3

# directory for the input simulation files
catalogDir = PWD+"/examples/"

# void catalog output directory
voidOutputDir = PWD+"/examples/example_simulation/"

# output directory for log files
logDir = PWD+"/logs/example_simulation/"

# output directory for figures
figDir = PWD+"/figs/example_simulation/"

# where to place the pipeline scripts
scriptDir = PWD+"/example_simulation/"

# don't change
dataType = "simulation"

# available formats for simulation: gadget, sdf, multidark
dataFormat = "multidark"

# units of position in Mpc/h
dataUnit = 1

# place particles on the lightcone (z-axis in sims)?
useLightCone = False

# add peculiar velocities?
doPecVel = False

# optimization: maximum number of parallel threads to use
numZobovThreads = 2

# optimization: number of subdivisions of the box
numZobovDivisions = 2

# maximum density for merging voids
#   0 (equivalent to infinitely large value) -> Merge everything (no threshold)
#   1e-9 (or smaller != 0) -> Do not merge anything
mergingThreshold = 1e-9

# prefix to give all outputs
prefix = "sim_"

# how many independent slices along the z-axis?
numSlices = 1

# how many subdivisions along the x- and y- axis?
#   ( = 2 will make 4 subvolumes for each slice, = 3 will make 9, etc.)
numSubvolumes = 1


###############################################################################
# Particles

# common filename of particle files
particleFileBase = "example_simulation_NNNN.dat"

# this flag will be replaced by values in fileNums list below
particleFileDummy = 'NNNN'

# list of file numbers for the particle files
fileNums = ["z0.0"]

# redshift of each file in the above fileNums list
redshifts = ["0.0"]

# list of desired subsamples - these are in unts of h Mpc^-3!
subSamples = [1.0]

# if True, do the subsampling in preparation (available for sdf and multidark)
doSubSamplingInPrep = False

# if 'absolute', subSamples are given in particles per cubic Mpc/h
# if 'relative', subSamples are given as a fraction of input particles
subSampleMode = "relative"

# shift the z-coord of sims with redshift
shiftSimZ = False

###############################################################################
# Halos

# common filename of halo files, leave blank to ignore halos
haloFileBase = ""
#haloFileBase = "mf_4s_1G_1k_bgc2_NNNNN.sdf"

# this flag will be replaced by values in fileNums list above
haloFileDummy = ''
#haloFileDummy = 'NNNNN'

# minimum halo mass cuts to apply for the halo catalog
#   use "none" to get all halos
minHaloMasses = []
#minHaloMasses = ["none", 1.2e13]

# locations of data in the halo catalog
haloFileMCol  = 6    # mass
haloFileXCol  = 0    # x
haloFileYCol  = 1    # y
haloFileZCol  = 2    # z
haloFileVXCol = 3    # v_x
haloFileVYCol = 4    # v_y
haloFileVZCol = 5    # v_z
haloFileColSep = ',' # separator
haloFileNumComLines = 0 # number of comments before data


###############################################################################
# simulation information

numPart = 1024*1024*1024
lbox = 999.983 # Mpc/h
omegaM = 0.2847979853038958
hubble = 0.6962 # h_0


###############################################################################
# HOD

# each of the HOD sets will be applied to each halo catalog defined above
hodParmList = [
  #{'name'       : "LowRes", #BOSS: Manera et al. 2012, eq. 26
  # 'Mmin'       : 0.0,
  # 'M1'         : 1.e14,
  # 'sigma_logM' : 0.596,
  # 'alpha'      : 1.0127,
  # 'Mcut'       : 1.19399e13,
  # 'galDens'    : 0.0002,
  #},
]

# END CONFIGURATION
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
