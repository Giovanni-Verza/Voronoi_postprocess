#!/usr/bin/env python
#+
#   VIDE -- Void IDentification and Examination -- ./python_tools/void_pipeline/datasets/example_observation.py
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
import numpy as np
from vide.backend.classes import *

# if True, will scan log files for last known completed state and run from there
continueRun = False

# stages:
#   1 : extract redshift slices from data
#   2 : void extraction using zobov
#   3 : removal of small voids and voids near the edge
startCatalogStage = 1
endCatalogStage   = 3

basePath = os.path.dirname(os.path.abspath(__file__))
basePath = os.path.abspath(os.path.join(basePath,"examples"))

# directory for input data files
inputDataDir = basePath

# void catalog output directory
workDir = os.path.join(basePath,"example_observation")

# output directory for log files
logDir = os.path.join(basePath,"logs","example_observation")

# output directory for figures
figDir = os.path.join(basePath,"figs","example_observation")

# optimization: maximum number of parallel threads to use
numZobovThreads = 2

# optimization: number of subdivisions of the box
numZobovDivisions = 2

# Maximum density for merging voids
#   0 (equivalent to infinitely large value) -> Merge everything (no threshold)
#   1e-9 (or smaller != 0) -> Do not merge anything
mergingThreshold = 1e-9

# don't change this
dataSampleList = []

# define your volume-limited samples
newSample = Sample(
                   # path to galaxy file is inputDataDir+dataFile
                   dataFile = "example_observation.dat",

                   # full name for this sample
                   fullName = "example_observation",

                   # a convenient nickname
                   nickName = "exobs",

                   # don't change this
                   dataType = "observation",

                   # assume sample is volume-limited?
                   volumeLimited = True,

                   # HEALpix mask file
                   maskFile = inputDataDir+"/example_observation_mask.fits",

                   # radial selection function (if not volume limited)
                   selFunFile = None,

                   # max and min redshifts of galaxies in your sample
                   zBoundary = (0.0, 0.15),

                   # max and min redshifts where you want to find voids
                   zRange = (0.1, 0.15),

                   # leave this at -1 for mean particle separation,
                   # or specify your own in Mpc/h
                   minVoidRadius = -1,

                   # density of mock particles in cubic Mpc/h
                   # (make this as high as you can afford)
                   fakeDensity = 0.05,

                   # if true, convert to comoving space using LCDM cosmology
                   useComoving = True,

                   # cosmology
                   omegaM = 0.3,

                   )
dataSampleList.append(newSample)

# repeat the above block for any other samples
