#+
#   VIDE -- Void IDentification and Examination -- ./python_tools/vide/voidUtil/profileUtil.py
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
__all__=['buildProfile','fitHSWProfile','getHSWProfile',]

from vide.backend.classes import *
from vide.voidUtil import *
from .plotDefs import *
import numpy as np
import os
import vide.apTools as vp
from scipy.optimize import curve_fit
from scipy.interpolate import interp1d

def HSWProfile(r, rs, dc):
  alpha = -2.0*rs + 4.0
  if rs < 0.91:
    beta = 17.5*rs - 6.5
  else:
    beta = -9.8*rs + 18.4
  return dc * (1 - (r/rs)**alpha) / (1+ (r)**beta) + 1

# -----------------------------------------------------------------------------
def buildProfile(catalog, rMin, rMax, nBins=10):

# builds a stacked radial density profile from the given catalog
#   catalog: void catalog
#   rMin: minimum void radius, in Mpc/h
#   rMax: maximum void radius, in Mpc/h
#   nBins: number of bins in profile (detaulf 10)
#
# returns:
#   binCenters: array of radii in binned profile
#   stackedProfile: the stacked density profile
#   sigmas: 1-sigma uncertainty in each bin

  rMaxProfile = rMin*3 + 2
  periodicLine = getPeriodic(catalog.sampleInfo)

  print("  Building particle tree...")
  partTree = getPartTree(catalog)

  print("  Selecting voids to stack...")
  voidsToStack = [v for v in catalog.voids if (v.radius > rMin and v.radius < rMax)]

  if len(voidsToStack) == 0:
    print("  No voids to stack!")
    return -1, -1, -1
    
  print("  Stacking voids...")
  allProfiles = []
  for void in voidsToStack:
    center = void.macrocenter
    
    localPart = catalog.partPos[ getBall(partTree, center, rMaxProfile) ]
    shiftedPart = shiftPart(localPart, center, periodicLine, catalog.ranges)

    dist = np.sqrt(np.sum(shiftedPart[:,:]**2, axis=1))
    thisProfile, radii = np.histogram(dist, bins=nBins, range=(0,rMaxProfile))
    deltaV = 4*np.pi/3*(radii[1:]**3-radii[0:(radii.size-1)]**3)
    thisProfile = np.float32(thisProfile)
    thisProfile /= deltaV
    thisProfile /= catalog.volNorm
    allProfiles.append(thisProfile)
    binCenters = 0.5*(radii[1:] + radii[:-1])

  nVoids = len(voidsToStack)
  stackedProfile = np.mean(allProfiles, axis=0) 
  sigmas = np.std(allProfiles, axis=0) / np.sqrt(nVoids)
  
  return binCenters, stackedProfile, sigmas


# -----------------------------------------------------------------------------
def fitHSWProfile(radii, densities, sigmas, rV):

# fits the given density profile to the HSW function
#   radii: array of radii
#   densities: array of densities in units of mean density
#   sigmas: array of uncertainties
#   rV: mean effective void radius
# 
# returns:
#   popt: best-fit values of rs and dc
#   pcov: covariance matrix
#   rVals: array of radii for best-fit curve
#   hswProfile: array of densities for best-fit curve in units of mean density

  popt, pcov = curve_fit(HSWProfile, radii/rV, densities,
                         sigma=sigmas,
                         maxfev=10000, xtol=5.e-3,
                         p0=[1.0,-1.0])

  # return best-fits
  rVals = np.linspace(0.0, radii[-1], 100) / rV
  return popt, pcov, rVals*rV, HSWProfile(rVals,popt[0],popt[1])

# -----------------------------------------------------------------------------
def getHSWProfile(density, radius):

# returns the HSW profile for the given sample density and void size
#   will interpolate/extrapole the radius

#  density: choice of sample (see arXiv:1309.5087):
#           maxDM: DM at 1 particles per cubic Mpc/h
#           fullDM: DM at 0.01 particles per cubic Mpc/h
#           denseDM: DM at 4.e-3 particles per cubic Mpc/h
#           sparseDM: DM at 3.e-4 particles per cubic Mpc/h
#           
#           denseHalos: halos at 4.e-3 particles per cubic Mpc/h
#           sparseHalos: halos at 3.e-4 particles per cubic Mpc/h
#          
#           denseGal: galaxies at 4.e-3 particles per cubic Mpc/h
#           sparseGal: galaxies at 3.e-4 particles per cubic Mpc/h
#
#  radius: void size in Mpc/h

# returns:
#   (rs, dc): best-fit values
#   binCenters: array of radii in binned profile
#   stackedProfile: the density profile

  samples = [
    {'name': 'maxDM',
     'rv': [8.74041095,  11.65557095,  15.54746657,  20.94913774, 28.41063131,  38.61523696,  51.85944898,  69.42297033],
     'rs': [0.74958738,  0.79650829,  0.86245251,  0.93960051, 1.01595177,  1.13159483,  1.31457096,  1.65611709],
     'dc': [-0.95353184, -0.94861939, -0.91888455, -0.84480086, -0.73431544, -0.62614422, -0.54908132, -0.4912146],
    },
    {'name': 'fullDM',
     'rv': [10, 15, 20, 25, 30, 35],
     'rs': [ 0.76986779,  0.80980775,  0.86590177,  0.93732629,  1.02643542,
        1.12875503],
     'dc': [-0.81021   , -0.78115474, -0.78326026, -0.78670109, -0.76626508,
       -0.72531084],
    },
    {'name': 'denseDM',
     'rv': [10, 15, 20, 25, 30, 35, 40],
     'rs': [0.7477462 ,  0.79932797,  0.84369297,  0.90309363,  0.92990401,
        1.06970842,  1.16393474],
     'dc': [-0.78333107, -0.75780925, -0.71586397, -0.74669512, -0.74902649,
       -0.75342964, -0.76598043],
    },
    {'name': 'sparseDM',
     'rv': [25, 30, 35, 40, 45, 50, 55, 60],
     'rs': [0.78351117,  0.79751047,  0.8225573 ,  0.83751894,  0.85443167,
        0.86346031,  0.85692501,  0.91470448],
     'dc': [-0.67407548, -0.62389586, -0.59125414, -0.55341724, -0.54659457,
       -0.5297821 , -0.534683  , -0.51055946],
    },
    {'name': 'denseHalos',
     'rv': [15, 20, 25, 30, 35, 40, 45, 50, 55],
     'rs': [0.75393528,  0.7758442 ,  0.79720886,  0.81560553,  0.83797299,
        0.84377082,  0.84900783,  0.8709897 ,  0.86886687],
     'dc': [-0.81348968, -0.77362777, -0.74336192, -0.72571135, -0.67928029,
       -0.6279349 , -0.6313316 , -0.55188564, -0.48096026],
    },
    {'name': 'sparseHalos',
     'rv': [25, 30, 35, 40, 45, 50, 55, 60, 65, 70],
     'rs': [0.76420703,  0.78939067,  0.80480265,  0.82315275,  0.8158607 ,
        0.82553517,  0.83843323,  0.85759226,  0.8471268 ,  0.89286939],
     'dc': [-0.79317192, -0.81661415, -0.76770778, -0.7151494 , -0.718561  ,
       -0.70858856, -0.68995608, -0.67415305, -0.63706798, -0.5303759],
    },
    {'name': 'denseGal',
     'rv': [10, 15, 20, 25, 30, 35, 40, 45],
     'rs': [0.70048139,  0.73717884,  0.75338516,  0.76782043,  0.79292536,
        0.80157122,  0.8207239 ,  0.8091386],
     'dc': [-0.83166549, -0.86505329, -0.81066899, -0.7662453 , -0.72840363,
       -0.65163607, -0.57937656, -0.57125164],
    },
    {'name': 'sparseGal',
     'rv': [25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75],
     'rs': [0.75928922,  0.76622648,  0.77695425,  0.79963152,  0.8045125 ,
        0.81892965,  0.83439691,  0.86600085,  0.83166875,  0.85283258,
        0.83971344],
     'dc': [-0.82413212, -0.87483536, -0.8221596 , -0.78459706, -0.75290061,
       -0.77513988, -0.70012913, -0.67487994, -0.69903308, -0.65811992,
       -0.57929526],
    },
  ]

  mySample = next((item for item in samples if item['name'] == density), None)
  if mySample == None:
    print("Sample", density," not found! Use one of ", [item['name'] for item in samples])
    return 

  # interpolate the radii
  rsFunc = interp1d( mySample['rv'], mySample['rs'], kind='cubic' )
  dcFunc = interp1d( mySample['rv'], mySample['dc'], kind='cubic' )
    
  rs = rsFunc(radius)
  dc = dcFunc(radius)

  # return best-fits
  rVals = np.linspace(0.0, 3*radius, 100) / radius
  return (rs,dc), rVals*radius, HSWProfile(rVals,rs,dc)
  
