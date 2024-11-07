#+
#   VIDE -- Void IDentification and Examination -- ./python_tools/vide/apTools/profiles/getSurveyProps.py
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
import numpy as np
import healpy as healpy
import scipy.integrate
import vide as ct
import os

__all__=['getSurveyProps']
   
# returns the volume and galaxy density for a given redshit slice
def getSurveyProps(maskFile, zmin, zmax, selFunMin, selFunMax, portion, selectionFuncFile=None, useComoving=False):

  LIGHT_SPEED = 299792.458

  mask = healpy.read_map(maskFile)
  area = (1.*np.size(np.where(mask > 0)) / np.size(mask)) * 4.*np.pi

  if useComoving:
    zmin = LIGHT_SPEED/100.*ct.angularDiameter(zmin, Om=0.27)
    zmax = LIGHT_SPEED/100.*ct.angularDiameter(zmax, Om=0.27)
    selFunMin = LIGHT_SPEED/100.*ct.angularDiameter(selFunMin, Om=0.27)
    selFunMax = LIGHT_SPEED/100.*ct.angularDiameter(selFunMax, Om=0.27)
  else:
    zmin = zmin * 3000
    zmax = zmax * 3000
    selFunMin *= 3000
    selFunMax *= 3000
  volume = area * (zmax**3  - zmin**3) / 3

  if selectionFuncFile == None:
    nbar = 1.0

  elif not os.access(selectionFuncFile, os.F_OK):
    print("   Warning, selection function file %s not found, using default of uniform selection." % selectionFuncFile)
    nbar = 1.0

  else:
    selfunc = np.genfromtxt(selectionFuncFile)
    selfunc = np.array(selfunc)
    selfunc[:,0] = selfunc[:,0]/100.
    selfuncUnity = selfunc
    selfuncUnity[:,1] = 1.0
    selfuncMin = selfunc[0,0]
    selfuncMax = selfunc[-1,0]
    selfuncDx = selfunc[1,0] - selfunc[0,0]
    selfuncN = np.size(selfunc[:,0])
  
    selFunMin = max(selFunMin, selfuncMin)
    selFunMax = min(selFunMax, selfuncMax)
  
  
    def f(z): return selfunc[np.ceil((z-selfuncMin)/selfuncDx), 1]*z**2 
    def fTotal(z): return selfuncUnity[np.ceil((z-selfuncMin)/selfuncDx), 1]*z**2 
  
    zrange = np.linspace(selFunMin, selFunMax)
  
    nbar = scipy.integrate.quad(f, selFunMin, selFunMax)
    nbar = nbar[0]
  
    ntotal = scipy.integrate.quad(fTotal, 0.0, max(selfuncUnity[:,0]))
    ntotal = ntotal[0]
  
    nbar = ntotal / area / nbar


  #print "PROPERTIES: ", volume, nbar
 
  return (volume, nbar)
