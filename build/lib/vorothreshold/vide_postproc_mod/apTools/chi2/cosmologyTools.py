#+
#   VIDE -- Void IDentification and Examination -- ./python_tools/vide/apTools/chi2/cosmologyTools.py
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
# a suite of functions to compute expansion rates, angular diameter 
# distances, and expected void stretching

import numpy as np
import scipy.integrate as integrate
from vide.backend import *

__all__=['expansion', 'angularDiameter', 'aveExpansion']

# returns 1/E(z) for the given cosmology
def expansion(z, Om = 0.27, Ot = 1.0, w0 = -1.0, wa = 0.0):
  ez = Om * (1+z)**3 + (Ot-Om)# * (1+z)**(3.+3*wz)
  #ez = Om * (1+z)**3 + (Ot-Om)# * integrade.quad(eosDE, 0.0, z, args=(w0,wa))[0]
  ez = 1./np.sqrt(ez)
  return ez

# returns DE value at redshift z
def eosDE(z, w0 = -1.0, wa = 0.0):
  return w0 + wa*z/(1+z)
  
# returns D_A(z) for the given cosmology
def angularDiameter(z, Om = 0.27, Ot = 1.0, w0 = -1.0, wa = 0.0):
  da = integrate.quad(expansion, 0.0, z, args=(Om, Ot, w0, wa))[0]
  return da
  
# -----------------------------------------------------------------------------
# returns average expected expansion for a given redshift range
def aveExpansion(zStart, zEnd, Om = 0.27, Ot = 1.0, w0 = -1.0, wa = 0.0):
  if zStart == 0.0: zStart = 1.e-6
  ave = integrate.quad(expansion, zStart, zEnd, args=(Om, Ot, w0, wa))[0]
  ave = (zEnd-zStart)/ave
  return ave


