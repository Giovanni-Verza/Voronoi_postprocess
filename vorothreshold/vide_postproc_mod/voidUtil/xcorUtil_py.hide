#+
#   VIDE -- Void IDentification and Examination -- ./python_tools/vide/voidUtil/xcorUtil.py
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
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib import rc
from . import xcorlib
from vide.voidUtil import getArray

def computeXcor(catalog,
                figDir="./",
                Nmesh = 256,
                Nbin = 100
                ):

# Computes and plots void-void and void-matter(galaxy) correlations
#   catalog: catalog to analyze
#   figDir: where to place plots
#   Nmesh: number of grid cells in cic mesh-interpolation
#   Nbin:  number of bins in final plots

  # Parameters
  Lbox = catalog.boxLen[0] # Boxlength
  Lboxcut = 0.
  Lbox -= 2*Lboxcut
  
  # Input particle arrays of shape (N,3)
  xm = catalog.partPos # Halos / Galaxies / Dark matter
  xv = getArray(catalog.voids, 'macrocenter')
  
  
  # Interpolate to mesh
  dm, wm, ws = xcorlib.cic(xm, Lbox, Lboxcut = Lboxcut, Nmesh = Nmesh, weights = None)
  dv, wm, ws = xcorlib.cic(xv, Lbox, Lboxcut = Lboxcut, Nmesh = Nmesh, weights = None)
  
  # Fourier transform
  dmk = np.fft.rfftn(dm)
  dvk = np.fft.rfftn(dv)
  
  # 1D Power spectra & correlation functions
  ((Nm, km, Pmm, SPmm),(Nmx, rm, Xmm, SXmm)) = xcorlib.powcor(dmk, dmk, Lbox, Nbin, 'lin', True, True, 1)
  ((Nm, km, Pvm, SPvm),(Nmx, rm, Xvm, SXvm)) = xcorlib.powcor(dvk, dmk, Lbox, Nbin, 'lin', True, True, 1)
  ((Nm, km, Pvv, SPvv),(Nmx, rm, Xvv, SXvv)) = xcorlib.powcor(dvk, dvk, Lbox, Nbin, 'lin', True, True, 1)
  
  # Number densities
  nm = np.empty(len(km))
  nv = np.empty(len(km))
  nm[:] = len(xm)/Lbox**3
  nv[:] = len(xv)/Lbox**3
  
  
  # Plots
  mpl.rc('font', family='serif')
  ms = 2.5
  fs = 16
  mew = 0.1
  margin = 1.2
  kmin = km.min()/margin
  kmax = km.max()*margin
  rmin = rm.min()/margin
  rmax = rm.max()*margin
  
  
  # Density fields (projected)
  plt.imshow(np.sum(dm[:,:,:]+1,2),extent=[0,Lbox,0,Lbox],aspect='equal',cmap='YlGnBu_r',interpolation='gaussian')
  plt.xlabel(r'$x \;[h^{-1}\mathrm{Mpc}]$')
  plt.ylabel(r'$y \;[h^{-1}\mathrm{Mpc}]$')
  plt.title(r'Dark matter')
  plt.savefig(figDir+'/dm.eps', bbox_inches="tight")
  plt.savefig(figDir+'/dm.pdf', bbox_inches="tight")
  plt.savefig(figDir+'/dm.png', bbox_inches="tight")
  plt.clf()
  
  plt.imshow(np.sum(dv[:,:,:]+1,2)/Nmesh,extent=[0,Lbox,0,Lbox],aspect='equal',cmap='YlGnBu_r',interpolation='gaussian')
  plt.xlabel(r'$x \;[h^{-1}\mathrm{Mpc}]$')
  plt.ylabel(r'$y \;[h^{-1}\mathrm{Mpc}]$')
  plt.title(r'Voids')
  plt.savefig(figDir+'/dv.eps', bbox_inches="tight") #, dpi=300
  plt.savefig(figDir+'/dv.pdf', bbox_inches="tight") #, dpi=300
  plt.savefig(figDir+'/dv.png', bbox_inches="tight") #, dpi=300
  plt.clf()
  
  
  # Power spectra & correlation functions
  pa ,= plt.plot(km, Pmm, 'k-o', ms=0.8*ms, mew=mew, mec='k')
  #plt.plot(km, Pmm-1./nm, 'k--', ms=ms, mew=mew)
  plt.fill_between(km, Pmm+SPmm, abs(Pmm-SPmm), color='k', alpha=0.2)
  pb ,= plt.plot(km, Pvm, 'm-D', ms=ms, mew=mew, mec='k')
  plt.plot(km, -Pvm, 'mD', ms=ms, mew=mew, mec='k')
  plt.fill_between(km, abs(Pvm+SPvm), abs(Pvm-SPvm), color='m', alpha=0.2)
  pc ,= plt.plot(km, Pvv, 'b-p', ms=1.3*ms, mew=mew, mec='k')
  #plt.plot(km, Pvv-1./nv, 'b--', ms=ms, mew=mew)
  plt.fill_between(km, Pvv+SPvv, abs(Pvv-SPvv), color='b', alpha=0.2)
  plt.xlabel(r'$k \;[h\mathrm{Mpc}^{-1}]$')
  plt.ylabel(r'$P(k) \;[h^{-3}\mathrm{Mpc}^3]$')
  plt.title(r'Power spectra')
  plt.xscale('log')
  plt.yscale('log')
  plt.xlim(kmin,kmax)
  plt.ylim(10**np.floor(np.log10(abs(Pvm[1:]).min()))/margin, max(10**np.ceil(np.log10(Pmm.max())),10**np.ceil(np.log10(Pvv.max())))*margin)
  plt.legend([pa, pb, pc],['tt', 'vt', 'vv'],'best',prop={'size':12})
  plt.savefig(figDir+'/power.eps', bbox_inches="tight") 
  plt.savefig(figDir+'/power.pdf', bbox_inches="tight") 
  plt.savefig(figDir+'/power.png', bbox_inches="tight") 
  plt.clf()
  
  pa ,= plt.plot(rm, Xmm, 'k-o', ms=0.8*ms, mew=mew)
  plt.fill_between(rm, abs(Xmm+SXmm), abs(Xmm-SXmm), color='k', alpha=0.2)
  pb ,= plt.plot(rm, Xvm, 'm-D', ms=ms, mew=mew)
  plt.plot(rm, -Xvm, 'mD', ms=ms, mew=mew)
  plt.fill_between(rm, abs(Xvm+SXvm), abs(Xvm-SXvm), color='m', alpha=0.2)
  pc ,= plt.plot(rm, Xvv, 'b-p', ms=1.3*ms, mew=mew)
  plt.plot(rm, -Xvv, 'bp', ms=ms, mew=1.3*mew)
  plt.fill_between(rm, abs(Xvv+SXvv), abs(Xvv-SXvv), color='b', alpha=0.2)
  plt.xlabel(r'$r \;[h^{-1}\mathrm{Mpc}]$')
  plt.ylabel(r'$\xi(r)$')
  plt.title(r'Correlation functions')
  plt.xscale('log')
  plt.yscale('log')
  plt.xlim(rmin,rmax)
  plt.ylim(min(10**np.floor(np.log10(abs(Xvm).min())),10**np.floor(np.log10(abs(Xmm).min())))/margin, max(10**np.ceil(np.log10(Xmm.max())),10**np.ceil(np.log10(Xvv.max())))*margin)
  plt.legend([pa, pb, pc],['tt', 'vt', 'vv'],'best',prop={'size':12})
  plt.savefig(figDir+'/correlation.eps', bbox_inches="tight") 
  plt.savefig(figDir+'/correlation.pdf', bbox_inches="tight")
  plt.savefig(figDir+'/correlation.png', bbox_inches="tight")
  plt.clf()
  
  
  return 
