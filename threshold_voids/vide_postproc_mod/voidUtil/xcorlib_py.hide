#+
#   VIDE -- Void IDentification and Examination -- ./python_tools/vide/voidUtil/xcorlib.py
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

# CIC interpolation
def cic(x, Lbox, Lboxcut = 0, Nmesh = 128, weights = None):

 if weights == None: weights = 1
 wm = np.mean(weights)
 ws = np.mean(weights**2)

 d = np.mod(x/(Lbox+2*Lboxcut)*Nmesh,1)

 box = ([Lboxcut,Lbox+Lboxcut],[Lboxcut,Lbox+Lboxcut],[Lboxcut,Lbox+Lboxcut])

 rho = np.histogramdd(x, range = box, bins = Nmesh, weights = weights*(1-d[:,0])*(1-d[:,1])*(1-d[:,2]))[0] \
     + np.roll(np.histogramdd(x, range = box, bins = Nmesh, weights = weights*d[:,0]*(1-d[:,1])*(1-d[:,2]))[0],1,0) \
     + np.roll(np.histogramdd(x, range = box, bins = Nmesh, weights = weights*(1-d[:,0])*d[:,1]*(1-d[:,2]))[0],1,1) \
     + np.roll(np.histogramdd(x, range = box, bins = Nmesh, weights = weights*(1-d[:,0])*(1-d[:,1])*d[:,2])[0],1,2) \
     + np.roll(np.roll(np.histogramdd(x, range = box, bins = Nmesh, weights = weights*d[:,0]*d[:,1]*(1-d[:,2]))[0],1,0),1,1) \
     + np.roll(np.roll(np.histogramdd(x, range = box, bins = Nmesh, weights = weights*d[:,0]*(1-d[:,1])*d[:,2])[0],1,0),1,2) \
     + np.roll(np.roll(np.histogramdd(x, range = box, bins = Nmesh, weights = weights*(1-d[:,0])*d[:,1]*d[:,2])[0],1,1),1,2) \
     + np.roll(np.roll(np.roll(np.histogramdd(x, range = box, bins = Nmesh, weights = weights*d[:,0]*d[:,1]*d[:,2])[0],1,0),1,1),1,2)

 rho /= wm
 
 rho = rho/rho.mean() - 1.
 
 return (rho, wm, ws)


# Power spectra & correlation functions
def powcor(d1, d2, Lbox, Nbin = 10, scale = 'lin', cor = False, cic = True, dim = 1):

 Nmesh = len(d1)

 # CIC correction
 if cic:
  wid = np.indices(np.shape(d1))
  wid[np.where(wid >= Nmesh/2)] -= Nmesh
  wid = wid*np.pi/Nmesh + 1e-100
  wcic = np.prod(np.sin(wid)/wid,0)**2

 # Shell average power spectrum
 dk = 2*np.pi/Lbox
 Pk = np.conj(d1)*d2*(Lbox/Nmesh**2)**3
 if cic: Pk /= wcic**2
 
 (Nm, km, Pkm, SPkm) = shellavg(np.real(Pk), dk, Nmesh, Nbin = Nbin, xmin = 0., xmax = Nmesh*dk/2, scale = scale, dim = dim)
 
 # Inverse Fourier transform and shell average correlation function
 if cor:
  if cic: Pk *= wcic**2 # Undo cic-correction in correlation function
  dx = Lbox/Nmesh
  Xr = np.fft.irfftn(Pk)*(Nmesh/Lbox)**3

  (Nmx, rm, Xrm, SXrm) = shellavg(np.real(Xr), dx, Nmesh, Nbin = Nbin/2, xmin = dx, xmax = 140., scale = scale, dim = dim)
  
  return ((Nm, km, Pkm, SPkm),(Nmx, rm, Xrm, SXrm))
 
 else: return (Nm, km, Pkm, SPkm)


# Shell averaging
def shellavg(f, dx, Nmesh, Nbin = 10, xmin = 0., xmax = 1., scale = 'lin', dim = 1):
 
 x = np.indices(np.shape(f))
 x[np.where(x >= Nmesh/2)] -= Nmesh
 f = f.flatten()
 
 if scale == 'lin': bins = xmin+(xmax-xmin)* np.linspace(0,1,Nbin+1)
 if scale == 'log': bins = xmin*(xmax/xmin)**np.linspace(0,1,Nbin+1)

 if dim == 1: # 1D
  x = dx*np.sqrt(np.sum(x**2,0)).flatten()
  Nm = np.histogram(x, bins = bins)[0]
  xm = np.histogram(x, bins = bins, weights = x)[0]/Nm
  fm = np.histogram(x, bins = bins, weights = f)[0]/Nm
  fs = np.sqrt((np.histogram(x, bins = bins, weights = f**2)[0]/Nm - fm**2)/(Nm-1))
  return (Nm, xm, fm, fs)

 elif dim == 2: # 2D
  xper = dx*np.sqrt(x[0,:,:,:]**2 + x[1,:,:,:]**2 + 1e-100).flatten()
  xpar = dx*np.abs(x[2,:,:,:]).flatten()
  x = dx*np.sqrt(np.sum(x**2,0)).flatten()
  Nm = np.histogram2d(xper, xpar, bins = [bins,bins])[0]
  xmper = np.histogram2d(xper, xpar, bins = [bins,bins], weights = xper)[0]/Nm
  xmpar = np.histogram2d(xper, xpar, bins = [bins,bins], weights = xpar)[0]/Nm
  fm = np.histogram2d(xper, xpar, bins = [bins,bins], weights = f)[0]/Nm
  return (Nm, xmper, xmpar, fm)
