# vorothreshold

Numba-python module to postprocess voids using the Voronoi tessellation and Delaunay scheme of the discrete tracers distribution. 

## Installation

The module can be installed by running

`pip install .`

## Dependencies

The excursion_set_functions make use of

- `numpy`
- `numba`
- `healpy`





## Getting started

The directory `examples` contains a Jupyter notebook for getting started, which is based on the output of the example of `VIDE`, available at [VIDE]. 


The module `voronoi_threshold` contains the min function. The module `main` contatins the `voronoi_threshold_finder` class, with authomatize the computation of the final void catalog.

The module `masks` contains functions to detect voids with Voronois cell touching borders (for lightcone use only).

The module `ovelaps` contains functions to find overlapping voids.

The module `read_funcs` contains functions to read `VIDE` quantities and the Voronoi scheme from `ZOBOV`.

The module `utilities` contains functions to convert from cartesian to spherical coordinates (both in R.A. - DEC or $\theta$ - $\phi$) and to map redshift in comoving distances and vice versa.


## WARNINGS

This is a beta version:

- The class `voronoi_threshold_finder` is currently implemented for ligthcones only, using the vide output.

- The (lightcone) example uses the output of the example of `VIDE`, which for the lightcone case is not optimal, as it too small, and most of the voids contains Voronoi cells touching borders. 

- The (lightcone) example shows a constat galaxy nuber density, which is usually not the case for surveys or mock catalogs.



[VIDE]: https://bitbucket.org/cosmicvoids/vide_public/


