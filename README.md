# vorothreshold

Numba-python module to postprocess of find voids using the Voronoi tessellation and Delaunay scheme of the discrete tracers distribution, implemented for simulation boxes and lightcones. The code can handle survey masks and weights.

## Installation

The module can be installed by running

`pip install .`

## Dependencies

The excursion_set_functions make use of

- `numpy`
- `numba`
- `healpy`





## Getting started

The directory `examples` contains two Jupyter notebooks for getting started, for both the simulation box and lightcone cases. They are both based on the output of the example of `VIDE`, available at [VIDE]. 

The module `voronoi_threshold` contains the min function. The module `main` contatins the `voronoi_threshold_finder` class, with authomatize the computation of the final void catalog.

The module `masks` contains functions to detect voids with Voronois cell touching borders (for lightcone use only).

The module `ovelaps` contains functions to find overlapping voids.

The module `read_funcs` contains functions to read `VIDE` quantities and the Voronoi scheme from `ZOBOV`.

The module `utilities` contains functions to convert from cartesian to spherical coordinates (both in R.A. - DEC or $\theta$ - $\phi$) and to map redshift in comoving distances and vice versa.



[VIDE]: https://bitbucket.org/cosmicvoids/vide_public/


