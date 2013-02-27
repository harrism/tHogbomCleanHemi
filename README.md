tHogbomCleanHemi
================

### Portable CUDA / OpenMP implementation of the Hogbom Clean Benchmark

This is a port of a Hogbom image cleaning algorithm benchmark originally written by Ben Humphreys of CSIRO ATNF. The port uses Hemi (http://github.com/harrism/hemi) to provide both a fast parallel CUDA implementation for GPUs and an OpenMP parallel implementation for CPUs.

Hemi enables most of the code to be used for both implementations.

The Hemi port was done by [Mark Harris](http://github.com/harrism).

Prerequisites
-------------

You must clone the Hemi headers from github. Hemi is available at http://github.com/harrism/hemi. Then edit the Makefile and change the value of `HEMIDIR` to the relative path of your clone of hemi.

To run on the GPU, you need a CUDA-capable GPU. More info about [CUDA](http://developer.nvidia.com/cuda-toolkit)

Compile
-----------

You should update the makefile to change the `-arch` flag to nvcc to specify the GPU architecture you are interested in. In the distribution it is set to `-arch=sm_20`

By default, running `make` will build the code to run on a CUDA GPU. To build for CPUs, comment out the first line below and uncomment the second line.

    HEMICC = nvcc -x cu
    #HEMICC = $(CXX)

Then run
    > make

You can also change the value of `CXX` to `icc` to use the Intel compiler, if you have it installed.

Run
-------

Make sure the dirty.img and psf.img files are in the same location as the executable, and then run it.
    > tHogbomCleanHemi

tHogbomClean License
--------------------

@copyright (c) 2011 CSIRO
Australia Telescope National Facility (ATNF)
Commonwealth Scientific and Industrial Research Organisation (CSIRO)
PO Box 76, Epping NSW 1710, Australia
atnf-enquiries@csiro.au

This file is part of the ASKAP software distribution.

The ASKAP software distribution is free software: you can redistribute it
and/or modify it under the terms of the GNU General Public License as
published by the Free Software Foundation; either version 2 of the License,
or (at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program; if not, write to the Free Software
Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307 USA

@author Ben Humphreys <ben.humphreys@csiro.au>

@author Ported to use Hemi & CUDA by Mark Harris <mharris@nvidia.com>

Hemi License
------------
Hemi is licensed under the Apache License, version 2.0
Hemi is available at http://github.com/harrism/hemi
