# Copyright (c) 2010 CSIRO
# Australia Telescope National Facility (ATNF)
# Commonwealth Scientific and Industrial Research Organisation (CSIRO)
# PO Box 76, Epping NSW 1710, Australia
# atnf-enquiries@csiro.au
#
# This file is part of the ASKAP software distribution.
#
# The ASKAP software distribution is free software: you can redistribute it
# and/or modify it under the terms of the GNU General Public License as
# published by the Free Software Foundation; either version 2 of the License,
# or (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program; if not, write to the Free Software
# Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307 USA
#
# To build normally:
# $ make
#

CXX = g++ 
CFLAGS = -g -O3 -fstrict-aliasing -Wall -Wextra 

HEMIDIR ?= ../hemi
CUBDIR  ?= ../cub/cub

# SWAP these lines to switch from CUDA to CPU compiler
HEMICC = nvcc -x cu
#HEMICC = $(CXX)

ifeq ($(strip $(CXX)), icc)
	CFLAGS += -openmp  
	LINKFLAGS = -openmp
	LIBS = 
ifneq ($(strip $(HEMICC)), nvcc -x cu)
	HEMICFLAGS = -DHEMI_CUDA_DISABLE
endif
else
	CFLAGS += -fopenmp
	LIBS = -L/usr/local/cuda/lib64 -lcudart
endif

ifeq ($(strip $(HEMICC)), nvcc -x cu)
	HEMICFLAGS += -g -O3 -arch=sm_20 --ptxas-options=-v
	LINK = $(CXX) 
	LIBS = -L/usr/local/cuda/lib64 -lcudart
	LINKFLAGS += $(LIBS) 
else 
	HEMICFLAGS += $(CFLAGS)
	LINK = $(CXX)
	LINKFLAGS += $(LIBS) -fopenmp
endif

INCLUDE = -I./ -I/usr/local/cuda/include -I$(HEMIDIR) -I$(CUBDIR)

EXENAME = tHogbomCleanHemi
OBJS = $(EXENAME).o Stopwatch.o HogbomGolden.o HogbomHemi.o

all:		$(EXENAME)

HogbomHemi.o: HogbomHemi.h HogbomHemi.cc Parameters.h
			$(HEMICC) $(HEMICFLAGS) $(INCLUDE) -c HogbomHemi.cc

%.o:		%.cc %.h Parameters.h
			$(CXX) $(CFLAGS) $(INCLUDE) -c $<

$(EXENAME):	$(OBJS)
		$(LINK) $(LINKFLAGS) -o $(EXENAME) $(OBJS) $(LIBS)

clean:
		rm -f $(EXENAME) *.o

