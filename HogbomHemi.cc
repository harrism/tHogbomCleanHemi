/// @copyright (c) 2011 CSIRO
/// Australia Telescope National Facility (ATNF)
/// Commonwealth Scientific and Industrial Research Organisation (CSIRO)
/// PO Box 76, Epping NSW 1710, Australia
/// atnf-enquiries@csiro.au
///
/// This file is part of the ASKAP software distribution.
///
/// The ASKAP software distribution is free software: you can redistribute it
/// and/or modify it under the terms of the GNU General Public License as
/// published by the Free Software Foundation; either version 2 of the License,
/// or (at your option) any later version.
///
/// This program is distributed in the hope that it will be useful,
/// but WITHOUT ANY WARRANTY; without even the implied warranty of
/// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
/// GNU General Public License for more details.
///
/// You should have received a copy of the GNU General Public License
/// along with this program; if not, write to the Free Software
/// Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307 USA
///
/// @author Ben Humphreys <ben.humphreys@csiro.au>

// Include own header file first
#include "HogbomHemi.h"

// System includes
#include <vector>
#include <iostream>
#include <cstddef>
#include <cmath>

// Local includes
#include "Parameters.h"
#ifdef HEMI_CUDA_COMPILER
    #include "cudaBlockMax.h"
    #ifdef USE_THRUST
        #include <thrust/device_vector.h>
        #include <thrust/extrema.h>
    #endif
#else
    #include <omp.h>
#endif

using namespace std;

HogbomHemi::HogbomHemi(std::vector<float>& residual,
                       std::vector<float>& psf)
{
    reportDevice();
    m_psf = new hemi::Array<float>(&psf[0], psf.size());
    m_blockMaxVal = new hemi::Array<float>(m_findPeakNBlocks, true);
    m_blockMaxPos = new hemi::Array<int>(m_findPeakNBlocks, true);
}

HogbomHemi::~HogbomHemi()
{
    delete m_psf;
    delete m_blockMaxVal;
    delete m_blockMaxPos;

#ifdef HEMI_CUDA_COMPILER
    checkCuda( cudaDeviceReset() );
#endif
}

void HogbomHemi::deconvolve(const vector<float>& dirty,
                            const size_t dirtyWidth,    
                            const vector<float>& psf,
                            const size_t psfWidth,
                            vector<float>& model,
                            vector<float>& residual)
{
    residual = dirty;
    hemi::Array<float> d_residual(&residual[0], residual.size());

    // Find the peak of the PSF
    float psfPeakVal = 0.0;
    size_t psfPeakPos = 0;
    findPeak(m_psf->readOnlyPtr(), m_psf->size(), psfPeakVal, psfPeakPos);
    
    cout << "Found peak of PSF: " << "Maximum = " << psfPeakVal
         << " at location " << idxToPos(psfPeakPos, psfWidth).x << ","
         << idxToPos(psfPeakPos, psfWidth).y << endl;

    for (unsigned int i = 0; i < g_niters; ++i) {
        // Find the peak in the residual image
        float absPeakVal = 0.0;
        size_t absPeakPos = 0;
        findPeak(d_residual.readOnlyPtr(), d_residual.size(), absPeakVal, absPeakPos);

        //cout << "Iteration: " << i + 1 << " - Maximum = " << absPeakVal
        //    << " at location " << idxToPos(absPeakPos, dirtyWidth).x << ","
        //    << idxToPos(absPeakPos, dirtyWidth).y << endl;

        // Check if threshold has been reached
        if (abs(absPeakVal) < g_threshold) {
            cout << "Reached stopping threshold" << endl;
            break;
        }

        // Add to model
        model[absPeakPos] += absPeakVal * g_gain;

        // Subtract the PSF from the residual image
        subtractPSF(m_psf->readOnlyPtr(), psfWidth, d_residual.ptr(), dirtyWidth, 
                    absPeakPos, psfPeakPos, absPeakVal, g_gain);
    }

    // force copy of residual back to host
    d_residual.readOnlyHostPtr();
}

#ifdef USE_THRUST
struct compare_abs
{
  HEMI_DEV_CALLABLE_INLINE_MEMBER
  bool operator()(float lhs, float rhs) { return abs(lhs) < abs(rhs); }
};
#endif

HEMI_KERNEL(findPeakLoop)(float *maxVal, int *maxPos, const float* image, int size)
{
#ifndef HEMI_DEV_CODE
    *maxVal = 0.0f;
    *maxPos = 0;

    #pragma omp parallel
#endif
    {
        float threadAbsMaxVal = 0.0;
        int threadAbsMaxPos = 0;
        #pragma omp for schedule(static)
        for (int i = hemiGetElementOffset(); i < size; i += hemiGetElementStride()) {
            if (abs(image[i]) > abs(threadAbsMaxVal)) {
                threadAbsMaxVal = image[i];
                threadAbsMaxPos = i;
            }
        }

#ifdef HEMI_DEV_CODE
        blockMax<HogbomHemi::FindPeakBlockSize>(maxVal, maxPos, threadAbsMaxVal, threadAbsMaxPos);
#else
        // Avoid using the double-checked locking optimization here unless
        // we can be certain that the load of a float is atomic
        #pragma omp critical
        if (abs(threadAbsMaxVal) > abs(*maxVal)) {
            *maxVal = threadAbsMaxVal;
            *maxPos = threadAbsMaxPos;
        }
#endif
    }
}

void HogbomHemi::findPeak(const float* image, size_t size, float& maxVal, size_t& maxPos)
{
#ifdef USE_THRUST
    thrust::device_vector<float>::iterator base_iter = 
        (thrust::device_vector<float>::iterator)thrust::device_pointer_cast((float*)image);
    
    compare_abs compare;
    thrust::device_vector<float>::iterator iter = 
        thrust::max_element(base_iter, base_iter + size, compare);
    
    maxPos = iter - base_iter;
    maxVal = *iter;
#else
    HEMI_KERNEL_LAUNCH(findPeakLoop, m_findPeakNBlocks, FindPeakBlockSize, 0, 0,
                       m_blockMaxVal->writeOnlyPtr(), m_blockMaxPos->writeOnlyPtr(),
                       image, size);   

    const float *mv = m_blockMaxVal->readOnlyHostPtr();
    const int *mp = m_blockMaxPos->readOnlyHostPtr();
    maxVal= mv[0];
    maxPos = mp[0];
#ifdef HEMI_CUDA_COMPILER
    for (int i = 1; i < m_findPeakNBlocks; ++i) {
        if (abs(mv[i]) > abs(maxVal)) {
            maxVal = mv[i];
            maxPos = mp[i];
        }
    }
#endif
#endif
}

HEMI_KERNEL(subtractPSFLoop)(const float* psf, const int psfWidth,
                             float* residual, const int residualWidth,
                             const int startx, const int starty,
                             int const stopx, const int stopy,
                             const int diffx, const int diffy,
                             const float absPeakVal,
                             const float gain)
{
    #pragma omp parallel for default(shared) schedule(static)
    for (int y = starty + hemiGetElementYOffset(); y <= stopy; y += hemiGetElementYStride()) {
        for (int x = startx + hemiGetElementXOffset(); x <= stopx; x += hemiGetElementXStride()) {
            residual[HogbomHemi::posToIdx(residualWidth, HogbomHemi::Position(x, y))] -= absPeakVal * gain
                * psf[HogbomHemi::posToIdx(psfWidth, HogbomHemi::Position(x - diffx, y - diffy))];
        }
    }    
}

void HogbomHemi::subtractPSF(const float* psf,
                             const size_t psfWidth,
                             float* residual,
                             const size_t residualWidth,
                             const size_t peakPos, 
                             const size_t psfPeakPos,
                             const float absPeakVal,
                             const float gain)
{
    const int rx = idxToPos(peakPos, residualWidth).x;
    const int ry = idxToPos(peakPos, residualWidth).y;

    const int px = idxToPos(psfPeakPos, psfWidth).x;
    const int py = idxToPos(psfPeakPos, psfWidth).y;

    const int diffx = rx - px;
    const int diffy = ry - py;

    const int startx = max(0, rx - px);
    const int starty = max(0, ry - py);

    const int stopx = min(residualWidth - 1, rx + (psfWidth - px - 1));
    const int stopy = min(residualWidth - 1, ry + (psfWidth - py - 1));

    const dim3 blockDim(32, 4);
        
    // Note: Both start* and stop* locations are inclusive.
    const int blocksx = ceil((stopx-startx+1.0f) / static_cast<float>(blockDim.x));
    const int blocksy = ceil((stopy-starty+1.0f) / static_cast<float>(blockDim.y));
    const dim3 gridDim(blocksx, blocksy);

    HEMI_KERNEL_LAUNCH(subtractPSFLoop, gridDim, blockDim, 0, 0, 
                       psf, psfWidth, residual, residualWidth, 
                       startx, starty, stopx, stopy, diffx, diffy, absPeakVal, gain);
}


void HogbomHemi::reportDevice(void)
{
#ifdef HEMI_CUDA_COMPILER
    std::cout << "+++++ Forward processing (CUDA) +++++" << endl;
    // Report the type of device being used
    int device;
    cudaDeviceProp devprop;
    cudaGetDevice(&device);
    cudaGetDeviceProperties(&devprop, device);
    std::cout << "    Using CUDA Device " << device << ": "
        << devprop.name << "( " << devprop.multiProcessorCount << " SMs)" << std::endl;

    m_findPeakNBlocks = 2 * devprop.multiProcessorCount;
#else
    std::cout << "+++++ Forward processing (OpenMP) +++++" << endl;
    std::cout << "    Using " << omp_get_max_threads() << " OpenMP threads" << endl;

    m_findPeakNBlocks =  1;
#endif
}
