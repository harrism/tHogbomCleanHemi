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
    #include <cub.cuh>
#else
    #include <omp.h>
#endif

using namespace std;

HogbomHemi::HogbomHemi(std::vector<float>& psf)
{
    reportDevice();
    m_psf = new hemi::Array<float>(&psf[0], psf.size());
    m_blockMax = new hemi::Array<MaxCandidate>(m_findPeakNBlocks, true);
}

HogbomHemi::~HogbomHemi()
{
    delete m_psf;
    delete m_blockMax;

#ifdef HEMI_CUDA_COMPILER
    checkCuda( cudaDeviceReset() );
#endif
}

void HogbomHemi::deconvolve(const vector<float>& dirty,
                            const size_t dirtyWidth,    
                            const size_t psfWidth,
                            vector<float>& model,
                            vector<float>& residual)
{
    residual = dirty;
    hemi::Array<float> d_residual(&residual[0], residual.size());

    // Find the peak of the PSF
    MaxCandidate psfPeak = {0.0, 0};
    findPeak(m_psf->readOnlyPtr(), m_psf->size(), psfPeak);
    
    cout << "Found peak of PSF: " << "Maximum = " << psfPeak.value
         << " at location " << idxToPos(psfPeak.index, psfWidth).x << ","
         << idxToPos(psfPeak.index, psfWidth).y << endl;

    for (unsigned int i = 0; i < g_niters; ++i) {
        // Find the peak in the residual image
        MaxCandidate absPeak = {0.0, 0};
        findPeak(d_residual.readOnlyPtr(), d_residual.size(), absPeak);

        //cout << "Iteration: " << i + 1 << " - Maximum = " << absPeak.value
        //    << " at location " << idxToPos(absPeak.index, dirtyWidth).x << ","
        //    << idxToPos(absPeak.index, dirtyWidth).y << endl;

        // Check if threshold has been reached
        if (abs(absPeak.value) < g_threshold) {
            cout << "Reached stopping threshold" << endl;
            break;
        }

        // Add to model
        model[absPeak.index] += absPeak.value * g_gain;

        // Subtract the PSF from the residual image
        subtractPSF(m_psf->readOnlyPtr(), psfWidth, d_residual.ptr(), dirtyWidth, 
                    absPeak.index, psfPeak.index, absPeak.value, g_gain);
    }

    // force copy of residual back to host
    d_residual.readOnlyHostPtr();
}

HEMI_DEV_CALLABLE_INLINE
int blockId()
{
#ifdef HEMI_CUDA_COMPILER
    return blockIdx.x;
#else
    return omp_get_thread_num();
#endif
}

// For CUB
struct MaxOp
{
    HEMI_DEV_CALLABLE_INLINE_MEMBER
    MaxCandidate operator()(const MaxCandidate &a, const MaxCandidate &b)
    {
        return (abs(b.value) > abs(a.value)) ? b : a;
    }
};

HEMI_DEV_CALLABLE_INLINE
void findPeakReduce(MaxCandidate *peak, MaxCandidate threadMax)
{
#ifdef HEMI_DEV_CODE
    typedef cub::BlockReduce<MaxCandidate, HogbomHemi::FindPeakBlockSize> BlockMax;
    __shared__ typename BlockMax::TempStorage temp_storage;
    MaxOp op;
    threadMax = BlockMax(temp_storage).Reduce(threadMax, op);
    if (threadIdx.x == 0)
#endif
    {           
        peak[blockId()] = threadMax;
    }
}

HEMI_KERNEL(findPeakLoop)(MaxCandidate *peak, const float* image, int size)
{
    peak->value = 0.0f;
    peak->index = 0;

    #pragma omp parallel
    {
        MaxCandidate threadMax = {0.0, 0};
        
        // parallel raking reduction (independent threads)
        #pragma omp for schedule(static)
        for (int i = hemiGetElementOffset(); i < size; i += hemiGetElementStride()) {
            if (abs(image[i]) > abs(threadMax.value)) {
                threadMax.value = image[i];
                threadMax.index = i;
            }
        }

        findPeakReduce(peak, threadMax);
    }
}

void HogbomHemi::findPeak(const float* image, size_t size, MaxCandidate &peak)
{
    HEMI_KERNEL_LAUNCH(findPeakLoop, m_findPeakNBlocks, FindPeakBlockSize, 0, 0,
                       m_blockMax->writeOnlyPtr(), image, size);   

    const MaxCandidate *maximum = m_blockMax->readOnlyHostPtr();
    
    peak = maximum[0];

    // serial final reduction
    for (int i = 1; i < m_findPeakNBlocks; ++i) {
        if (abs(maximum[i].value) > abs(peak.value))
            peak = maximum[i];
    }

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

    m_findPeakNBlocks =  omp_get_max_threads();
#endif
}
