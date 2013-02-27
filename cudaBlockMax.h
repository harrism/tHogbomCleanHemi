__device__ inline
float warpMax(volatile float *maxVal, float v, int idx)
{
    maxVal[idx] = v = max(v, maxVal[idx + 32]);
    maxVal[idx] = v = max(v, maxVal[idx + 16]);
    maxVal[idx] = v = max(v, maxVal[idx +  8]);
    maxVal[idx] = v = max(v, maxVal[idx +  4]);
    maxVal[idx] = v = max(v, maxVal[idx +  2]);
    maxVal[idx] = v = max(v, maxVal[idx +  1]);
    return v;
}

template <int blockSize>
__device__ inline
void blockMax(float *blockMaxVal, int *blockMaxPos, float threadMax, int threadPos)
{
    __shared__ float maxVal[blockSize];
    float v = maxVal[threadIdx.x] = abs(threadMax);
    
    __syncthreads();

    if (blockSize >= 1024) {
        if (threadIdx.x < 512) { maxVal[threadIdx.x] = v = max(v, maxVal[threadIdx.x + 512]); }
        __syncthreads();
    }
    if (blockSize >= 512) {
        if (threadIdx.x < 256) { maxVal[threadIdx.x] = v = max(v, maxVal[threadIdx.x + 256]); }
        __syncthreads();
    }
    if (blockSize >= 256) {
        if (threadIdx.x < 128) { maxVal[threadIdx.x] = v = max(v, maxVal[threadIdx.x + 128]); }
        __syncthreads();
    }
    if (blockSize >= 128) {
        if (threadIdx.x < 64)  { maxVal[threadIdx.x] = v = max(v, maxVal[threadIdx.x + 64]); }
        __syncthreads();
    }
    if (threadIdx.x < 32)      { v = warpMax(maxVal, v, threadIdx.x); }

    __syncthreads();

    if (maxVal[0] == abs(threadMax)) {
        blockMaxVal[blockIdx.x] = threadMax;
        blockMaxPos[blockIdx.x] = threadPos;
    }
}