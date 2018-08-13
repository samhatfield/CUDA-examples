#include "params.h"

// ODE device function
__device__ half2 dXdT(half2 x__2, half2 x__1, half2 x, half2 x_1) {
    return __hfma2(__hadd2(x_1, __hneg2(x__2)), x__1, __hadd2(__hneg2(x), f));
}

// Index shifter device function
__device__ int shft(int n, int m) {
    return (n + m + N/2)%(N/2);
}

// Main step kernel function
__global__ void step(half2* __restrict__ in, half2* __restrict__ out) {
    // Get global thread ID
    int tid = threadIdx.x + blockDim.x*blockIdx.x;

    // Shared work array
    __shared__ half2 work[N/2];

    // Intermediate steps
    half2 k1, k2, k3, k4;


    if (tid < N/2) {
        // Compute k1
        k1 = dXdT(
                in[shft(tid,-1)],
                __halves2half2(__high2half(in[shft(tid,-1)]),__low2half(in[tid])),
                in[tid],
                __halves2half2(__high2half(in[tid]), __low2half(in[shft(tid,1)]))
        );
        work[tid] = __hfma2(half_dt, k1, in[tid]);
        __syncthreads();

        // Compute k2
        k2 = dXdT(
                work[shft(tid,-1)],
                __halves2half2(__high2half(work[shft(tid,-1)]),__low2half(work[tid])),
                work[tid],
                __halves2half2(__high2half(work[tid]), __low2half(work[shft(tid,1)]))
        );
        work[tid] = __hfma2(half_dt, k2, in[tid]);
        __syncthreads();

        // Compute k3
        k3 = dXdT(
                work[shft(tid,-1)],
                __halves2half2(__high2half(work[shft(tid,-1)]),__low2half(work[tid])),
                work[tid],
                __halves2half2(__high2half(work[tid]), __low2half(work[shft(tid,1)]))
        );
        work[tid] = __hfma2(dt, k3, in[tid]);
        __syncthreads();

        // Compute k4
        k4 = dXdT(
                work[shft(tid,-1)],
                __halves2half2(__high2half(work[shft(tid,-1)]),__low2half(work[tid])),
                work[tid],
                __halves2half2(__high2half(work[tid]), __low2half(work[shft(tid,1)]))
        );

        // Step forwards
        out[tid] = __hfma2(one_six_dt, __hadd2(k1, __hfma2(two, k2, __hfma2(two, k3, k4))), in[tid]);
    }
}
