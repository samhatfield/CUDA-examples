#include "params.h"

// ODE device function
__device__ PREC dXdT(PREC x__2, PREC x__1, PREC x, PREC x_1) {
    return (x_1 - x__2)*x__1 - x + F;
}

// Index shifter device function
__device__ int shft(int n, int m) {
    return (n + m + N)%N;
}

// Main step kernel function
__global__ void step(PREC* __restrict__ in, PREC* __restrict__ out) {
    // Get global thread ID
    int tid = threadIdx.x + blockDim.x*blockIdx.x;

    // Shared work array
    __shared__ PREC work[N];

    // Intermediate steps
    PREC k1, k2, k3, k4;

    if (tid < N) {
        // Compute k1
        k1 = dXdT(in[shft(tid,-2)], in[shft(tid,-1)], in[tid], in[shft(tid,1)]);
        work[tid] = in[tid] + HALF*DT*k1;
        __syncthreads();

        // Compute k2
        k2 = dXdT(work[shft(tid,-2)], work[shft(tid,-1)], work[tid], work[shft(tid,1)]);
        work[tid] = in[tid] + HALF*DT*k2;
        __syncthreads();

        // Compute k3
        k3 = dXdT(work[shft(tid,-2)], work[shft(tid,-1)], work[tid], work[shft(tid,1)]);
        work[tid] = in[tid] + DT*k3;
        __syncthreads();

        // Compute k4
        k4 = dXdT(work[shft(tid,-2)], work[shft(tid,-1)], work[tid], work[shft(tid,1)]);

        // Step forwards
        out[tid] = in[tid] + (ONE/SIX)*DT*(k1 + TWO*k2 + TWO*k3 + k4);
    }
}
