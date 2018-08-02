#include <stdio.h>
#include <helper_cuda.h>
#include <stdlib.h>
#include <cuda_fp16.h>
#include "fp16_conversion.h"

#if TYPE == 0
#define CONVERTTODEVTYPE(X) X
#define CONVERTBACK(X) X
#elif TYPE == 1
#define CONVERTTODEVTYPE(X) (float)X
#define CONVERTBACK(X) (double)X
#elif TYPE == 2
#define CONVERTTODEVTYPE(X) approx_float_to_half(X)
#define CONVERTBACK(X) (double)half_to_float(X)
#endif

// Compile time constants
#define HALF (PREC)0.5
#define ONE (PREC)1.0
#define TWO (PREC)2.0
#define SIX (PREC)6.0
#define F (PREC)8.0
#define DT (PREC)0.05
#define N 512

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

int main(int argc, const char **argv) {
    // Simulation parameters
    int length = 5000;

    // Storage vectors
    double *h_hist;
    PREC *h_state, *d_prev, *d_next, *d_temp;

    // Kernel parameters
    int nThreadsPerBlock = N;
    int nBlocks = 1;

    // Initialise card
    findCudaDevice(argc, argv);

    // Allocate memory on host and device
    h_state = (PREC *)malloc(sizeof(PREC)*N);
    h_hist  = (double *)malloc(sizeof(double)*length);

    checkCudaErrors(cudaMalloc((void **)&d_prev, sizeof(PREC)*N));
    checkCudaErrors(cudaMalloc((void **)&d_next, sizeof(PREC)*N));

    // Set initial conditions
    for (int i = 0; i < N; i++) {
        h_state[i] = CONVERTTODEVTYPE((double)rand()/RAND_MAX);
    }

    // Copy initial conditions to device
    checkCudaErrors(cudaMemcpy(d_prev, h_state, sizeof(PREC)*N, cudaMemcpyHostToDevice));

    // Set initial condition in history array
    h_hist[0] = CONVERTBACK(h_state[0]);
    printf("%.12f\n", h_hist[0]);

    // Run forecast
    printf("Running forecast with %d blocks and %d threads per block\n", nBlocks, nThreadsPerBlock);
    for (int i = 1; i < length; i++) {
        // Step forward once
        step<<<nBlocks, nThreadsPerBlock>>>(d_prev, d_next);
        getLastCudaError("step execution failed\n");

        // Store one variable
        checkCudaErrors(cudaMemcpy(&h_state[0], &d_next[0], sizeof(PREC), cudaMemcpyDeviceToHost));
        h_hist[i] = CONVERTBACK(h_state[0]);

        printf("%.12f\n", h_hist[i]);

        // Swap prev and next pointers
        d_temp = d_prev; d_prev = d_next; d_next = d_temp;
    }

    // Free up memory
    free(h_hist);
    free(h_state);
    checkCudaErrors(cudaFree(d_prev));
    checkCudaErrors(cudaFree(d_next));

    printf("Finished successfully\n");

    cudaDeviceReset();
}
