#include <stdio.h>
#include <helper_cuda.h>
#include <stdlib.h>

// Dimension of model
__constant__ int N;

// Timestep
__constant__ PREC dt;

// Forcing
__constant__ PREC F;

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

    // Intermediate steps
    PREC k1, k2;

    if (tid < N) {
        // Compute k1
        k1 = dXdT(in[shft(tid,-2)], in[shft(tid,-1)], in[tid], in[shft(tid,1)]);

        // Add h*k1 to step
        in[tid] += dt*k1;

        __syncthreads();

        // Compute k2
        k2 = dXdT(in[shft(tid,-2)], in[shft(tid,-1)], in[tid], in[shft(tid,1)]);

        // Get local state
        out[tid] = in[tid] + 0.5*dt*(k2 - k1);
    }
}

int main(int argc, const char **argv) {
    // Simulation parameters
    int h_N = 40;
    PREC h_dt = 0.05;
    PREC h_F = 10.0;
    int length = 1000;

    // Storage vectors
    PREC *h_state, *h_hist, *d_prev, *d_next, *d_temp;

    // Kernel parameters
    int nThreadsPerBlock = 64;
    int nBlocks = 1 + ((h_N - 1)/nThreadsPerBlock);

    // Initialise card
    findCudaDevice(argc, argv);

    // Move global constants to device
    checkCudaErrors(cudaMemcpyToSymbol(N, &h_N, sizeof(h_N)));
    checkCudaErrors(cudaMemcpyToSymbol(dt, &h_dt, sizeof(h_dt)));
    checkCudaErrors(cudaMemcpyToSymbol(F, &h_F, sizeof(h_F)));

    // Allocate memory on host and device
    h_state = (PREC *)malloc(sizeof(PREC)*h_N);
    h_hist  = (PREC *)malloc(sizeof(PREC)*length);

    checkCudaErrors(cudaMalloc((void **)&d_prev, sizeof(PREC)*h_N));
    checkCudaErrors(cudaMalloc((void **)&d_next, sizeof(PREC)*h_N));

    // Set initial conditions
    for (int i = 0; i < h_N; i++) {
        h_state[i] = (PREC)rand()/RAND_MAX;
    }

    printf("%f %f\n", h_state[0], h_state[1]);

    // Copy initial conditions to device
    checkCudaErrors(cudaMemcpy(d_prev, h_state, sizeof(PREC)*h_N, cudaMemcpyHostToDevice));

    // Set initial condition in history array
    h_hist[0] = h_state[0];

    // Run forecast
    printf("Running forecast with %d blocks and %d threads per block\n", nBlocks, nThreadsPerBlock);
    for (int i = 1; i < length; i++) {
        // Step forward once
        step<<<nBlocks, nThreadsPerBlock>>>(d_prev, d_next);
        getLastCudaError("step execution failed\n");

        // Store one variable
        checkCudaErrors(cudaMemcpy(&h_hist[i], &d_next[0], sizeof(PREC), cudaMemcpyDeviceToHost));

        printf("%f\n", h_hist[i]);

        // Swap prev and next pointers
        d_temp = d_prev; d_prev = d_next; d_next = d_temp;
    }

    // Copy back results
    checkCudaErrors(cudaMemcpy(h_state, d_next, sizeof(PREC)*h_N,cudaMemcpyDeviceToHost));

    // Free up memory
    free(h_state);
    free(h_hist);
    checkCudaErrors(cudaFree(d_prev));
    checkCudaErrors(cudaFree(d_next));

    cudaDeviceReset();
}
