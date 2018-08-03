#include <stdio.h>
#include <helper_cuda.h>
#include <stdlib.h>
#include <cuda_fp16.h>
#include "fp16_conversion.h"
#include "device.h"
#include "params.h"

int main(int argc, const char **argv) {
    // Storage vectors
    double *h_hist;
    half *h_state, *d_prev, *d_next, *d_temp;

    // Kernel parameters
    int nThreadsPerBlock = N;
    int nBlocks = 1;

    // Initialise CUDA timing
    float milli;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Initialise card
    findCudaDevice(argc, argv);

    // Allocate memory on host and device
    h_state = (half *)malloc(sizeof(half)*N);
    h_hist  = (double *)malloc(sizeof(double)*LEN);

    checkCudaErrors(cudaMalloc((void **)&d_prev, sizeof(half)*N));
    checkCudaErrors(cudaMalloc((void **)&d_next, sizeof(half)*N));

    // Set initial conditions
    for (int i = 0; i < N; i++) {
        h_state[i] = approx_float_to_half((float)rand()/RAND_MAX);
    }

    // Copy initial conditions to device
    checkCudaErrors(cudaMemcpy(d_prev, h_state, sizeof(half)*N, cudaMemcpyHostToDevice));

    // Set initial condition in history array
    h_hist[0] = (double)half_to_float(h_state[0]);

    // Run forecast
    printf("Running forecast with %d blocks and %d threads per block\n", nBlocks, nThreadsPerBlock);
    printf("%.12f\n", h_hist[0]);
    cudaEventRecord(start);
    for (int i = 1; i < LEN; i++) {
        // Step forward once
        step<<<nBlocks, nThreadsPerBlock>>>(d_prev, d_next);
        getLastCudaError("step execution failed\n");

        // Store one variable
        checkCudaErrors(cudaMemcpy(&h_state[0], &d_next[0], sizeof(half), cudaMemcpyDeviceToHost));
        h_hist[i] = (double)half_to_float(h_state[0]);

        printf("%.12f\n", h_hist[i]);

        // Swap prev and next pointers
        d_temp = d_prev; d_prev = d_next; d_next = d_temp;
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milli, start, stop);

    printf("Forecast took %f ms to execute\n", milli);

    // Free up memory
    free(h_hist);
    free(h_state);
    checkCudaErrors(cudaFree(d_prev));
    checkCudaErrors(cudaFree(d_next));

    printf("Finished successfully\n");

    cudaDeviceReset();
}
