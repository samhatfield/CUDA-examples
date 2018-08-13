#include <stdio.h>
#include <helper_cuda.h>
#include <stdlib.h>
#include <cuda_fp16.h>
#include "fp16_conversion.h"
#include "device.h"
#include "params.h"

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

int main(int argc, const char **argv) {
    // Storage vectors
    double *h_hist;
    PREC *h_state, *d_prev, *d_next, *d_temp;

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
    h_state = (PREC *)malloc(sizeof(PREC)*N);
    h_hist  = (double *)malloc(sizeof(double)*LEN);

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

    // Run forecast
    printf("Running forecast with %d blocks and %d threads per block\n", nBlocks, nThreadsPerBlock);
    printf("%.12f\n", h_hist[0]);
    cudaEventRecord(start);
    for (int i = 1; i < LEN; i++) {
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
