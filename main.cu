#include <stdio.h>
#include <helper_cuda.h>
#include <stdlib.h>
#include <cuda_fp16.h>
#include "fp16_conversion.h"
#include "device.h"
#include "params.h"

int main(int argc, const char **argv) {
    // Model parameters
    half2 h_half_dt, h_one_six_dt, h_two, h_f, h_dt;

    // Initialise model parameters
    h_half_dt    = __float2half2_rn(0.5*(float)DT);
    h_one_six_dt = __float2half2_rn((float)DT*1.0f/6.0f);
    h_two        = __float2half2_rn(2.0f);
    h_f          = __float2half2_rn(F);
    h_dt         = __float2half2_rn(DT);

    // Storage vectors
    float *h_hist;
    half2 *h_state, *d_prev, *d_next, *d_temp;

    // Kernel parameters
    int nThreadsPerBlock = N/2;
    int nBlocks = 1;

    // Initialise CUDA timing
    float milli;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Initialise card
    findCudaDevice(argc, argv);

    // Send constants to device
    checkCudaErrors(cudaMemcpyToSymbol(half_dt,    &h_half_dt,    sizeof(h_half_dt)));
    checkCudaErrors(cudaMemcpyToSymbol(one_six_dt, &h_one_six_dt, sizeof(h_one_six_dt)));
    checkCudaErrors(cudaMemcpyToSymbol(two,       &h_two,         sizeof(h_two)));
    checkCudaErrors(cudaMemcpyToSymbol(f,         &h_f,           sizeof(h_f)));
    checkCudaErrors(cudaMemcpyToSymbol(dt,        &h_dt,          sizeof(h_dt)));

    // Allocate memory on host and device
    h_state = (half2 *)malloc(sizeof(half2)*N/2);
    h_hist  = (float *)malloc(sizeof(float)*LEN);

    checkCudaErrors(cudaMalloc((void **)&d_prev, sizeof(half2)*N/2));
    checkCudaErrors(cudaMalloc((void **)&d_next, sizeof(half2)*N/2));

    // Set initial conditions
    for (int i = 0; i < N/2; i++) {
        h_state[i] = __floats2half2_rn((float)rand()/RAND_MAX, (float)rand()/RAND_MAX);
    }

    // Copy initial conditions to device
    checkCudaErrors(cudaMemcpy(d_prev, h_state, sizeof(half2)*N/2, cudaMemcpyHostToDevice));

    // Set initial condition in history array
    h_hist[0] = (float)__low2float(h_state[0]);

    // Run forecast
    printf("Running forecast with %d blocks and %d threads per block\n", nBlocks, nThreadsPerBlock);
    printf("%.12f\n", h_hist[0]);
    cudaEventRecord(start);
    for (int i = 1; i < LEN; i++) {
        // Step forward once
        step<<<nBlocks, nThreadsPerBlock>>>(d_prev, d_next);
        getLastCudaError("step execution failed\n");

        // Store one variable
        checkCudaErrors(cudaMemcpy(&h_state[0], &d_next[0], sizeof(half2), cudaMemcpyDeviceToHost));
        h_hist[i] = (float)__low2float(h_state[0]);

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
