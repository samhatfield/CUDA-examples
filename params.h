#ifndef PARAMS_H
#define PARAMS_H

// CUDA constants
__constant__ half2 half_dt, one_six_dt, two, f, dt;

// Compile time constants
#define F 8.0
#define DT 0.05
#define N 512
#define LEN 500

#endif
