#ifndef PARAMS_H
#define PARAMS_H

// Compile time constants
#if TYPE == 0
#define HALF 0.5
#define ONE 1.0
#define TWO 2.0
#define SIX 6.0
#define F 8.0
#define DT 0.05
#else
#define HALF 0.5f
#define ONE 1.0f
#define TWO 2.0f
#define SIX 6.0f
#define F 8.0f
#define DT 0.05f
#endif

#define N 512
#define LEN 5000

#endif
