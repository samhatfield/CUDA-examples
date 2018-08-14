# lorenz96-cuda

A version of [Lorenz '96](https://en.wikipedia.org/wiki/Lorenz_96_model) written in CUDA. The code is parallelised over gridpoints - each thread gets one gridpoint. Because shared memory is used in the kernel, only one block can be run. This means the maximum dimension of the model is 1024 (or 2048 for half2).

The model can be run with different precisions for comparison: double, single, half and half2 (vectorised half-precision). For double and single, use the master branch and compile with `make double` or `make single`. For half or half2, checkout the respective branch and simply `make`.

The model is integrated with a standard 4th order Runge-Kutta scheme.
