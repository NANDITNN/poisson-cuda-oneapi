# poisson-cuda-oneapi
This repository contains GPU codes to solve an unsteady poisson equation. Second order central difference scheme is used for spatial discretization and an explicit Euler scheme for time integration. It contains the following files: 

poisson.cu -- CUDA code that runs only on NVIDIA GPUs.
poisson_unoptimized.dp.cpp -- unoptimized SYCL code directly obtained from SCYLomatic tool.
poisson_optimized.dp.cpp -- optimized SYCL code.
