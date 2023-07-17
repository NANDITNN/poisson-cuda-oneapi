// ---------------------------------------------------------------
// This program solves unsteady Poisson equation using CUDA
// Second order centred schemes for spatial discretisation
// First order explicit Euler scheme for time marching
//
// The source term is chosen such that the exact solution when
// steady state reached is
// u(x,y) = sin(PI*x)sin(PI*y)
//
// Somendra kumar, Y. Sudhakar
// Indian Institute of Technology Goa
//
// Tested with nvhpc-22.11 on 02 March 2023
// ---------------------------------------------------------------
#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include <algorithm>
#include <time.h>
#include <iostream>
#include <fstream>
#include <stdio.h>
#include <bits/stdc++.h>
#include <chrono>
#include <cmath>
 // for computation of cpu time

#define BLOCK_SIZE 32

// ---
// Helper function to convert 2D indices to 1D array index
// ---
int Indexkernel(const int i, const int j, const int width)
{
	return i*width + j;
}

// ---
// Compute the source term of the Poisson equation
// ---
void sourcekernel(double *F,const int nx,int ny,const double dx,const double dy,
                  sycl::nd_item<3> item_ct1) 
{
        int i = item_ct1.get_local_id(2) +
                item_ct1.get_group(2) * item_ct1.get_local_range(2);
        if (i > 0 && i < nx )
	{
                int j = item_ct1.get_local_id(1) +
                        item_ct1.get_group(1) * item_ct1.get_local_range(1);
                if (j > 0 && j < ny )
		{
			const int ind= Indexkernel(i, j, ny+1);
                        F[ind] = 2.0 * M_PI * M_PI *
                                 (sin(M_PI * i * dx) *
                                  sin(M_PI * j * dy));
                }
	}
}

// ---
// Solution update from one time level to the next
// ---
void  evolve_kernel( double* U, double* U_prev,double* F, const int nx,const int ny,
							double dx2inv,double dy2inv, const double alpha,const double Dt, double *residual,
							sycl::nd_item<3> item_ct1)
{
        int i = item_ct1.get_local_id(2) +
                item_ct1.get_group(2) * item_ct1.get_local_range(2);
        int j = item_ct1.get_local_id(1) +
                item_ct1.get_group(1) * item_ct1.get_local_range(1);

        if (j > 0 && j < ny )
	{
		if (i > 0 && i < nx )
		{
			const int index = Indexkernel(i, j, ny+1);
			int n = Indexkernel(i-1, j, ny+1);
			int s = Indexkernel(i+1, j, ny+1);
			int e = Indexkernel(i, j+1, ny+1);
			int w = Indexkernel(i, j-1, ny+1);
            
			residual[index] = alpha*( (U_prev[n] - 2.0*U_prev[index] + U_prev[s])*dx2inv 
							+ (U_prev[w] - 2.0*U_prev[index] + U_prev[e])*dy2inv ) + F[index];
			U[index] = U_prev[index] + Dt*residual[index];
		}
	}
}

 

int main()
{
  dpct::device_ext &dev_ct1 = dpct::get_current_device();
  sycl::property_list props{sycl::ext::oneapi::property::queue::discard_events{},
	                            sycl::property::queue::in_order{}};
  sycl::queue q_ct1( props );  
        // ----- set these input parameters -----------
	// no of elements in each direction
	// no of nodes will be (nx+1) and (ny+1)
	const int nx = 512, ny = 512; 
	// diffusion coefficient
	const double alpha = 1.0;
	// convergence criterion
	const double epsilon = 1e-6;
	// --------------------------------------------
	
	// mesh spacing
	const double dx = 1.0/nx, dy = 1.0/ny;
	// the following is used in the discretisation
	const double dx2 = dx*dx, dy2 = dy*dy;
	const double dx2inv = 1.0/dx2, dy2inv = 1.0/dy2;

	// time step based on stability criterion
	const double dt = 0.5*dx*dx*dy*dy/alpha/(dx*dx+dy*dy);

	// allocate host memory for unknown U and exact solution Uexact
    // and compute the exact solution
	int np = (nx+1)*(ny+1);
	double* U        = (double*)calloc(np, sizeof(double));
	double* Uexact   = (double*)calloc(np, sizeof(double));
	for( int ii=0; ii<=nx; ii++ )
	{
		for( int jj=0; jj<=ny; jj++ )
		{
			int index = Indexkernel(ii, jj, ny+1);
			U[index] = 0.0;
			Uexact[index] = sin(M_PI*ii*dx)*sin(M_PI*jj*dy);
		}
	}

	// allocate device memory
	// since we perform all calculations on the device, we need
	// to allocate memory for source as well as previous time step value
	double* d_U, *d_Up, *d_source;
        d_U = sycl::malloc_device<double>((np), q_ct1);
        d_Up = sycl::malloc_device<double>((np), q_ct1);
        d_source = sycl::malloc_device<double>((np), q_ct1);

        sycl::range<3> numBlocks(1, (ny + 1) / BLOCK_SIZE + 1,
                                 (nx + 1) / BLOCK_SIZE + 1);
        sycl::range<3> threadsPerBlock(1, BLOCK_SIZE, BLOCK_SIZE);

        // [device] compute source term
        /*
	DPCT1049:0: The work-group size passed to the SYCL kernel may
         * exceed the limit. To get the device limit, query
         * info::device::max_work_group_size. Adjust the work-group size if
         * needed.
	*/
        q_ct1.parallel_for(
            sycl::nd_range<3>(numBlocks * threadsPerBlock, threadsPerBlock),
            [=](sycl::nd_item<3> item_ct1) {
                    sourcekernel(d_source, nx, ny, dx, dy, item_ct1);
            });

        // Copy the values of U from host to device
	// This is to copy the boundary/initial conditions set on the host
        q_ct1.memcpy(d_U, U, (np) * sizeof(double));
        q_ct1.memcpy(d_Up, U, (np) * sizeof(double)); 
        q_ct1.wait();
        // --- Max error norm
	// This is computed in the device but this is needed in the host
	// to check for the convergence
	double*residual;
        residual = sycl::malloc_shared<double>(np, q_ct1);

    int iter = 0;
	double maxNorm = 1e6;
	std::chrono::duration<double> cpu_time_total;
	std::chrono::time_point<std::chrono::system_clock> cpustart;
	cpustart = std::chrono::system_clock::now();
	while( 1 )
	{
                /*
		DPCT1049:1: The work-group size passed to the
                 * SYCL kernel may exceed the limit. To get the device limit,
                 * query info::device::max_work_group_size. Adjust the
                 * work-group size if needed.
		*/
        q_ct1.parallel_for(
            sycl::nd_range<3>(numBlocks * threadsPerBlock,
                              threadsPerBlock),
            [=](sycl::nd_item<3> item_ct1) {
                    evolve_kernel(d_U, d_Up, d_source, nx, ny, dx2inv,
                                  dy2inv, alpha, dt, residual,
                                  item_ct1);
            });
        q_ct1.wait();

        if( iter%100 == 0)
		{
			maxNorm = *std::max_element( residual, residual + np, [](double a, double b) {
                                                                return std::abs(a) < std::abs(b);});
		}
		//if(iter%1000 == 0) std::cout <<iter << "  " << maxNorm <<"\n";

		std::swap(d_U, d_Up);
		iter++;

		if( maxNorm < epsilon )
			break;
	}
	// copy the solution from devide to host
    q_ct1.memcpy(U, d_U, (np) * sizeof(double));
    q_ct1.wait();
        // L-infty error between exact and computed
	double errnorm=0.0;
	for (int i=0; i<np; i++)  
	{
                errnorm = std::max(errnorm,abs(U[i]-Uexact[i]));
        }
	std::cout <<"norm = "<<errnorm <<"\n";

	// total computational time
	cpu_time_total = std::chrono::system_clock::now() - cpustart;
	std::cout<<" total simulation time = "<< cpu_time_total.count() <<" seconds \n"; //delete sudh

	// Write tecplot output
	std::ofstream outfile;
	outfile.open ("poisson.dat");
	outfile << "VARIABLES=X,Y,T\n";
	outfile << "ZONE T=\"0\" i=" <<nx+1<<",j="<<ny+1<<",ZONETYPE=ORDERED,DATAPACKING=POINT\n";
	for( int ii=0; ii<=nx; ii++ )
	{
		for( int jj=0; jj<=ny; jj++ )
		{
			int index = Indexkernel(ii, jj, ny+1);
			outfile << ii*dx <<"\t" << jj*dy <<"\t" << U[index] <<"\n";
		}
	}
	outfile.close();

	// Release the memory
	free(U);
	free(Uexact);
        sycl::free(d_U, q_ct1);
        sycl::free(d_Up, q_ct1);
        sycl::free(d_source, q_ct1);

        return 0;
}
