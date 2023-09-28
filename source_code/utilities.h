
#ifndef BDG_UTILITIES_H
#define BDG_UTILITIES_H
#include <string>
#include "field_types.h"
#include <cuda_runtime.h>
#include <stdio.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <complex>

namespace util{

	//Summarizes input parameters
	extern void helper(int argc, char* argv[]);


	// Calculates how many Y blocks one can utilize given the GPU free memory and the system size.
	extern int gpuYblocks(const size_t SIZE_2N, const size_t SIZE_N, const size_t SIZE_N_REAL, const size_t SIZE_3N, int dev, const int N, const bool GAUGE_FIELD);

	//Import simulation parameters from command line
	extern void inputParser(Hamiltonian &hamiltonian,
							std::string out_file, int argc,char* argv[]);

	//Coefficient rescaling to have eigenvalues within [-1,1]
	extern void rescaleCoefficient(Hamiltonian &hamiltonian, const std::string out_file);

	//Rescaling the results back
	extern void rescaleInverse(field2 *D, const Hamiltonian hamiltonian);
	extern void rescaleInverse(field3 *D, const Hamiltonian hamiltonian);


}

namespace array{
	// initialize complex valued array on gpu
	extern __global__ void init_gpu(field2 *A, int N, field2 value = {1.0,0.0});

	// initialize real valued array on gpu
	extern __global__ void init_gpu(field *A, int N, field value = 1.0);

	//Swap arrays on GPU (by swapping their references)
	extern void swap(field2 ** a_ref, field2 ** b_ref);
	extern void swap(field ** a_ref, field ** b_ref);

}

namespace geometry{

	extern void setGeometry(int *h_neighbors, const Hamiltonian &hamiltonian, std::string out_file);

}

#endif
