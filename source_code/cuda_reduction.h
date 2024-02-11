#ifndef REDUCTION_ALGORITHMS_H
#define REDUCTION_ALGORITHMS_H
#include "field_types.h"



//////////////////////////////////REDUCTION ALGORITHMS//////////////////////////////////////////////////////////
namespace rdx{
//Function reducing an array into a scalar
extern void sumArray(field2 *in, field2* out, int N);

//Convergence check. USE BEFORE SWAPPING DELTA
extern float checkConvergence(const field2 *D, const field2* D_new, int N, std::string quantity = "Delta", bool print = true );

//Convergence check for populations. USE BEFORE SWAPPING N
extern float checkConvergence(field *n, const field* n_new, int N, std::string quantity = "N", bool print = true );


extern field caclulateStep(const field2 *d_A, const field2 *d_A_prev, field2 *d_A_grad_prev,
                           const field3 *d_J, const int *neighbor_list, const int N, const int Nx, const field Bext);
}

#endif