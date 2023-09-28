#ifndef CPLX_CUDA_GUARD
#define CPLX_CUDA_GUARD
#include "field_types.h"
#include <math.h>
////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////   COMPLEX NUMBERS FUNCTIONS ////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////


__device__ __forceinline__
field modCx(field2 &z)
{
    return sqrt((z.x*z.x + z.y*z.y));
}


 __device__ __forceinline__
field2 conj(const field2 &z, const field scal = 1.0)
{
    return { scal * z.x, - scal * z.y };
}

 __device__ __forceinline__
field2 mult(const field2 &z, const field scal = 1.0)
{
    return { scal * z.x, scal * z.y };
}

 __device__ __forceinline__
void multSum(field2 &z_out, const field2 &z1, const field2 &z2, 
             const field scal1 = 1.0,const field scal2 = 1.0)
{
    z_out.x = z1.x*scal1 + z2.x*scal2;
    z_out.y = z1.y*scal1 + z2.y*scal2;
}

/////////////////////////////////////////////////////////////////////////////////////
// This function multiplies a complex numbers with a scalar and sums to the output //
//                                                                                 //
//       multScS(*z_out, *z1_in, real) --> z_out = (z1_in)*(real)                  //
//                                                                                 //

 __device__ __forceinline__
void multScSum(field2 &z_out, const field2 &z1, const field scalar = 1.0) 
{
    z_out.x += z1.x * scalar;
    z_out.y += z1.y * scalar;
}


/////////////////////////////////////////////////////////////////////////////////////////////////////
// This function multiplies two complex numbers with eachother and a scalar and sums to the output //
//                                                                                                 //
//       multCxScS(*z_out, *z1_in, *z2_in, scalar) --> z_out = (z1_in * z2_in)*(field)(flag)   //
//                                                                                                 //

__device__ __forceinline__
void multCxScSum(field2 &z_out, const field2 &z1, const field2 &z2, const field scalar = 1.0)
{
    z_out.x += ( z1.x * z2.x - z1.y * z2.y ) * scalar;
    z_out.y += ( z1.x * z2.y + z1.y * z2.x ) * scalar;
}

#endif