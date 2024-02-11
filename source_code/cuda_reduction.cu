#include <math.h>
#include "spdlog/spdlog.h"
#include "spdlog/sinks/basic_file_sink.h"
#include "cuda_reduction.h"
#define warpSize 32





////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////   END OF NAMESPACE CX HOST   ///////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////

//================================================================================================//

////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////    NAMESPACE REDUCE    ////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////


//////////// AUXILIARY FUNCTIONS FOR REDUCTION. ONLY LOCAL /////////////

__device__ __forceinline__
field modCx(field2 &z)
{
    return sqrt((z.x*z.x + z.y*z.y));
}
////////////////////Reduction algorithm for field2//////////////////////#include 

//Uses shfl_down to sum a warp into a scalar. Input parameter is the warp first element 
 __device__
field2 warpReduceSum(field2 val) {
 
  for (unsigned int offset = warpSize/2; offset > 0; offset /= 2){ 
    val.x += __shfl_down_sync(0xFFFFFFFF,val.x, offset,warpSize);
    val.y += __shfl_down_sync(0xFFFFFFFF,val.y, offset,warpSize);
  }
  

  return val;
}

//////////////////////////Now we sum over the warps//////////////////////
// Using the warpReduceSum function we can now easily build a reduction across the entire block.
// To do this we first reduce within warps. 
// Then the first thread of each warp writes its partial sum to shared memory. 
// Finally, after synchronizing, the first warp reads from shared memory and reduces again. 
 __device__
field2 blockReduceSum(field2 val) {

  static __shared__ field2 shared[32]; // Shared mem for 32 partial sums. Each partial sum sums 32 elements, hence 32 is also the maximum number of warps in a block with 1024 threads.
  int lane = threadIdx.x % warpSize;
  int wid = threadIdx.x / warpSize;

  val = warpReduceSum(val);     // Each warp performs partial reduction

  if (lane==0) shared[wid]=val; // Write reduced value to shared memory

  __syncthreads();              // Wait for all partial reductions

  //read from shared memory only if that warp existed
  field2 zero = {0.0,0.0};

  val = (threadIdx.x < blockDim.x / warpSize) ? shared[lane] : zero;

  if (wid==0) val = warpReduceSum(val); //Final reduce within first warp

  return val;
}


__global__ void deviceReduceKernel(const field2 *in, field2* out, const int N) {

  field2 sum = {0.0,0.0};
  //reduce multiple elements per thread
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; 
       i < N; 
       i += blockDim.x * gridDim.x) {
    sum.x += in[i].x;
    sum.y += in[i].y;
  }
  sum = blockReduceSum(sum);
  if (threadIdx.x==0){
    out[blockIdx.x]=sum;
  }
}


/////////////////Modified summation kernel to estimate convergence///////////////
//Output is save in D hence this function must be used BEFORE the swap
__global__ void convergenceKernel(field2 *D, const field2* D_new, int N) {

  field2 sum = {0.0,0.0};
  //reduce multiple elements per thread
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; 
       i < N; 
       i += blockDim.x * gridDim.x) {
    
    sum.x += fabs(D_new[i].x - D[i].x)/(modCx(D[i]) + 1.0e-9 );
    sum.y += fabs(D_new[i].y - D[i].y)/(modCx(D[i]) + 1.0e-9 );
  }
  sum = blockReduceSum(sum);
  if (threadIdx.x==0){
    D[blockIdx.x].x = sum.x/N;
    D[blockIdx.x].y = sum.y/N;
  }
}


////////////////////Reduction algorithm for field//////////////////////
__device__
field warpReduceSumReal(field val) {
 
  for (unsigned int offset = warpSize/2; offset > 0; offset /= 2){ 
    val += __shfl_down_sync(0xFFFFFFFF,val, offset,warpSize);
    
  }
  

  return val;
}
 
 __device__
field blockReduceSumReal(field val) {

  static __shared__ field shared[32]; // Shared mem for 32 partial sums. Each partial sum sums 32 elements, hence 32 is also the maximum number of warps in a block with 1024 threads.
  int lane = threadIdx.x % warpSize;
  int wid = threadIdx.x / warpSize;

  val = warpReduceSumReal(val);     // Each warp performs partial reduction

  if (lane==0) shared[wid]=val; // Write reduced value to shared memory

  __syncthreads();              // Wait for all partial reductions

  //read from shared memory only if that warp existed
  val = (threadIdx.x < blockDim.x / warpSize) ? shared[lane] : 0.0;

  if (wid==0) val = warpReduceSumReal(val); //Final reduce within first warp

  return val;
}


__global__ void deviceReduceKernelReal(field *in, field* out, int N) {

  field sum = 0.0;
  //reduce multiple elements per thread
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; 
       i < N; 
       i += blockDim.x * gridDim.x) {
    sum += in[i];
  }
  sum = blockReduceSumReal(sum);
  if (threadIdx.x==0){
    out[blockIdx.x]=sum;
  }
}


__global__ void convergenceKernelReal(const field *n_new, field* n, int N) {

  field sum = 0.0;
  //reduce multiple elements per thread
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; 
       i < N; 
       i += blockDim.x * gridDim.x) {
    
    sum += fabs(n_new[i] - n[i])/( n[i] + 1.0e-9 );

  }
  sum = blockReduceSumReal(sum);
  if (threadIdx.x==0){
    n[blockIdx.x] = sum/N;

  }
}


//Kernel calculating optimal step for magnetic field
__global__ void stepKernel(const field2 *d_A, const field2 *d_A_prev, field2 *d_A_grad_prev, const field3 *d_J, const int *neighbor_list, 
                           const int N, const int Nx, const field Bext) 
{

  field2 dFA = {0.0,0.0};
  field2 sum = {0.0,0.0};
  uint3 isHoppingUp, isHoppingDown;
  uint3 d_up, d_down;
  uint2 d_up_lr, d_down_lr; //x-> positive x; y-> negative x


  //reduce multiple elements per thread
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; 
       i < N; 
       i += blockDim.x * gridDim.x) {
    
      //Calculation of the gradient
      {
        int current_neighbors = neighbor_list[i];
        
        //For neighbor convention see function isNeighbor. 
        //Here it is copied because it is faster
        
        isHoppingDown.x = (current_neighbors >> 0) & 1;
        isHoppingDown.y = (current_neighbors >> 2) & 1;
        isHoppingDown.z = (current_neighbors >> 4) & 1;
    
        isHoppingUp.x = (current_neighbors >> 1) & 1;
        isHoppingUp.y = (current_neighbors >> 3) & 1;
        isHoppingUp.z = (current_neighbors >> 5) & 1;
    
      }
    
    
      d_up.x              = ( i + 1 ) * isHoppingUp.x;   
      d_down.x            = ( i - 1 ) * isHoppingDown.x;     
      d_up.y              = ( i + Nx ) * isHoppingUp.y;
      d_down.y            = ( i - Nx ) * isHoppingDown.y;
    
      d_up_lr.x           = (d_up.y   + 1)*isHoppingUp.y*isHoppingUp.x;
      d_up_lr.y           = (d_up.y   - 1)*isHoppingUp.y*isHoppingDown.x;
      d_down_lr.x         = (d_down.y + 1)*isHoppingDown.y*isHoppingUp.x;
      d_down_lr.y         = (d_down.y - 1)*isHoppingDown.y*isHoppingDown.x;
    
    
    
    
      dFA.x = - d_J[i].x + ( - ( d_A[d_up.y].x - d_A[i].x  ) +  ( d_A[d_up.x].y - d_A[i].y ) - Bext ) *isHoppingUp.y*isHoppingUp.x 
              + ( (d_A[i].x - d_A[d_down.y].x) - (d_A[d_down_lr.x].y -d_A[d_down.y].y) + Bext ) *isHoppingDown.y*isHoppingUp.x;


      dFA.y = - d_J[i].y + ( ( d_A[d_up.y].x - d_A[i].x ) - ( d_A[d_up.x].y - d_A[i].y ) + Bext )*isHoppingUp.y*isHoppingUp.x
              + ( ( d_A[i].y - d_A[d_down.x].y )  - (d_A[d_up_lr.y].x - d_A[d_down.x].x  ) - Bext )*isHoppingDown.x*isHoppingUp.y;
                          
                            
      sum.x += ( d_A[i].x - d_A_prev[i].x )*( dFA.x - d_A_grad_prev[i].x ) + ( d_A[i].y - d_A_prev[i].y )*( dFA.y - d_A_grad_prev[i].y );
      sum.y += (dFA.x*dFA.x + dFA.y*dFA.y );


  }
  sum = blockReduceSum(sum);
  if (threadIdx.x==0){
    d_A_grad_prev[blockIdx.x].x = sum.x;
    d_A_grad_prev[blockIdx.x].y = sum.y;
  }
}



/////////////////Modified summation kernel to estimate convergence///////////////
//Output is save in D hence this function must be used BEFORE the swap
//////////////////////////////////ACCESSIBLE SUMMATION ALGORITHMS//////////////////////
//Function reducing an array into a scalar
void rdx::sumArray(field2 *in, field2* out, int N){
    
    const int threads = 512;
    const int blocks = min((N + threads - 1) / threads, 1024);
  
    deviceReduceKernel<<<blocks, threads>>>(in, out, N);
    deviceReduceKernel<<<1, 1024>>>(out, out, blocks);
}






//Convergence check. USE BEFORE SWAPPING DELTA
float rdx::checkConvergence(const field2 *D, const field2* D_new, int N, std::string quantity, bool print) {

    field2 *d_convergenceArray;
    cudaMalloc(&d_convergenceArray, N*sizeof(field2));
    cudaMemcpy(d_convergenceArray, D, N*sizeof(field2), cudaMemcpyDeviceToDevice);

    const int threads = 256;
    const int blocks = min((N + threads - 1) / threads, 1024);
  
    //Convergence kernel saves in D
    convergenceKernel<<<blocks, threads>>>(d_convergenceArray, D_new, N);
    deviceReduceKernel<<<1, 1024>>>(d_convergenceArray, d_convergenceArray, blocks);
   
    field2 convergence;
    cudaMemcpy(&convergence, d_convergenceArray, sizeof(field2), cudaMemcpyDeviceToHost);
    cudaFree(d_convergenceArray);
    float convergernceModulus = (float)sqrt(convergence.x*convergence.x + convergence.y*convergence.y);
   
    if(print){
    spdlog::debug("{:=^44}","    " + quantity + "    ");
    spdlog::debug("Convergence ==> {:.4E}",convergernceModulus);
    spdlog::debug("{:=^44}","");
    }
    return (convergernceModulus);
}

//Convergence check for populations. USE BEFORE SWAPPING N
float rdx::checkConvergence(field *n, const field* n_new, int N, std::string quantity, bool print) {
   
    const int threads = 256;
    const int blocks = min((N + threads - 1) / threads, 1024);
  
    convergenceKernelReal<<<blocks, threads>>>(n_new, n, N);
    deviceReduceKernelReal<<<1, 1024>>>(n, n, blocks);
   
    field convergence;
    cudaMemcpy(&convergence, n, sizeof(field), cudaMemcpyDeviceToHost);
    
   
    if(print){
    spdlog::debug("{:=^44}","    " + quantity + "    ");
    spdlog::debug("Convergence ==> {:.4E}",convergence);
    spdlog::debug("{:=^44}","");
    }
    return (convergence);
}

field rdx::caclulateStep(const field2 *d_A, const field2 *d_A_prev, field2 * d_A_grad_prev,
                         const field3 *d_J, const int *neighbor_list,
                         const int N, const int Nx, const field Bext){

    const int threads = 256;
    const int blocks = min((N + threads - 1) / threads, 1024);

    //Convergence kernel saves in D
    stepKernel<<<blocks, threads>>>(d_A, d_A_prev, d_A_grad_prev, 
                                    d_J, neighbor_list, N, Nx, Bext);
    deviceReduceKernel<<<1, 1024>>>(d_A_grad_prev, d_A_grad_prev, blocks);

    field2 step;
    cudaMemcpy(&step,d_A_grad_prev, sizeof(field2), cudaMemcpyDeviceToHost);
    
    return (field) abs(step.x/(step.y + 1.0e-8 ));

}