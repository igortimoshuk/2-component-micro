#include "chebyshev_self_consistent.h"
#include <stdio.h>
#include <iostream>
#include "complex_cuda.h"
#include "cuda_reduction.h"
#include "spdlog/spdlog.h"
#include "utilities.h"
#include "io.h"
#define PI 3.14159265359
////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////      GPU FUNCTIONS      ///////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////
__global__
void multiplyH(
    const field2 *t_x,const field2 *t_y,const field2 *t_z, const field *modulation,
    const field mu, const field H, const field V,
    const field2 *h, field2 *h_prev, const field2 *e, field2 *e_prev,
    const field2 *D, field2 *D_new, field *F,
    const field *n_up, const field *n_down, field *n_up_new, field *n_down_new, 
    const int *neighbor_list, const field T,
    const int N, const int Nx, const int Ny, const int Nz, const int site_offset,
    const field cheb_n, const field cheb_n_F, const field recursion_coeff, const bool HARTREE)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t blocky = blockIdx.y;
    size_t id = idx + blocky * 2 * N;

    const int site = blocky + site_offset;

    if(idx >= N) return;
    if(site >= N) return;

    uint3 isHoppingUp, isHoppingDown, isPeriodicUp, isPeriodicDown;
    uint3 d_up, d_down, hop_idx_down;

     //     NEIGHBOR CONVENTION  //
    //setting n = which_neighbor//
    //      n = 0 => x - 1      //
    //      n = 1 => x + 1      //
    //      n = 2 => y - 1      //
    //      n = 3 => y + 1      //
    //      n = 4 => z - 1      //
    //      n = 5 => z + 1      //
    //     n bigger:periodic boundaries //

    //Setting boundaries
     //Setting boundaries
     {
        int current_neighbors = neighbor_list[idx];

        //For neighbor convention see function isNeighbor.
        //Here it is copied because it is faster

        isHoppingDown.x = (current_neighbors >> 0) & 1;
        isHoppingDown.y = (current_neighbors >> 2) & 1;
        isHoppingDown.z = (current_neighbors >> 4) & 1;

        isHoppingUp.x = (current_neighbors >> 1) & 1;
        isHoppingUp.y = (current_neighbors >> 3) & 1;
        isHoppingUp.z = (current_neighbors >> 5) & 1;

        isPeriodicUp.x  = (current_neighbors >> 6)  & 1;
        isPeriodicUp.y  = (current_neighbors >> 8)  & 1;
        isPeriodicUp.z  = (current_neighbors >> 10) & 1;

        isPeriodicDown.x  = (current_neighbors >> 7)  & 1;
        isPeriodicDown.y  = (current_neighbors >> 9)  & 1;
        isPeriodicDown.z  = (current_neighbors >> 11) & 1;

    }


    // isPeriodic up is 1 only for the sites on the upper boundary of the respective dimension. I.e. in x when idx = Nx then isPeriodicUp = 1 if we have p. bc.
    // similarly when idx = 0 and we have pbc we have isPeriodicDown = 1
    //Setting displacement vectors for nearest neighbor hopping
    d_up.x              = ( id + 1 ) * isHoppingUp.x   + isPeriodicUp.x  * ( id - (Nx - 1)  - ( id + 1 ) * isHoppingUp.x );    //when idx = Nx-1, you cannot hop up, but you can hop to idx = 0 if periodic
    d_down.x            = ( id - 1 ) * isHoppingDown.x + isPeriodicDown.x *( id + (Nx - 1)  - ( id - 1 ) * isHoppingDown.x );   //when idx = 0, you cannot hop down, but you can hop to idx = Nx -1 if periodic

    // Similar structure for y and z

    d_up.y              = ( id + Nx ) * isHoppingUp.y   + isPeriodicUp.y  * (id - Nx * (Ny - 1)  - ( id + Nx ) * isHoppingUp.y );
    d_down.y            = ( id - Nx ) * isHoppingDown.y + isPeriodicDown.y * (id + Nx * (Ny - 1) - ( id - Nx ) * isHoppingDown.y);

    d_up.z              = ( id + Nx*Ny ) * isHoppingUp.z   + isPeriodicUp.z  *  (id - Nx * Ny * (Nz - 1) - ( id + Nx*Ny ) * isHoppingUp.z) ;
    d_down.z            = ( id - Nx*Ny ) * isHoppingDown.z + isPeriodicDown.z * (id + Nx * Ny * (Nz - 1) - ( id - Nx*Ny ) * isHoppingDown.z );

    //For hopping forward use index idx, for hopping abckwards use the following indeces;
    // The isPeriodicUp is not needed for the hopping because if isPeridicUp is non zero the hopping value will be non zero
    //                      Simple back-hop             if periodic boundary
    hop_idx_down.x = idx - isHoppingDown.x       + isPeriodicDown.x * ( (Nx - 1) + isHoppingDown.x ) ;
    hop_idx_down.y = idx - Nx * isHoppingDown.y  + isPeriodicDown.y * ( Nx * (Ny - 1) + Nx * isHoppingDown.y);
    hop_idx_down.z = idx - Nx*Ny*isHoppingDown.z + isPeriodicDown.z * ( Nx*Ny*(Nz - 1) + Nx*Ny*isHoppingDown.z);


    field2 h_temp = {0, 0};
    field2 e_temp = {0, 0};
    

    //Diagonal element
    multScSum(h_temp, h[id], ( -( mu + H + modulation[idx]) - V * n_down[idx] * HARTREE ) );
    multScSum(e_temp, e[id], ( -( mu + H + modulation[idx]) - V * n_down[idx] * HARTREE ) );


    //Hopping in x
    multCxScSum(h_temp, h[d_up.x], t_x[idx], isHoppingUp.x );
    multCxScSum(h_temp, h[d_down.x], conj(t_x[hop_idx_down.x]), isHoppingDown.x );

    multCxScSum(e_temp, e[d_up.x], t_x[idx], isHoppingUp.x);
    multCxScSum(e_temp, e[d_down.x], conj(t_x[hop_idx_down.x]), isHoppingDown.x);


    //Hopping in Y
    multCxScSum(h_temp, h[d_up.y], t_y[idx], isHoppingUp.y );
    multCxScSum(h_temp, h[d_down.y], conj(t_y[hop_idx_down.y]), isHoppingDown.y );

    multCxScSum(e_temp, e[d_up.y], t_y[idx], isHoppingUp.y);
    multCxScSum(e_temp, e[d_down.y], conj(t_y[hop_idx_down.y]), isHoppingDown.y);

    //Hopping in z
    multCxScSum(h_temp, h[d_up.z], t_z[idx], isHoppingUp.z);
    multCxScSum(h_temp, h[d_down.z], conj(t_z[hop_idx_down.z]), isHoppingDown.z);

    multCxScSum(e_temp, e[d_up.z], t_z[idx], isHoppingUp.z);
    multCxScSum(e_temp, e[d_down.z], conj(t_z[hop_idx_down.z]), isHoppingDown.z);

    //Pairing interaction
    multCxScSum(h_temp, h[id + N], D[idx], 1.0);
    multCxScSum(e_temp, e[id + N], D[idx], 1.0);

    //Assigning value to new vector
    multSum(h_prev[id], h_temp, h_prev[id], recursion_coeff, -1.0);
    multSum(e_prev[id], e_temp, e_prev[id], recursion_coeff, -1.0);
    
    
    h_temp = {0,0};
    e_temp = {0,0};

    //Diagonal element

    multScSum(h_temp, h[id + N], ( (mu - H + modulation[idx] ) + V*n_up[idx] * HARTREE) );
    multScSum(e_temp, e[id + N], ( (mu - H + modulation[idx] ) + V*n_up[idx] * HARTREE) );


    //Hopping in x
    multCxScSum(h_temp, h[d_up.x + N], conj(t_x[idx], -1.0), isHoppingUp.x);
    multCxScSum(h_temp, h[d_down.x + N], mult(t_x[hop_idx_down.x], -1.0), isHoppingDown.x);

    multCxScSum(e_temp, e[d_up.x + N], conj(t_x[idx], -1.0), isHoppingUp.x);
    multCxScSum(e_temp, e[d_down.x + N], mult(t_x[hop_idx_down.x], -1.0), isHoppingDown.x);

    //Hopping in y
    multCxScSum(h_temp, h[d_up.y + N], conj(t_y[idx], -1.0), isHoppingUp.y);
    multCxScSum(h_temp, h[d_down.y + N], mult(t_y[hop_idx_down.y], -1.0), isHoppingDown.y);

    multCxScSum(e_temp, e[d_up.y + N], conj(t_y[idx], -1.0), isHoppingUp.y);
    multCxScSum(e_temp, e[d_down.y + N], mult(t_y[hop_idx_down.y], -1.0), isHoppingDown.y);

    //Hopping in z
    multCxScSum(h_temp, h[d_up.z + N], conj(t_z[idx], -1.0), isHoppingUp.z);
    multCxScSum(h_temp, h[d_down.z + N], mult(t_z[hop_idx_down.z], -1.0), isHoppingDown.z);

    multCxScSum(e_temp, e[d_up.z + N], conj(t_z[idx],-1.0), isHoppingUp.z);
    multCxScSum(e_temp, e[d_down.z + N], mult(t_z[hop_idx_down.z], -1.0), isHoppingDown.z);

    //Pairing interaction
    multCxScSum(h_temp, h[id], conj(D[idx]), 1.0);
    multCxScSum(e_temp, e[id], conj(D[idx]), 1.0);


    //Assigning value to new vector
    multSum(h_prev[id + N], h_temp, h_prev[id + N], recursion_coeff, -1.0);
    multSum(e_prev[id + N], e_temp, e_prev[id + N], recursion_coeff, -1.0); 

    //Maximum value of site_offset + blocky is N
    if(idx == site)
    {
        // multScSum(D_new[idx], h_prev[id], (- V * cheb_n ) ); //e(i).h_n*(i)
        multScSum(D_new[idx], h_prev[id], (- cheb_n ) ); //e(i).h_n*(i)

        //Free energy calculation
        F[idx] += - T * cheb_n_F * ( h_prev[id + N].x + e_prev[id].x);

        if(HARTREE){
        n_up_new[idx] +=  cheb_n * e_prev[id].x; // e(i).e_n(i)
        n_down_new[idx] -= cheb_n * h_prev[id + N].x;    //h(i).h_n*(i). n must be real
        }
        


    }
}

/*
__global__
void josephson(field2 *d_D_1, field2 *d_D_2, field2 *d_D_3, const field V1,const field V2, const field V3, const field V12, const field V13, const field V23, const int N)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= N) return;

    field2 d_D_1_temp = d_D_1[idx];
    field2 d_D_2_temp = d_D_2[idx];
    field2 d_D_3_temp = d_D_3[idx];


    d_D_1[idx].x = V1 * d_D_1_temp.x + V12 * d_D_2_temp.x + V13 * d_D_3_temp.x;
    d_D_1[idx].y = V1 * d_D_1_temp.y + V12 * d_D_2_temp.y + V13 * d_D_3_temp.y;

    d_D_2[idx].x = V2 * d_D_2_temp.x + V12 * d_D_1_temp.x  + V13 * d_D_3_temp.x;
    d_D_2[idx].y = V2 * d_D_2_temp.y + V12 * d_D_1_temp.y  + V13 * d_D_3_temp.y; 

    d_D_3[idx].x = V3 * d_D_3_temp.x + V13 * d_D_1_temp.x  + V23 * d_D_2_temp.x;
    d_D_3[idx].y = V3 * d_D_3_temp.y + V13 * d_D_1_temp.y  + V23 * d_D_2_temp.y; 
}
*/

__global__
void josephson(field2 *d_D_1, field2 *d_D_2, const field V1,const field V2, const field V12, const int N)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= N) return;

    field2 d_D_1_temp = d_D_1[idx];
    field2 d_D_2_temp = d_D_2[idx];

    d_D_1[idx].x = V1 * d_D_1_temp.x + V12 * d_D_2_temp.x;
    d_D_1[idx].y = V1 * d_D_1_temp.y + V12 * d_D_2_temp.y;

    d_D_2[idx].x = V2 * d_D_2_temp.x + V12 * d_D_1_temp.x;
    d_D_2[idx].y = V2 * d_D_2_temp.y + V12 * d_D_1_temp.y; 
}

/////?///just fot J//////not for you
//Calculation of e_{n+1} by recursion relation of Chebyshev polinomial. Function also adds partial result
__global__
void update_J(
    const field2 *t_x_1,const field2 *t_y_1,const field2 *t_z_1,
    const field2 *t_x_2,const field2 *t_y_2,const field2 *t_z_2,
    //const field2 *t_x_3,const field2 *t_y_3, const field2 *t_z_3,
    const field2 *e_1, field2 *h_1, const field2 *e_2, field2 *h_2, 
    //const field2 *e_3, field2 *h_3, 
    field3 *d_J,
    const int *neighbor_list,
    const int N, const int Nx, const int Ny, const int Nz, const int site_offset,
    const field cheb_n)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t blocky = blockIdx.y;
    size_t id = idx + blocky * 2 * N;
    const int site = blocky + site_offset;

    if(idx >= N) return;

    if(site >= N) return;

    int3 isHoppingUp, isPeriodicUp;
    int3 d_up;

    //     NEIGHBOR CONVENTION  //
   //setting n = which_neighbor//
   //      n = 0 => x - 1      //
   //      n = 1 => x + 1      //
   //      n = 2 => y - 1      //
   //      n = 3 => y + 1      //
   //      n = 4 => z - 1      //
   //      n = 5 => z + 1      //
   //Setting boundaries
    {
        int current_neighbors = neighbor_list[idx];

        isHoppingUp.x = (current_neighbors >> 1) & 1;
        isHoppingUp.y = (current_neighbors >> 3) & 1;
        isHoppingUp.z = (current_neighbors >> 5) & 1;

        isPeriodicUp.x  = (current_neighbors >> 6)  & 1;
        isPeriodicUp.y  = (current_neighbors >> 8)  & 1;
        isPeriodicUp.z  = (current_neighbors >> 10) & 1;

    }


    // isPeriodic up is 1 only for the sites on the upper boundary of the respective dimension. I.e. in x when idx = Nx then isPeriodicUp = 1 if we have p. bc.
    // similarly when idx = 0 and we have pbc we have isPeriodicDown = 1
    //Setting displacement vectors for nearest neighbor hopping
    d_up.x              = ( id + 1 ) * isHoppingUp.x   + isPeriodicUp.x  * (id - (Nx - 1) );    //when idx = Nx-1, you cannot hop up, but you can if periodic
    d_up.y              = ( id + Nx ) * isHoppingUp.y   + isPeriodicUp.y  * (id - Nx * (Ny - 1) );
    d_up.z              = ( id + Nx*Ny ) * isHoppingUp.z   + isPeriodicUp.z  * (id - Nx * Ny * (Nz - 1) );



    field2 tx_1 = t_x_1[idx];
    field2 ty_1 = t_y_1[idx];
    field2 tz_1 = t_z_1[idx];
    field2 tx_2 = t_x_2[idx];
    field2 ty_2 = t_y_2[idx];
    field2 tz_2 = t_z_2[idx];
    /*
    field2 tx_3 = t_x_3[idx];
    field2 ty_3 = t_y_3[idx];
    field2 tz_3 = t_z_3[idx];
    */
    //Maximum value of site_offset + blocky is N
    if(idx == site)
    {

        // Current contribution from component 1
        d_J[idx].x += cheb_n * 2 * (tx_1.x * e_1[d_up.x].y + tx_1.y * e_1[d_up.x].x)*isHoppingUp.x;
        d_J[idx].y += cheb_n * 2 * (ty_1.x * e_1[d_up.y].y + ty_1.y * e_1[d_up.y].x)*isHoppingUp.y;
        d_J[idx].z += cheb_n * 2 * (tz_1.x * e_1[d_up.z].y + tz_1.y * e_1[d_up.z].x)*isHoppingUp.z;

        d_J[idx].x += cheb_n * 2 * (tx_1.x * h_1[d_up.x + N].y - tx_1.y * h_1[d_up.x + N].x)*isHoppingUp.x;
        d_J[idx].y += cheb_n * 2 * (ty_1.x * h_1[d_up.y + N].y - ty_1.y * h_1[d_up.y + N].x)*isHoppingUp.y;
        d_J[idx].z += cheb_n * 2 * (tz_1.x * h_1[d_up.z + N].y - tz_1.y * h_1[d_up.z + N].x)*isHoppingUp.z;

         // Current contribution from component 2
        d_J[idx].x += cheb_n * 2 * (tx_2.x * e_2[d_up.x].y + tx_2.y * e_2[d_up.x].x)*isHoppingUp.x;
        d_J[idx].y += cheb_n * 2 * (ty_2.x * e_2[d_up.y].y + ty_2.y * e_2[d_up.y].x)*isHoppingUp.y;
        d_J[idx].z += cheb_n * 2 * (tz_2.x * e_2[d_up.z].y + tz_2.y * e_2[d_up.z].x)*isHoppingUp.z;
 
        d_J[idx].x += cheb_n * 2 * (tx_2.x * h_2[d_up.x + N].y - tx_2.y * h_2[d_up.x + N].x)*isHoppingUp.x;
        d_J[idx].y += cheb_n * 2 * (ty_2.x * h_2[d_up.y + N].y - ty_2.y * h_2[d_up.y + N].x)*isHoppingUp.y;
        d_J[idx].z += cheb_n * 2 * (tz_2.x * h_2[d_up.z + N].y - tz_2.y * h_2[d_up.z + N].x)*isHoppingUp.z;

        // Current contribution from component 3
        /*
        d_J[idx].x += cheb_n * 2 * (tx_3.x * e_3[d_up.x].y + tx_3.y * e_3[d_up.x].x)*isHoppingUp.x;
        d_J[idx].y += cheb_n * 2 * (ty_3.x * e_3[d_up.y].y + ty_3.y * e_3[d_up.y].x)*isHoppingUp.y;
        d_J[idx].z += cheb_n * 2 * (tz_3.x * e_3[d_up.z].y + tz_3.y * e_3[d_up.z].x)*isHoppingUp.z;

        d_J[idx].x += cheb_n * 2 * (tx_3.x * h_3[d_up.x + N].y - tx_3.y * h_3[d_up.x + N].x)*isHoppingUp.x;
        d_J[idx].y += cheb_n * 2 * (ty_3.x * h_3[d_up.y + N].y - ty_3.y * h_3[d_up.y + N].x)*isHoppingUp.y;
        d_J[idx].z += cheb_n * 2 * (tz_3.x * h_3[d_up.z + N].y - tz_3.y * h_3[d_up.z + N].x)*isHoppingUp.z;
        */
    }
}




//Update of the vector potential in the superconductor
__global__
void updateA2d(
    field2 *t_x_1, field2 *t_y_1, field2 *t_x_2, field2 *t_y_2, //field2 *t_x_3, field2 *t_y_3, 
    const field Bext, const field q,
    const field3 *d_J, const field2 *d_A, field2 *d_A_new,
    field step, const int *neighbor_list,  const int N, const int Nx, const field Emax)
{

    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= N) return;

    uint3 isHoppingUp, isHoppingDown;
    uint3 d_up, d_down;
    uint2 d_up_lr, d_down_lr; //x-> positive x; y-> negative x
    field2 dFA = {0.0,0.0};
    {
        int current_neighbors = neighbor_list[idx];

        //For neighbor convention see function isNeighbor.
        //Here it is copied because it is faster

        isHoppingDown.x = (current_neighbors >> 0) & 1;
        isHoppingDown.y = (current_neighbors >> 2) & 1;
        isHoppingDown.z = (current_neighbors >> 4) & 1;

        isHoppingUp.x = (current_neighbors >> 1) & 1;
        isHoppingUp.y = (current_neighbors >> 3) & 1;
        isHoppingUp.z = (current_neighbors >> 5) & 1;

    }


    d_up.x              = ( idx + 1 ) * isHoppingUp.x;
    d_down.x            = ( idx - 1 ) * isHoppingDown.x;
    d_up.y              = ( idx + Nx ) * isHoppingUp.y;
    d_down.y            = ( idx - Nx ) * isHoppingDown.y;

    d_up_lr.x           = (d_up.y   + 1)*isHoppingUp.y*isHoppingUp.x;
    d_up_lr.y           = (d_up.y   - 1)*isHoppingUp.y*isHoppingDown.x;
    d_down_lr.x         = (d_down.y + 1)*isHoppingDown.y*isHoppingUp.x;
    d_down_lr.y         = (d_down.y - 1)*isHoppingDown.y*isHoppingDown.x;



    dFA.x = -q * Emax * d_J[idx].x + ( - ( d_A[d_up.y].x - d_A[idx].x  ) +  ( d_A[d_up.x].y - d_A[idx].y ) - Bext ) * isHoppingUp.y * isHoppingUp.x
                                   + ( (d_A[idx].x - d_A[d_down.y].x) - (d_A[d_down_lr.x].y -d_A[d_down.y].y) + Bext ) * isHoppingDown.y * isHoppingUp.x;


    dFA.y = -q * Emax * d_J[idx].y + ( ( d_A[d_up.y].x - d_A[idx].x ) - ( d_A[d_up.x].y - d_A[idx].y ) + Bext ) * isHoppingUp.y * isHoppingUp.x
                                   + ( ( d_A[idx].y - d_A[d_down.x].y )  - (d_A[d_up_lr.y].x - d_A[d_down.x].x  ) - Bext ) * isHoppingDown.x * isHoppingUp.y;

                       
    ////// FIXED STEP UPDATE /////
    d_A_new[idx].x = d_A[idx].x - step*dFA.x;
    d_A_new[idx].y = d_A[idx].y - step*dFA.y;
    /////////////////////////////

    // Update hopping component 1
    field2 t0_x_1 = t_x_1[idx];
    field2 t0_y_1 = t_y_1[idx];

    t_x_1[idx].x = t0_x_1.x * cos(q *  step*dFA.x) + t0_x_1.y * sin(q * step*dFA.x);
    t_x_1[idx].y = t0_x_1.y * cos(q *  step*dFA.x) - t0_x_1.x * sin(q * step*dFA.x);

    t_y_1[idx].x = t0_y_1.x * cos(q *  step*dFA.y) + t0_y_1.y * sin(q * step*dFA.y);
    t_y_1[idx].y = t0_y_1.y * cos(q *  step*dFA.y) - t0_y_1.x * sin(q * step*dFA.y);

    // Update hopping component 2
    field2 t0_x_2 = t_x_2[idx];
    field2 t0_y_2 = t_y_2[idx];

    t_x_2[idx].x = t0_x_2.x * cos(q *  step*dFA.x) + t0_x_2.y * sin(q * step*dFA.x);
    t_x_2[idx].y = t0_x_2.y * cos(q *  step*dFA.x) - t0_x_2.x * sin(q * step*dFA.x);

    t_y_2[idx].x = t0_y_2.x * cos(q *  step*dFA.y) + t0_y_2.y * sin(q * step*dFA.y);
    t_y_2[idx].y = t0_y_2.y * cos(q *  step*dFA.y) - t0_y_2.x * sin(q * step*dFA.y);

/*
    // Update hopping component 3
    field2 t0_x_3 = t_x_3[idx];
    field2 t0_y_3 = t_y_3[idx];

    t_x_3[idx].x = t0_x_3.x * cos(q *  step*dFA.x) + t0_x_3.y * sin(q * step*dFA.x);
    t_x_3[idx].y = t0_x_3.y * cos(q *  step*dFA.x) - t0_x_3.x * sin(q * step*dFA.x);

    t_y_3[idx].x = t0_y_3.x * cos(q *  step*dFA.y) + t0_y_3.y * sin(q * step*dFA.y);
    t_y_3[idx].y = t0_y_3.y * cos(q *  step*dFA.y) - t0_y_3.x * sin(q * step*dFA.y);
*/




}



//INITIALIZATION FUNCTION FOR h VECTOR
__global__
void init_h(field2 *Q_prev, field2 *Q_n, int N, int site_offset)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t blocky = blockIdx.y;
    size_t id = idx + blockIdx.y * 2 * N;

    if(idx >= N) return;
    if(blocky + site_offset >= N) return;

    Q_n[id]     = {0,0};
    Q_n[id + N] = { (field)(idx == site_offset + blocky), 0 };


    Q_prev[id]      = {0.0,0.0};
    Q_prev[id + N]  = {0.0,0.0};
}


//INITIALIZATION FUNCTION FOR e VECTOR
__global__
void init_e(field2 *Q_prev, field2 *Q_n, int N, int site_offset)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t blocky = blockIdx.y;
    size_t id = idx + blockIdx.y * 2 * N;

    if(idx >= N) return;
    if(blocky + site_offset >= N) return;

    Q_n[id]     = { (field)(idx == site_offset + blocky), 0 };
    Q_n[id + N] = {0,0};


    Q_prev[id]      = {0.0,0.0};
    Q_prev[id + N]  = {0.0,0.0};
}

__global__
void update_Delta(field2 *d_D, field2 *d_D_new, const float m, const int N)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= N) return;

    //field2 d_D_temp = d_D_new[idx];

    d_D_new[idx].x = m * d_D[idx].x + (1.0 - m) * d_D_new[idx].x;
    d_D_new[idx].y = m * d_D[idx].y + (1.0 - m) * d_D_new[idx].y;
}



////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////      CPU FUNCTIONS      ////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////


//Coefficients for Chebishev expansion at zero temperature
void zeroTcheb(field *chebCoefs, const int order, const field b, const field a)
{

    spdlog::info("{:=^44}", "");
    spdlog::info("{:=^44}", "    CHEBISHEV COEFFICIENT T=0    ");
    spdlog::info("{:=^44}", "");

    field kernel;
    chebCoefs[0] = ( PI - acos(-b/a) )/(PI);
    for(int i = 1; i < order; ++i) {

        kernel = ((order - i + 1) * cos(PI * i / (order + 1)) + sin(PI * i / (order + 1)) * cos(PI / (order + 1)) / sin(PI / (order + 1))) / (order + 1);
        chebCoefs[i] = -2*kernel*( sin( i * acos(-b/a) )/i )/(PI);
    }

}


//Coefficients for Chebishev expansion at arbitrary temperature for f(H)
//https://mathworld.wolfram.com/ChebyshevApproximationFormula.html
void generateChebTable(field *chebCoefs, const int order, const field coef)
{

    spdlog::info("{:=^44}", "");
    spdlog::info("{:=^44}", "    CHEBISHEV COEFFICIENT FOR f(H) T>0    ");
    spdlog::info("{:=^44}", "");

    field kernel;
    field x;
    for(int i = 0; i < order; ++i)
    {
        chebCoefs[i] = 0;
        for(int k = 1; k <= order; ++k)
        {
            x = coef * cos(PI * (k - 0.5) / (field)order);
            // chebCoefs[i] += 1.0 / (exp(x) + 1.0) * cos(i * PI * (k - 0.5) / (field)order);
            if(x < 0)
            {
                chebCoefs[i] += 1.0 / (exp(x) + 1.0) * cos(i * PI * (k - 0.5) / (field)order);
            }
            else
            {
                chebCoefs[i] += ( 1 - 1.0 / (exp( - x) + 1.0) ) * cos(i * PI * (k - 0.5) / (field)order);
            }
            
        }
        kernel = 1.0;
        // kernel = ((order - i + 1) * cos(PI * i / (order + 1)) + sin(PI * i / (order + 1)) * cos(PI / (order + 1)) / sin(PI / (order + 1))) / (order + 1);
        chebCoefs[i] *= kernel * 2.0 * (i == 0 ? (0.5) : 1.0) / order;
    }
}

//Coefficients for Chebishev expansion at arbitrary temperature for f(H)ln(H)
void generateChebTableF(field *chebCoefs, const int order, const field coef)
{

    spdlog::info("{:=^44}", "");
    spdlog::info("{:=^44}", "    CHEBISHEV COEFFICIENT FOR ln(H)f(H) T>0    ");
    spdlog::info("{:=^44}", "");

    field kernel;
    field x;
    for(int i = 0; i < order; ++i)
    {
        chebCoefs[i] = 0;
        for(int k = 1; k <= order; ++k)
        {
            x = coef * cos( PI * (k - 0.5) / (field)order );
            if(x > 0)
            {
                chebCoefs[i] +=   log( exp( -x ) + 1 ) * cos( ( i * PI * (k - 0.5) ) / (field)order );
            }
            else
            {
                chebCoefs[i] += - ( x  + log( 1.0 / (exp(x) + 1.0) ) ) * cos( ( i * PI * (k - 0.5) ) / (field)order );
            }
            
           
        }
        kernel = 1.0; //((order - i + 1) * cos(PI * i / (order + 1)) + sin(PI * i / (order + 1)) * cos(PI / (order + 1)) / sin(PI / (order + 1))) / (order + 1);
        chebCoefs[i] *= kernel * 2.0 * (i == 0 ? (0.5) : 1.0) / order;
       
    }
}

__global__ 
void modkernel(field *modulation, const int N, const Hamiltonian hamiltonian)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int rx,ry,rz;
    field x,y,z;

    if(idx >= N) return;

    int Nx = hamiltonian.Nx;
    int Ny = hamiltonian.Ny;
    int Nz = hamiltonian.Nz;

    rx = (idx % Nx);
    ry = (idx / Nx) % Ny;
    rz = (idx / (Ny * Nx) );

    x = (field) 2*( rx - Nx/2 )/Nx;
    y = (field) 2*( ry - Ny/2 )/Ny;
    z = (field) 2*( rz - Nz/2 )/Nz;

   // if( (x < -0.7 || x>0.7) && ( y > -0.1 && y < 0.1 ) )
   // {
   // modulation[idx] = 2.0 ;
   // }
   // else
   // {
   // modulation[idx] = 0.0;
   // }

    if( (x > -4.0/Nx  &&  x < 4.0/Nx) && ( y > -4.0/Ny && y < 4.0/Ny ) )
    {
    modulation[idx] = 2.0 ;
    }
    else
    {
    modulation[idx] = 0.0;
    }

    //rescaling
    modulation[idx] *= 1/ hamiltonian.Emax;
    
}



//SELF CONCISTENCE LOOP
void mean_field::selfConsistent(
    const int *h_geometry, Hamiltonian &hamiltonian, field2 *h_A, field3 *h_J, field *h_F,
    field2 *h_D_1, field *h_n_up_1, field *h_n_down_1, field2 *h_T_x_1, field2 *h_T_y_1, field2 *h_T_z_1,
    field2 *h_D_2, field *h_n_up_2, field *h_n_down_2, field2 *h_T_x_2, field2 *h_T_y_2, field2 *h_T_z_2,
    //field2 *h_D_3, field *h_n_up_3, field *h_n_down_3, field2 *h_T_x_3, field2 *h_T_y_3, field2 *h_T_z_3,
    const size_t SIZE_N_REAL, const size_t SIZE_N, const size_t SIZE_3N, const size_t SIZE_2N_XY,
    const int X_BLOCKS, const int TPB, const int Y_BLOCKS,
    const int CHEB_ORDER, const int MAX_ITER, const float CONVERGED, const float memory_par,
    int &totalIter, field &convergenceDelta_1, field &convergenceDelta_2, //field &convergenceDelta_3,
    field &convergenceNup_1, field &convergenceNdown_1, field &convergenceNup_2, field &convergenceNdown_2, 
    //field &convergenceNup_3, field &convergenceNdown_3, 
    field &convergenceA,
    const bool PARTIAL_PRINT, const int PARTIAL_SAVE_ITERATIONS,  const bool GAUGE_FIELD, const bool HARTREE, const bool MEMORY, 
    std::string file)
    {

    const int N = hamiltonian.Nx*hamiltonian.Ny*hamiltonian.Nz;

    spdlog::info("{:=^44}", "");
    spdlog::info("{:=^44}", "    COMPUTING CHEBISHEV COEFFICIENT    ");
    spdlog::info("{:=^44}", "");


   
    ///////////// EXPANSION COEFFICIENT CALCULATION /////////////
    field *chebTab  = (field*)malloc(CHEB_ORDER * sizeof(field));
    field *chebTabF = (field*)malloc(CHEB_ORDER * sizeof(field));

    // Generate expansion coefficient for f
    if(hamiltonian.T == 0.0) zeroTcheb(chebTab, CHEB_ORDER, 0.0, hamiltonian.Emax);
    else             generateChebTable(chebTab, CHEB_ORDER, hamiltonian.Emax / hamiltonian.T);

    // Generate expansion coefficients for the free energy calculation
    generateChebTableF(chebTabF, CHEB_ORDER, hamiltonian.Emax / hamiltonian.T);


    // Component 1
    field2 *d_D_1           = nullptr;
    field2 *d_Dnew_1        = nullptr;
    field2 *d_h_prev_1      = nullptr;
    field2 *d_h_n_1         = nullptr;
    field2 *d_e_prev_1      = nullptr;
    field2 *d_e_n_1         = nullptr;
    field2 *d_T_x_1         = nullptr;
    field2 *d_T_y_1         = nullptr;
    field2 *d_T_z_1         = nullptr;
    field  *d_n_up_1        = nullptr;
    field  *d_n_down_1      = nullptr;
    field  *d_n_up_new_1    = nullptr;
    field  *d_n_down_new_1  = nullptr;

    // Component 2
    field2 *d_D_2           = nullptr;
    field2 *d_Dnew_2        = nullptr;
    field2 *d_h_prev_2      = nullptr;
    field2 *d_h_n_2         = nullptr;
    field2 *d_e_prev_2      = nullptr;
    field2 *d_e_n_2         = nullptr;
    field2 *d_T_x_2         = nullptr;
    field2 *d_T_y_2         = nullptr;
    field2 *d_T_z_2         = nullptr;
    field  *d_n_up_2        = nullptr;
    field  *d_n_down_2      = nullptr;
    field  *d_n_up_new_2    = nullptr;
    field  *d_n_down_new_2  = nullptr;
/*
    // Component 3
    field2 *d_D_3           = nullptr;
    field2 *d_Dnew_3        = nullptr;
    field2 *d_h_prev_3      = nullptr;
    field2 *d_h_n_3         = nullptr;
    field2 *d_e_prev_3      = nullptr;
    field2 *d_e_n_3         = nullptr;
    field2 *d_T_x_3         = nullptr;
    field2 *d_T_y_3         = nullptr;
    field2 *d_T_z_3         = nullptr;
    field  *d_n_up_3        = nullptr;
    field  *d_n_down_3      = nullptr;
    field  *d_n_up_new_3    = nullptr;
    field  *d_n_down_new_3  = nullptr;
*/


    int    *d_neighbors   = nullptr;
    field2 *d_A           = nullptr;
    field2 *d_A_new       = nullptr;
    field3 *d_J           = nullptr;
    field  *d_F           = nullptr;
    field  *d_modulation  = nullptr;
    
    
    
    
    //GPU variables needed for the recursion relations
    cudaMalloc(&d_neighbors,  N*sizeof(int));
    cudaMalloc(&d_F,           SIZE_N_REAL);
    cudaMalloc(&d_modulation,  SIZE_N_REAL);
    

    if(GAUGE_FIELD)
    {
        cudaMalloc(&d_J,           SIZE_3N);
        cudaMalloc(&d_A,           SIZE_N);
        cudaMalloc(&d_A_new,       SIZE_N);
    }
    
    
    // Allocation variables first component
    cudaMalloc(&d_D_1,           SIZE_N);
    cudaMalloc(&d_Dnew_1,        SIZE_N);
    cudaMalloc(&d_T_x_1,         SIZE_N);
    cudaMalloc(&d_T_y_1,         SIZE_N);
    cudaMalloc(&d_T_z_1,         SIZE_N);
    cudaMalloc(&d_n_up_1,        SIZE_N_REAL);
    cudaMalloc(&d_n_down_1,      SIZE_N_REAL);
    cudaMalloc(&d_n_up_new_1,    SIZE_N_REAL);
    cudaMalloc(&d_n_down_new_1,  SIZE_N_REAL);
    cudaMalloc(&d_e_prev_1,      SIZE_2N_XY);
    cudaMalloc(&d_e_n_1,         SIZE_2N_XY);
    cudaMalloc(&d_h_prev_1,      SIZE_2N_XY);
    cudaMalloc(&d_h_n_1,         SIZE_2N_XY);

    // Allocation variables for second component
    cudaMalloc(&d_D_2,           SIZE_N);
    cudaMalloc(&d_Dnew_2,        SIZE_N);
    cudaMalloc(&d_T_x_2,         SIZE_N);
    cudaMalloc(&d_T_y_2,         SIZE_N);
    cudaMalloc(&d_T_z_2,         SIZE_N);
    cudaMalloc(&d_n_up_2,        SIZE_N_REAL);
    cudaMalloc(&d_n_down_2,      SIZE_N_REAL);
    cudaMalloc(&d_n_up_new_2,    SIZE_N_REAL);
    cudaMalloc(&d_n_down_new_2,  SIZE_N_REAL);
    cudaMalloc(&d_e_prev_2,      SIZE_2N_XY);
    cudaMalloc(&d_e_n_2,         SIZE_2N_XY);
    cudaMalloc(&d_h_prev_2,      SIZE_2N_XY);
    cudaMalloc(&d_h_n_2,         SIZE_2N_XY);
/*
    // Allocation variables for second component
    cudaMalloc(&d_D_3,           SIZE_N);
    cudaMalloc(&d_Dnew_3,        SIZE_N);
    cudaMalloc(&d_T_x_3,         SIZE_N);
    cudaMalloc(&d_T_y_3,         SIZE_N);
    cudaMalloc(&d_T_z_3,         SIZE_N);
    cudaMalloc(&d_n_up_3,        SIZE_N_REAL);
    cudaMalloc(&d_n_down_3,      SIZE_N_REAL);
    cudaMalloc(&d_n_up_new_3,    SIZE_N_REAL);
    cudaMalloc(&d_n_down_new_3,  SIZE_N_REAL);
    cudaMalloc(&d_e_prev_3,      SIZE_2N_XY);
    cudaMalloc(&d_e_n_3,         SIZE_2N_XY);
    cudaMalloc(&d_h_prev_3,      SIZE_2N_XY);
    cudaMalloc(&d_h_n_3,         SIZE_2N_XY);
*/
    

    //Initialization of vector potential
    if(GAUGE_FIELD)
    {
        cudaMemcpy(d_A, h_A, SIZE_N, cudaMemcpyHostToDevice);
        cudaMemset(d_A_new,0, SIZE_N);
    }
    
    // Initialization of component 1
    array::init_gpu<<< X_BLOCKS, TPB >>>(d_n_up_1, N, 0.0);
    array::init_gpu<<< X_BLOCKS, TPB >>>(d_n_down_1, N, 0.0);
    cudaMemcpy(d_D_1, h_D_1, SIZE_N, cudaMemcpyHostToDevice);
    cudaMemcpy(d_T_x_1, h_T_x_1, SIZE_N, cudaMemcpyHostToDevice);
    cudaMemcpy(d_T_y_1, h_T_y_1, SIZE_N, cudaMemcpyHostToDevice);
    cudaMemcpy(d_T_z_1, h_T_z_1, SIZE_N, cudaMemcpyHostToDevice);
    cudaMemset(d_n_down_new_1, 0, SIZE_N_REAL);
    cudaMemset(d_n_up_new_1, 0, SIZE_N_REAL);
    cudaMemset(d_n_down_1, 0, SIZE_N_REAL);
    cudaMemset(d_n_up_1, 0, SIZE_N_REAL);

    // Initialization of component 2
    array::init_gpu<<< X_BLOCKS, TPB >>>(d_n_up_2, N, 0.0);
    array::init_gpu<<< X_BLOCKS, TPB >>>(d_n_down_2, N, 0.0);
    cudaMemcpy(d_D_2, h_D_2, SIZE_N, cudaMemcpyHostToDevice);
    cudaMemcpy(d_T_x_2, h_T_x_2, SIZE_N, cudaMemcpyHostToDevice);
    cudaMemcpy(d_T_y_2, h_T_y_2, SIZE_N, cudaMemcpyHostToDevice);
    cudaMemcpy(d_T_z_2, h_T_z_2, SIZE_N, cudaMemcpyHostToDevice);
    cudaMemset(d_n_down_new_2, 0, SIZE_N_REAL);
    cudaMemset(d_n_up_new_2, 0, SIZE_N_REAL);
    cudaMemset(d_n_down_2, 0, SIZE_N_REAL);
    cudaMemset(d_n_up_2, 0, SIZE_N_REAL);
/*
     // Initialization of component 2
     array::init_gpu<<< X_BLOCKS, TPB >>>(d_n_up_3, N, 0.0);
     array::init_gpu<<< X_BLOCKS, TPB >>>(d_n_down_3, N, 0.0);
     cudaMemcpy(d_D_3, h_D_3, SIZE_N, cudaMemcpyHostToDevice);
     cudaMemcpy(d_T_x_3, h_T_x_3, SIZE_N, cudaMemcpyHostToDevice);
     cudaMemcpy(d_T_y_3, h_T_y_3, SIZE_N, cudaMemcpyHostToDevice);
     cudaMemcpy(d_T_z_3, h_T_z_3, SIZE_N, cudaMemcpyHostToDevice);
     cudaMemset(d_n_down_new_3, 0, SIZE_N_REAL);
     cudaMemset(d_n_up_new_3, 0, SIZE_N_REAL);
     cudaMemset(d_n_down_3, 0, SIZE_N_REAL);
     cudaMemset(d_n_up_3, 0, SIZE_N_REAL);
*/
    //Copying geometry to GPU
    cudaMemcpy(d_neighbors, h_geometry, N*sizeof(int), cudaMemcpyHostToDevice);


    if(HARTREE)
    {
        spdlog::info("{:=^44}", "");
        spdlog::info("{:=^44}", "    RUNNING WITH HARTREE TERM    ");
        spdlog::info("{:=^44}", "");
    }
    else
    {
        spdlog::info("{:=^44}", "");
        spdlog::info("{:=^44}", "    RUNNING WITHOUT HARTREE TERM ");
        spdlog::info("{:=^44}", "");
    }
  

    /// Modulation kernel
    ///modkernel<<< X_BLOCKS, TPB >>> (d_modulation, N, hamiltonian);
    cudaDeviceSynchronize();
    field *h_mod 	     = (field*)malloc(SIZE_N_REAL);
    cudaMemcpy(h_mod,d_modulation,SIZE_N_REAL, cudaMemcpyDeviceToHost);
    io::printField(h_mod,"modulation", N, file);
    free(h_mod);


    field step = 0.1;
  

    ///// ITERATIONS ////;
    for(int iter = 0; iter < MAX_ITER; ++iter)
    {

        spdlog::trace("Iteration: {}", iter);

        //Re-Initialize vectors
        cudaMemset(d_Dnew_1, 0, SIZE_N);
        cudaMemset(d_Dnew_2, 0, SIZE_N);
        //cudaMemset(d_Dnew_3, 0, SIZE_N);

        array::init_gpu<<< X_BLOCKS, TPB >>>(d_F, N, -3*2*chebTabF[0]*hamiltonian.T);
        
        if(GAUGE_FIELD) cudaMemset(d_J ,0 ,SIZE_3N);

        if(HARTREE)
        {
            array::init_gpu<<< X_BLOCKS, TPB >>>(d_n_up_new_1, N, chebTab[0]);
            array::init_gpu<<< X_BLOCKS, TPB >>>(d_n_up_new_2, N, chebTab[0]);
            //array::init_gpu<<< X_BLOCKS, TPB >>>(d_n_up_new_3, N, chebTab[0]);
            array::init_gpu<<< X_BLOCKS, TPB >>>(d_n_down_new_1, N, (1-chebTab[0]));
            array::init_gpu<<< X_BLOCKS, TPB >>>(d_n_down_new_2, N, (1-chebTab[0]));
            //array::init_gpu<<< X_BLOCKS, TPB >>>(d_n_down_new_3, N, (1-chebTab[0]));
        }


        for(int i = 0; i < N; i += Y_BLOCKS)
        {
            
            spdlog::trace("Site quota: {}", ((float)i)/((float)(N))*100.0);

            init_h<<<dim3(X_BLOCKS,Y_BLOCKS,1), dim3(TPB,1,1)>>>(d_h_prev_1, d_h_n_1, N, i);
            init_h<<<dim3(X_BLOCKS,Y_BLOCKS,1), dim3(TPB,1,1)>>>(d_h_prev_2, d_h_n_2, N, i);
            //init_h<<<dim3(X_BLOCKS,Y_BLOCKS,1), dim3(TPB,1,1)>>>(d_h_prev_3, d_h_n_3, N, i);
            
            init_e<<<dim3(X_BLOCKS,Y_BLOCKS,1), dim3(TPB,1,1)>>>(d_e_prev_1, d_e_n_1, N, i);
            init_e<<<dim3(X_BLOCKS,Y_BLOCKS,1), dim3(TPB,1,1)>>>(d_e_prev_2, d_e_n_2, N, i);
            //init_e<<<dim3(X_BLOCKS,Y_BLOCKS,1), dim3(TPB,1,1)>>>(d_e_prev_3, d_e_n_3, N, i);
            


            for(int n = 1; n < CHEB_ORDER ; ++n)
            {
                // Multiplication for component 1
                multiplyH<<<dim3(X_BLOCKS,Y_BLOCKS,1), dim3(TPB,1,1)>>>(
                    d_T_x_1, d_T_y_1, d_T_z_1, d_modulation,
                    hamiltonian.mu[0], hamiltonian.H[0], hamiltonian.V[0],
                    d_h_n_1,  d_h_prev_1, d_e_n_1, d_e_prev_1,
                    d_D_1, d_Dnew_1, d_F,
                    d_n_up_1, d_n_down_1, d_n_up_new_1, d_n_down_new_1,  
                    d_neighbors, hamiltonian.T,
                    N, hamiltonian.Nx, hamiltonian.Ny, hamiltonian.Nz, i,
                    chebTab[n], chebTabF[n],  2.0*(n > 1 ? 1.0 : 0.5), HARTREE
                );
                
                // Multiplication for component 2
                multiplyH<<<dim3(X_BLOCKS,Y_BLOCKS,1), dim3(TPB,1,1)>>>(
                    d_T_x_2, d_T_y_2, d_T_z_2, d_modulation,
                    hamiltonian.mu[1], hamiltonian.H[1], hamiltonian.V[1],
                    d_h_n_2,  d_h_prev_2, d_e_n_2, d_e_prev_2,
                    d_D_2, d_Dnew_2, d_F,
                    d_n_up_2, d_n_down_2, d_n_up_new_2, d_n_down_new_2, 
                    d_neighbors, hamiltonian.T,
                    N, hamiltonian.Nx, hamiltonian.Ny, hamiltonian.Nz, i,
                    chebTab[n], chebTabF[n],  2.0*(n > 1 ? 1.0 : 0.5), HARTREE
                );
                /*
                 // Multiplication for component 3
                 multiplyH<<<dim3(X_BLOCKS,Y_BLOCKS,1), dim3(TPB,1,1)>>>(
                    d_T_x_3, d_T_y_3, d_T_z_3, d_modulation,
                    hamiltonian.mu[2], hamiltonian.H[2], hamiltonian.V[2],
                    d_h_n_3,  d_h_prev_3, d_e_n_3, d_e_prev_3,
                    d_D_3, d_Dnew_3, d_F,
                    d_n_up_3, d_n_down_3, d_n_up_new_3, d_n_down_new_3, 
                    d_neighbors, hamiltonian.T,
                    N, hamiltonian.Nx, hamiltonian.Ny, hamiltonian.Nz, i,
                    chebTab[n], chebTabF[n],  2.0*(n > 1 ? 1.0 : 0.5), HARTREE
                );
*/
                cudaDeviceSynchronize();

                if(GAUGE_FIELD)
                {
                ///////////
                /*
                update_J<<<dim3(X_BLOCKS,Y_BLOCKS,1), dim3(TPB,1,1)>>>(
                    d_T_x_1, d_T_y_1, d_T_z_1, d_T_x_2, d_T_y_2, d_T_z_2, d_T_x_3, d_T_y_3, d_T_z_3,
                    d_e_prev_1, d_h_prev_1, d_e_prev_2, d_h_prev_2, d_e_prev_3, d_h_prev_3,
                    d_J,
                    d_neighbors,
                    N, hamiltonian.Nx, hamiltonian.Ny, hamiltonian.Nz, i,
                    chebTab[n]);
                */
                update_J<<<dim3(X_BLOCKS,Y_BLOCKS,1), dim3(TPB,1,1)>>>(
                    d_T_x_1, d_T_y_1, d_T_z_1, d_T_x_2, d_T_y_2, d_T_z_2,
                    d_e_prev_1, d_h_prev_1, d_e_prev_2, d_h_prev_2,
                    d_J,
                    d_neighbors,
                    N, hamiltonian.Nx, hamiltonian.Ny, hamiltonian.Nz, i,
                    chebTab[n]);
                ///////////
                }
            
                array::swap(&d_h_n_1, &d_h_prev_1);
                array::swap(&d_h_n_2, &d_h_prev_2);
                //array::swap(&d_h_n_3, &d_h_prev_3);
                array::swap(&d_e_n_1, &d_e_prev_1);
                array::swap(&d_e_n_2, &d_e_prev_2);  
                //array::swap(&d_e_n_3, &d_e_prev_3);  
            }
        }

        cudaDeviceSynchronize();
        //Kernel introducing josephson interband mixing
        //josephson<<<X_BLOCKS, TPB>>>(d_Dnew_1, d_Dnew_2, d_Dnew_3, hamiltonian.V[0], hamiltonian.V[1], hamiltonian.V[2],
        //                             hamiltonian.V_int[0], hamiltonian.V_int[1], hamiltonian.V_int[2], N);

        josephson<<<X_BLOCKS, TPB>>>(d_Dnew_1, d_Dnew_2, hamiltonian.V[0], hamiltonian.V[1],
                                     hamiltonian.V_int[0], N);

        spdlog::trace("Site quota: 100%");
        spdlog::trace("");
        
        
        convergenceDelta_1   = rdx::checkConvergence(d_D_1, d_Dnew_1, N, "Delta 1", PARTIAL_PRINT);
        convergenceDelta_2   = rdx::checkConvergence(d_D_2, d_Dnew_2, N, "Delta 2", PARTIAL_PRINT);
        //convergenceDelta_3   = rdx::checkConvergence(d_D_3, d_Dnew_3, N, "Delta 3", PARTIAL_PRINT);

        // Self consistency
        update_Delta<<<X_BLOCKS, TPB>>>(d_D_1, d_Dnew_1, memory_par, N);
        update_Delta<<<X_BLOCKS, TPB>>>(d_D_2, d_Dnew_2, memory_par, N);
        //update_Delta<<<X_BLOCKS, TPB>>>(d_D_3, d_Dnew_3, memory_par, N);
        array::swap(&d_D_1, &d_Dnew_1);
        array::swap(&d_D_2, &d_Dnew_2);
        //array::swap(&d_D_3, &d_Dnew_3);

        if(HARTREE)
        {
            convergenceNup_1     = rdx::checkConvergence(d_n_up_1, d_n_up_new_1, N, "Up 1", PARTIAL_PRINT);
            convergenceNdown_1   = rdx::checkConvergence(d_n_down_1, d_n_down_new_1, N, "Down 1", PARTIAL_PRINT);
            convergenceNup_2     = rdx::checkConvergence(d_n_up_2, d_n_up_new_2, N, "Up 2", PARTIAL_PRINT);
            convergenceNdown_2   = rdx::checkConvergence(d_n_down_2, d_n_down_new_2, N, "Down 2", PARTIAL_PRINT);
            //convergenceNup_3     = rdx::checkConvergence(d_n_up_3, d_n_up_new_3, N, "Up 3", PARTIAL_PRINT);
            //convergenceNdown_3   = rdx::checkConvergence(d_n_down_3, d_n_down_new_3, N, "Down 3", PARTIAL_PRINT);
        
            array::swap(&d_n_up_1, &d_n_up_new_1);
            array::swap(&d_n_up_2, &d_n_up_new_2);
            //array::swap(&d_n_up_3, &d_n_up_new_3);
            array::swap(&d_n_down_1, &d_n_down_new_1);
            array::swap(&d_n_down_2, &d_n_down_new_2);
            //array::swap(&d_n_down_3, &d_n_down_new_3);
        }
        

        if(GAUGE_FIELD)
        {
            updateA2d<<<X_BLOCKS, TPB>>>(d_T_x_1, d_T_y_1, d_T_x_2, d_T_y_2, //d_T_x_3, d_T_y_3, 
                                    hamiltonian.Bext.z, hamiltonian.q,
                                    d_J, d_A, d_A_new, step, d_neighbors, N, hamiltonian.Nx, hamiltonian.Emax);
            convergenceA = rdx::checkConvergence(d_A, d_A_new, N,"A", PARTIAL_PRINT);
            array::swap(&d_A, &d_A_new);
        }


        if(HARTREE && GAUGE_FIELD)
        {
            if( ( convergenceDelta_1 < CONVERGED ) &&
                ( convergenceDelta_2 < CONVERGED ) &&
               // ( convergenceDelta_3 < CONVERGED ) &&
                ( convergenceNup_1   < CONVERGED ) &&
                ( convergenceNup_2   < CONVERGED ) &&
               // ( convergenceNup_3   < CONVERGED ) &&
                ( convergenceNdown_1 < CONVERGED ) &&
                ( convergenceNdown_2 < CONVERGED ) &&
               // ( convergenceNdown_3 < CONVERGED ) &&
                ( convergenceA       < CONVERGED) )
            {
                totalIter = iter;
                break;
            }
        }
        else if(HARTREE)
        {
            if( 
                ( convergenceDelta_1 < CONVERGED ) &&
                ( convergenceDelta_2 < CONVERGED ) &&
               // ( convergenceDelta_3 < CONVERGED ) &&
                ( convergenceNup_1   < CONVERGED ) &&
                ( convergenceNup_2   < CONVERGED ) &&
               // ( convergenceNup_3   < CONVERGED ) &&
                ( convergenceNdown_1 < CONVERGED ) &&
                ( convergenceNdown_2  < CONVERGED ) //&&
               // ( convergenceNdown_3 < CONVERGED ) 
            )
            {
                totalIter = iter;
                break;
            }
        }
        else if(GAUGE_FIELD)
        {
            if(
                ( convergenceDelta_1 < CONVERGED) && 
                ( convergenceDelta_2 < CONVERGED) && 
              //  ( convergenceDelta_3 < CONVERGED) && 
                ( convergenceA < CONVERGED)
            ){
                totalIter = iter;
                break;
            }
        }
        else
        {
            if( convergenceDelta_1 < CONVERGED && convergenceDelta_2 < CONVERGED ){ //&& convergenceDelta_3 < CONVERGED ){
                totalIter = iter;
                break;
            }
        }
       
       

        if(iter%PARTIAL_SAVE_ITERATIONS==0)
        {//Put here file output

            cudaMemcpy(h_D_1, d_D_1, SIZE_N, cudaMemcpyDeviceToHost);
            cudaMemcpy(h_D_2, d_D_2, SIZE_N, cudaMemcpyDeviceToHost);
            //cudaMemcpy(h_D_3, d_D_3, SIZE_N, cudaMemcpyDeviceToHost);
            io::printField(h_D_1, "delta_1", N, file);
            io::printField(h_D_2, "delta_2", N, file);
            //io::printField(h_D_3, "delta_3", N, file);
            
            cudaMemcpy(h_F, d_F, SIZE_N_REAL, cudaMemcpyDeviceToHost);
            io::printField(h_F, "F", N, file);

            if(GAUGE_FIELD)
            {
                cudaMemcpy(h_J, d_J, SIZE_3N, cudaMemcpyDeviceToHost);
                cudaMemcpy(h_A, d_A, SIZE_N, cudaMemcpyDeviceToHost);
                io::printField(h_J, "J", N, file);
                io::printField(h_A, "A", N, file);
            }
            

            if(HARTREE)
            {
                cudaMemcpy(h_n_up_1, d_n_up_1, SIZE_N_REAL, cudaMemcpyDeviceToHost);
                cudaMemcpy(h_n_down_1, d_n_down_1, SIZE_N_REAL, cudaMemcpyDeviceToHost);
                cudaMemcpy(h_n_up_2, d_n_up_2, SIZE_N_REAL, cudaMemcpyDeviceToHost);
                cudaMemcpy(h_n_down_2, d_n_down_2, SIZE_N_REAL, cudaMemcpyDeviceToHost);
               // cudaMemcpy(h_n_up_3, d_n_up_3, SIZE_N_REAL, cudaMemcpyDeviceToHost);
               // cudaMemcpy(h_n_down_3, d_n_down_3, SIZE_N_REAL, cudaMemcpyDeviceToHost);
                io::printField(h_n_up_1, "n_up_1", N, file);
                io::printField(h_n_up_2, "n_up_2", N, file);
               // io::printField(h_n_up_3, "n_up_3", N, file);
                io::printField(h_n_down_1, "n_down_1", N, file);
                io::printField(h_n_down_2, "n_down_2", N, file);
               // io::printField(h_n_down_3, "n_down_3", N, file);
            }
            

            
            spdlog::info("Iteration: {}", iter);
            spdlog::info("{:=^44}","    Delta 1   ");
            spdlog::info("Convergence ==> {:.4E}", convergenceDelta_1);
            spdlog::info("{:=^44}","");
            spdlog::info("{:=^44}","    Delta 2   ");
            spdlog::info("Convergence ==> {:.4E}", convergenceDelta_2);
            spdlog::info("{:=^44}","");
            /*
            spdlog::info("{:=^44}","    Delta 3   ");
            spdlog::info("Convergence ==> {:.4E}", convergenceDelta_3);
            spdlog::info("{:=^44}","");
            */

            if(GAUGE_FIELD)
            {
                spdlog::info("{:=^44}","    Gauge     ");
                spdlog::info("Convergence ==> {:.4E}",convergenceA);
                spdlog::info("{:=^44}","");
            }

            if(HARTREE)
            {
                spdlog::info("{:=^44}","    N_up 1  ");
                spdlog::info("Convergence ==> {:.4E}",convergenceNup_1);
                spdlog::info("{:=^44}","");
                spdlog::info("{:=^44}","    N_down 1  ");
                spdlog::info("Convergence ==> {:.4E}",convergenceNdown_1);
                spdlog::info("{:=^44}","");
                spdlog::info("{:=^44}","    N_up 2  ");
                spdlog::info("Convergence ==> {:.4E}",convergenceNup_2);
                spdlog::info("{:=^44}","");
                spdlog::info("{:=^44}","    N_down 2  ");
                spdlog::info("Convergence ==> {:.4E}",convergenceNdown_2);
                spdlog::info("{:=^44}","");
                /*
                spdlog::info("{:=^44}","    N_up 3  ");
                spdlog::info("Convergence ==> {:.4E}",convergenceNup_3);
                spdlog::info("{:=^44}","");
                spdlog::info("{:=^44}","    N_down 3  ");
                spdlog::info("Convergence ==> {:.4E}",convergenceNdown_3);
                spdlog::info("{:=^44}","");
                */
            }

            
        }

    }


    cudaMemcpy(h_D_1, d_D_1, SIZE_N, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_D_2, d_D_2, SIZE_N, cudaMemcpyDeviceToHost);
    //cudaMemcpy(h_D_3, d_D_3, SIZE_N, cudaMemcpyDeviceToHost);
    io::printField(h_D_1, "delta_1", N, file);
    io::printField(h_D_2, "delta_2", N, file);
    //io::printField(h_D_3, "delta_3", N, file);
    
    if(GAUGE_FIELD)
    {
        cudaMemcpy(h_J, d_J, SIZE_3N, cudaMemcpyDeviceToHost);
        cudaMemcpy(h_A, d_A, SIZE_N, cudaMemcpyDeviceToHost);
        io::printField(h_J, "J", N, file);
        io::printField(h_A, "A", N, file);
    }

    // We print them anyway in case there is population imbalance
    cudaMemcpy(h_n_up_1, d_n_up_1, SIZE_N_REAL, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_n_down_1, d_n_down_1, SIZE_N_REAL, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_n_up_2, d_n_up_2, SIZE_N_REAL, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_n_down_2, d_n_down_2, SIZE_N_REAL, cudaMemcpyDeviceToHost);
    //cudaMemcpy(h_n_up_3, d_n_up_3, SIZE_N_REAL, cudaMemcpyDeviceToHost);
    //cudaMemcpy(h_n_down_3, d_n_down_3, SIZE_N_REAL, cudaMemcpyDeviceToHost);
    io::printField(h_n_up_1, "n_up_1", N, file);
    io::printField(h_n_up_2, "n_up_2", N, file);
    //io::printField(h_n_up_3, "n_up_3", N, file);
    io::printField(h_n_down_1, "n_down_1", N, file);
    io::printField(h_n_down_2, "n_down_2", N, file);
    //io::printField(h_n_down_3, "n_down_3", N, file);

    cudaMemcpy(h_F, d_F, SIZE_N_REAL, cudaMemcpyDeviceToHost);
    io::printField(h_F, "F", N, file);


    

    //Variables that are always declared
    cudaFree(d_neighbors);
    if(GAUGE_FIELD)
    {
        cudaFree(d_J);
        cudaFree(d_A);
        cudaFree(d_A_new);
    }
    

    cudaFree(d_F);

    cudaFree(d_D_1);
    cudaFree(d_Dnew_1);
    cudaFree(d_n_down_1);
    cudaFree(d_n_up_1);
    cudaFree(d_n_up_new_1);
    cudaFree(d_n_down_new_1);
    cudaFree(d_T_x_1);
    cudaFree(d_T_y_1);
    cudaFree(d_T_z_1);
    cudaFree(d_h_prev_1);
    cudaFree(d_h_n_1);
    cudaFree(d_e_prev_1);
    cudaFree(d_e_n_1);

    cudaFree(d_D_2);
    cudaFree(d_Dnew_2);
    cudaFree(d_n_down_2);
    cudaFree(d_n_up_2);
    cudaFree(d_n_up_new_2);
    cudaFree(d_n_down_new_2);
    cudaFree(d_T_x_2);
    cudaFree(d_T_y_2);
    cudaFree(d_T_z_2);
    cudaFree(d_h_prev_2);
    cudaFree(d_h_n_2);
    cudaFree(d_e_prev_2);
    cudaFree(d_e_n_2);
/*
    cudaFree(d_D_3);
    cudaFree(d_Dnew_3);
    cudaFree(d_n_down_3);
    cudaFree(d_n_up_3);
    cudaFree(d_n_up_new_3);
    cudaFree(d_n_down_new_3);
    cudaFree(d_T_x_3);
    cudaFree(d_T_y_3);
    cudaFree(d_T_z_3);
    cudaFree(d_h_prev_3);
    cudaFree(d_h_n_3);
    cudaFree(d_e_prev_3);
    cudaFree(d_e_n_3);

    */
    cudaFree(d_modulation);

}
