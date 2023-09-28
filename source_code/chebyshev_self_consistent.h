#ifndef CHEB_FUNCTION_GUARD_NEW
#define CHEB_FUNCTION_GUARD_NEW
#include "field_types.h"
#include <string>
#define PI 3.14159265359 

namespace mean_field{

extern void selfConsistent(
    const int *h_geometry, Hamiltonian &hamiltonian, field2 *h_A, field3 *h_J, field *h_F,
    field2 *h_D_1, field *h_n_up_1, field *h_n_down_1, field2 *h_T_x_1, field2 *h_T_y_1, field2 *h_T_z_1,
    field2 *h_D_2, field *h_n_up_2, field *h_n_down_2, field2 *h_T_x_2, field2 *h_T_y_2, field2 *h_T_z_2,
    //field2 *h_D_3, field *h_n_up_3, field *h_n_down_3, field2 *h_T_x_3, field2 *h_T_y_3, field2 *h_T_z_3,
    const size_t SIZE_N_REAL, const size_t SIZE_N, const size_t SIZE_3N, const size_t SIZE_2N_XY,
    const int X_BLOCKS, const int TPB, const int Y_BLOCKS,
    const int CHEB_ORDER, const int MAX_ITER, const float CONVERGED,
    int &totalIter, field &convergenceDelta_1, field &convergenceDelta_2, //field &convergenceDelta_3,
    field &convergenceNup_1, field &convergenceNdown_1, field &convergenceNup_2, field &convergenceNdown_2, 
    //field &convergenceNup_3, field &convergenceNdown_3, 
    field &convergenceA,
    const bool PARTIAL_PRINT, const int PARTIAL_SAVE_ITERATIONS, const bool GAUGE_FIELD, const bool HARTREE, const bool MEMORY,
    std::string file);

}


#endif