#ifndef INITIALIZATION_HEADER_GUARD
#define INITIALIZATION_HEADER_GUARD

#include "field_types.h"
#include "io.h"
#include "argparser.h"
#include "spdlog/spdlog.h"
#include <stdio.h>
#include <stdlib.h>
#include <ctime>

namespace init
{


extern void generateInitialguess(field2 *Initial_guess, const field amplitude, const field phase, const Hamiltonian &hamiltonian, int argc,char* argv[]);

extern void importInitialguess(field2 **h_D, field **h_n_up, field **h_n_down, field3 **h_J, field2 **h_A, field **h_F, int **h_geometry,
                        const std::string in_file, std::string &out_file, Hamiltonian &hamiltonian, int component=1);

extern void generateHoppings(field2 **h_T_x, field2 **h_T_y, field2 **h_T_z, const field2 *h_A, const int *neighbor_list, Hamiltonian hamiltonian, int component);

extern void generateGauge(field2 *h_A, const int *neighbor_list, const Hamiltonian &hamiltonian);

extern void importState2(field2 **h_D, field **h_n_up, field **h_n_down, field3 **h_J, field2 **h_A, field **h_F, int **h_geometry,
                       const std::string in_file, std::string &out_file, Hamiltonian &hamiltonian, int argc, char* argv[]);

extern void addVortex(field2 *Initial_guess, const Hamiltonian &hamiltonian, field x0, field y0, field xi, int n);
extern void addDomainWall(field2 *Initial_guess, const Hamiltonian &hamiltonian);
extern void addDomainWallV(field2 *Initial_guess, const Hamiltonian &hamiltonian);
extern void setUniform(field2 *Initial_guess, const Hamiltonian &hamiltonian, const field amplitude, const field phase);

}
#endif