#ifndef INPUT_OUTPUT_HEADER_GUARD
#define INPUT_OUTPUT_HEADER_GUARD
#include <stdio.h>
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <complex>
#include <iomanip>
#include <field_types.h>

namespace io{

//Sets simulation output file
void setOutputFile(const bool HARTREE, const bool GAUGE_FIELD, std::string &out_file);

//Exports the geometry of the simulation
extern void writeGeometry(const int *geometry, const int N, const std::string out_file );

extern void writeHamiltonian(const Hamiltonian hamiltonian, const std::string out_file );

extern void updateEmax(const Hamiltonian hamiltonian, const std::string out_file);

extern void printField(const field3 *Z, std::string name, const int N, std::string out_file);

extern void printField(const field2 *Z, std::string name, const int N, std::string out_file);

extern void printField(const field *Z, std::string name, const int N, std::string out_file);
}

#endif
