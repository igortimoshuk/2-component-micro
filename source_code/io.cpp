#include "io.h"
#include <iostream>
#include <h5pp/h5pp.h>
#include <typeinfo>

///// OUTPUT FUNCTION ////////

//Sets simulation output file
void io::setOutputFile(const bool HARTREE, const bool GAUGE_FIELD, std::string &out_file)
{

     // if the file already exists then creates one with (name)-1 and so on..
    h5pp::File file(out_file, h5pp::FilePermission::RENAME);

    // update the name of the current output file
    out_file = file.getFilePath();

    if(HARTREE) file.writeDataset((int) 1, "settings/hartree_settings");
    else file.writeDataset((int) 0, "settings/hartree_settings");


    if(GAUGE_FIELD) file.writeDataset((int) 1, "settings/gauge_settings");
    else file.writeDataset((int) 0, "settings/gauge_settings");

}

//Saves the geometry used in the simulation
void io::writeGeometry(const int *geometry, const int N, const std::string out_file )
{

    // if the file already exists then creates one with (name)-1 and so on..
    h5pp::File file(out_file, h5pp::FilePermission::READWRITE);
   
    // update the name of the current file
    file.writeDataset(geometry, "geometry", N);

}

//Saves the hamiltonian of the simulation
void io::writeHamiltonian(const Hamiltonian hamiltonian, const std::string out_file )
{

    // if the file already exists then creates one with (name)-1 and so on..
    h5pp::File file(out_file, h5pp::FilePermission::READWRITE);
    // update the name of the current file
    
    //Parameters are written not rescaled.
     
    file.writeDataset(hamiltonian.Nx,  "Nx");
    file.writeDataset(hamiltonian.Ny,  "Ny");
    file.writeDataset(hamiltonian.Nz,  "Nz");
    file.writeDataset(hamiltonian.t_x[0], "hamiltonian/t1x");
    file.writeDataset(hamiltonian.t_y[0], "hamiltonian/t1y");
    file.writeDataset(hamiltonian.t_z[0], "hamiltonian/t1z");
    file.writeDataset(hamiltonian.t_x[1], "hamiltonian/t2x");
    file.writeDataset(hamiltonian.t_y[1], "hamiltonian/t2y");
    file.writeDataset(hamiltonian.t_z[1], "hamiltonian/t2z");
   // file.writeDataset(hamiltonian.t_x[2], "hamiltonian/t3x");
   // file.writeDataset(hamiltonian.t_y[2], "hamiltonian/t3y");
   // file.writeDataset(hamiltonian.t_z[2], "hamiltonian/t3z");
    file.writeDataset(hamiltonian.V[0],   "hamiltonian/V1");
    file.writeDataset(hamiltonian.V[1],   "hamiltonian/V2");
   // file.writeDataset(hamiltonian.V[2],   "hamiltonian/V3");
    file.writeDataset(hamiltonian.H[0],   "hamiltonian/H1");
    file.writeDataset(hamiltonian.H[1],   "hamiltonian/H2");
   // file.writeDataset(hamiltonian.H[2],   "hamiltonian/H3");
    file.writeDataset(hamiltonian.mu[0],  "hamiltonian/mu1");
    file.writeDataset(hamiltonian.mu[1],  "hamiltonian/mu2");
   // file.writeDataset(hamiltonian.mu[2],  "hamiltonian/mu3");
    file.writeDataset(hamiltonian.V_int[0],   "hamiltonian/Vint12");
   // file.writeDataset(hamiltonian.V_int[1],   "hamiltonian/Vint13");
   // file.writeDataset(hamiltonian.V_int[2],   "hamiltonian/Vint23");
    file.writeDataset(hamiltonian.T,   "hamiltonian/T");
    file.writeDataset(hamiltonian.Emax,"hamiltonian/Emax");
    file.writeDataset(hamiltonian.Bext,"hamiltonian/Bext");
    file.writeDataset(hamiltonian.periodic,"hamiltonian/periodic");
    // Modified!!!
    file.writeDataset(-hamiltonian.q,    "hamiltonian/q");

    //Check if the code is working in float or double
    if(typeid(field) == typeid(float))  file.writeDataset("float", "settings/type");
    if(typeid(field) == typeid(double)) file.writeDataset("double", "settings/type");

}

void io::updateEmax(const Hamiltonian hamiltonian, const std::string out_file)
{
    
    //Write the value of Emax
    h5pp::File file(out_file, h5pp::FilePermission::READWRITE);
    file.writeDataset(hamiltonian.Emax,"hamiltonian/Emax");

}



/////////Printing single field/////////
void io::printField(const field3 *Z, std::string name, const int N, std::string out_file)
{
    h5pp::File file(out_file, h5pp::FilePermission::READWRITE);
    file.writeDataset(Z, name, N);     
}

void io::printField(const field2 *Z, std::string name, const int N, std::string out_file)
{
    h5pp::File file(out_file, h5pp::FilePermission::READWRITE);
    file.writeDataset(Z, name, N);     
}

void io::printField(const field *Z, std::string name, const int N, std::string out_file)
{
    h5pp::File file(out_file, h5pp::FilePermission::READWRITE);
    file.writeDataset(Z, name, N);     
}


