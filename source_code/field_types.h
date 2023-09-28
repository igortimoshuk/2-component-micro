#ifndef FIELD_TYPES_GUARD
#define FIELD_TYPES_GUARD


template<typename T>
struct scalar2{
    T x,y;
};

template<typename T>
struct scalar3{
    T x,y,z;
};
//////////////////////////////////////////////////////////////////////
/////  CHANGE ONLY THIS LINE TO CHANGE SIMULATION TYPE TO DOUBLE /////

using field  = float;

//////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////

using field2 = scalar2<field>;
using field3 = scalar3<field>;


struct Hamiltonian
{
    int Nx = 200;
    int Ny = 1;
    int Nz = 1;

    //Default parameters. Convention {comp1, comp2, comp3}
    field2 t_x[2]         = { {-1.0, 0.0}, {-1.0, 0.0}};//, {-1.0, 0.0} };
    field2 t_y[2]         = { {-1.0, 0.0}, {-1.0, 0.0}};//, {-1.0, 0.0} };
    field2 t_z[2]         = { {-0.0, 0.0}, {-0.0, 0.0}};//, {-0.0, 0.0} };
    field mu[2]           = { 0.5, 0.5};//, 0.5 };
    field H[2]            = { 0.0, 0.0};//, 0.0 };
    field V[2]            = { 3.0, 3.0};//, 3.0 };
    
                          //{ V12, V13, V23 }
    field V_int[2]        = { -0.5, -0.5};//, -0.5 };
    field q               = 0.0;
    field T               = 0.1;
    field3 Bext           = {0.,0.,0.0};
    scalar3<int> periodic = {0,0,0};
    field Emax            = 1.0;

};

#endif
