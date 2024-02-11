#include "initialization.h"
#include "utilities.h"
#include "argparser.h"
#define PI 3.14159265359

//x0 and y0 need to be within -1 and 1
void init::addVortex(field2 *Initial_guess, const Hamiltonian &hamiltonian, field x0, field y0, field xi, int n)
{
    const int Nx = hamiltonian.Nx;
    const int Ny = hamiltonian.Ny;
    const int Nz = hamiltonian.Nz;

    field mod_psi, theta;
    int index;
    field x,y,z;
    field2 psi_old;
    for(int k=0; k<Nz; ++k)
    {
        for(int j=0; j<Ny; ++j)
        {
		    for(int i=0; i<Nx; ++i)
            {

			x = (field) 2*( i - Nx/2 )/Nx;
            y = (field) 2*( j - Ny/2 )/Ny;
            z = (field) 2*( k - Nz/2 )/Nz;
            //Our index convention
            index = i + j*Nx + k*Nx*Ny;
            //Put a for loop here if one wants more vortices

            mod_psi = tanh( sqrt( pow((x-x0)/xi,2) + pow((y-y0)/xi,2) ) );
            theta   = (field) ( n*atan2( (y-y0), (x-x0) ) );
            psi_old = Initial_guess[index];
            
            Initial_guess[index].x = mod_psi * ( psi_old.x * cos(theta) - psi_old.y * sin(theta) );
            Initial_guess[index].y = mod_psi * ( psi_old.x * sin(theta) + psi_old.y * cos(theta) );
           
            }
        }   
    }
}

//This function initializes the field to uniform, and must be always used. For more complicated guesses add modulation or vortices
void init::setUniform(field2 *Initial_guess, const Hamiltonian &hamiltonian, const field amplitude, const field phase)
{
    const int Nx = hamiltonian.Nx;
    const int Ny = hamiltonian.Ny;
    const int Nz = hamiltonian.Nz;

    int index;
    field x,y,z;
    for(int k=0; k<Nz; ++k)
    {
        for(int j=0; j<Ny; ++j)
        {
		    for(int i=0; i<Nx; ++i)
            {

			x = (field) 2*( i - Nx/2 )/Nx;
            y = (field) 2*( j - Ny/2 )/Ny;
            z = (field) 2*( k - Nz/2 )/Nz;
            //Our index convention
            index = i + j*Nx + k*Nx*Ny;
            //Put a for loop here if one wants more vortices

            Initial_guess[index].x = amplitude*cos(phase);
            Initial_guess[index].y = amplitude*sin(phase);
           
            }
        }   
    }

}

//this function must be used after initializing the field to uniform.
void addModulation(field2 *Initial_guess, const Hamiltonian &hamiltonian, std::string modType = "LO")
{


    const int Nx = hamiltonian.Nx;
    const int Ny = hamiltonian.Ny;
    const int Nz = hamiltonian.Nz;

    //Obtain amplitude value from first element of ther field 
    const field amplitude = Initial_guess[0].x;

    field x,y,z;
    int index;
    field kx,ky;

    std::cout << amplitude << std::endl;
    for(int k=0; k<Nz; ++k)
    {
        for(int j=0; j<Ny; ++j)
        {
		    for(int i=0; i<Nx; ++i)
            {

			x = (field) 2*( i - Nx/2 )/Nx;
            y = (field) 2*( j - Ny/2 )/Ny;
            z = (field) 2*( k - Nz/2 )/Nz;
            //Our index convention
            index = i + j*Nx + k*Nx*Ny;
            //Put a for loop here if one wants more vortices
            if(modType == "LO")
            {
                kx = 2 * PI * ( 1 );
                ky = 2 * PI * ( 1 );
                //const
                Initial_guess[index].x = amplitude * cos( kx * x + ky *  y );
                Initial_guess[index].y = 0.0;

            }

            if(modType == "FF")
            {
                kx = 2 * PI * ( 1 );
                ky = 2 * PI * ( 1 );
                //const
                Initial_guess[index].x = Initial_guess[index].x * cos( kx * x + ky *  y );
                Initial_guess[index].y = amplitude * sin( kx * x + ky *  y );

            }
           
            }
        }   
    }
}

void addCircularSkyrmion(field2 *Initial_guess, field r, int N_vortices, int n, const Hamiltonian &hamiltonian, 
field dwWidth = 0.05, field x_offset = 0.0, field y_offset = 0.0)
{
    const int Nx = hamiltonian.Nx;
    const int Ny = hamiltonian.Ny;
    const int Nz = hamiltonian.Nz;

    int index;
    field x,y,z;
    field x0, y0;
    field dTheta = (field) 2*PI / N_vortices;
    

    for(int k=0; k<Nz; ++k)
    {
        for(int j=0; j<Ny; ++j)
        {
		    for(int i=0; i<Nx; ++i)
            {

			x = (field) 2*( i - Nx/2 )/Nx;
            y = (field) 2*( j - Ny/2 )/Ny;
            z = (field) 2*( k - Nz/2 )/Nz;
            //Our index convention
            index = i + j*Nx + k*Nx*Ny;
           
            // Circular domain wall creation
            if( sqrt( ( x - x_offset ) * ( x - x_offset ) + ( y - y_offset ) * ( y - y_offset ) ) < ( r  + dwWidth/2 ) ) Initial_guess[index].y *= -1.0;

            }
        }   
    }
    // Adding vortices
    for(int m=0; m<N_vortices; ++m)
    {  
                x0 = ( r  + dwWidth/2 ) * cos( m * dTheta ) + x_offset;
                y0 = ( r  + dwWidth/2 ) * sin( m * dTheta ) + y_offset;
                init::addVortex(Initial_guess, hamiltonian, x0,y0, dwWidth, n);
    }
}

void init::addDomainWall(field2 *Initial_guess, const Hamiltonian &hamiltonian)
{
    const int Nx = hamiltonian.Nx;
    const int Ny = hamiltonian.Ny;
    const int Nz = hamiltonian.Nz;

    int index;
    field x,y,z;
    for(int k=0; k<Nz; ++k)
    {
        for(int j=0; j<Ny; ++j)
        {
		    for(int i=0; i<Nx; ++i)
            {

			x = (field) 2*( i - Nx/2 )/Nx;
            y = (field) 2*( j - Ny/2 )/Ny;
            z = (field) 2*( k - Nz/2 )/Nz;
            //Our index convention
            index = i + j*Nx + k*Nx*Ny;

            // smarter way to initialise a domain wall
            //if(y<0) Initial_guess[index].x *= -1.0;
            if(y<0) Initial_guess[index].y *= -1.0;
            
            }
        }   
    }
}

void init::addDomainWallV(field2 *Initial_guess, const Hamiltonian &hamiltonian)
{
    const int Nx = hamiltonian.Nx;
    const int Ny = hamiltonian.Ny;
    const int Nz = hamiltonian.Nz;

    int index;
    field x,y,z;
    for(int k=0; k<Nz; ++k)
    {
        for(int j=0; j<Ny; ++j)
        {
		    for(int i=0; i<Nx; ++i)
            {

			x = (field) 2*( i - Nx/2 )/Nx;
            y = (field) 2*( j - Ny/2 )/Ny;
            z = (field) 2*( k - Nz/2 )/Nz;
            //Our index convention
            index = i + j*Nx + k*Nx*Ny;

            // smarter way to initialise a domain wall
            if(x<0.05) Initial_guess[index].x *= -1.0;
            if(x<0.05) Initial_guess[index].y *= -1.0;
            
            }
        }   
    }
}

// Function generating initial guess from scratches
void init::generateInitialguess(field2 *Initial_guess, const field amplitude, const field phase, const Hamiltonian &hamiltonian, int argc, char* argv[])
{

    [[maybe_unused]] field x,y,z,x0,y0,z0;

    const int Nx = hamiltonian.Nx;
    const int Ny = hamiltonian.Ny;
    const int Nz = hamiltonian.Nz;

    
    
    const field xi = 0.05;
    srand((unsigned) time(0));

    //Always start by setting field to uniform
    init::setUniform(Initial_guess, hamiltonian, amplitude, phase);

    std::string ig = input::parse_arg_s(argc, argv, "-ig", "uniform");

    x0 = (field) input::parse_arg_d(argc, argv, "-x0", 0.0);
    y0 = (field) input::parse_arg_d(argc, argv, "-y0", 0.0);
    z0 = 0.0;

    //Initialize single vortex in the system
    int n = (int)input::parse_arg_i(argc, argv, "-n", 1);

    if(ig == "vortex") init::addVortex(Initial_guess, hamiltonian, x0,y0, xi, n);


    if(ig == "dw") init::addDomainWall(Initial_guess, hamiltonian);

    if(ig == "dwv")
    {
        init::addDomainWall(Initial_guess, hamiltonian);
        init::addVortex(Initial_guess, hamiltonian, x0,y0, xi, n);
    }

    if(ig == "skyrmion")
    {
        field r = (field) input::parse_arg_d(argc, argv, "-r", 0.4);
        addCircularSkyrmion(Initial_guess, r, n , 1, hamiltonian, xi);
    }

    if(ig == "skyrmionConcentric")
    {
        field r = (field) input::parse_arg_d(argc, argv, "-r", 0.4);
        field r2 = (field) input::parse_arg_d(argc, argv, "-r2", 0.2);
        int w2 = (field) input::parse_arg_i(argc, argv, "-w2", 1);
        int n2 = (field) input::parse_arg_i(argc, argv, "-n2", 1);

        addCircularSkyrmion(Initial_guess, r, n , 1, hamiltonian, xi);
        addCircularSkyrmion(Initial_guess, r2, n2 , w2, hamiltonian, xi);
    }

    if(ig == "skyrmionVortex")
    {
        field r = (field) input::parse_arg_d(argc, argv, "-r", 0.5);
        int n2 = (field) input::parse_arg_i(argc, argv, "-n2", 1);
        
        addCircularSkyrmion(Initial_guess, r, n , 1, hamiltonian, xi);
        init::addVortex(Initial_guess, hamiltonian, x0,y0, xi, n2);
        
    }

    //Initialize with LO modulation
    if(ig == "LO" or ig == "lo") addModulation(Initial_guess,hamiltonian, "LO");

    //Initialize with FF modulation
    if(ig == "FF" or ig == "ff") addModulation(Initial_guess, hamiltonian, "FF");

    // add n vortices in random position
    if(ig == "random")
    {
        int border_distance_x = 15;
        int border_distance_y = (int) ( (field) border_distance_x * ((field) Ny) / ((field) Nx) );

        for(int i = 0; i < n; ++i)
        {
            x0 = (field) 2*( ( rand() % ( Nx - border_distance_x )) - (Nx - border_distance_x)/2 ) /Nx;
            y0 = (field) 2*( ( rand() % ( Ny - border_distance_y )) - (Ny - border_distance_y)/2 ) /Ny;

            addVortex(Initial_guess, hamiltonian, x0, y0, xi, 1);
        }

    }


}



//Must be called after coefficient rescaling. Sets hopping coefficients!!!
void init::generateHoppings(field2 **h_T_x, field2 **h_T_y, field2 **h_T_z, const field2 *h_A, const int *neighbor_list, Hamiltonian hamiltonian, int component)
{
    const int N = hamiltonian.Nx*hamiltonian.Ny*hamiltonian.Nz;
    const size_t SIZE_N      = N*sizeof(field2);

    if(*h_T_x        == nullptr) *h_T_x 	             = (field2*)malloc(SIZE_N);
    if(*h_T_y        == nullptr) *h_T_y 	             = (field2*)malloc(SIZE_N);
    if(*h_T_z        == nullptr) *h_T_z 	             = (field2*)malloc(SIZE_N);
    

    int index;
    int current_neighbors;
    int3 isHoppingUp;

	for(int k=0; k<hamiltonian.Nz; ++k){
        for(int j=0; j<hamiltonian.Ny; ++j){
		    for(int i=0; i<hamiltonian.Nx; ++i){

		
            //Our index convention
            index = i + j*hamiltonian.Nx + k*hamiltonian.Nx*hamiltonian.Ny;


            current_neighbors = neighbor_list[index];

            isHoppingUp.x = (current_neighbors >> 1) & 1;
            isHoppingUp.y = (current_neighbors >> 3) & 1;
            isHoppingUp.z = (current_neighbors >> 5) & 1;

            (*h_T_x)[index].x = hamiltonian.t_x[component].x * cos( hamiltonian.q * h_A[index].x ) * isHoppingUp.x;
            (*h_T_x)[index].y = hamiltonian.t_x[component].x * sin( hamiltonian.q * h_A[index].x ) * isHoppingUp.x;

            (*h_T_y)[index].x = hamiltonian.t_y[component].x * cos( hamiltonian.q * h_A[index].y ) * isHoppingUp.y;
            (*h_T_y)[index].y = hamiltonian.t_y[component].x * sin( hamiltonian.q * h_A[index].y ) * isHoppingUp.y;

            (*h_T_z)[index].x = hamiltonian.t_z[component].x*isHoppingUp.z;
            (*h_T_z)[index].y = 0.0;

		    }
        }
    }


    //Throw an error if external magnetic field coexhists with periodic boundary conditions
    if(hamiltonian.periodic.x or hamiltonian.periodic.y or hamiltonian.periodic.z)
    {
        if(hamiltonian.Bext.x !=0 or hamiltonian.Bext.y !=0 or hamiltonian.Bext.z !=0)
        {
            throw std::runtime_error("Periodic boundary conditions with external magnetic field are not allowed");
        }
    }


}

//Sets the external gague field
void init::generateGauge(field2 *h_A, const int *neighbor_list, const Hamiltonian &hamiltonian)
{
    const int N = hamiltonian.Nx*hamiltonian.Ny*hamiltonian.Nz;
    int index;
    int current_neighbors;
    int3 isHoppingUp;
	for(int k=0; k<hamiltonian.Nz; ++k){
        for(int j=0; j<hamiltonian.Ny; ++j){
		    for(int i=0; i<hamiltonian.Nx; ++i){
            //Our index convention
            index = i + j*hamiltonian.Nx + k*hamiltonian.Nx*hamiltonian.Ny;
            current_neighbors = neighbor_list[index];

            isHoppingUp.x = (current_neighbors >> 1) & 1;
            isHoppingUp.y = (current_neighbors >> 3) & 1;
            // isHoppingUp.z = (current_neighbors >> 5) & 1;

            h_A[index].x = - 0.5 * j * hamiltonian.Bext.z*isHoppingUp.x;
            h_A[index].y =   0.5 * i * hamiltonian.Bext.z*isHoppingUp.y;

		    }
        }
    }
}

