#include "utilities.h"
#include <math.h>
#include "argparser.h"
#include <sys/stat.h>
#include <sys/types.h>
#include <spdlog/spdlog.h>
#include "io.h"
#define MB 1048576L


///////////////////////     AUXILIARY FUNCTIONS     ////////////////////


field modCx(field2 &z)
{
    return sqrt((z.x*z.x + z.y*z.y));
}

field modCx1(field2 z)
{
    return sqrt((z.x*z.x + z.y*z.y));
}

////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////     NAMESPACE UTIL     ////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////
//Summarizes input parameters
void util::helper(int argc, char* argv[])
{
    if( (bool) input::parse_arg_bool(argc, argv, "-h",    false) ||
        (bool) input::parse_arg_bool(argc, argv, "-help", false) ||
        (bool) input::parse_arg_bool(argc, argv, "--help",false))
    {
        std::cout << R"(

        ///////   BDG SOLVER THROUGH CHEBYSHEV POLYNOMIAL EXPANSION   ///////

        To assign value to parameters -parameter=value. If boolean -parameter
        ======    Code settings    =====:
        To enable self consistency for Hartree term, and maintain efficiency
        it is necessary recompile the code. Default settings is ENABLED.
        To disable Hartree recursion relations, go to CmakeLists.txt and set
        at the top of the file ENABLE_HARTREE to OFF.
        Then recompile using ./build.sh -c

        ///////         LIST OF POSSIBLE INPUT PARAMETERS             ///////

        ======    Run settings    =====:

        -verbosity          : 0-6. 0-minimal info are printed.        Default 4.
        -dev                : 1-2 Sets GPU.                           Default 2.
        -iter               : Maximum number of iterations.           Default 10000.
        -converged          : Convergence criterion.                  Default 1e-6.
        -order              : Polynomial expansion order.             Default 400.
        -partial            : Partial save level. 0-no partial saves.
                            1-only Delta is saved. 2-Everything.    Default 2.
        -partialIter        : Sets every how many iterations to save. Default 15.




        ======Simulation settings======:

        DEFAULT VALUES ARE IN THE FILE "field_types.h". The same file contains the numeric precision

        -output             : name of output file (the extension is added automatically). Default "test".
        -input              : name of input file. If not entered, initial guess is selected. Default "".
        -Nx                 : number of sites along x. Minimal value 1. 
        -Ny                 : number of sites along y. Minimal value 1. 
        -Nz                 : number of sites along z. Minimal value 1. 
        -tx                 : Hopping coefficient (Re part) along x.
        -Itx                : Hopping coefficient (Im part) along x.
        -ty                 : Hopping coefficient (Re part) along y.
        -Ity                : Hopping coefficient (Im part) along y.
        -tz                 : Hopping coefficient (Re part) along z.
        -Itz                : Hopping coefficient (Im part) along z.
        -mu                 : Chemical potential. 
        -V                  : (minus) Interaction potential. 
        -T                  : Temperature. 
        -H                  : Zeeman field. 
        -Bext               : External magnetic field.
        -q                  : Electric charge.
        -periodicx          : Periodic boundary conditions along x. 
        -periodicy          : Periodic boundary conditions along x. 
        -periodicz          : Periodic boundary conditions along x.


        ======Initial guess settings======:

        GAUGE FIELD INITIAL GUESS:

        -A1: sets initial uniform B field in the sample to 0.5*Bext
             without it the initial guage field is set to zero 
        
        THERE ARE MULTIPLE OPTION TO OBTAIN AN INITIAL GUESS
        
        IF NO -input IS GIVEN:

        -ig = FF; LO; vortex; random;
        
        if random the additional parameter -n specifies 
        the number of vortices to add.

        IF -input IS SPECIFIED THERE ARE TWO OPTIONS

        To continue a simulation with the same parameters
        -input= (path of input file to continue)

        -input = (path to file) -change_parameters (optional -addVortex)
        
        IN THIS WAY THE PARAMETERS USED FOR THE SIMULATION
        COME FROM THE COMMAND LINE/DEFAULT VALUES.
        THE OPTION -addVortex adds a vortex to the imported state


        ======Plotting settings======:
        run: python3 post_processing.py -file={name of the file to plot}
        Note: if the simulation is 3d the python code will output vtk files
        to be opened using mayavi




        )" << std::endl;

        exit(0);
    }

}



// THIS FUNCTION NEEDS UPDATE SINCE WE DECLARE MANY MORE VARIABLES
// Calculates how many Y blocks one can utilize given the GPU free memory and the system size.
int util::gpuYblocks(const size_t SIZE_2N, const size_t SIZE_N, const size_t SIZE_N_REAL, const size_t SIZE_3N, int dev, const int N, const bool GAUGE_FIELD){

    //Variables saved on GPU for single site:
    // d_Q_n, d_Q_prev --> size each 2N*sizeof(field2)
    // d_D, d_Dnew --> size each N*sizeof(field2)

    size_t titanFreeMemory, titanTotalMemory;
	//Obtaining information on TITAN V
	cudaSetDevice(dev);
    cudaMemGetInfo(&titanFreeMemory, &titanTotalMemory);

    size_t usage = ( 12*SIZE_2N );

    size_t safety = 500*MB;
    size_t deltaSize = 20* SIZE_N + 18*SIZE_N_REAL;
    if(GAUGE_FIELD) deltaSize += SIZE_3N + 2*N*sizeof(int);

    size_t ourFreeMemory = titanFreeMemory -  deltaSize - safety;

    

    int expectedYblocks = (int)( ourFreeMemory / usage);

	spdlog::info("{:=^44}", "");
	spdlog::info("GPU Initial Available memory: {}",ourFreeMemory/MB);
	spdlog::info("Of total memory: {}",titanTotalMemory/MB);
    spdlog::info("{:=^44}", "");


    if(expectedYblocks > N) expectedYblocks = N;

    while( ourFreeMemory < (expectedYblocks*usage))
    {
        expectedYblocks -= 1;
        if(expectedYblocks < 0) throw std::runtime_error("Negative Y blocks!");
		spdlog::warn("{:=^44}", "");
		spdlog::warn("Reducing blocks --> : {}",expectedYblocks);
    	spdlog::warn("{:=^44}", "");

    }

	spdlog::info("{:=^44}", "");
	spdlog::info("Available blocks: {}",expectedYblocks);
	spdlog::info("Expected remaining memory: {}",( ourFreeMemory - expectedYblocks*usage)/MB);
	spdlog::info("{:=^44}", "");

    return expectedYblocks;
}


//Rescale coefficient to have energies within [-1,1]
void util::rescaleCoefficient(Hamiltonian &hamiltonian, const std::string out_file)
{

    field max_mu = max( abs(hamiltonian.mu[0]), abs(hamiltonian.mu[1]));//, abs(hamiltonian.mu[2]));
    field max_H  = max( abs(hamiltonian.H[0]), abs(hamiltonian.H[1])) ;//, abs(hamiltonian.H[2] ));
    field max_V  = max( abs(hamiltonian.V[0]), abs(hamiltonian.V[1]));//, abs(hamiltonian.V[2] ));

    field max_V_int = abs(hamiltonian.V_int[0]);//, abs(hamiltonian.V_int[2] ));

    field max_tx_mod = max(modCx(hamiltonian.t_x[0]),modCx(hamiltonian.t_x[1]));//, modCx(hamiltonian.t_x[2]));
    field max_ty_mod = max(modCx(hamiltonian.t_y[0]),modCx(hamiltonian.t_y[1]));//, modCx(hamiltonian.t_y[2]));
    field max_tz_mod = max(modCx(hamiltonian.t_z[0]),modCx(hamiltonian.t_z[1]));//, modCx(hamiltonian.t_z[2]));


	hamiltonian.Emax  = 2*max_tx_mod*(hamiltonian.Nx > 1) +
				  		2*max_ty_mod*(hamiltonian.Ny > 1) +
				  		2*max_tz_mod*(hamiltonian.Nz > 1) +
				  		max_mu + max_H + max_V + max_V_int + 4;



    hamiltonian.Emax *= 2;

    for(int i=0; i<2; i++)

    {
        if(hamiltonian.Nx > 1) hamiltonian.t_x[i] = { hamiltonian.t_x[i].x/hamiltonian.Emax, hamiltonian.t_x[i].y/hamiltonian.Emax};
        if(hamiltonian.Ny > 1) hamiltonian.t_y[i] = { hamiltonian.t_y[i].x/hamiltonian.Emax, hamiltonian.t_y[i].y/hamiltonian.Emax};
        if(hamiltonian.Nz > 1) hamiltonian.t_z[i] = { hamiltonian.t_z[i].x/hamiltonian.Emax, hamiltonian.t_z[i].y/hamiltonian.Emax};

        hamiltonian.V[i]   =hamiltonian.V[i]/hamiltonian.Emax;
        hamiltonian.V_int[i]   =hamiltonian.V_int[i]/hamiltonian.Emax;
        hamiltonian.mu[i]  =hamiltonian.mu[i]/hamiltonian.Emax;
        hamiltonian.H[i]   =hamiltonian.H[i]/hamiltonian.Emax;
    }   

	spdlog::info("{:=^44}", "    RESCALING FACTOR   ");
	spdlog::info("Emax = {:.4f}", hamiltonian.Emax);
    spdlog::info("{:=^44}", "");

    io::updateEmax(hamiltonian, out_file);
    

}


void util::inputParser(Hamiltonian &hamiltonian,
					std::string out_file, int argc,char* argv[])
{
	hamiltonian.Nx    		= (int)  input::parse_arg_i(argc, argv, "-Nx",   hamiltonian.Nx);
	hamiltonian.Ny    		= (int)  input::parse_arg_i(argc, argv, "-Ny",   hamiltonian.Ny);
    hamiltonian.Nz    		= (int)  input::parse_arg_i(argc, argv, "-Nz",   hamiltonian.Nz);
    
    // Component 1
    hamiltonian.t_x[0].x    = (field)input::parse_arg_d(argc, argv, "-t1x",  hamiltonian.t_x[0].x);
    hamiltonian.t_y[0].x    = (field)input::parse_arg_d(argc, argv, "-t1y",  hamiltonian.t_y[0].x);
    hamiltonian.t_z[0].x    = (field)input::parse_arg_d(argc, argv, "-t1z",  hamiltonian.t_z[0].x);
    hamiltonian.mu[0]  		= (field)input::parse_arg_d(argc, argv, "-mu1",  hamiltonian.mu[0]);
    hamiltonian.V[0] 		= (field)input::parse_arg_d(argc, argv, "-V1",   hamiltonian.V[0]);
    hamiltonian.H[0]  	  	= (field)input::parse_arg_d(argc, argv, "-H1",   hamiltonian.H[0]);

    // Component 2
	hamiltonian.t_x[1].x    = (field)input::parse_arg_d(argc, argv, "-t2x",  hamiltonian.t_x[1].x);
	hamiltonian.t_y[1].x    = (field)input::parse_arg_d(argc, argv, "-t2y",  hamiltonian.t_y[1].x);
    hamiltonian.t_z[1].x    = (field)input::parse_arg_d(argc, argv, "-t2z",  hamiltonian.t_z[1].x);
    hamiltonian.mu[1]  		= (field)input::parse_arg_d(argc, argv, "-mu2",  hamiltonian.mu[1]);
    hamiltonian.V[1]    	= (field)input::parse_arg_d(argc, argv, "-V2",   hamiltonian.V[1]);
    hamiltonian.H[1]  	  	= (field)input::parse_arg_d(argc, argv, "-H2",   hamiltonian.H[1]);
/*
    // Component 3
    hamiltonian.t_x[2].x    = (field)input::parse_arg_d(argc, argv, "-t3x",  hamiltonian.t_x[2].x);
	hamiltonian.t_y[2].x    = (field)input::parse_arg_d(argc, argv, "-t3y",  hamiltonian.t_y[2].x);
    hamiltonian.t_z[2].x    = (field)input::parse_arg_d(argc, argv, "-t3z",  hamiltonian.t_z[2].x);
    hamiltonian.mu[2]  		= (field)input::parse_arg_d(argc, argv, "-mu3",  hamiltonian.mu[2]);
    hamiltonian.V[2]    	= (field)input::parse_arg_d(argc, argv, "-V3",   hamiltonian.V[2]);
    hamiltonian.H[2]  	  	= (field)input::parse_arg_d(argc, argv, "-H3",   hamiltonian.H[2]);
*/

    hamiltonian.V_int[0]   	= (field)input::parse_arg_d(argc, argv, "-V12", hamiltonian.V_int[0]);
//   hamiltonian.V_int[1]   	= (field)input::parse_arg_d(argc, argv, "-V13", hamiltonian.V_int[1]);
 //   hamiltonian.V_int[2]   	= (field)input::parse_arg_d(argc, argv, "-V23", hamiltonian.V_int[2]);

	hamiltonian.T  	  		= (field)input::parse_arg_d(argc, argv, "-T",    hamiltonian.T);
    hamiltonian.Bext.z      = (field)input::parse_arg_d(argc, argv, "-Bext", hamiltonian.Bext.z);
    hamiltonian.q           = -(field)input::parse_arg_d(argc, argv, "-q",    hamiltonian.q);
	hamiltonian.periodic.x  = (int)  input::parse_arg_bool(argc, argv, "-periodicx",(bool) hamiltonian.periodic.x);
	hamiltonian.periodic.y  = (int)  input::parse_arg_bool(argc, argv, "-periodicy",(bool) hamiltonian.periodic.y);
	hamiltonian.periodic.z  = (int)  input::parse_arg_bool(argc, argv, "-periodicz",(bool) hamiltonian.periodic.z);

	spdlog::info("{:=^44}", "");
    spdlog::info("{:=^44}", "    SYSTEM PARAMETERS   ");
    spdlog::info("Bext.z =  {:.4f}", hamiltonian.Bext.z);
    spdlog::info("q      =  {:.4f}", hamiltonian.q);
    spdlog::info("T      =  {:.4f}", hamiltonian.T);
    spdlog::info("V_12   =  {:.4f}", hamiltonian.V_int[0]);
 //   spdlog::info("V_13   =  {:.4f}", hamiltonian.V_int[1]);
 //   spdlog::info("V_23   =  {:.4f}", hamiltonian.V_int[2]);
    spdlog::info("{:=^44}", "    Component 1  ");
    spdlog::info("{:=^44}", "");
	spdlog::info("t1_x    =  {:.4f}, {:.4f}",hamiltonian.t_x[0].x,hamiltonian.t_x[0].y);
	spdlog::info("t1_y    =  {:.4f}, {:.4f}",hamiltonian.t_y[0].x,hamiltonian.t_y[0].y);
	spdlog::info("t1_z    =  {:.4f}, {:.4f}",hamiltonian.t_z[0].x,hamiltonian.t_z[0].y);
	spdlog::info("mu1     =  {:.4f}", hamiltonian.mu[0]);
	spdlog::info("V1      =  {:.4f}", hamiltonian.V[0]);
    spdlog::info("H1      =  {:.4f}", hamiltonian.H[0]);
    spdlog::info("{:=^44}", "    Component 2  ");
    spdlog::info("{:=^44}", "");
	spdlog::info("t2_x    =  {:.4f}, {:.4f}",hamiltonian.t_x[1].x, hamiltonian.t_x[1].y);
	spdlog::info("t2_y    =  {:.4f}, {:.4f}",hamiltonian.t_y[1].x, hamiltonian.t_y[1].y);
	spdlog::info("t2_z    =  {:.4f}, {:.4f}",hamiltonian.t_z[1].x, hamiltonian.t_z[1].y);
	spdlog::info("mu2     =  {:.4f}", hamiltonian.mu[1]);
	spdlog::info("V2      =  {:.4f}", hamiltonian.V[1]);
    spdlog::info("H2      =  {:.4f}", hamiltonian.H[1]);
    /*
    spdlog::info("{:=^44}", "    Component 2  ");
    spdlog::info("{:=^44}", "");
	spdlog::info("t2_x    =  {:.4f}, {:.4f}",hamiltonian.t_x[2].x, hamiltonian.t_x[2].y);
	spdlog::info("t2_y    =  {:.4f}, {:.4f}",hamiltonian.t_y[2].x, hamiltonian.t_y[2].y);
	spdlog::info("t2_z    =  {:.4f}, {:.4f}",hamiltonian.t_z[2].x, hamiltonian.t_z[2].y);
	spdlog::info("mu2     =  {:.4f}", hamiltonian.mu[2]);
	spdlog::info("V2      =  {:.4f}", hamiltonian.V[2]);
    spdlog::info("H2      =  {:.4f}", hamiltonian.H[2]);
    */
    



	if(hamiltonian.periodic.x) spdlog::info("x-periodic");
	if(hamiltonian.periodic.y) spdlog::info("y-periodic");
	if(hamiltonian.periodic.z) spdlog::info("z-periodic");

	spdlog::info("Saving in: {:}", out_file);
    spdlog::info("{:=^44}", "");

    io::writeHamiltonian(hamiltonian, out_file);

    //Rescale the coefficients of the hamiltonian before starting the simulation
    util::rescaleCoefficient(hamiltonian, out_file);

}


//Rescaling the results back
void util::rescaleInverse(field2 *D, const Hamiltonian hamiltonian)
{
	const int N = hamiltonian.Nx*hamiltonian.Ny*hamiltonian.Nz;


    //Rescale delta
	for(int i = 0; i < N; ++i)
	{
	D[i].x *= hamiltonian.Emax;
    D[i].y *= hamiltonian.Emax;

    }




}

void util::rescaleInverse(field3 *D, const Hamiltonian hamiltonian)
{
	const int N = hamiltonian.Nx*hamiltonian.Ny*hamiltonian.Nz;


    //Rescale delta
	for(int i = 0; i < N; ++i)
	{
	D[i].x *= hamiltonian.Emax;
    D[i].y *= hamiltonian.Emax;
    D[i].z *= hamiltonian.Emax;

    }


}



////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////     NAMESPACE ARRAY     ////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////
__global__
void array::init_gpu(field2 *A, int N, field2 value)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if(idx >= N) return;

    A[idx]     = {value.x,value.y};
}

__global__
void array::init_gpu(field *A, int N, field value)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if(idx >= N) return;

    A[idx]     = value;
}

void array::swap(field2 ** a_ref, field2 ** b_ref)
{
	field2 *addr;
	addr = *a_ref;
    *a_ref = *b_ref;
    *b_ref = addr;
}

void array::swap(field ** a_ref, field ** b_ref)
{
	field *addr;
	addr = *a_ref;
    *a_ref = *b_ref;
    *b_ref = addr;
}


////////////////////////////////////////////////////////////////////////////////////////////////////


////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////     NAMESPACE GEOMETRY   /////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////

    //        CONVENTION        //
    //setting n = which_neighbor//
    //         n = 0 => x - 1         //
    //         n = 1 => x + 1         //
    //         n = 2 => y - 1         //
    //         n = 3 => y + 1         //
    //         n = 4 => z - 1         //
    //         n = 5 => z + 1         //
    //  n in (6-11) are for periodic  //


//To check if a value is neighbor or not
// return ( (neighbor_list >> which_neighbor) & 1 );



__global__
void writeNeighborsTable(int *neighbor, const Hamiltonian hamiltonian){
    int rx,ry,rz;
    uint3 isHoppingUp, isHoppingDown;

    int idx = threadIdx.x + blockDim.x*blockIdx.x;

    int Nx = hamiltonian.Nx;
    int Ny = hamiltonian.Ny;
    int Nz = hamiltonian.Nz;

    if(idx >= Nx*Ny*Nz) return;

    rx = (idx % Nx);
    ry = (idx / Nx) % Ny;
    rz = (idx / (Ny * Nx) );
    neighbor[idx] = 0;    //No hopping at all

    //Setting boundaries
    isHoppingDown.x = rx > 0;
    isHoppingDown.y = ry > 0;
    isHoppingDown.z = rz > 0;

    isHoppingUp.x = rx < Nx - 1;
    isHoppingUp.y = ry < Ny - 1;
    isHoppingUp.z = rz < Nz - 1;


    //        CONVENTION        //
    //setting n = which_neighbor//
    //      n = 0 => x - 1      //
    //      n = 1 => x + 1      //
    //      n = 2 => y - 1      //
    //      n = 3 => y + 1      //
    //      n = 4 => z - 1      //
    //      n = 5 => z + 1      //

    //Nearest neighbor in x
    if(isHoppingDown.x) neighbor[idx] += 1<<0;     //(2^0)
    if(isHoppingUp.x)   neighbor[idx] += 1<<1;     //(2^1)

    //Newarest neighbor in y
    if(isHoppingDown.y) neighbor[idx] += 1<<2;     //(2^2)
    if(isHoppingUp.y)   neighbor[idx] += 1<<3;     //(2^3)

    //Nearest neighbor in z
    if(isHoppingDown.z) neighbor[idx] += 1<<4;     //(2^4)
    if(isHoppingUp.z)   neighbor[idx] += 1<<5;     //(2^5)

    //Check for periodic boundaries
    if( hamiltonian.periodic.x && ( rx == Nx - 1 ) )
    {
        neighbor[idx] += 1<<6;    //Toggle periodic hopping
        neighbor[idx] += 1<<1;    // Allow hopping up through boundary

    }
    if( hamiltonian.periodic.x && ( rx == 0  ) )
    {
        neighbor[idx] += 1<<7;      //Toggle periodic hopping
        neighbor[idx] += 1<<0;      //Allow hopping in that direction

    }


    if( hamiltonian.periodic.y && ( ry == Ny - 1 ) )
    {
        neighbor[idx]  += 1<<8;    //Toggle periodic hopping yUp
        neighbor[idx]  += 1<<3;     //Allow hopping in that direction

    }

    if( hamiltonian.periodic.y && ( ry == 0  ) )
    {
        neighbor[idx] += 1<<9;      //Toggle periodic hopping yDown
        neighbor[idx] += 1<<2;      //allow hopping in that direction

    }

    if( hamiltonian.periodic.z && ( rz == Nz - 1 ) )
    {
        neighbor[idx] += 1<<10;   //Toggle periodic hoppinh zUp
        neighbor[idx] += 1<<5;    //Allow hopping
    }

    if( hamiltonian.periodic.z && ( rz == 0  ) )
    {
        neighbor[idx] += 1<<11;     //Toggle peiodic hopping zDown
        neighbor[idx] += 1<<4;      //Allow hopping
    }



}


void geometry::setGeometry(int *h_geometry, const Hamiltonian &hamiltonian, std::string out_file){

	//Rectangular geometries are built on GPU
	//Other geometries might be imported

	spdlog::info("{:=^44}", "");
    spdlog::info("{:=^44}", "    BUILDING GEOMETRY  ");
	spdlog::info("{:=^44}", "");

	const int N = hamiltonian.Nx*hamiltonian.Ny*hamiltonian.Nz;
	int   *d_neighbors   = nullptr;
	cudaMalloc(&d_neighbors,  N*sizeof(int));

	//GPU calculation
	const int TPB = 512;
	const int X_BLOCKS = 1 + (N - 1)/TPB;

	writeNeighborsTable<<< X_BLOCKS, TPB >>>(d_neighbors, hamiltonian);

	cudaMemcpy(h_geometry, d_neighbors, N*sizeof(int), cudaMemcpyDeviceToHost);
    cudaFree(d_neighbors);

    //write the geometry file
    io::writeGeometry(h_geometry, N, out_file);



}
