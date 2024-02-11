#include <string>
#include <math.h>
#include <stdio.h>
#include <iostream>
#include <complex>
#include "field_types.h"
#include "cuda_reduction.h"
#include "utilities.h"
#include "spdlog/spdlog.h"
#include "argparser.h"
#include "spdlog/sinks/basic_file_sink.h"
#include "chebyshev_self_consistent.h"
#include "io.h"
#include "initialization.h"
#define PI 3.14159265359
#define BILLION 1000000000L
#define MB 1048576L



int main(int argc,char* argv[])
{
    util::helper(argc, argv);
    ///////////////////////////////////////////////////////////////////////
    //////////////////////       INITIAL SETUP        /////////////////////
    ///////////////////////////////////////////////////////////////////////
    struct timespec start , stop ; // variables for timing
    double accum ; // elapsed time variable


    ////////////////////////// PRINTING SETTINGS //////////////////////////
    spdlog::set_pattern("%v");
    int levelZeroToSix = 5 - (int)input::parse_arg_i(argc, argv, "-verbosity", 3);
    auto lvlEnum = static_cast<spdlog::level::level_enum>(levelZeroToSix);
    spdlog::set_level(lvlEnum);
    //Levels: trace -> debug -> info -> warn -> error -> critical

    //////////////////////////// CHOOSING GPU /////////////////////////////

    int dev=0; //Set 1 for RTX2080Ti or 0 for TITANV
    dev = (int)( (int)input::parse_arg_i(argc, argv, "-dev", dev) %   3);
    cudaSetDevice(dev);
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, dev);

    ///////////////////////  SETTING OUTPUT FOLDER  ////////////////////////

    //Setting output folder PATH
    std::string out_file = "simulations/" +
                            input::parse_arg_s(argc, argv, "-output", "test") + ".h5";


    //CHECKING SELF CONSISTENT FIELDS
    const bool MEMORY = (bool)input::parse_arg_i(argc, argv, "-memory", 0);
    const bool HARTREE = (bool)input::parse_arg_i(argc, argv, "-hartree", 0);
    const bool GAUGE_FIELD   = (bool)input::parse_arg_i(argc, argv, "-gauge"  , 0);
    ///////////////////////////////////////////////////////////////////////
    //////////////////////     PARAMETERS LOADING     /////////////////////
    ///////////////////////////////////////////////////////////////////////

    //System variable declaration
    Hamiltonian hamiltonian;
    int    *h_geometry = nullptr;
    field2 *h_A        = nullptr;
    field3 *h_J        = nullptr;
    field  *h_F        = nullptr;

    // Component 1
    field2 *h_D_1 	   = nullptr;
    field  *h_n_up_1   = nullptr;
    field  *h_n_down_1 = nullptr;
    field2 *h_T_x_1    = nullptr;
    field2 *h_T_y_1    = nullptr;
    field2 *h_T_z_1    = nullptr;

    // Component 2 
    field2 *h_D_2 	   = nullptr;
    field  *h_n_up_2   = nullptr;
    field  *h_n_down_2 = nullptr;
    field2 *h_T_x_2    = nullptr;
    field2 *h_T_y_2    = nullptr;
    field2 *h_T_z_2    = nullptr;

    // Component 3
    /*
    field2 *h_D_3 	   = nullptr;
    field  *h_n_up_3   = nullptr;
    field  *h_n_down_3 = nullptr;
    field2 *h_T_x_3    = nullptr;
    field2 *h_T_y_3    = nullptr;
    field2 *h_T_z_3    = nullptr;
    */




    //CHECK IF THERE IS AN INPUT STATE
    std::string input_file = input::parse_arg_s(argc, argv, "-input", "");
    bool IMPORT_INITIAL_STATE = false;
    if(input_file != "") IMPORT_INITIAL_STATE = true;

    //CREATE OUTOUT FILE
    else io::setOutputFile(HARTREE, GAUGE_FIELD, out_file);
     

    //IMPORT INITIAL STATE IF REQUESTED
    if(IMPORT_INITIAL_STATE)
    {
        throw std::runtime_error("Initial guess import not impleted yet");
    } 

    
                                        
    //READ PARAMETERS FROM COMMAND LINE OTHERWISE
    if(!IMPORT_INITIAL_STATE)   util::inputParser(hamiltonian, out_file, argc,argv);


    //AT THIS POINT OF THE CODE THE HAMILTONIAN PARAMETERS ARE RESCALED. THE RESCALING CONSTANT IS SAVED IN Emax

    ///////////////////////////////////////////////////////////////////////
    //////////////////////    SIMULTATION SETTINGS    /////////////////////
    ///////////////////////////////////////////////////////////////////////

    //Get the total size
    const int N  = hamiltonian.Nx*hamiltonian.Ny*hamiltonian.Nz;

    //Set parameters for iterations and convergence
    const int   MAX_ITER_DEFAULT  = 100000;
    const int   MAX_ITER = (int)input::parse_arg_i(argc,argv,"-iter", MAX_ITER_DEFAULT);
    const float CONVERGED = max( (float)input::parse_arg_d(argc,argv,"-converged", 1.0e-8), 1.0e-9);

    // Pairing potential variation
    const float variation = (float)input::parse_arg_d(argc,argv,"-variation", 0.0);

    // Delta memory parameter
    const float memory_par = (float)input::parse_arg_d(argc,argv,"-memory_par", 0.0);

    //Set Chebyshev expansion order
    const int CHEB_ORDER_DEFAULT = 1000;
    const int CHEB_ORDER = (int) input::parse_arg_i(argc, argv, "-order", CHEB_ORDER_DEFAULT);

    //Set level and occurrence of partial save
    // PARTIAL_SAVE_LEVEL sets how much is saved of the partial resusts of the simulation
    // 0 - no partial data are saved
    // 1 - only Delta is saved
    // 2 - Delta, nUp, nDown are saved.
    // This can have a strong effect in the time needed form memCpy for big systems
    const bool PARTIAL_PRINT      =  input::parse_arg_bool(argc, argv,"-verbose", false) ;
    const int PARTIAL_SAVE_ITERATIONS = ( ( (int) input::parse_arg_i(argc, argv,"-partialIter", 15) ) % (MAX_ITER + 1));


    spdlog::info("{:=^44}", "");
    spdlog::info("{:=^44}", "    SIMULATIONS SETTINGS   ");
    spdlog::info("{:=^44}", "");
    spdlog::info("N SITES          ===>   {}",N);
    spdlog::info("MAX_ITER         ===>   {}",MAX_ITER);
    spdlog::info("EXPANSION ORDER  ===>   {}",CHEB_ORDER);
    spdlog::info("DEVICE: {} --- {}",dev, deviceProp.name);
    spdlog::info("VERBOSE PRINTING: {}", PARTIAL_PRINT);
    spdlog::info("PARTIAL SAVE EVERY  {}  ITERATIONS",PARTIAL_SAVE_ITERATIONS);
    spdlog::info("CONVERGENCE GAPS  ===> {}",CONVERGED);
    spdlog::info("POTENTIAL VARIATION ===> {}",variation);
    if(HARTREE) spdlog::info("{:=^44}", "SELF CONSISTENT HARTREE ENABLED");
    if(GAUGE_FIELD) spdlog::info("{:=^44}", "SELF CONSISTENT GAUGE ENABLED");
    
  
    /////////////////////  VARIABLES SIZE CALCULATION  ////////////////////
    const size_t SIZE_3N     = N*sizeof(field3);
    const size_t SIZE_N      = N*sizeof(field2);
    const size_t SIZE_N_REAL = N*sizeof(field);
    const size_t SIZE_2N     = 2*N*sizeof(field2);
    const int TPB = 512;
    const int X_BLOCKS = 1 + (N - 1)/TPB;
    const int Y_BLOCKS = util::gpuYblocks(SIZE_2N, SIZE_N, SIZE_N_REAL, SIZE_3N, dev, N, GAUGE_FIELD);
    const size_t SIZE_2N_XY =  SIZE_2N*Y_BLOCKS;


    spdlog::info("{:=^44}", "");
    spdlog::info("{:=^44}", "    BLOCK SETTINGS   ");
    spdlog::info("{:=^44}", "");
    spdlog::info("X BLOCKS  ===>   {}",X_BLOCKS);
    spdlog::info("TPB       ===>   {}",TPB);
    spdlog::info("Y BLOCKS  ===>   {}",Y_BLOCKS);

    // If initial state is not imported, allocate memory for simulation
    if(!IMPORT_INITIAL_STATE)
    {

        if(h_D_1        == nullptr) h_D_1 	         = (field2*)malloc(SIZE_N);
        if(h_n_up_1     == nullptr) h_n_up_1 	     = (field*)malloc(SIZE_N_REAL);
        if(h_n_down_1   == nullptr) h_n_down_1       = (field*)malloc(SIZE_N_REAL);

        if(h_D_2        == nullptr) h_D_2 	         = (field2*)malloc(SIZE_N);
        if(h_n_up_2     == nullptr) h_n_up_2 	     = (field*)malloc(SIZE_N_REAL);
        if(h_n_down_2   == nullptr) h_n_down_2       = (field*)malloc(SIZE_N_REAL);

        /*
        if(h_D_3        == nullptr) h_D_3 	         = (field2*)malloc(SIZE_N);
        if(h_n_up_3     == nullptr) h_n_up_3 	     = (field*)malloc(SIZE_N_REAL);
        if(h_n_down_3   == nullptr) h_n_down_3       = (field*)malloc(SIZE_N_REAL);
        */
       
        if(h_geometry == nullptr) h_geometry         = (int*)malloc(N*sizeof(int));
        if(h_A        == nullptr) h_A                = (field2*)malloc(SIZE_N);
        if(h_J        == nullptr) h_J                = (field3*)malloc(SIZE_3N);
        if(h_F        == nullptr) h_F                = (field*)malloc(SIZE_N_REAL);
        
        memset(h_A, 0, SIZE_N);
        memset(h_F, 0, SIZE_N_REAL);
        memset(h_J, 0, SIZE_3N);
        //Construct and save system geometry
        
        geometry::setGeometry(h_geometry, hamiltonian, out_file);
    
        //Generate initial guess
        if(input::parse_arg_s(argc, argv, "-ig", "uniform") == "dwfv")
        {
            init::setUniform(h_D_1, hamiltonian, 0.1, - PI / 2 );
            init::setUniform(h_D_2, hamiltonian, 0.1, PI / 2 );

            init::addDomainWall(h_D_2, hamiltonian);
            //init::addDomainWall(h_D_1, hamiltonian);

            init::addVortex(h_D_1, hamiltonian, 0.0, 0.0, 2.5, 1);
        }

        else if(input::parse_arg_s(argc, argv, "-ig", "uniform") == "ddwfv")
        {

            init::setUniform(h_D_1, hamiltonian, 0.1, - PI / 2 );
            init::setUniform(h_D_2, hamiltonian, 0.1, PI / 2 );

            init::addDomainWall(h_D_2, hamiltonian);
            init::addDomainWallV(h_D_2, hamiltonian);

            init::addVortex(h_D_1, hamiltonian, 0.0, 0.0, 2.5, 1);
        }

        else if(input::parse_arg_s(argc, argv, "-ig", "uniform") == "fv")
        {

            init::setUniform(h_D_1, hamiltonian, 0.1, - PI / 2 );
            init::setUniform(h_D_2, hamiltonian, 0.1, PI / 2 );

            init::addVortex(h_D_1, hamiltonian, 0.0, 0.0, 2.5, 1);
        }

        else if(input::parse_arg_s(argc, argv, "-ig", "uniform") == "fvortex")
        {
            int nf1 = (int)input::parse_arg_i(argc, argv, "-nf1", 1);
            int nf2 = (int)input::parse_arg_i(argc, argv, "-nf2", 1);

            init::setUniform(h_D_1, hamiltonian, 0.1, 0.0 );
            init::setUniform(h_D_2, hamiltonian, 0.1, 0.0 );            

            init::addVortex(h_D_1, hamiltonian, 0.0, 0.0, 0.05, nf1);
            init::addVortex(h_D_2, hamiltonian, 0.0, 0.0, 0.05, nf2);
        }

        else
        {
            const float trsbDev = input::parse_arg_d(argc,argv,"-trsbdev", 0.0);
            init::generateInitialguess(h_D_1, 0.1, 0.0, hamiltonian, argc, argv);
            init::generateInitialguess(h_D_2, 0.1, ( (2*PI/3) + PI/3 * trsbDev ), hamiltonian, argc, argv);
            //init::generateInitialguess(h_D_3, 0.1, ( (4*PI/3) - PI/3 * trsbDev ), hamiltonian, argc, argv);
        }
        
        

        io::printField(h_D_1, "delta_1_ig", N, out_file);
        io::printField(h_D_2, "delta_2_ig", N, out_file);
        //io::printField(h_D_3, "delta_3_ig", N, out_file);

        memset(h_n_up_1,0,SIZE_N_REAL);
        memset(h_n_down_1, 0, SIZE_N_REAL);

        memset(h_n_up_2,0,SIZE_N_REAL);
        memset(h_n_down_2, 0, SIZE_N_REAL);
 /*
        memset(h_n_up_3,0,SIZE_N_REAL);
        memset(h_n_down_3, 0, SIZE_N_REAL);
*/ 
      
        //Generate gauge field
        if(input::parse_arg_bool(argc, argv, "-A1", false))
            init::generateGauge(h_A, h_geometry, hamiltonian);

    }

    //THIS FUNCTION INITIALIZES THE HOPPING COEFFICIENTS
    init::generateHoppings(&h_T_x_1, &h_T_y_1, &h_T_z_1, h_A, h_geometry, hamiltonian, 0);
    init::generateHoppings(&h_T_x_2, &h_T_y_2, &h_T_z_2, h_A, h_geometry, hamiltonian, 1);
    //init::generateHoppings(&h_T_x_3, &h_T_y_3, &h_T_z_3, h_A, h_geometry, hamiltonian, 2);
    ///////////////////////////////////////////////////////////////////////
    //////////////////////         SIMULATION         /////////////////////
    ///////////////////////////////////////////////////////////////////////
   

    spdlog::info("{:=^44}", "");
    spdlog::info("{:=^44}", "    STARTING  SIMULATION    ");
    spdlog::info("{:=^44}", "");



    //Simulation timing
    int totalIter = 0;
    field convergenceDelta_1 = 1.0e7;
    field convergenceDelta_2 = 1.0e7;
    //field convergenceDelta_3 = 1.0e7;
    field convergenceNup_1   = 1.0e7;
    field convergenceNdown_1 = 1.0e7;
    field convergenceNup_2   = 1.0e7;
    field convergenceNdown_2 = 1.0e7;
    //field convergenceNup_3   = 1.0e7;
    //field convergenceNdown_3 = 1.0e7;
    field convergenceA = 1.0e7;
    clock_gettime ( CLOCK_REALTIME ,& start ); // start timer
   
    // // Iterations
    mean_field::selfConsistent(
        h_geometry, hamiltonian, h_A, h_J, h_F,
        h_D_1, h_n_up_1, h_n_down_1, h_T_x_1, h_T_y_1, h_T_z_1,
        h_D_2, h_n_up_2, h_n_down_2, h_T_x_2, h_T_y_2, h_T_z_2,
       // h_D_3, h_n_up_3, h_n_down_3, h_T_x_3, h_T_y_3, h_T_z_3,
        SIZE_N_REAL, SIZE_N, SIZE_3N, SIZE_2N_XY,
        X_BLOCKS, TPB, Y_BLOCKS,
        CHEB_ORDER, MAX_ITER, CONVERGED, memory_par,
        totalIter, convergenceDelta_1 , convergenceDelta_2, //convergenceDelta_3,
        convergenceNup_1, convergenceNdown_1, convergenceNup_2, convergenceNdown_2,
        //convergenceNup_3, convergenceNdown_3, 
        convergenceA,
        PARTIAL_PRINT, PARTIAL_SAVE_ITERATIONS, GAUGE_FIELD, HARTREE, MEMORY, out_file
    );



    spdlog::info("{:=^44}", "    END OF SIMULATION    ");
    spdlog::info("Convergence Delta 1 ===>   {:.4E}",convergenceDelta_1);
    spdlog::info("Convergence Delta 2 ===>   {:.4E}",convergenceDelta_2);
    //spdlog::info("Convergence Delta 3 ===>   {:.4E}",convergenceDelta_3);
    if(HARTREE){
    spdlog::info("Convergence N_up   ===>   {:.4E}",convergenceNup_1);
    spdlog::info("Convergence N_down ===>   {:.4E}",convergenceNdown_1);
    }
    if(GAUGE_FIELD){
    spdlog::info("Convergence A      ===>   {:.4E}",convergenceA);
    }
    spdlog::info("Total iterations   ===>   {}", totalIter);
    spdlog::info("{:=^44}", "");
    spdlog::info("{:=^44}", "    COPYING FIELDS...    ");



    spdlog::info("Saved in: {:}", out_file);
    spdlog::info("{:=^44}", "");


    clock_gettime( CLOCK_REALTIME ,&stop); // stop timer
    accum =( stop.tv_sec - start.tv_sec ) + (stop.tv_nsec - start.tv_nsec )/(double)BILLION;
    spdlog::info("Total time     ===>   {}s", accum);
    spdlog::info("Time/Iteration ===>   {}s", accum/totalIter);


    free(h_D_1);
    free(h_n_up_1);
    free(h_n_down_1);
    free(h_T_x_1);
    free(h_T_y_1);
    free(h_T_z_1);

    free(h_D_2);
    free(h_n_up_2);
    free(h_n_down_2);
    free(h_T_x_2);
    free(h_T_y_2);
    free(h_T_z_2);
/*
    free(h_D_3);
    free(h_n_up_3);
    free(h_n_down_3);
    free(h_T_x_3);
    free(h_T_y_3);
    free(h_T_z_3);
*/
    free(h_A);
    free(h_geometry);
    free(h_J);
    free(h_F);
}


