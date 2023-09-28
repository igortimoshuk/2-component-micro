# BdG solver through C
BDG Solver Through Chebyshev Polynomial Expansion

In this version it is possible to have self consistent solution to BdG equations including:
Hartree term
Self consistent gauge field
Self consistent Zeeman splitting  

### Prerequisites
```
cmake 3.24
nvcc 10
python 3.6.9
```

### CUDA-Settings
Default cuda compatibility:
```
 -arch_sm=70
 ```
To modify io go into CmakeLists.txt

### Compiling
Build using any of the provided CMake presets. Use `cmake --list-presets` to view the available build preests.
```
cmake --preset=release-native
```


### Running
Release mode:
```
./build/<preset>/polynomial_bdg
```

### Code settings
To disable self consistency for Hartree term, and maintain efficiency
it is necessary recompile the code. Default settings is ENABLED.
To disable Hartree recursion relations, go to CmakeLists.txt and set:
```
option(ENABLE_HARTREE     "Enables Hartree term compilation "  OFF)
```

To disable self consistency for the gauge field, and maintain efficiency
it is necessary recompile the code. Default settings is ENABLED.
To disable self coefficient recursion relations, go to CmakeLists.txt and set:
```
option(ENABLE_GAUGE    "Enables gauge term term compilation "  OFF)
```

To disable self consistency for the Zeeman splitting, and maintain efficiency
it is necessary recompile the code. Default settings is ENABLED.
To disable self coefficient recursion relations, go to CmakeLists.txt and set:
```
option(ENABLE_ZEEMAN   "Enables gauge term term compilation "  OFF)
```

Then recompile from scratch:


##  Input parameters
To assign value to parameters:
```
 -parameter=value. 
 ```
 If boolean:
 ```
  -parameter
  ```

### Run settings
```
-verbosity          : 0-6. 0-minimal info are printed.        Default 4.
-dev                : 1-2 Sets GPU.                           Default 2.
-iter               : Maximum number of iterations.           Default 10000.
-converged          : Convergence criterion.                  Default 1e-6.
-order              : Polynomial expansion order.             Default 400.
-partial            : Partial save level. 0-no partial saves.
                    1-only Delta is saved. 2-Everything.    Default 2.
-partialIter        : Sets every how many iterations to save. Default 15.
```


### Simulation settings
The default values are in the file "field_types.h". The same file contains the numeric precision
```
-output             : name of output file (the extension is added automatically). Default "test".
-input              : name of input file. If not entered, initial guess is selected. Default "".
-Nx                 : number of sites along x. Minimal value 1. 
-Ny                 : number of sites along y. Minimal value 1. 
-Nz                 : number of sites along z. Minimal value 1. 
-Rtx                : Hopping coefficient (Re part) along x.
-Itx                : Hopping coefficient (Im part) along x.
-Rty                : Hopping coefficient (Re part) along y.
-Ity                : Hopping coefficient (Im part) along y.
-Rtz                : Hopping coefficient (Re part) along z.
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

```
### Initial guess settings  
There are multiple options to initialize the system

If no ``` -input ``` is given:

```
-ig = FF; LO; vortex; random; LOV;
```
```LOV``` mean LO+vortex
if random the additional parameter ``` -n ``` specifies 
the number of vortices to add.

There are also two options for initializing the external magnetic field.
0) The default option is zero magnetic field in the sysem and ```Bext``` outside
1) If the flag ```-A1```is added then the magnetic field is initialized to uniform ```Bext``` throughout the entire system.

If ```-input``` is specified there are two options:

To continue a simulation with the same parameters:
```
-input= (path of input file to continue)

-input = (path to file) -change_parameters (optional -addVortex)
```
In this way the parameters used for the simulation come from the command line/default value:

The option ```-addVortex``` adds a vortex to the imported state.

### Plotting
To plots the data:
```
python3 post_processing.py -file={name of the file to plot}
```
Note: if the simulation is 3d the python code will output vtk files
to be opened using mayavi

### Rescaling
To be able to use Chebyshev formalism, one needs to ensure the eigenvalues of the hamiltonian to 
be withing [-1,1]. Hence we perform a rescaling according to Gershogorin theorem.
The output file contain the original parameters entered in the hamiltonian.
Also the rescaling factor is saved within hamiltonian/Emax.
The order parameter Delta and the current J are saved when still rescaled, hence it is necessary to 
rescale them back before visualization. This is performed in the ```-post_processing.py```. 
The gauge field A is not rescaled.