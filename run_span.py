##########   RUNNING INSTRUCTIONS   ###########
#
# python3 run.py compute  ==> Executes freefem
# python3 run.py plot     ==> Plots the result
#
###############################################
import os
from re import X
import shutil
import numpy as np
from dataclasses import dataclass
import argparse

parser = argparse.ArgumentParser()
# System size
parser.add_argument('-Nx','--Nx', help='x-size of the system', required=True)
parser.add_argument('-Ny','--Ny', help='y-size of the system', default="1")
# Initial guess
parser.add_argument('-ig', '--ig', help='Initial guess (LO/uniform/vortex)', default="uniform")
parser.add_argument('-nv', '--nv', help='Number of vortex(es) winding', default="1")
parser.add_argument('-x0', '--x0', help='x-position of the vortices [-1,1]', default="0.0")
parser.add_argument('-y0', '--y0', help='y-position of the vortices [-1,1]', default="0.0")
parser.add_argument('-nf1', '--nf1', help='Number of vortices in first component', default="1")
parser.add_argument('-nf2', '--nf2', help='Number of vortices in second component', default="0")
# Convergence parameters
parser.add_argument('-convergence', '--convergence', help="Gaps convergence value", default="1.0e-6")
parser.add_argument('-maxiter', '--maxiter', help='Maximal iterations', default="10000")
# Simulation features
parser.add_argument('-trsbdev', '--trsbdev', help='0: s+is; 1: s++-; -2: s+++', default="0.0")
parser.add_argument("-hartree", action="store_true", help="Hartree term activation", default=False)
parser.add_argument("-gauge", action="store_true", help="Gauge field term activation", default=False)
# Saving data
parser.add_argument('-folder', '--folder', help='Output saving folder', required=True)
parser.add_argument('-order', '--order', help='Polynomial expansion order', default="700")
# if ig="skyrmion"
parser.add_argument('-r', '--r', help='Radius of the skyrmion', default="0.4")
# if ig="skyrmionConcentric"
parser.add_argument('-n2', '--n2', help='Number of vortices in second circle', default="1")
parser.add_argument('-r2', '--r2', help='Radius of the second circle', default="0.25")

args = parser.parse_args()

Nx          = str(args.Nx)
Ny          = str(args.Ny)
IG 			= str(args.ig)
X0          = str(args.x0)
Y0          = str(args.y0)
FOLDER_NAME = str(args.folder)
CHEB_ORDER  = str(args.order)
NV          = str(args.nv)
CONVERGENCE = str(args.convergence)
MAX_ITER    = str(args.maxiter)
trsb_deviation = str(args.trsbdev)
HARTREE = str(int(args.hartree))
GAUGE_FIELD = str(int(args.gauge))
R              = str(args.r)
N2             = str(args.n2)
R2             = str(args.r2)
#W2             = str(args.w2)
Nf1            = str(args.nf1)
Nf2            = str(args.nf2)


################################
##### PARAMETERS SELECTION #####
################################

# SYSTEM PARAMETERS SETUP
#### Simulations with Mats critical temperature
T_c = 0.45974743
@dataclass
class Hamiltonian:
    #Convention [ band#1, band#2]
    t_x    = np.array( [-1.0,-1.0] )
    t_y    = np.array( [-1.0,-1.0] )
    t_z    = np.array( [-0.0,-0.0] )
    mu     = np.array( [ 0.0, 0.0] )
    H      = np.array( [ 0.0, 0.0] )
    V      = np.array( [ 2.0, 2.0] ) # v
    Vint   = np.array( [-0.02] ) # u
    #common parameters
    T      = 0.35* T_c
    q      = 0.15 * 3
    Bext   = 0.00

# SELECT HERE THE PARAMETERS TO SPAN OVER

p_NAME  = "Vint"
p_start = 0.0024
p_end   = 0.01
p_N     = 6

################################

hamiltonian = Hamiltonian()
p = np.linspace(p_start,p_end, p_N)
# Create simulation folder 
#p = np.array([60,80])
#p_N = len(p)

if not os.path.isdir("./simulations/" + FOLDER_NAME):
    os.mkdir("./simulations/" + FOLDER_NAME)
else:
    raise OSError(1, 'Directory alaready exists')


if os.path.isdir("./logs/" + FOLDER_NAME):
        shutil.rmtree("./logs/" + FOLDER_NAME)
os.mkdir("./logs/" + FOLDER_NAME)


for i in range(p_N):

    sim_name = p_NAME + "=" + f"{p[i]}"
    out_name = FOLDER_NAME + "/" + sim_name
    run_file = open(f"srunfile.sh", "w")

    
    ########## PARAMETER TO BE CHANGED ##########
    # hamiltonian.q =  p[i]
    # hamiltonian.V      = np.array( [ 3.0 - p[i] , 3.0 - p[i] , 3.0 - p[i] ] ) # v
    hamiltonian.Vint   = np.array( [-p[i]] ) # u
    # NV = str( int( p[i] ) )
    #Nx = str(p[i])
    #Ny = str(p[i])
    #############################################
    
    
    
    header_list = [ "#!/bin/bash -l",
                "#SBATCH -J " + out_name,
                "#SBATCH --kill-on-invalid-dep=yes",
                "#SBATCH --output=logs/"+out_name+".out",
                "#SBATCH --error=logs/"+out_name+".err",
                "#SBATCH --gres=gpu:1",
                "#SBATCH --time=48:00:00",
                "#SBATCH --qos=gpu",
                "#SBATCH --exclude=boltzmann,alpha",
                "#SBATCH --clusters=kraken",
                "#SBATCH --partition=gpu"
                ]

    run_list = ["srun ./build/Release/polynomial_bdg",
		"-Nx="        + Nx,
        "-Ny="        + Ny,
		"-output="    + out_name,
		"-ig="        + IG,
        "-order="     + CHEB_ORDER,
        "-iter="      + MAX_ITER,
        "-n="         + NV,
        "-x0="        + X0,
        "-y0="        + Y0,
        "-converged=" + CONVERGENCE,
        "-hartree="   + HARTREE,
        "-gauge="     + GAUGE_FIELD,
        "-t1x="       + str(hamiltonian.t_x[0]),
        "-t2x="       + str(hamiltonian.t_x[1]),
        "-t1y="       + str(hamiltonian.t_y[0]),
        "-t2y="       + str(hamiltonian.t_y[1]),
        "-t1z="       + str(hamiltonian.t_z[0]),
        "-t2z="       + str(hamiltonian.t_z[1]),
        "-mu1="       + str(hamiltonian.mu[0]),
        "-mu2="       + str(hamiltonian.mu[1]),
		"-H1="        + str(hamiltonian.H[0]),
		"-H2="        + str(hamiltonian.H[1]),
		"-V1="        + str(hamiltonian.V[0]),
		"-V2="        + str(hamiltonian.V[1]),
        "-V12="       + str(hamiltonian.Vint[0]),
		"-T="         + str(hamiltonian.T),
        "-Bext="      + str(hamiltonian.Bext),
        "-q="         + str(hamiltonian.q),
        "-trsbdev="   + str(trsb_deviation),
        "-r="         + R,
        "-r2="        + R2,
        "-n2="        + N2,
        "-nf1="       + Nf1,
        "-nf2="       + Nf2]
    
    # Writing the runfile
    run_file.writelines([ "\n".join(header_list) , "\n\n" , " ".join(run_list) ])
    run_file.close()
    # Submitting runfile
    os.system("sbatch srunfile.sh")
    # Deleting runfile
    os.remove("srunfile.sh")

    
							
