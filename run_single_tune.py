import os
import numpy as np
from dataclasses import dataclass
import argparse

#If the biggest eignevalue of the potential matrix is negative, 
#the algorithm needs to run with memory
def memory(V, Vint):
    V_mtrx = 0.5*np.diag(V)
    V_mtrx[0,1] = Vint[0]
    V_mtrx = V_mtrx + V_mtrx.transpose()

    w, v = np.linalg.eig(V_mtrx)
    if(np.max(w)<0.0): 
        print("Run with memory")
        return "1"
    else: 
        print("Run without memory")
        return "0"

##########        RUNNING SCRIPT      ##########

################################################


parser = argparse.ArgumentParser()
parser.add_argument('-name','--name', help='Name of the simulation', default="test")
parser.add_argument('-Nx','--Nx', help='x-size of the system', required=True)
parser.add_argument('-Ny','--Ny', help='y-size of the system', default="1")
parser.add_argument('-ig', '--ig', help='Initial guess (vortex/dw/dwv)', default="uniform")
parser.add_argument('-nv', '--nv', help='Number of vortices', default="1")
parser.add_argument('-nf1', '--nf1', help='Number of vortices in first component', default="1")
parser.add_argument('-nf2', '--nf2', help='Number of vortices in second component', default="0")
parser.add_argument('-order', '--order', help="Chebyshev expansion order", default=500)
parser.add_argument('-convergence', '--convergence', help="Gaps convergence value", default="1.0e-6")
parser.add_argument('-maxiter', '--maxiter', help='Maximal iterations', default="10000")
parser.add_argument('-trsbdev', '--trsbdev', help='0: s+is; 1 s++-; -2 s+++', default="0")
parser.add_argument("-hartree", action="store_true", help="Hartree term activation", default=False)
parser.add_argument("-gauge", action="store_true", help="Gauge field term activation", default=False)
parser.add_argument("-memory_par", "--memory_par", help="Update delta step", default="0.5")

# if ig="skyrmion"
parser.add_argument('-r', '--r', help='Radius of the skyrmion', default="0.4")
# if ig="skyrmionConcentric"
parser.add_argument('-n2', '--n2', help='Number of vortices in second circle', default="1")
parser.add_argument('-r2', '--r2', help='Radius of the second circle', default="0.1")
args = parser.parse_args()

SIM_NAME       = str(args.name)
Nx             = str(args.Nx)
Ny             = str(args.Ny)
IG 			   = str(args.ig)
NV             = str(args.nv)
CHEB_ORDER     = str(args.order) 
CONVERGENCE    = str(args.convergence)
MAX_ITER       = str(args.maxiter)
trsb_deviation = str(args.trsbdev)
HARTREE        = str(int(args.hartree))
GAUGE_FIELD    = str(int(args.gauge))
R              = str(args.r)
N2             = str(args.n2)
R2             = str(args.r2)
Nf1            = str(args.nf1)
Nf2            = str(args.nf2)
memory_par     = str(args.memory_par)



################################
##### PARAMETERS SELECTION #####
################################

#### Simulations with Mats critical temperature
T_c = 0.45974743
# SYSTEM PARAMETERS SETUP
@dataclass
class Hamiltonian:
    t_x    = np.array( [ 1.0, 1.0] )
    t_y    = np.array( [ 1.0, 1.0] )
    t_z    = np.array( [-0.0,-0.0] )
    mu     = np.array( [ 0.0, 0.0] )
    H      = np.array( [ 0.0, 0.0] )
    V      = np.array( [ 3.2, 2.2] ) # [V11,V22] v
    Vint   = np.array( [-0.05] ) # [V12] u
    #common parameters
    T      = 0.5*T_c
    q      = 0.25
    Bext   = 0.0
    


###############################
hamiltonian = Hamiltonian()
run_file = open("srunfile.sh", "w")

header_list = [ "#!/bin/bash -l",
                "#SBATCH -J " + SIM_NAME,
                "#SBATCH --kill-on-invalid-dep=yes",
                "#SBATCH --output=logs/cuda_"+SIM_NAME+".out",
                "#SBATCH --error=logs/cuda_"+SIM_NAME+".err",
                "#SBATCH --gres=gpu:1",
                "#SBATCH --time=72:00:00",
                "#SBATCH --qos=gpu",
                "#SBATCH --exclude=boltzmann,alpha",
#                "#SBATCH --nodelist=dirac",
                "#SBATCH --clusters=kraken",
                "#SBATCH --partition=gpu"]

run_list = ["./build/Release/polynomial_bdg",
		"-Nx="        + Nx,
        "-Ny="        + Ny,
		"-output="    + SIM_NAME,
		"-ig="        + IG,
        "-order="     + CHEB_ORDER,
        "-n="         + NV,
        "-converged=" + CONVERGENCE,
        "-iter="      + MAX_ITER,
        "-hartree="   + HARTREE,
        "-gauge="     + GAUGE_FIELD,
        "-memory="    + memory(hamiltonian.V, hamiltonian.Vint),
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
        "-nf2="       + Nf2, 
        "-memory_par=" + memory_par]

# Writing the runfile
run_file.writelines([ "\n".join(header_list) , "\n\n" , " ".join(run_list) ])
run_file.close()
# Submitting runfile
os.system("sbatch srunfile.sh")
# Deleting runfile
os.remove("srunfile.sh")
