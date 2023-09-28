#!/usr/bin/env python3
import h5py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from dataclasses import dataclass
import matplotlib.colorbar
from mpl_toolkits.mplot3d import Axes3D
import shutil
import os
import matplotlib.gridspec as gridspec
import threading
# from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
# from mpl_toolkits.axes_grid1.colorbar import colorbar
import operator
import re

#Plotting parameters
params = {
   'font.family' : 'STIXGeneral',
   'mathtext.fontset': 'stix',
   'axes.labelsize': 10,
   'legend.fontsize': 9,
   'xtick.labelsize': 10,
   'ytick.labelsize': 10,
   'text.usetex': True,
   'figure.figsize': [3.44, 3.44],
   'axes.grid' : False
   }

plt.rcParams.update(params)
plt.close('all')


@dataclass
class Hamiltonian:

    t_x    = np.array([-1.0, -1.0, -1.0], dtype=complex)
    t_y    = np.array([-1.0, -1.0, -1.0], dtype=complex)
    t_z    = np.array([-0.0, -0.0, 0.0], dtype=complex)
    mu     = np.array([-0.0, -0.00, 0.0])
    H      = np.array([-0.0, -0.00, 0.0])
    V      = np.zeros((3,3))
    T      = 0.3
    q      = 0.1
    Bext   = 0.0  
    
    Emax   = 1

class dataProcessing:

    def __init__(self, folder_path, verbose=False):
        self.folder_path = folder_path
        self.input_file = None
        self.verbose = verbose

        self.ourTypeComplex = np.dtype(np.complex64)
        self.ourTypeReal    = np.dtype(np.float32)
        self.hamiltonian    = Hamiltonian()
        self.Nx             = 1 
        self.Ny             = 1
        self.Nz             = 1
        self.N              = 1
        self.neighbors      = None
        self.isHartree      = True
        self.isGauge        = True
        self.dimensionality  = 1

        # Fields
        self.D_1              = None 
        self.D_2              = None 
        self.D_3              = None 
        self.nUp_1            = None
        self.nUp_2            = None 
        self.nUp_3            = None 
        self.nDown_1          = None 
        self.nDown_2          = None 
        self.nDown_3          = None 
        self.Jx               = None 
        self.Jy               = None 
        self.Jz               = None
        self.B                = None
        self.Ax               = None
        self.Ay               = None
        self.FreeEnergy       = None
        self.FreeEnergyDensity= None
        self.f1               = None
        self.f2               = None
        self.D_1_ig                = None
        self.D_2_ig                = None
        self.D_3_ig                = None        
          
       
        # To be removed
        self.Zeeman         = None
        self.V_inter        = None
        self.V_intra        = None

        # Plotting parameters
        self.xlim           = None
        self.ylim           = None
        self.dpi            = 200

        # Spanning parameters
        self.p1 = [] #T
        self.list_names = [] #List of the names of the file
        self.p1_NAME = ""
      
        self.importandsort()
        

     

        
# Following functions lists all the files and sorts as a function of the spanned quantity
    def importandsort(self):
        self.list_names = os.listdir()
        if "tmp.h5" in self.list_names: self.list_names.remove("tmp.h5")
        self.p1_NAME = self.list_names[0].strip().split("=")[0]
        self.p1 = [float(i.strip().split("=")[1].replace(".h5","")) for i in self.list_names]

        self.p1 = np.array(self.p1)
        self.list_names = np.array(self.list_names)
        sort_idx = np.array(self.p1).argsort()
        
        self.p1 = self.p1[sort_idx]
        self.list_names = self.list_names[sort_idx]

        print(f"SPAN PARAMETER ==> {self.p1_NAME}")
        print(f"{self.p1_NAME} VALUES:")
        print(self.p1)
        print(f"RELATIVE FILES")
        print(self.list_names)

  

        
    def importSimData(self, p1_index , verbose=True):
        self.input_file = self.list_names[p1_index]
        self.open_inputFile()
        self.checkType()
        self.importSettings()
        self.importGeometry()
        self.checkDimensions()
        self.importHamiltonian()
        self.importFields()
        self.importFreeEnergy()

        if(self.dimesionality == 2):
            self.reshapeFields()
        self.close_inputFile()


#############  Previous functions readapted  ################
    # Input file managing functions
# Input file managing functions
    def open_inputFile(self):
        shutil.copy(self.input_file, "tmp.h5")
        self.file  = h5py.File("tmp.h5", 'r')
   
    def close_inputFile(self):
        self.file.close()
        os.remove("tmp.h5")
    

    #Importing functions
    def checkType(self):
        
        if(self.file['settings/type'][()].decode("utf-8") == 'float'):
            if self.verbose: print("SIMULATION TYPE ==> FLOAT")
            self.ourTypeComplex = np.dtype(np.complex64)
            self.ourTypeReal    = np.dtype(np.float32)

        if(self.file['settings/type'][()].decode("utf-8") == 'double'):
            if self.verbose: print("SIMULATION TYPE ==> DOUBLE")
            self.ourTypeComplex = np.dtype(np.complex128)
            self.ourTypeReal    = np.dtype(np.float64)

    def importSettings(self):
        self.isHartree = bool(np.asarray(self.file['settings/hartree_settings'][()].view(dtype=np.int32)))
        self.isGauge   = bool(np.asarray(self.file['settings/gauge_settings'][()].view(dtype=np.int32)))
        if self.verbose: print(f"SELF CONSISTENT [GAUGE, HARTREE]  ==> {[self.isGauge,self.isHartree]}")
    
    def importGeometry(self):
        if self.verbose: print("Importing geometry...")
        self.neighbors = np.asarray(self.file['geometry'][()].view(dtype=np.int32))
        self.Nx = np.asarray(self.file['Nx'][()].view(dtype=np.int32))
        self.Ny = np.asarray(self.file['Ny'][()].view(dtype=np.int32))
        self.Nz = np.asarray(self.file['Nz'][()].view(dtype=np.int32))
        self.xlim = self.Nx
        self.ylim = self.Ny

    def importHamiltonian(self):
        if self.verbose: print("Importing Hamiltonian...")
        # Component 1
        self.hamiltonian.t_x[0] = np.asarray(self.file['hamiltonian/t1x'][()].view(dtype=self.ourTypeComplex))
        self.hamiltonian.t_y[0] = np.asarray(self.file['hamiltonian/t1y'][()].view(dtype=self.ourTypeComplex))
        self.hamiltonian.t_z[0] = np.asarray(self.file['hamiltonian/t1z'][()].view(dtype=self.ourTypeComplex))
        self.hamiltonian.mu[0]  = np.asarray(self.file['hamiltonian/mu1'][()].view(dtype=self.ourTypeReal))
        self.hamiltonian.H[0]   = np.asarray(self.file['hamiltonian/H1'][()].view(dtype=self.ourTypeReal))
        # Component 2
        self.hamiltonian.t_x[1] = np.asarray(self.file['hamiltonian/t2x'][()].view(dtype=self.ourTypeComplex))
        self.hamiltonian.t_y[1] = np.asarray(self.file['hamiltonian/t2y'][()].view(dtype=self.ourTypeComplex))
        self.hamiltonian.t_z[1] = np.asarray(self.file['hamiltonian/t2z'][()].view(dtype=self.ourTypeComplex))
        self.hamiltonian.H[1]   = np.asarray(self.file['hamiltonian/H2'][()].view(dtype=self.ourTypeReal))
        # Component 3
        self.hamiltonian.t_x[2] = np.asarray(self.file['hamiltonian/t3x'][()].view(dtype=self.ourTypeComplex))
        self.hamiltonian.t_y[2] = np.asarray(self.file['hamiltonian/t3y'][()].view(dtype=self.ourTypeComplex))
        self.hamiltonian.t_z[2] = np.asarray(self.file['hamiltonian/t3z'][()].view(dtype=self.ourTypeComplex))
        self.hamiltonian.H[2]   = np.asarray(self.file['hamiltonian/H3'][()].view(dtype=self.ourTypeReal))

        self.hamiltonian.T   = np.asarray(self.file['hamiltonian/T'][()].view(dtype=self.ourTypeReal))
        self.hamiltonian.Emax=  np.asarray(self.file['hamiltonian/Emax'][()].view(dtype=self.ourTypeReal))
        self.hamiltonian.q   =  np.asarray(self.file['hamiltonian/q'][()].view(dtype=self.ourTypeReal))
        self.hamiltonian.Bext=  np.asarray(self.file['hamiltonian/Bext'][()])
        
        # The diagonal elements have a 0.5 factor because to build the 
        # symmeric matrix we sum with the transpose
        self.hamiltonian.V = np.zeros((3,3))
        self.hamiltonian.V[0,0] = 0.5 * np.asarray(self.file['hamiltonian/V1'][()].view(dtype=self.ourTypeReal))
        self.hamiltonian.V[1,1] = 0.5 * np.asarray(self.file['hamiltonian/V2'][()].view(dtype=self.ourTypeReal))
        self.hamiltonian.V[2,2] = 0.5 * np.asarray(self.file['hamiltonian/V3'][()].view(dtype=self.ourTypeReal))
        self.hamiltonian.V[0,1] = np.asarray(self.file['hamiltonian/Vint12'][()].view(dtype=self.ourTypeReal))
        self.hamiltonian.V[0,2] = np.asarray(self.file['hamiltonian/Vint13'][()].view(dtype=self.ourTypeReal))
        self.hamiltonian.V[1,2] = np.asarray(self.file['hamiltonian/Vint23'][()].view(dtype=self.ourTypeReal))
        self.hamiltonian.V = ( self.hamiltonian.V + self.hamiltonian.V.transpose() ) 
    
    def importFields(self):
        if self.verbose: print("Importing Fields...")
        self.D_1  =  self.hamiltonian.Emax * np.asarray(self.file['delta_1'][()].view(dtype=self.ourTypeComplex))
        self.D_2  =  self.hamiltonian.Emax * np.asarray(self.file['delta_2'][()].view(dtype=self.ourTypeComplex))
        self.D_3  =  self.hamiltonian.Emax * np.asarray(self.file['delta_3'][()].view(dtype=self.ourTypeComplex))
        self.D_1_ig =  self.hamiltonian.Emax * np.asarray(self.file['delta_1_ig'][()].view(dtype=self.ourTypeComplex))
        self.D_2_ig =  self.hamiltonian.Emax * np.asarray(self.file['delta_2_ig'][()].view(dtype=self.ourTypeComplex))
        self.D_3_ig =  self.hamiltonian.Emax * np.asarray(self.file['delta_3_ig'][()].view(dtype=self.ourTypeComplex))
    
        
        if(self.isHartree):
            self.nUp_1   = np.asarray(self.file['n_up_1'][()].view(dtype=self.ourTypeReal))
            self.nUp_2   = np.asarray(self.file['n_up_2'][()].view(dtype=self.ourTypeReal))
            self.nUp_3   = np.asarray(self.file['n_up_3'][()].view(dtype=self.ourTypeReal))
            self.nDown_1 = np.asarray(self.file['n_down_1'][()].view(dtype=self.ourTypeReal))
            self.nDown_2 = np.asarray(self.file['n_down_2'][()].view(dtype=self.ourTypeReal))
            self.nDown_3 = np.asarray(self.file['n_down_3'][()].view(dtype=self.ourTypeReal))

        if(self.isGauge): 
            Jtot = self.hamiltonian.Emax * np.asarray(self.file['J'][()].view(dtype=self.ourTypeReal))
            self.Jx = Jtot[0::3]
            self.Jy = Jtot[1::3]
            self.Jz = Jtot[2::3]

            A = np.asarray(self.file['A'][()].view(dtype=self.ourTypeReal))
            self.Ax = A[0::2]
            self.Ay = A[1::2]
        
    def importFreeEnergy(self):
      
        self.f1 =  np.asarray(self.file['F'][()].view(dtype=self.ourTypeReal))     
        self.f2 = np.zeros( len(self.f1) )

        try:
            V_inv = np.linalg.inv(self.hamiltonian.V)
            for i in range( len(self.f1) ):
                deltas = np.array([self.D_1[i], self.D_2[i], self.D_3[i]])
                self.f2[i] = np.real(np.dot(np.conj(deltas), np.dot(V_inv, deltas)))
        except:
            print("Singular")
            eval, evec = np.linalg.eigh(self.hamiltonian.V)

            for i in range(len(eval)):
                e  = eval[i]
                ev = evec[:,i] 
                
                if np.abs(e) > 1.0e-3:
                    for j in range(self.Nx * self.Ny):          
                        deltas = np.array([self.D_1[j], self.D_2[j], self.D_3[j]])
                        proj = np.dot( np.conj(deltas), ev ) 
                        self.f2[j] += np.abs(proj)**2 / e 

                        
        if(self.dimesionality == 2):
            self.f1 = self.f1.reshape(self.Ny, self.Nx)
            self.f2 = self.f2.reshape(self.Ny, self.Nx)
        
        self.FreeEnergyDensity = self.f1 + self.f2
        self.FreeEnergy = np.sum(self.FreeEnergyDensity)
        
        if self.verbose: print("Free energy imported")
       

    def reshapeFields(self):
        if self.verbose: print("Reshaping Fields...")
        if(self.Nz == 1 ):
            self.D_1      = self.D_1.reshape(self.Ny, self.Nx)
            self.D_2      = self.D_2.reshape(self.Ny, self.Nx)
            self.D_3      = self.D_3.reshape(self.Ny, self.Nx)
            self.D_1_ig   = self.D_1_ig.reshape(self.Ny, self.Nx)
            self.D_2_ig   = self.D_2_ig.reshape(self.Ny, self.Nx)
            self.D_3_ig   = self.D_3_ig.reshape(self.Ny, self.Nx)
           

            if(self.isHartree):
                self.nUp_1   = self.nUp_1.reshape(self.Ny, self.Nx)
                self.nUp_2   = self.nUp_2.reshape(self.Ny, self.Nx)
                self.nUp_3   = self.nUp_3.reshape(self.Ny, self.Nx)

                self.nDown_1 = self.nDown_1.reshape(self.Ny, self.Nx)
                self.nDown_2 = self.nDown_2.reshape(self.Ny, self.Nx)
                self.nDown_3 = self.nDown_3.reshape(self.Ny, self.Nx)

            if(self.isGauge):
                self.Jx     = self.Jx.reshape(self.Ny, self.Nx)
                self.Jy     = self.Jy.reshape(self.Ny, self.Nx)
                self.Ax     = self.Ax.reshape(self.Ny, self.Nx)
                self.Ay     = self.Ay.reshape(self.Ny, self.Nx)
                self.B = self.curl(self.Ax, self.Ay)

        else:
            raise ValueError('2d simulations must have Nz=1!')

    def checkDimensions(self):
        dims = np.array([self.Nx, self.Ny, self.Nz])
        is1d = sum(dims[dims == 1])
        self.dimesionality = 3-is1d
    
    def importTestPotentials(self):
        self.open_inputFile()
        self.V_intra  =  np.asarray(self.file['Vintra'][()].view(dtype=self.ourTypeReal)).reshape(self.Ny, self.Nx)
        self.V_inter  =  np.asarray(self.file['Vinter'][()].view(dtype=self.ourTypeReal)).reshape(self.Ny, self.Nx)
        self.close_inputFile()

 #math functions
    def div(self,J1,J2):
            divJ = np.zeros(self.Ny*self.Nx).reshape(self.Ny,self.Nx)

            for i in range(self.Ny):
                for j in range(self.Nx):

                    xDown = j - 1
                    yDown = i - 1

                    x_hopDown = 1
                    y_hopDown = 1

                    if(j==0):
                        xDown = 0
                        x_hopDown = 0
                    if(i==0):
                        yDown = 0
                        y_hopDown = 0

                    maxDivJ = max([ abs(J1[i][xDown]*x_hopDown), abs( J1[i][j]), abs(J2[yDown][j]*y_hopDown), abs(J2[i][j]) ]) + 1e-4
                    #
                    #
                    divJ[i][j] = - 100*( (J1[i][xDown]*x_hopDown - J1[i][j]) + (J2[yDown][j]*y_hopDown - J2[i][j] )) / maxDivJ

            return divJ

    def curl(self,A1,A2):
        B = np.zeros((self.Ny)*(self.Nx)).reshape((self.Ny),(self.Nx))

        for i in range(self.Ny-1):
            for j in range(self.Nx-1):

                B[i][j] = ( A1[i][j] - A1[i + 1][j]  + A2[i][j+1] - A2[i][j])

        Bnew = np.zeros((self.Ny-1)*(self.Nx-1)).reshape((self.Ny-1),(self.Nx-1))

        for i in range(self.Ny-1):
            for j in  range(self.Nx-1):
                Bnew[i][j] = B[i][j]

        return Bnew

    def perimeterCirculation(self,N1,N2,A1,A2):

        res = 0

        for i in range(N1-1):
            res += A1[0][i] - A1[-1][i]
        for i in range(N2-1):
            res += -A2[i][0] + A2[i][-1] 

        return res

    def shiftToSites(self):
        # Resulting current is defined within [1,N-2] x [1, N-2]
        # Fixing current lattice
        J1old = +self.Jx
        J2old = +self.Jy
        J1    = np.zeros(self.Ny*self.Nx).reshape(self.Ny,self.Nx)
        J2    = np.zeros(self.Ny*self.Nx).reshape(self.Ny,self.Nx)

        for i in range(self.Ny):
            for j in range(1,self.Nx):
                J1[i][j] = 0.5*(J1old[i][j] + J1old[i][j-1])

        for i in range(self.Nx):
            for j in range(1,self.Ny):
                J2[j][i] = 0.5*(J2old[j-1][i] + J2old[j][i])

        J1 = np.delete(J1,0,0)
        J1 = np.delete(J1,self.Ny-2,0)
        J1 = np.delete(J1,0,1)
        J1 = np.delete(J1,self.Nx-2,1)

        J2 = np.delete(J2,0,0)
        J2 = np.delete(J2,self.Ny-2,0)
        J2 = np.delete(J2,0,1)
        J2 = np.delete(J2,self.Nx-2,1)

        return J1, J2
    
    # Plotting functions

    def phaseDiff(self,Z_a, Z_b):
        X_a = np.real(Z_a)
        Y_a = np.imag(Z_a)
        X_b = np.real(Z_b)
        Y_b = np.imag(Z_b)
        return np.arctan2( ( Y_a * X_b - X_a * Y_b ), ( X_a * X_b + Y_a * Y_b ) )
     
    def printHamiltonian(self):

        def matprint(mat, fmt="g"):
            col_maxes = [max([len(("{:"+fmt+"}").format(x)) for x in col]) for col in mat.T]
            for x in mat:
                for i, y in enumerate(x):
                    print(("{:"+str(col_maxes[i])+fmt+"}").format(y), end="  ")
                print("")

        print("COMMON VARIABLES:")
        print(f"T    ==>  {self.hamiltonian.T}")
        print(f"q    ==>  {self.hamiltonian.q}")
        print(f"Bext ==>  {self.hamiltonian.Bext}")
        print(f"Emax ==>  {self.hamiltonian.Emax}")
        
        print("[Component 1, Component 2, Component 3]")
        print(f"t_x  ==>  {self.hamiltonian.t_x}")
        print(f"t_y  ==>  {self.hamiltonian.t_y}")
        print(f"mu   ==>  {self.hamiltonian.mu}")
        print(f"h    ==>  {self.hamiltonian.H}")
        
        print("Coupling matrix V:")
        matprint(self.hamiltonian.V)
        print("Inverse of V:")
        try:
            matprint(np.linalg.inv(self.hamiltonian.V))
        except:
            print("Singular matrix")
        print(f"detV ==> {np.linalg.det(self.hamiltonian.V)}")

    def printVdata(self):

        def matprint(mat, fmt="g"):
            col_maxes = [max([len(("{:"+fmt+"}").format(x)) for x in col]) for col in mat.T]
            for x in mat:
                for i, y in enumerate(x):
                    print(("{:"+str(col_maxes[i])+fmt+"}").format(y), end="  ")
                print("")
        print("Matrix V:")
        matprint(self.hamiltonian.V)
        print("Inverse of V:")
        matprint(np.linalg.inv(self.hamiltonian.V))
        print(f"detV ==> {np.linalg.det(self.hamiltonian.V)}")
        print("The eigenvalues and eigenvectors of V are")
        eval, evec = np.linalg.eigh(self.hamiltonian.V)

        for i in range (len(eval)):
            print(f"Eval: {eval[i]:.5f} with Evec: {evec[:,i]}")

