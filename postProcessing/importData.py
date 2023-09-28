
from operator import mod
from tkinter.messagebox import NO
from traceback import print_tb
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
from mpl_toolkits.axes_grid1 import make_axes_locatable

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
   'axes.grid' : False,
   }

plt.rcParams.update(params)
plt.close('all')


@dataclass
class Hamiltonian:
  
    t_x    = np.array([-1.0, -1.0], dtype=complex)
    t_y    = np.array([-1.0, -1.0], dtype=complex)
    t_z    = np.array([-0.0, -0.0], dtype=complex)
    mu     = np.array([-0.0, -0.0])
    H      = np.array([-0.0, -0.0])
    V      = np.zeros((2,2))
    T      = 0.3
    q      = 0.1
    Bext   = 0.0  
    
    Emax   = 1


class dataProcessing:

    def __init__(self, input_path, verbose=False):

        self.input_path = input_path
        self.verbose = verbose
        # Simulation settings
        self.ourTypeComplex = np.dtype(np.complex64)
        self.ourTypeReal    = np.dtype(np.float32)
        self.hamiltonian    = Hamiltonian()
        self.Nx             = 1 
        self.Ny             = 1
        self.Nz             = 1
        self.neighbors      = None
        self.isHartree      = True
        self.isGauge        = True
        self.dimesionality  = None

        # Fields
        self.D_1              = None 
        self.D_2              = None 
        self.D_3              = None 
        self.D_1_ig                = None
        self.D_2_ig                = None
        #self.D_3_ig                = None                
        self.nUp_1            = None
        self.nUp_2            = None
        #self.nUp_3            = None 
        self.nDown_1          = None 
        self.nDown_2          = None
        #self.nDown_3          = None 
        self.Jx               = None 
        self.Jy               = None 
        self.Jz               = None
        self.B                = None
        self.Ax               = None
        self.Ay               = None
        self.mod              = None   

        # Hopping Fields
        self.t1x              = None
        self.t1y              = None
        self.t2x              = None
        self.t2y              = None
        #self.t3x              = None
        #self.t3y              = None


        self.f1               = None
        self.f2               = None    
        self.fB               = None 
        self.FreeEnergyDensity= None 
        self.FreeEnergy       = None

        # To be removed
        self.Zeeman         = None
        self.V_inter        = None
        self.V_intra        = None

        # Plotting parameters
        self.xlim           = None
        self.ylim           = None
        self.dpi            = 200

        self.open_inputFile()
        self.checkType()
        self.importSettings()
        self.importGeometry()
        self.checkDimensions()
        self.importHamiltonian()
        self.importFields()
        self.importFreeEnergy()
        self.importHoppingFields()
        if(self.dimesionality == 2):
            self.reshapeFields()
            self.addMagneticFreeEnergy()
        self.close_inputFile()


        


    # Input file managing functions
    def open_inputFile(self):
        shutil.copy(self.input_path, "simulations/tmp.h5")
        self.file  = h5py.File("simulations/tmp.h5", 'r')
   
    def close_inputFile(self):
        self.file.close()
        os.remove("simulations/tmp.h5")
    

    #Importing functions
    def checkType(self):
        print(self.file['settings/type'][()])
        if(self.file['settings/type'][()] == 'float'):
            print("SIMULATION TYPE ==> FLOAT")
            self.ourTypeComplex = np.dtype(np.complex64)
            self.ourTypeReal    = np.dtype(np.float32)

        if(self.file['settings/type'][()] == 'double'):
            print("SIMULATION TYPE ==> DOUBLE")
            self.ourTypeComplex = np.dtype(np.complex128)
            self.ourTypeReal    = np.dtype(np.float64)

    def importSettings(self):
        self.isHartree = bool(np.asarray(self.file['settings/hartree_settings'][()].view(dtype=np.int32)))
        self.isGauge   = bool(np.asarray(self.file['settings/gauge_settings'][()].view(dtype=np.int32)))
        print(f"SELF CONSISTENT [GAUGE, HARTREE]  ==> {[self.isGauge,self.isHartree]}")
    
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
        self.hamiltonian.mu[1]  = np.asarray(self.file['hamiltonian/mu2'][()].view(dtype=self.ourTypeReal))
        
        self.hamiltonian.T   = np.asarray(self.file['hamiltonian/T'][()].view(dtype=self.ourTypeReal))
        self.hamiltonian.Emax=  np.asarray(self.file['hamiltonian/Emax'][()].view(dtype=self.ourTypeReal))
        self.hamiltonian.q   =  np.asarray(self.file['hamiltonian/q'][()].view(dtype=self.ourTypeReal))
        self.hamiltonian.Bext=  np.asarray(self.file['hamiltonian/Bext'][()])
        
        # The diagonal elements have a 0.5 factor because to build the 
        # symmeric matrix we sum with the transpose
        self.hamiltonian.V[0,0] = 0.5 * np.asarray(self.file['hamiltonian/V1'][()].view(dtype=self.ourTypeReal))
        self.hamiltonian.V[1,1] = 0.5 * np.asarray(self.file['hamiltonian/V2'][()].view(dtype=self.ourTypeReal))
        #self.hamiltonian.V[2,2] = 0.5 * np.asarray(self.file['hamiltonian/V3'][()].view(dtype=self.ourTypeReal))
        self.hamiltonian.V[0,1] = np.asarray(self.file['hamiltonian/Vint12'][()].view(dtype=self.ourTypeReal))
        #self.hamiltonian.V[0,2] = np.asarray(self.file['hamiltonian/Vint13'][()].view(dtype=self.ourTypeReal))
        #self.hamiltonian.V[1,2] = np.asarray(self.file['hamiltonian/Vint23'][()].view(dtype=self.ourTypeReal))
        self.hamiltonian.V = ( self.hamiltonian.V + self.hamiltonian.V.transpose() ) 
    
    def importFields(self):
        if self.verbose: print("Importing Fields...")
        self.D_1  =  self.hamiltonian.Emax * np.asarray(self.file['delta_1'][()].view(dtype=self.ourTypeComplex))
        self.D_2  =  self.hamiltonian.Emax * np.asarray(self.file['delta_2'][()].view(dtype=self.ourTypeComplex))
        #self.D_3  =  self.hamiltonian.Emax * np.asarray(self.file['delta_3'][()].view(dtype=self.ourTypeComplex))
        self.D_1_ig =  self.hamiltonian.Emax * np.asarray(self.file['delta_1_ig'][()].view(dtype=self.ourTypeComplex))
        self.D_2_ig =  self.hamiltonian.Emax * np.asarray(self.file['delta_2_ig'][()].view(dtype=self.ourTypeComplex))
        #self.D_3_ig =  self.hamiltonian.Emax * np.asarray(self.file['delta_3_ig'][()].view(dtype=self.ourTypeComplex))
        self.mod = self.hamiltonian.Emax * np.asarray(self.file['modulation'][()].view(dtype=self.ourTypeReal))
       

        if(self.isHartree):
            self.nUp_1   = np.asarray(self.file['n_up_1'][()].view(dtype=self.ourTypeReal))
            self.nUp_2   = np.asarray(self.file['n_up_2'][()].view(dtype=self.ourTypeReal))
            #self.nUp_3   = np.asarray(self.file['n_up_3'][()].view(dtype=self.ourTypeReal))
            self.nDown_1 = np.asarray(self.file['n_down_1'][()].view(dtype=self.ourTypeReal))
            self.nDown_2 = np.asarray(self.file['n_down_2'][()].view(dtype=self.ourTypeReal))
            #self.nDown_3 = np.asarray(self.file['n_down_3'][()].view(dtype=self.ourTypeReal))

        if(self.isGauge): 
            # according to the last conversation with Albert, the current the code computes is - the real current, hence we add the "-" sign
            Jtot = self.hamiltonian.Emax * np.asarray(self.file['J'][()].view(dtype=self.ourTypeReal))
            self.Jx = Jtot[0::3]
            self.Jy = Jtot[1::3]
            self.Jz = Jtot[2::3]
            # according to the last conversation with Albert, the current the code computes is - the real current, hence we add the "-" sign
            A = np.asarray(self.file['A'][()].view(dtype=self.ourTypeReal))
            self.Ax = A[0::2]
            self.Ay = A[1::2]

    def importHoppingFields(self):
        try:
            self.t1x = np.asarray( self.file['t1x'][()].view(dtype=self.ourTypeComplex) )
            self.t1y = np.asarray( self.file['t1y'][()].view(dtype=self.ourTypeComplex) )
            self.t2x = np.asarray( self.file['t2x'][()].view(dtype=self.ourTypeComplex) )
            self.t2y = np.asarray( self.file['t2y'][()].view(dtype=self.ourTypeComplex) )
            #self.t3x = np.asarray( self.file['t3x'][()].view(dtype=self.ourTypeComplex) )
            #self.t3y = np.asarray( self.file['t3y'][()].view(dtype=self.ourTypeComplex) )
        except:
            print("The current simulation does not export HoppingFields")

    def importFreeEnergy(self):
      
        self.f1 =  np.asarray(self.file['F'][()].view(dtype=self.ourTypeReal))    
        self.f2 = np.zeros( len( self.f1 ) )

        try:
            V_inv = np.linalg.inv(self.hamiltonian.V)
            for i in range( len(self.f1) ):
                deltas = np.array([self.D_1[i], self.D_2[i]])
                self.f2[i] = np.real(np.dot(np.conj(deltas), np.dot(V_inv, deltas)))
        except:
            print("Singular")
            eval, evec = np.linalg.eigh(self.hamiltonian.V)

            for i in range(len(eval)):
                e  = eval[i]
                ev = evec[:,i] 
                
                if np.abs(e) > 1.0e-3:
                    for j in range(self.Nx * self.Ny):          
                        deltas = np.array([self.D_1[j], self.D_2[j]])
                        proj = np.dot( np.conj(deltas), ev ) 
                        self.f2[j] += np.abs(proj)**2 / e 
        
        if(self.dimesionality == 2):
            self.f1 = self.f1.reshape(self.Ny, self.Nx)
            self.f2 = self.f2.reshape(self.Ny, self.Nx)
        
        self.FreeEnergyDensity = self.f1 + self.f2
        self.FreeEnergy = np.sum(self.FreeEnergyDensity)
        if self.verbose: print("Free energy imported")
    
    def addMagneticFreeEnergy(self):
        print("Note: Magnetic free energy density not added to free energy density because defined on plaquettes")
        if(self.isGauge):
            self.fB = 0.5*self.B**2
            self.FreeEnergy += np.sum(0.5*self.B**2)

    def reshapeFields(self):
        if self.verbose: print("Reshaping Fields...")
        if(self.Nz == 1 ):
            self.D_1      = self.D_1.reshape(self.Ny, self.Nx)
            self.D_2      = self.D_2.reshape(self.Ny, self.Nx)
            #self.D_3      = self.D_3.reshape(self.Ny, self.Nx)
            self.D_1_ig   = self.D_1_ig.reshape(self.Ny, self.Nx)
            self.D_2_ig   = self.D_2_ig.reshape(self.Ny, self.Nx)
            #self.D_3_ig   = self.D_3_ig.reshape(self.Ny, self.Nx)
            self.mod      = self.mod.reshape(self.Ny, self.Nx)
           

            if(self.isHartree):
                self.nUp_1   = self.nUp_1.reshape(self.Ny, self.Nx)
                self.nUp_2   = self.nUp_2.reshape(self.Ny, self.Nx)
                #self.nUp_3   = self.nUp_3.reshape(self.Ny, self.Nx)

                self.nDown_1 = self.nDown_1.reshape(self.Ny, self.Nx)
                self.nDown_2 = self.nDown_2.reshape(self.Ny, self.Nx)
                #self.nDown_3 = self.nDown_3.reshape(self.Ny, self.Nx)

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
        self.V_intra  =  np.asarray(self.file['V_intra'][()].view(dtype=self.ourTypeReal)).reshape(self.Ny, self.Nx)
        self.V_inter  =  np.asarray(self.file['V_inter'][()].view(dtype=self.ourTypeReal)).reshape(self.Ny, self.Nx)
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
    
    def plotDelta(self, save=False):
        fig, axes = plt.subplots(nrows=3,ncols=2,figsize=(4*3,3.5*3),
                        gridspec_kw=dict( {'wspace':0.3, 'top':0.8, 'bottom':0.15} ) )

        if(self.dimesionality == 2):
        #Order parameter modulus
            
            p = axes[0,0].imshow(np.absolute(self.D_1), origin='lower',vmin=np.amin(np.absolute(self.D_1)), vmax = np.amax(np.absolute(self.D_1)), cmap='viridis')
            axes[0,0].set_ylabel("$y$")
            axes[0,0].set_xlim(0, self.xlim-1)
            axes[0,0].set_ylim(0, self.ylim-1)
            divider = make_axes_locatable(axes[0,0])
            cax = divider.append_axes('right', size='5%', pad=0.05)
            cbar = fig.colorbar(p, cax=cax, orientation='vertical')
            cbar.ax.set_title("$|\Delta_1|$")
        else:
            axes[0,0].plot(np.absolute(self.D_1), label=r"$|\Delta_1|$")
            axes[0,0].set_xlabel("$x$")
            axes[0,0].set_xlim(0, self.xlim-1)
            axes[0,0].legend(fontsize=18)
        

        # Component 2
        if(self.dimesionality == 2):
        #Order parameter modulus
            
            p = axes[0,1].imshow(np.absolute(self.D_2), origin='lower',vmin=np.amin(np.absolute(self.D_2)), vmax = np.amax(np.absolute(self.D_2)), cmap='viridis')
            axes[0,1].set_ylabel("$y$")
            axes[0,1].set_xlim(0, self.xlim-1)
            axes[0,1].set_ylim(0, self.ylim-1)
            divider = make_axes_locatable(axes[0,1])
            cax = divider.append_axes('right', size='5%', pad=0.05)
            cbar = fig.colorbar(p, cax=cax, orientation='vertical')
            cbar.ax.set_title("$|\Delta_2|$")
            
        else:
            axes[0,1].plot(np.absolute(self.D_2), label=r"$|\Delta_2|$")
            axes[0,1].set_xlabel("$x$")
            axes[0,1].set_xlim(0, self.xlim-1)
            axes[0,1].legend(fontsize=18)

        # 1d ticks
        values = ('-1','-2/3','-1/2','-1/3', '0', '1/3','1/2', '2/3', '1')
        y_pos = np.linspace(-1,1,len(values))
     
        # Phase difference 21
        if(self.dimesionality == 2):
        #Order parameter modulus
            
            p = axes[1,0].imshow(self.phaseDiff(self.D_2, self.D_1), origin='lower', vmin=-np.pi, vmax = np.pi,cmap = 'twilight')
            axes[1,0].set_ylabel("$y$")
            axes[1,0].set_xlim(0, self.xlim-1)
            axes[1,0].set_ylim(0, self.ylim-1)
            divider = make_axes_locatable(axes[1,0])
            cax = divider.append_axes('right', size='5%', pad=0.05)
            cbar = fig.colorbar(p, cax=cax, orientation='vertical')
            cbar.ax.set_title(r"$\phi_2-\phi_1$")
            
        else:
            axes[1,0].plot(self.phaseDiff(self.D_2, self.D_1)/np.pi, label=r"$(\phi_2-\phi_1)/\pi$")
            axes[1,0].set_xlabel("$x$")
            axes[1,0].set_xlim(0, self.xlim-1)
            axes[1,0].legend(fontsize=18)
            axes[1,0].set_yticks(y_pos, values)

        if(self.dimesionality == 2):
        #Order parameter modulus
            
            p = axes[1,1].imshow(self.phaseDiff(self.D_2, self.D_1), origin='lower', vmin=-np.pi, vmax = np.pi,cmap = 'twilight')
            axes[1,1].set_ylabel("$y$")
            axes[1,1].set_xlim(0, self.xlim-1)
            axes[1,1].set_ylim(0, self.ylim-1)
            divider = make_axes_locatable(axes[1,0])
            cax = divider.append_axes('right', size='5%', pad=0.05)
            cbar = fig.colorbar(p, cax=cax, orientation='vertical')
            cbar.ax.set_title(r"$\phi_2-\phi_1$")
            
        else:
            axes[1,1].plot(self.phaseDiff(self.D_2, self.D_1)/np.pi, label=r"$(\phi_2-\phi_1)/\pi$")
            axes[1,1].set_xlabel("$x$")
            axes[1,1].set_xlim(0, self.xlim-1)
            axes[1,1].legend(fontsize=18)
            axes[1,1].set_yticks(y_pos, values)

        if(self.dimesionality == 2):
        #Order parameter modulus
            
            p=axes[2,0].imshow(np.angle(self.D_1), origin='lower',cmap = 'hsv')
            axes[2,0].set_ylabel("$y$")
            axes[2,0].set_xlim(0, self.xlim-1)
            axes[2,0].set_ylim(0, self.ylim-1)
            divider = make_axes_locatable(axes[2,0])
            cax = divider.append_axes('right', size='5%', pad=0.05)
            cbar = fig.colorbar(p, cax=cax, orientation='vertical')
            cbar.ax.set_title(r"$\phi_1$")
        else:
            axes[2,0].plot(np.angle(self.D_1), label=r"$\phi_1$")
            axes[2,0].set_xlabel("$x$")
            axes[2,0].set_xlim(0, self.xlim-1)
            axes[2,0].legend(fontsize=18)
            axes[2,0].set_yticks(y_pos, values)       
    

        if(self.dimesionality == 2):
        #Order parameter modulus
            
            p=axes[2,1].imshow(np.angle(self.D_2), origin='lower',cmap = 'hsv')
            axes[2,1].set_ylabel("$y$")
            axes[2,1].set_xlim(0, self.xlim-1)
            axes[2,1].set_ylim(0, self.ylim-1)
            divider = make_axes_locatable(axes[2,1])
            cax = divider.append_axes('right', size='5%', pad=0.05)
            cbar = fig.colorbar(p, cax=cax, orientation='vertical')
            cbar.ax.set_title(r"$\phi_2$")
        else:
            axes[2,1].plot(np.angle(self.D_2), label=r"$\phi_2$")
            axes[2,1].set_xlabel("$x$")
            axes[2,1].set_xlim(0, self.xlim-1)
            axes[2,1].legend(fontsize=18)
            axes[2,1].set_yticks(y_pos, values)

        if save != False:
            plt.savefig(save +".pdf",bbox_inches='tight')
        
        plt.show()
      


    def plotJ(self,cutoff,save=False):

        if(not self.isGauge):
            print("\nGauge field not active")
            return
       
        
        fig = plt.figure(dpi=self.dpi)
        ax = plt.gca()
        X,Y = np.meshgrid(np.linspace(0,self.xlim,self.xlim),np.linspace(0,self.ylim,self.ylim))
        Q = ax.quiver(X, Y, self.Jx, self.Jy)
        ax.set_title("$\mathbf{J}$")
        if(save): plt.savefig(save + "J.pdf",bbox_inches='tight')
        else:
            plt.draw()
            plt.pause(0.001)
 
        # Current divergence
        fig = plt.figure(dpi=self.dpi)
        divJ = self.div(self.Jx, self.Jy)
        divJ_adj = divJ * (self.Jx**2 + self.Jy**2 > cutoff)
        plt.imshow(divJ_adj, origin='lower',vmin=np.amin(divJ_adj), vmax =np.amax(divJ_adj),cmap="Greys_r")
        plt.xlabel("$x$")
        plt.ylabel("$y$")
        cbar = plt.colorbar()
        plt.title("$\\nabla \\cdot J / J_{max} $")
        cbar.ax.set_title("$(\%)$")
        if(not save):
            plt.draw()
            plt.pause(0.001)

        fig = plt.figure(dpi=self.dpi)
        ax = plt.gca()
        J1,J2 = self.shiftToSites()
        j = np.sqrt(J1**2 + J2**2)
        plt.imshow(j, origin='lower',vmin=np.amin(j), vmax = np.amax(j),cmap="magma")
        #title = "$T = " + str(round(self.hamiltonian.T,3)) + ", V = " + str(round(self.hamiltonian.V,3)) + ", \mu = " + str(round(self.hamiltonian.mu,3)) +"$"
        #plt.title(title)
        plt.xlabel("$x$")
        plt.ylabel("$y$")
        cbar = plt.colorbar()
        cbar.ax.set_title("$|J|$")
        if(save): plt.savefig(save + "mod(J).pdf",bbox_inches='tight')
        else:
            plt.draw()
            plt.pause(0.001)
   
    def plotA(self):
        if(not self.isGauge):
            print("\nGauge field not active")
            return
    
        fig = plt.figure(dpi=self.dpi)
        ax = plt.gca()
        X,Y = np.meshgrid(np.linspace(0,self.xlim,self.xlim),np.linspace(0,self.ylim,self.ylim))
        Q = ax.quiver(X, Y, self.Ax, self.Ay)
        ax.set_title("$\mathbf{A}$")
        plt.draw()
        plt.pause(0.001)
 
    def plotB(self,save=False):
        if(not self.isGauge):
            print("\nGauge field not active")
            return
        
        fig = plt.figure(dpi=self.dpi)
        ax = plt.gca()
        plt.imshow(self.B, origin='lower')
        plt.xlabel("$x$")
        plt.ylabel("$y$")
        cbar = plt.colorbar()
        cbar.ax.set_title("$B_z$")
        plt.title(r"$\Phi_B/\Phi_0 = " + str(round(self.hamiltonian.q*np.sum(self.B)/(np.pi),4)) + "$")
        if(save): plt.savefig(save + "B.pdf",bbox_inches='tight')
        else:
            plt.draw()
            plt.pause(0.001)

    def plotF(self,save=False):
        Free_energy = np.sum(self.f)/(self.Nx * self.Ny)
        title = "$ F/N = " + str(round(Free_energy,4)) + "$"

        fig  = plt.figure(dpi=self.dpi)
        if(self.dimesionality == 2):
            ax = plt.gca()
            plt.imshow(self.f, vmin = np.amin(self.f), vmax = np.amax(self.f), cmap="magma")
            plt.title(title)
            plt.xlabel("$x$")
            plt.ylabel("$y$")
            plt.xlim(0, self.xlim-1)
            plt.ylim(0, self.ylim-1)
            cbar = plt.colorbar()
            cbar.ax.set_title("$\\mathcal{F}$")
        else:
            plt.plot(self.f)
            plt.xlabel("$x$")
            plt.ylabel("$\\mathcal{F}$")
            plt.xlim(0, self.xlim - 1)
            plt.title(title)

        plt.draw()
        plt.pause(0.001)
        
    def plotImbalance(self, save=False):
        if(not self.isHartree):
            print("\nHartree term not self consistent")
            return
        if(not save):
            print("\nmax(N_UP)   = ", np.amax(self.nUp))
            print("max(N_DOWN) = ", np.amax(self.nDown))
     
        title = "Title"
        #Spin imbalance
        fig = plt.figure(dpi=self.dpi)
        DeltaN_1 = self.nUp_1 - self.nDown_1
        if(self.dimesionality == 2):
            ax = plt.gca()
            plt.imshow(DeltaN_1, vmin=np.amin(DeltaN_1), vmax = np.amax(DeltaN_1),cmap = "Greys_r")
            plt.title(title)
            plt.xlabel("$x$")
            plt.ylabel("$y$")
            cbar = plt.colorbar()
            cbar.ax.set_title("$n_{1\\uparrow} - n_{1\\downarrow}$")
        else:
            plt.plot(DeltaN_1)
            plt.xlabel("$x$")
            plt.ylabel("$n_{1\\uparrow} - n_{1\\downarrow}$")
            plt.xlim(0, self.xlim - 1)
            plt.title(title)
        if(save): plt.savefig(save + "deltaN_1.pdf",bbox_inches='tight')
        else:
            plt.draw()
            plt.pause(0.001)
    
        #Spin imbalance
        fig = plt.figure(dpi=self.dpi)
        DeltaN_2 = self.nUp_2 - self.nDown_2
        if(self.dimesionality == 2):
            ax = plt.gca()
            plt.imshow(DeltaN_2, vmin=np.amin(DeltaN_2), vmax = np.amax(DeltaN_2),cmap = "Greys_r")
            plt.title(title)
            plt.xlabel("$x$")
            plt.ylabel("$y$")
            cbar = plt.colorbar()
            cbar.ax.set_title("$ n_{2\\uparrow} - n_{2\\downarrow}  $")
        else:
            plt.plot(DeltaN_2)
            plt.xlabel("$x$")
            plt.ylabel("$n_{2\\uparrow} - n_{2\\downarrow}$")
            plt.xlim(0, self.xlim - 1)
            plt.title(title)
        if(save): plt.savefig(save + "deltaN_2.pdf",bbox_inches='tight')
        else:
            plt.draw()
            plt.pause(0.001)

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
        
        print("[Component 1, Component 2]")
        print(f"t_x  ==>  {self.hamiltonian.t_x}")
        print(f"t_y  ==>  {self.hamiltonian.t_y}")
        print(f"mu   ==>  {self.hamiltonian.mu}")
        print(f"h    ==>  {self.hamiltonian.H}")
        
        print("Coupling matrix V:")
        matprint(self.hamiltonian.V)
        print("Inverse of V:")
        try:
            matprint(np.linalg.inv(self.hamiltonian.V))
            print(f"detV ==> {np.linalg.det(self.hamiltonian.V)}")
        except:
            print("Singular Hamiltonian")
        
    def plotInitialGuess(self, save=False):
        fig, axes = plt.subplots(nrows=3,ncols=2,figsize=(4*3,3.5*3),
                        gridspec_kw=dict( {'wspace':0.3, 'top':0.8, 'bottom':0.15} ) )

        if(self.dimesionality == 2):
        #Order parameter modulus
            
            p = axes[0,0].imshow(np.absolute(self.D_1_ig), origin='lower',vmin=np.amin(np.absolute(self.D_1_ig)), vmax = np.amax(np.absolute(self.D_1_ig)), cmap='viridis')
            axes[0,0].set_ylabel("$y$")
            axes[0,0].set_xlim(0, self.xlim-1)
            axes[0,0].set_ylim(0, self.ylim-1)
            divider = make_axes_locatable(axes[0,0])
            cax = divider.append_axes('right', size='5%', pad=0.05)
            cbar = fig.colorbar(p, cax=cax, orientation='vertical')
            cbar.ax.set_title("$|\Delta_1|$")
        else:
            axes[0,0].plot(np.absolute(self.D_1_ig), label=r"$|\Delta_1|$")
            axes[0,0].set_xlabel("$x$")
            axes[0,0].set_xlim(0, self.xlim-1)
            axes[0,0].legend(fontsize=18)
        

        # Component 2
        if(self.dimesionality == 2):
        #Order parameter modulus
            
            p = axes[0,1].imshow(np.absolute(self.D_2_ig), origin='lower',vmin=np.amin(np.absolute(self.D_2_ig)), vmax = np.amax(np.absolute(self.D_2_ig)), cmap='viridis')
            axes[0,1].set_ylabel("$y$")
            axes[0,1].set_xlim(0, self.xlim-1)
            axes[0,1].set_ylim(0, self.ylim-1)
            divider = make_axes_locatable(axes[0,1])
            cax = divider.append_axes('right', size='5%', pad=0.05)
            cbar = fig.colorbar(p, cax=cax, orientation='vertical')
            cbar.ax.set_title("$|\Delta_2|$")
            
        else:
            axes[0,1].plot(np.absolute(self.D_2_ig), label=r"$|\Delta_2|$")
            axes[0,1].set_xlabel("$x$")
            axes[0,1].set_xlim(0, self.xlim-1)
            axes[0,1].legend(fontsize=18)

        # 1d ticks
        values = ('-1','-2/3','-1/2','-1/3', '0', '1/3','1/2', '2/3', '1')
        y_pos = np.linspace(-1,1,len(values))
     
        # Phase difference 21
        if(self.dimesionality == 2):
        #Order parameter modulus
            
            p = axes[1,0].imshow(self.phaseDiff(self.D_2_ig, self.D_1_ig), origin='lower', vmin=-np.pi, vmax = np.pi,cmap = 'twilight')
            axes[1,0].set_ylabel("$y$")
            axes[1,0].set_xlim(0, self.xlim-1)
            axes[1,0].set_ylim(0, self.ylim-1)
            divider = make_axes_locatable(axes[1,0])
            cax = divider.append_axes('right', size='5%', pad=0.05)
            cbar = fig.colorbar(p, cax=cax, orientation='vertical')
            cbar.ax.set_title(r"$\phi_2-\phi_1$")
            
        else:
            axes[1,0].plot(self.phaseDiff(self.D_2_ig, self.D_1_ig)/np.pi, label=r"$(\phi_2-\phi_1)/\pi$")
            axes[1,0].set_xlabel("$x$")
            axes[1,0].set_xlim(0, self.xlim-1)
            axes[1,0].legend(fontsize=18)
            axes[1,0].set_yticks(y_pos, values)

        if(self.dimesionality == 2):
        #Order parameter modulus
            
            p = axes[1,1].imshow(self.phaseDiff(self.D_2_ig, self.D_1_ig), origin='lower', vmin=-np.pi, vmax = np.pi,cmap = 'twilight')
            axes[1,1].set_ylabel("$y$")
            axes[1,1].set_xlim(0, self.xlim-1)
            axes[1,1].set_ylim(0, self.ylim-1)
            divider = make_axes_locatable(axes[1,0])
            cax = divider.append_axes('right', size='5%', pad=0.05)
            cbar = fig.colorbar(p, cax=cax, orientation='vertical')
            cbar.ax.set_title(r"$\phi_2-\phi_1$")
            
        else:
            axes[1,1].plot(self.phaseDiff(self.D_2_ig, self.D_1_ig)/np.pi, label=r"$(\phi_2-\phi_1)/\pi$")
            axes[1,1].set_xlabel("$x$")
            axes[1,1].set_xlim(0, self.xlim-1)
            axes[1,1].legend(fontsize=18)
            axes[1,1].set_yticks(y_pos, values)
       
        if(self.dimesionality == 2):
        #Order parameter modulus
            
            p=axes[2,0].imshow(np.angle(self.D_1_ig), origin='lower',cmap = 'hsv')
            axes[2,0].set_ylabel("$y$")
            axes[2,0].set_xlim(0, self.xlim-1)
            axes[2,0].set_ylim(0, self.ylim-1)
            divider = make_axes_locatable(axes[2,0])
            cax = divider.append_axes('right', size='5%', pad=0.05)
            cbar = fig.colorbar(p, cax=cax, orientation='vertical')
            cbar.ax.set_title(r"$\phi_1$")
        else:
            axes[2,0].plot(np.angle(self.D_1_ig), label=r"$\phi_1$")
            axes[2,0].set_xlabel("$x$")
            axes[2,0].set_xlim(0, self.xlim-1)
            axes[2,0].legend(fontsize=18)
            axes[2,0].set_yticks(y_pos, values)

        if(self.dimesionality == 2):
        #Order parameter modulus
            
            p=axes[2,1].imshow(np.angle(self.D_2_ig), origin='lower',cmap = 'hsv')
            axes[2,1].set_ylabel("$y$")
            axes[2,1].set_xlim(0, self.xlim-1)
            axes[2,1].set_ylim(0, self.ylim-1)
            divider = make_axes_locatable(axes[2,1])
            cax = divider.append_axes('right', size='5%', pad=0.05)
            cbar = fig.colorbar(p, cax=cax, orientation='vertical')
            cbar.ax.set_title(r"$\phi_2$")
        else:
            axes[2,1].plot(np.angle(self.D_2_ig), label=r"$\phi_2$")
            axes[2,1].set_xlabel("$x$")
            axes[2,1].set_xlim(0, self.xlim-1)
            axes[2,1].legend(fontsize=18)
            axes[2,1].set_yticks(y_pos, values)           