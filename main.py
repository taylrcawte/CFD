# from HeatTransfer1D.classes import HeatTransfer1D
from modules.classes.classes1d import HeatTransfer1D
from modules.classes.classes2d import HeatTransfer2D
import matplotlib.pyplot as plt 
import numpy as np

def one_dee(): 
    
    sim = HeatTransfer1D(
        x_nodes=5, 
        length=0.02, 
        k=0.5,
        area=1, 
        bc_w=100, 
        bc_e=200,
        T_inf=0,
        q=1000E3, 
        h=0,
        p=0 
        )
    sim.calculate_coefficients()
    temp = sim.solve()
    print(temp)

    plt.figure()
    plt.plot(np.linspace(0, sim.length, sim.x_nodes), temp)
    plt.xlabel('Distance [m]')
    plt.ylabel('Temperature [C]')
    plt.show()

def two_dee(): 

    sim = HeatTransfer2D(
        x_nodes=30,
        y_nodes=40,  
        x_length=0.3, 
        y_length=0.4, 
        k=1000,
        bc_n=100, 
        bc_s=0,  # i think these are technically the first guesses of the boundary temperatures ... 
        bc_e=0,
        bc_w=0,
        thickness=0.01,
        q=0
    )
    # sim.create_identity_grid()
    # sim.identify_boundary_nodes()
    sim.calculate_coefficients()
    temp = sim.solve()
    sim.visualize()
    print(temp) 
    # TODO: replace tdma with np.linalg 

def main():

    one_dee()
    two_dee()


if __name__ == '__main__': 
    main()