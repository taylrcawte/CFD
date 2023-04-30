from classes import HeatTransfer1D, HeatTransfer2D
import matplotlib.pyplot as plt 
import numpy as np

def one_d(): 

    sim = HeatTransfer1D(
        x_nodes=5, 
        length=0.5, 
        k=1000,
        area=10E-3, 
        bc_w=100, 
        bc_e=500,
        T_inf=0,
        q=0, 
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

def two_d():

    sim = HeatTransfer2D(
        x_nodes=5,
        y_nodes=5,  
        x_length=0.5, 
        y_length=0.5, 
        k=1000,
        area=10E-3, 
        bc_n=100, 
        bc_s=500,
        bc_e=0,
        bc_w=0,
        T_inf=0,
        q=0, 
        h=0,
        p=0,
        initial_guess=0
    )
    sim.create_identity_grid()
    sim.identify_boundary_nodes()
    # sim.calculate_coefficients()
    # temp = sim.solve() 


def main(): 
    # one_d()
    two_d()

if __name__ == '__main__': 
    main()