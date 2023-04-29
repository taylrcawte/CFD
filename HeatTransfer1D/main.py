from classes import HeatTransfer1D
import matplotlib.pyplot as plt 
import numpy as np


def main(): 

    sim = HeatTransfer1D(
        x_nodes=5, 
        length=0.5, 
        k=1000,
        area=10E-3, 
        bc1=100, 
        bc2=500,
        T_inf=0,
        q=0, 
        h=0,
        p=0 
        )
    sim.calculate_coefficients()
    temp = sim.solve() 

    plt.figure()
    plt.plot(np.linspace(0, sim.length, sim.x_nodes), temp)
    plt.xlabel('Distance [m]')
    plt.ylabel('Temperature [C]')
    plt.show()

if __name__ == '__main__': 
    main()