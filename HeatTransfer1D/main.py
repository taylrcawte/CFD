from classes import HeatTransfer1D
import matplotlib.pyplot as plt 
import numpy as np


def main(): 

    sim = HeatTransfer1D(
        x_nodes=5000, 
        length=0.02, 
        k=0.5,
        area=1, 
        bc1=100, 
        bc2=200,
        T_inf = 0
        q=1000E3, 
        hp=0 
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