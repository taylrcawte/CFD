from classes import HeatTransfer1D
import matplotlib.pyplot as plt 
import numpy as np


def main(): 

    sim = HeatTransfer1D(
        x_nodes=5000, 
        length=1, 
        k=22407.695572217,
        area=7.069E-6, 
        bc1=100, 
        bc2=20,
        T_inf=20,
        q=1000E6, 
        h=60,
        p=0.066 
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