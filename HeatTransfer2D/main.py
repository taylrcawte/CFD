from classes import HeatTransfer2D
import matplotlib.pyplot as plt 
import numpy as np

def main(): 

    sim = HeatTransfer2D(
        x_nodes=4,
        y_nodes=3,  
        x_length=0.4, 
        y_length=0.3, 
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

if __name__ == '__main__': 
    main()