from classes import HeatTransfer1D
import matplotlib.pyplot as plt 


def main(): 

    sim = HeatTransfer1D()
    sim.calculate_coefficients()
    sim.solve() 

if __name__ == '__main__': 
    main()