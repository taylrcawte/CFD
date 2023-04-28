from classes import HeatTransfer1D
import matplotlib.pyplot as plt 


def main(): 

    sim = HeatTransfer1D(
        x_nodes=5, 
        length=0.02, 
        k=0.5, 
        area=1, 
        bc1=100, 
        bc2=200,
        q=1000E3
        )
    sim.calculate_coefficients()
    sim.solve() 

if __name__ == '__main__': 
    main()