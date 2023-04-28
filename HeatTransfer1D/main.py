from classes import HeatTransfer1D
import matplotlib.pyplot as plt 


def main(): 

    sim = HeatTransfer1D(
        x_nodes=5, 
        length=0.5, 
        k=1000, 
        area=10E-3, 
        bc1=100, 
        bc2=500,
        q=0
        )
    sim.calculate_coefficients()
    sim.solve() 

if __name__ == '__main__': 
    main()