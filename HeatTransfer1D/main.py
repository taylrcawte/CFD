from classes import HeatTransfer1D


def main(): 

    sim = HeatTransfer1D()
    sim.calculate_coefficients()
    sim.solve() 

if __name__ == '__main__': 
    main()