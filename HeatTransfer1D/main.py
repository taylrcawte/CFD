from classes import HeatTransfer1D


def main(): 

    sim = HeatTransfer1D()
    sim.calculate_coefficients() 

if __name__ == '__main__': 
    main()