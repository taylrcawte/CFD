# from HeatTransfer1D.classes import HeatTransfer1D
from modules.classes.classes1d import HeatTransfer1D
from modules.functions.functions2d import const_flux_boundary, const_temp_boundary, calculate_internal_a_e, calculate_internal_a_p, calculate_internal_a_w, calculate_internal_a_s, insulated_boundary, calculate_internal_a_n
from modules.classes.classes2d import HeatTransfer2D, PhysicalProperties
import matplotlib.pyplot as plt 
import numpy as np

def one_dee(): 
    
    sim = HeatTransfer1D(
        x_nodes=5, 
        length=0.02, 
        k=0.5,
        area=1, 
        bc_w=100, 
        bc_e=200,
        T_inf=0,
        q=1000E3, 
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

def two_dee():

    # TODO: make a physical props class or dict to define the physical props and feed them to the heat transfer sim 
    physical_props = PhysicalProperties(
        x_nodes = 300, 
        y_nodes = 400,
        x_length = 0.3,  
        y_length = 0.4, 
        thickness = 0.01, 
        k = 1000,
        bt_n=100, 
        bt_s=0, 
        bt_e=0,
        bt_w=0, 
        q = 500E3
    )

    boundaries = {
        'north_boundary': {
            's_u': const_temp_boundary(k=physical_props.k, area=physical_props.dx*physical_props.thickness, bc=physical_props.bt_n, dist=physical_props.dy), 
            's_p': -1*const_temp_boundary(k=physical_props.k, area=physical_props.dx*physical_props.thickness, bc=physical_props.bt_n, dist=physical_props.dy) / physical_props.bt_n, 
            'a_n': 0,
            'a_s': calculate_internal_a_s(area=physical_props.thickness*physical_props.dx, k=physical_props.k, dist=physical_props.dy),
            'a_e': calculate_internal_a_e(area=physical_props.thickness*physical_props.dy, k=physical_props.k, dist=physical_props.dx), 
            'a_w': calculate_internal_a_w(area=physical_props.thickness*physical_props.dy, k=physical_props.k, dist=physical_props.dx)
        }, 
        'south_boundary': {
            's_u': insulated_boundary(), 
            's_p': insulated_boundary(), 
            'a_n': calculate_internal_a_n(area=physical_props.thickness*physical_props.dx, k=physical_props.k, dist=physical_props.dy),
            'a_s': 0,
            'a_e': calculate_internal_a_e(area=physical_props.thickness*physical_props.dy, k=physical_props.k, dist=physical_props.dx), 
            'a_w': calculate_internal_a_w(area=physical_props.thickness*physical_props.dy, k=physical_props.k, dist=physical_props.dx)
        }, 
        'west_boundary': {
            's_u': const_flux_boundary(q=physical_props.q, area=physical_props.dy*physical_props.thickness), 
            's_p': insulated_boundary(), 
            'a_n': calculate_internal_a_n(area=physical_props.thickness*physical_props.dx, k=physical_props.k, dist=physical_props.dy),
            'a_s': calculate_internal_a_s(area=physical_props.thickness*physical_props.dx, k=physical_props.k, dist=physical_props.dy),
            'a_e': calculate_internal_a_e(area=physical_props.thickness*physical_props.dy, k=physical_props.k, dist=physical_props.dx), 
            'a_w': 0
        }, 
        'east_boundary': {
            's_u': insulated_boundary(), 
            's_p': insulated_boundary(), 
            'a_n': calculate_internal_a_n(area=physical_props.thickness*physical_props.dx, k=physical_props.k, dist=physical_props.dy),
            'a_s': calculate_internal_a_s(area=physical_props.thickness*physical_props.dx, k=physical_props.k, dist=physical_props.dy),
            'a_e': 0, 
            'a_w': calculate_internal_a_w(area=physical_props.thickness*physical_props.dy, k=physical_props.k, dist=physical_props.dx)
        },
        'northwest_boundary': { 
            's_u': const_flux_boundary(q=physical_props.q, area=physical_props.dy*physical_props.thickness) + (const_temp_boundary(physical_props.k, physical_props.dx*physical_props.thickness, physical_props.bt_n, physical_props.dy)), 
            's_p': -1*const_temp_boundary(physical_props.k, physical_props.dx*physical_props.thickness, physical_props.bt_n, physical_props.dx) / physical_props.bt_n, 
            'a_n': 0,
            'a_s': calculate_internal_a_s(area=physical_props.thickness*physical_props.dx, k=physical_props.k, dist=physical_props.dy),
            'a_e': calculate_internal_a_e(area=physical_props.thickness*physical_props.dy, k=physical_props.k, dist=physical_props.dx), 
            'a_w': 0
        }, 
        'northeast_boundary': {
            's_u': const_temp_boundary(physical_props.k, physical_props.dx*physical_props.thickness, physical_props.bt_n, physical_props.dy),
            's_p': -1*const_temp_boundary(physical_props.k, physical_props.dx*physical_props.thickness, physical_props.bt_n, physical_props.dy) / physical_props.bt_n , 
            'a_n': 0,
            'a_s': calculate_internal_a_s(area=physical_props.thickness*physical_props.dx, k=physical_props.k, dist=physical_props.dy),
            'a_e': 0, 
            'a_w': calculate_internal_a_w(area=physical_props.thickness*physical_props.dy, k=physical_props.k, dist=physical_props.dx)
        },
        'southwest_boundary': {
            's_u': const_flux_boundary(q=physical_props.q, area=physical_props.dy*physical_props.thickness),
            's_p': insulated_boundary(), 
            'a_n': calculate_internal_a_n(area=physical_props.thickness*physical_props.dx, k=physical_props.k, dist=physical_props.dy),
            'a_s': 0,
            'a_e': calculate_internal_a_e(area=physical_props.thickness*physical_props.dy, k=physical_props.k, dist=physical_props.dx), 
            'a_w': 0
        },
        'southeast_boundary': {
            's_u': insulated_boundary(), 
            's_p': insulated_boundary(), 
            'a_n': calculate_internal_a_n(area=physical_props.thickness*physical_props.dx, k=physical_props.k, dist=physical_props.dy),
            'a_s': 0,
            'a_e': 0, 
            'a_w': calculate_internal_a_w(area=physical_props.thickness*physical_props.dy, k=physical_props.k, dist=physical_props.dx)
        }
    }

    sim = HeatTransfer2D(physical_properties=physical_props, boundary_dict=boundaries)
    sim.calculate_coefficients()
    temp = sim.solve()
    sim.visualize()
    print(temp) 

def main():

    # one_dee()
    two_dee()


if __name__ == '__main__': 
    main()