import numpy as np 
from functions import calculate_internal_a_w, calculate_internal_a_e, calculate_internal_a_p, calculate_internal_s_p, calculate_internal_s_u

class HeatTransfer1D(object): 

    def __init__(self, x_nodes, length, k, area, bc1, bc2) -> None:
        
        self.x_nodes = x_nodes
        self.length = length  # meters
        self.k = k  # W / m.K 
        self.area = area  # m^2
        self.bc1 = bc1  # K 
        self.bc2 = bc2  # K

        # init variables 
        self.a_p = np.empty(self.x_nodes, dtype=float) 
        self.a_e = np.empty(self.x_nodes, dtype=float)
        self.a_w = np.empty(self.x_nodes, dtype=float)
        self.s_p = np.empty(self.x_nodes, dtype=float)
        self.s_u = np.empty(self.x_nodes, dtype=float)
        self.dx = self.length / self.x_nodes
    
    def identify_boundary_nodes(self) -> dict: 
        # since this is 1D we can assume the first and last nodes are the boundary nodes 
        # this process will become more complilicicated as dimensions increase 
        return {'west_boundary': [0], 'east_boundary': [self.x_nodes - 1]}

    def calculate_coefficients(self) -> None:
        
        # do the common values first i.e. all internal nodes
        # TODO: can maybe turn the things used to calculate a_w, a_e, etc. into functions
        a_w = calculate_internal_a_w(self.area, self.k, self.dx)
        a_e = calculate_internal_a_e(self.area, self.k, self.dx)
        a_p = calculate_internal_a_p(a_w, a_e) 
        # since we are assuming source free heat then we know that the internal 
        # nodes have no heat generation term to them 
        s_p = calculate_internal_s_p()
        s_u = calculate_internal_s_u() 
        
        # fill arrays with the values 
        self.a_w.fill(a_w)
        self.a_e.fill(a_e)
        self.a_p.fill(a_p)
        self.s_p.fill(s_p)
        self.s_u.fill(s_u)

        # account for boundary nodes 

        boundary_nodes = self.identify_boundary_nodes()

        for key in boundary_nodes.keys(): 

            for node in boundary_nodes[key]: 

                if key == 'west_boundary':

                    # source term 
                    self.s_u[node] = (2*self.k*self.area*self.bc1) / self.dx
                    self.s_p[node] = ((-2*self.k*self.area)/self.dx) 
                    # coefs 
                    self.a_w[node] = 0
                    self.a_e[node] = a_e
                    self.a_p[node] = self.a_w[node] + self.a_e[node] - self.s_p[node]

                else: 

                    # source term 
                    self.s_u[node] = (2*self.k*self.area*self.bc2) / self.dx
                    self.s_p[node] = ((-2*self.k*self.area)/self.dx) 
                    # coefs 
                    self.a_e[node] = 0
                    self.a_w[node] = a_w
                    self.a_p[node] = self.a_w[node] + self.a_e[node] - self.s_p[node]
        
        # calculate a_p 
        # TODO: change the logic here so that a_p gets calcualted at the very end using vector addition 
        # self.a_p = self.a_w + self.a_p + self.s_p

        print(f'a_w:{self.a_w}')
        print(f'a_e:{self.a_e}')
        print(f'a_p:{self.a_p}')
        print(f's_p:{self.s_p}')
        print(f's_u:{self.s_u}')

    def solve(self) -> None:

        # create tridiag matrix 
        matrix_a_w = np.diag(self.a_w[1:], k=1)
        matrix_a_e = np.diag(self.a_e[:-1], k=-1)
        matrix_a_p = np.diag(self.a_p, k=0)
        matrix_A = -1*matrix_a_e + matrix_a_p + -1*matrix_a_w

        print(matrix_A)

        temp = np.linalg.solve(a=matrix_A, b=self.s_u)
        print(temp)

