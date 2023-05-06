import numpy as np
import sys 
from modules.functions.functions1d import calculate_internal_a_w, calculate_internal_a_e, calculate_internal_a_p, \
    calculate_internal_s_p, calculate_internal_s_u, calculate_boundary_s_p, calculate_boundary_s_u, \
    calculate_internal_a_n, calculate_internal_a_s

class HeatTransfer1D(object): 

    def __init__(self, x_nodes, length, k, q, area, bc_w, bc_e, h, p, T_inf) -> None:
        
        self.x_nodes = x_nodes
        self.length = length  # meters
        self.k = k  # W / m.K 
        self.area = area  # m^2
        self.bc1 = bc_w  # K 
        self.bc2 = bc_e  # K
        self.q = q  # heat energy transfer 
        self.T_inf = T_inf
        self.h = h
        self.p = p 

        self.hp = self.h*self.p  # need the h and p to calculate hp here, wont use n2 

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
        a_w = calculate_internal_a_w(self.area, self.k, self.dx)
        a_e = calculate_internal_a_e(self.area, self.k, self.dx)
        s_p = calculate_internal_s_p(self.hp, self.dx)

        # since we are assuming source free heat then we know that the internal 
        # nodes have no heat generation term to them 
        s_u = calculate_internal_s_u(self.area, self.dx, self.q, self.hp, self.T_inf) 
        a_p = calculate_internal_a_p(a_w, a_e, s_p) 

        # fill arrays with the values 
        self.a_w.fill(a_w)
        self.a_e.fill(a_e)
        self.a_p.fill(a_p)
        self.s_p.fill(s_p)
        self.s_u.fill(s_u)

        # account for boundary nodes 

        boundary_nodes = self.identify_boundary_nodes()

        # TODO: clean up this for loop into something smarter, maybe use 
        # for key, value in boundary_nodes.items() 
        for key in boundary_nodes.keys(): 

            for node in boundary_nodes[key]: 

                if key == 'west_boundary':

                    # source term 
                    self.s_u[node] = calculate_boundary_s_u(self.k, self.area, self.dx, self.bc1, q=self.q, T_inf=self.T_inf, hp=self.hp)
                    self.s_p[node] = calculate_boundary_s_p(self.k, self.area, self.hp, self.dx)
                    # coefs
                    self.a_w[node] = 0
                    # self.a_e[node] = a_e  # removed this because it's not a necessary operation as it was filled in last step 
                    self.a_p[node] = self.a_w[node] + self.a_e[node] - self.s_p[node]

                else: 

                    # source term 
                    self.s_u[node] = calculate_boundary_s_u(self.k, self.area, self.dx, self.bc2, q=self.q, T_inf=self.T_inf, hp=self.hp)
                    self.s_p[node] = calculate_boundary_s_p(self.k, self.area, self.hp, self.dx)
                    # coefs 
                    self.a_e[node] = 0
                    # self.a_w[node] = a_w  # removed this because it's not a necesarry operation as it was filled in previous step
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

        solver = Tdma(-1*self.a_w, self.a_p, -1*self.a_e, self.s_u)
        self.temp = solver.solve() 

        return self.temp

class Tdma(object): 

    def __init__(self, A, B, C, D):

        self.A = A 
        self.B = B 
        self.C = C 
        self.D = D

        if not (len(self.A) == len(self.B) == len(self.C) == len(self.D)): 
            raise ValueError(f'All vectors must be same length,\
                             provided dimensions {len(self.A), len(self.B), len(self.C), len(self.D)}')
        else:
            self.Dim = len(self.A)
            self.X = np.empty(self.Dim, dtype=float)

    def solve(self):

        for i in range(1, self.Dim, 1): 
            w = self.A[i] / self.B[i-1]
            self.B[i] = self.B[i] - w*self.C[i-1]
            self.D[i] = self.D[i] - w*self.D[i-1]

        self.X[self.Dim-1] = self.D[self.Dim-1] / self.B[self.Dim-1]

        for i in range(self.Dim-2, -1, -1):
            self.X[i] = (self.D[i]-self.C[i]*self.X[i+1]) / self.B[i]

        return self.X
