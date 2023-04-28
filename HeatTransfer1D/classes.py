import numpy as np 
from functions import calculate_internal_a_w, calculate_internal_a_e, calculate_internal_a_p, calculate_internal_s_p, calculate_internal_s_u, calculate_boundary_s_p, calculate_boundary_s_u

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
        a_w = calculate_internal_a_w(self.area, self.k, self.dx, condition='case1')
        a_e = calculate_internal_a_e(self.area, self.k, self.dx, condition='case1')
        a_p = calculate_internal_a_p(a_w, a_e, condition='case1') 
        # since we are assuming source free heat then we know that the internal 
        # nodes have no heat generation term to them 
        s_p = calculate_internal_s_p(condition='case1')
        s_u = calculate_internal_s_u(condition='case1') 
        
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
                    self.s_u[node] = calculate_boundary_s_u(self.k, self.area, self.dx, self.bc1, condition='case1')
                    self.s_p[node] = calculate_boundary_s_p(self.k, self.area, self.dx, condition='case1')
                    # coefs 
                    self.a_w[node] = 0
                    # self.a_e[node] = a_e  # removed this because it's not a necessary operation as it was filled in last step 
                    self.a_p[node] = self.a_w[node] + self.a_e[node] - self.s_p[node]

                else: 

                    # source term 
                    self.s_u[node] = calculate_boundary_s_u(self.k, self.area, self.dx, self.bc2, condition='case1')
                    self.s_p[node] = calculate_boundary_s_p(self.k, self.area, self.dx, condition='case1')
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

        # create tridiag matrix 
        # matrix_a_w = np.diag(self.a_w[1:], k=1)
        # matrix_a_e = np.diag(self.a_e[:-1], k=-1)
        # matrix_a_p = np.diag(self.a_p, k=0)
        # matrix_A = -1*matrix_a_e + matrix_a_p + -1*matrix_a_w

        # print(matrix_A)

        # temp = np.linalg.solve(a=matrix_A, b=self.s_u)
        # print(temp)
        solver = Tdma(self.a_w, self.a_p, self.a_e, self.s_u)
        temp = solver.solve() 


        return temp

class Tdma(object): 

    def __init__(self, A, B, C, D):

        self.A = -1*A 
        self.B = B 
        self.C = -1*C 
        self.D = D

        if not (len(self.A) == len(self.B) == len(self.C) == len(self.D)): 
            raise ValueError(f'All vectors must be same length,\
                             provided dimensions {len(self.A), len(self.B), len(self.C), len(self.D)}')
        else:
            self.Dim = len(self.A)
            self.X = np.empty(self.Dim)

    def solve(self):

        for i in range(1, self.Dim, 1): 
            print(i)
            w = self.A[i] / self.B[i-1]
            self.B[i] = self.B[i] - w*self.C[i-1]
            self.D[i] = self.D[i] - w*self.D[i-1]

        self.X[self.Dim-1] = self.D[self.Dim-1] / self.B[self.Dim-1]

        for i in range(self.Dim-2, -1, -1):
            print(i)
            self.X[i] = (self.D[i]-self.C[i]*self.X[i+1]) / self.B[i]

        print(self.X) 

        return self.X
