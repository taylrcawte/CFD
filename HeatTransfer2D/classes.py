import numpy as np
import sys 
sys.path.insert(0, '/home/taylr/code_dir/CFD/HeatTransfer1D/') 
from functions import calculate_internal_a_w, calculate_internal_a_e, calculate_internal_a_p, \
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

class HeatTransfer2D(object):

    def __init__(self, x_nodes, y_nodes, x_length, y_length, k, q, bc_n, bc_s, bc_e, bc_w, h, p, T_inf, thickness) -> None:
        
        self.x_nodes = x_nodes
        self.y_nodes = y_nodes 
        self.x_length = x_length  # meters
        self.y_length = y_length
        self.k = k  # W / m.K 
        # m^2
        self.bc_n = bc_n  # K TODO: turn all boundary conditions into a class, collapse the q, tinf, h, p, and bc_temps into this class
        self.bc_s = bc_s  # K
        self.bc_e = bc_e
        self.bc_w = bc_w  
        self.q = q  # heat energy transfer 
        self.T_inf = T_inf
        self.h = h
        self.p = p
        self.thickness = thickness
 
        self.hp = self.h*self.p  # need the h and p to calculate hp here, wont use n2 

        # init variables 
        self.a_p = np.empty(self.x_nodes*self.y_nodes, dtype=float) 
        self.a_e = np.empty(self.x_nodes*self.y_nodes, dtype=float)
        self.a_w = np.empty(self.x_nodes*self.y_nodes, dtype=float)
        self.a_n = np.empty(self.x_nodes*self.y_nodes, dtype=float)
        self.a_s = np.empty(self.x_nodes*self.y_nodes, dtype=float)
        self.s_p = np.empty(self.x_nodes*self.y_nodes, dtype=float)
        self.s_u = np.empty(self.x_nodes*self.y_nodes, dtype=float)
        self.phi = np.empty(self.x_nodes*self.y_nodes, dtype=float)
        self.phi.fill(0)  # the temperature array 
        self.dx = self.x_length / self.x_nodes
        self.dy = self.y_length / self.y_nodes

        self.ident_grid = self.create_identity_grid()

    def create_identity_grid(self): 
        """
        in a square grid the bottom left corner is the first node, increases occur right to left and bottom to top
        """
        count = 0 
        grid = []
        
        for i in range(self.x_nodes):

            row = []

            for j in range(self.y_nodes):
                row.append(count)
                count += 1
            
            grid.append(row)

        return np.array(grid) 


    def identify_boundary_nodes(self) -> dict:
        """
        since considering the 0th row S, and 0th column W, can just slice the identityt grid to retrieve the boundary nodes
        this method of counting only allows for one node to have one boundary condition, N/S boundaries take precedent  
        """
        boundary_nodes = {}
        boundary_nodes['south_boundary'] = self.ident_grid[0]
        boundary_nodes['north_boundary'] = self.ident_grid[-1]
        boundary_nodes['west_boundary'] = [self.ident_grid[i][0] for i in range(1, self.y_nodes, 1)]
        boundary_nodes['east_boundary'] = [self.ident_grid[i][-1] for i in range(1, self.y_nodes, 1)]

        return boundary_nodes
    
    
    def calculate_coefficients(self) -> None:

        # TODO: i will need to create a sweep for the internal nodes too
        
        # do the common values first i.e. all internal nodes
        a_w = calculate_internal_a_w(self.thickness*self.dx, self.k, self.dx)
        a_e = calculate_internal_a_e(self.thickness*self.dx, self.k, self.dx)
        a_n = calculate_internal_a_n(self.thickness*self.dy, self.k, self.dy)
        a_s = calculate_internal_a_s(self.thickness*self.dy, self.k, self.dy)
        s_p = calculate_internal_s_p(self.hp, self.dx) 
        # TODO; this will need to be changed, should probably just asign all coefficients iterating over the identity grid 
        s_u = calculate_internal_s_u(self.thickness*self.dx, self.dx, self.q, self.hp, self.T_inf) 
        a_p = calculate_internal_a_p(a_w, a_e, s_p) 

        # fill arrays with the values 
        self.a_w.fill(a_w)
        self.a_e.fill(a_e)
        self.a_n.fill(a_n)
        self.a_s.fill(a_s)
        self.a_p.fill(a_p)
        self.s_p.fill(s_p)
        self.s_u.fill(s_u)

        # account for boundary nodes 

        boundary_nodes = self.identify_boundary_nodes()

        # TODO: clean up this for loop into something smarter, maybe use 
        # for key, value in boundary_nodes.items() 
        for key in boundary_nodes.keys(): 
                
            if key == 'north_boundary':
                for node in boundary_nodes[key]: 
                    # source term 
                    self.s_u[node] = calculate_boundary_s_u(k=self.k, dist=self.dy, bc=self.bc_n, q=self.q, T_inf=self.T_inf, hp=self.hp, area=self.dy*self.thickness)
                    self.s_p[node] = calculate_boundary_s_p(k=self.k, hp=self.hp, dist=self.dy,  area=self.dy*self.thickness)
                    # coefs
                    self.a_n[node] = 0.0
                    # self.a_e[node] = a_e  # removed this because it's not a necessary operation as it was filled in last step 
                    self.a_p[node] = self.a_w[node] + self.a_e[node] + self.a_n[node] + self.a_s[node] - self.s_p[node]

            elif key == 'south_boundary': 
                for node in boundary_nodes[key]: 
                    # source term 
                    self.s_u[node] = calculate_boundary_s_u(k=self.k, dist=self.dy, bc=self.bc_s, q=self.q, T_inf=self.T_inf, hp=self.hp, area=self.dy*self.thickness)
                    self.s_p[node] = calculate_boundary_s_p(k=self.k, hp=self.hp, dist=self.dy,  area=self.dy*self.thickness)
                    # coefs
                    self.a_s[node] = 0.0
                    # self.a_e[node] = a_e  # removed this because it's not a necessary operation as it was filled in last step 
                    self.a_p[node] = self.a_w[node] + self.a_e[node] + self.a_n[node] + self.a_s[node] - self.s_p[node] 

            elif key == 'west_boundary':
                for node in boundary_nodes[key]: 
                        # source term 
                    self.s_u[node] = calculate_boundary_s_u(k=self.k, dist=self.dx, bc=self.bc_w, q=500, T_inf=self.T_inf, hp=self.hp, area=self.dx*self.thickness)
                    self.s_p[node] = calculate_boundary_s_p(k=self.k, hp=self.hp, dist=self.dx, area=self.dx*self.thickness)
                    # coefs
                    self.a_w[node] = 0.0
                    # self.a_e[node] = a_e  # removed this because it's not a necessary operation as it was filled in last step 
                    self.a_p[node] = self.a_w[node] + self.a_e[node] + self.a_n[node] + self.a_s[node] - self.s_p[node]

            else: 
                for node in boundary_nodes[key]: 
                        # source term 
                    self.s_u[node] = calculate_boundary_s_u(k=self.k, dist=self.dx, bc=self.bc_e, q=self.q, T_inf=self.T_inf, hp=self.hp,  area=self.dx*self.thickness)
                    self.s_p[node] = calculate_boundary_s_p(k=self.k, hp=self.hp, dist=self.dx, area=self.dx*self.thickness)
                    # coefs 
                    self.a_e[node] = 0.0
                    # self.a_w[node] = a_w  # removed this because it's not a necesarry operation as it was filled in previous step
                    self.a_p[node] = self.a_w[node] + self.a_e[node] + self.a_n[node] + self.a_s[node] - self.s_p[node]
        
        # calculate a_p 
        # TODO: change the logic here so that a_p gets calcualted at the very end using vector addition 
        # self.a_p = self.a_w + self.a_p + self.s_p

        print(f'a_w:{self.a_w}')        
        print(f'a_e:{self.a_e}')
        print(f'a_n:{self.a_n}')
        print(f'a_s:{self.a_s}')
        print(f'a_p:{self.a_p}')
        print(f's_p:{self.s_p}')
        print(f's_u:{self.s_u}')

    def solve(self): 
        
        # calculate the convergence after a full pass has created, iteratate the for loop with a while loop to meet convergence 
        lines = [self.ident_grid[:, i] for i in range(self.x_nodes)]  # choose x nodes because we sweep W-E
        passes = 0 
        
        for i in range(len(lines)-1):
            
            bee = self.s_p[lines[i]] + self.s_u[lines[i]]
            alpha = self.a_n[lines[i]]
            beta = self.a_s[lines[i]]
            dee = self.a_p[lines[i]]
            cee = self.a_e[lines[i]]*self.phi[lines[i+1]]+self.a_w[lines[i]]*self.phi[lines[i-1]]+bee 
            solver = Tdma(-1*alpha, beta, -1*dee, cee)

            temp = solver.solve()
            print(temp)

            self.phi[lines[i]] = temp

        passes += 1
        print(f'Completed pass: {passes}')

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
