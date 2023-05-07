import numpy as np
import matplotlib.pyplot as plt
from modules.functions.functions2d import calculate_internal_a_w, calculate_internal_a_e, calculate_internal_a_p, \
    calculate_internal_s_p, calculate_internal_s_u, \
    calculate_internal_a_n, calculate_internal_a_s, const_flux_boundary, const_temp_boundary, insulated_boundary, tdma_cons, tdma_noncons

class HeatTransfer2D(object):

    def __init__(self, x_nodes, y_nodes, x_length, y_length, k, bc_n, bc_s, bc_e, bc_w, q, thickness) -> None:
        
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
        self.thickness = thickness
        self.q = q
 

        # init variables 
        self.a_p = np.zeros(self.x_nodes*self.y_nodes) 
        self.a_e = np.zeros(self.x_nodes*self.y_nodes)
        self.a_w = np.zeros(self.x_nodes*self.y_nodes)
        self.a_n = np.zeros(self.x_nodes*self.y_nodes)
        self.a_s = np.zeros(self.x_nodes*self.y_nodes)
        self.s_p = np.zeros(self.x_nodes*self.y_nodes)
        self.s_u = np.zeros(self.x_nodes*self.y_nodes)
        self.phi = np.zeros(self.x_nodes*self.y_nodes)
        # fill all arrays with initial guess 0 
        # the temperature array 
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
        # TODO: use slicing and intersects to select the boundary nodes 
        boundary_nodes = {}
        boundary_nodes['south_boundary'] = [self.ident_grid[i][0] for i in range(1, self.x_nodes-1, 1)]
        boundary_nodes['north_boundary'] = [self.ident_grid[i][-1] for i in range(1, self.x_nodes-1, 1)]
        boundary_nodes['west_boundary'] = [self.ident_grid[0][i] for i in range(1, self.y_nodes-1, 1)]
        boundary_nodes['east_boundary'] = [self.ident_grid[-1][i] for i in range(1, self.y_nodes-1, 1)]
        boundary_nodes['southeast_boundary'] = [self.ident_grid[-1][0]]
        boundary_nodes['northeast_boundary'] = [self.ident_grid[-1][-1]] 
        boundary_nodes['southwest_boundary'] = [self.ident_grid[0][0]] 
        boundary_nodes['northwest_boundary'] = [self.ident_grid[0][-1]] 

        return boundary_nodes
    
    def calculate_coefficients(self) -> None:

        # TODO: i will need to create a sweep for the internal nodes too
        
        # do the common values first i.e. all internal nodes
        a_w = calculate_internal_a_w(self.thickness*self.dy, self.k, self.dx)
        a_e = calculate_internal_a_e(self.thickness*self.dy, self.k, self.dx)
        a_n = calculate_internal_a_n(self.thickness*self.dx, self.k, self.dy)
        a_s = calculate_internal_a_s(self.thickness*self.dx, self.k, self.dy)
        s_p = calculate_internal_s_p() 
        s_u = calculate_internal_s_u() 
        
        # TODO; this will need to be changed, should probably just asign all coefficients iterating over the identity grid 
        a_p = calculate_internal_a_p(a_w, a_e, a_n, a_s, s_p) 

        # fill arrays with the values 
        self.a_w.fill(a_w)
        self.a_e.fill(a_e)
        self.a_n.fill(a_n)
        self.a_s.fill(a_s)
        self.a_p.fill(a_p)
        self.s_p.fill(s_p)
        self.s_u.fill(s_u)

        # account for boundary nodes 

        self.boundary_nodes = self.identify_boundary_nodes()

        # TODO: clean up this for loop into something smarter, maybe use 
        # for key, value in boundary_nodes.items() 
        for key in self.boundary_nodes.keys():

            if key == 'north_boundary':
                for node in self.boundary_nodes[key]: 
                    # source term 
                    self.s_u[node] = const_temp_boundary(k=self.k, area=self.dx*self.thickness, bc=self.bc_n, dist=self.dy) 
                    self.s_p[node] = -1*const_temp_boundary(k=self.k, area=self.dx*self.thickness, bc=self.bc_n, dist=self.dy) / self.bc_n
                    # coefs
                    self.a_n[node] = 0
                    # self.a_e[node] = a_e  # removed this because it's not a necessary operation as it was filled in last step 
                    self.a_p[node] = self.a_w[node] + self.a_e[node] + self.a_n[node] + self.a_s[node] - self.s_p[node]

            elif key == 'south_boundary': 
                for node in self.boundary_nodes[key]: 
                    # source term 
                    self.s_u[node] = insulated_boundary()
                    self.s_p[node] = insulated_boundary()
                    # coefs
                    self.a_s[node] = 0
                    # self.a_e[node] = a_e  # removed this because it's not a necessary operation as it was filled in last step 
                    self.a_p[node] = self.a_w[node] + self.a_e[node] + self.a_n[node] + self.a_s[node] - self.s_p[node] 

            elif key == 'west_boundary':
                for node in self.boundary_nodes[key]: 
                        # source term 
                    self.s_u[node] = const_flux_boundary(q=500E3, area=self.dy*self.thickness)
                    self.s_p[node] = insulated_boundary()
                    # coefs
                    self.a_w[node] = 0
                    # self.a_e[node] = a_e  # emoved this because it's not a necessary operation as it was filled in last step 
                    self.a_p[node] = self.a_w[node] + self.a_e[node] + self.a_n[node] + self.a_s[node] - self.s_p[node]

            elif key == 'east_boundary': 
                for node in self.boundary_nodes[key]: 
                        # source term 
                    self.s_u[node] = insulated_boundary()
                    self.s_p[node] = insulated_boundary()
                    # coefs 
                    self.a_e[node] = 0
                    # self.a_w[node] = a_w  # removed this because it's not a necesarry operation as it was filled in previous step
                    self.a_p[node] = self.a_w[node] + self.a_e[node] + self.a_n[node] + self.a_s[node] - self.s_p[node]

            elif key == 'northwest_boundary': 
                
                for node in self.boundary_nodes[key]:
                    self.s_u[node] = const_flux_boundary(q=500E3, area=self.dy*self.thickness) + (const_temp_boundary(self.k, self.dx*self.thickness, self.bc_n, self.dy))
                    self.s_p[node] = -1*const_temp_boundary(self.k, self.dx*self.thickness, self.bc_n, self.dx) / self.bc_n
                    # coefs 
                    self.a_w[node] = 0
                    self.a_n[node] = 0
                    # self.a_w[node] = a_w  # removed this because it's not a necesarry operation as it was filled in previous step
                    self.a_p[node] = self.a_w[node] + self.a_e[node] + self.a_n[node] + self.a_s[node] - self.s_p[node]

            elif key == 'northeast_boundary': 
                
                for node in self.boundary_nodes[key]:
                    self.s_u[node] = const_temp_boundary(self.k, self.dx*self.thickness, self.bc_n, self.dy)
                    self.s_p[node] = -1*const_temp_boundary(self.k, self.dx*self.thickness, self.bc_n, self.dy) / self.bc_n
                    # coefs 
                    self.a_e[node] = 0
                    self.a_n[node] = 0
                    # self.a_w[node] = a_w  # removed this because it's not a necesarry operation as it was filled in previous step
                    self.a_p[node] = self.a_w[node] + self.a_e[node] + self.a_n[node] + self.a_s[node] - self.s_p[node]

            elif key == 'southwest_boundary': 
                
                for node in self.boundary_nodes[key]:
                    self.s_u[node] = const_flux_boundary(500E3, self.dx*self.thickness)
                    self.s_p[node] = insulated_boundary()
                    # coefs 
                    self.a_w[node] = 0
                    self.a_s[node] = 0
                    # self.a_w[node] = a_w  # removed this because it's not a necesarry operation as it was filled in previous step
                    self.a_p[node] = self.a_w[node] + self.a_e[node] + self.a_n[node] + self.a_s[node] - self.s_p[node]

            elif key == 'southeast_boundary': 
                for node in self.boundary_nodes[key]:
                    self.s_u[node] = insulated_boundary()
                    self.s_p[node] = insulated_boundary()
                    # coefs 
                    self.a_e[node] = 0
                    self.a_s[node] = 0
                    # self.a_w[node] = a_w  # removed this because it's not a necesarry operation as it was filled in previous step
                    self.a_p[node] = self.a_w[node] + self.a_e[node] + self.a_n[node] + self.a_s[node] - self.s_p[node]

            else: 
                raise ValueError(f'Unknown boundary key {key}')

    def solve(self): 
        
        # calculate the convergence after a full pass has created, iteratate the for loop with a while loop to meet convergence 
        # TODO: need to fix this part, there is something wrong with the recalc of the node coeffs and is resulting in 3x3 output instead of 1x12 or.w.e
        passes = 0 
        
        error = 1
        phi_old = np.empty(self.x_nodes*self.y_nodes, dtype=float)
        phi_old.fill(1)

        while error >= 0.01: 

            for i in range(len(self.ident_grid)):
                
                bee = self.s_u[self.ident_grid[i]]
                alpha = self.a_n[self.ident_grid[i]]
                beta = self.a_s[self.ident_grid[i]]
                dee = self.a_p[self.ident_grid[i]]

                ##
                if i == 0: 
                    ay = np.multiply(self.a_e[self.ident_grid[i]], self.phi[self.ident_grid[i+1]])
                    bay = 0 
                elif i == len(self.ident_grid)-1: 
                    ay = 0 
                    bay = np.multiply(self.a_w[self.ident_grid[i]], self.phi[self.ident_grid[i-1]])
                else: 
                    ay = np.multiply(self.a_e[self.ident_grid[i]], self.phi[self.ident_grid[i+1]])
                    bay = np.multiply(self.a_w[self.ident_grid[i]], self.phi[self.ident_grid[i-1]])

                cee = ay + bay + bee
                # solver = TdmaCons(-1*alpha, dee, -1*beta, cee)
                # temp = solver.solve()
                # TODO: write unit tests for the tdma algs 
                temp = tdma_noncons(-1*alpha, dee, -1*beta, cee)
                self.phi[self.ident_grid[i]] = temp.copy()
            
            error = np.average(np.absolute(np.divide(np.subtract(self.phi, phi_old), phi_old)))
            phi_old = self.phi.copy()

            passes += 1
            print(f'Completed pass: {passes}, error: {error}')

        return self.phi

    def visualize(self): 

        plt.figure()
        plt.imshow(self.phi.reshape(self.x_nodes, self.y_nodes), cmap='hot', interpolation='nearest')
        plt.show()

