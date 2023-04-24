import numpy as np 

class HeatTransfer1D(object): 

    def __init__(self) -> None:
        
        self.x_nodes = 5
        self.length = 0.5  # meters
        self.k = 1000  # W / m.K 
        self.area = 10E-3  # m^2
        self.bc1 = 500  # K 
        self.bc2 = 100  # K
        self.a_p = np.empty(self.x_nodes, dtype=float) 
        self.a_e = np.empty(self.x_nodes, dtype=float)
        self.a_w = np.empty(self.x_nodes, dtype=float)
        self.dx = self.length / self.x_nodes
    
    def identify_boundary_nodes(self) -> dict: 
        # since this is 1D we can assume the first and last nodes are the boundary nodes 
        # this process will become more complilicicated as dimensions increase 
        return {'west_boundary': [0], 'east_boundary': [self.x_nodes - 1]}

    def calculate_coefficients(self) -> None:

        a_w = self.area*(self.k/self.dx)
        a_e = self.area*(self.k/self.dx)
        a_p = a_w + a_e 

        self.a_w.fill(a_w)
        self.a_e.fill(a_e)
        self.a_p.fill(a_p)

        # account for boundary nodes 

        boundary_nodes = self.identify_boundary_nodes()

        for key in boundary_nodes.keys(): 

            for node in boundary_nodes[key]: 

                if key == 'west_boundary': 

                    self.a_w[node] = 0
                    self.a_e[node] = a_e 
                    s_p = ((-2*self.k*self.area)/self.dx) 
                    self.a_p[node] = self.a_w[node] + self.a_e[node] - s_p

                else: 

                    self.a_e[node] = 0
                    self.a_w[node] = a_w 
                    s_p = ((-2*self.k*self.area)/self.dx) 
                    self.a_p[node] = self.a_w[node] + self.a_e[node] - s_p

        print(f'a_w:{self.a_w}')
        print(f'a_e:{self.a_e}')
        print(f'a_p:{self.a_p}')

        