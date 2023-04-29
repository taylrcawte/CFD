import unittest
import numpy as np 
from classes import Tdma

class Test(unittest.TestCase):
    """
    Test scenarios to ensure TDMA solver works properly and fails safely 
    """
    
    def setUp(self): 
        # create tridiagonal matrices 
        self.a = np.array([0, -1, -1, -1, -1])
        self.b = np.array([2, 2, 2, 2, 2])
        self.c = np.array([-1, -1, -1, -1, 0])
        self.d = np.array([1, 0, 0, 0, 7])

        matrix_a = np.diag(self.a_w[1:], k=1)
        matrix_b = np.diag(self.a_e[:-1], k=-1)
        matrix_c = np.diag(self.a_p, k=0)
        self.coef_matrix = matrix_a + matrix_b + matrix_c

        self.solver = Tdma(self.a, self.b, self.c, self.d)

    def test_1_compare_tdma_to_linalg(self): 

        gt = np.linalg.solve(self.coef_matrix, self.d) 

        self.solver.calculate_coefficients()
        
        pr = self.solver.solve() 

        if gt != pr: 
            raise ValueError(f'ground truth and tdma solved linalg system should be equivalent, results were {gt} and {pr}')