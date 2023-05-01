import unittest
import numpy as np
import sys

sys.path.insert(0, '/home/taylr/code_dir/CFD/')
from HeatTransfer1D.classes import Tdma
from HeatTransfer2D.functions import tdma

class Test(unittest.TestCase):
    """
    Test scenarios to ensure TDMA solver works properly and fails safely 
    """
    
    def setUp(self): 
        # create tridiagonal matrices 
        
        self.a = np.array([0, -100, -100, -100, -100], dtype=float)
        self.b = np.array([300, 200, 200, 200, 300], dtype=float)
        self.c = np.array([-100, -100, -100, -100, 0], dtype=float)
        self.d = np.array([20000, 0, 0, 0, 100000], dtype=float)

        matrix_a = np.diag(self.a[1:], k=1)
        matrix_b = np.diag(self.b, k=0)
        matrix_c = np.diag(self.c[:-1], k=-1)
        self.coef_matrix = matrix_a + matrix_b + matrix_c

        self.tdma = Tdma(self.a, self.b, self.c, self.d)

    def test_1_compare_tdma_class_to_linalg(self): 

        gt = np.linalg.solve(self.coef_matrix, self.d) 

        pr = self.tdma.solve()
        
        for i in range(self.tdma.Dim): 
            self.assertAlmostEqual(gt[i], pr[i])

        print(f'Test one passed, tdma solved {pr}, linalg solved {gt}')

    def test_2_compare_tdma_func_to_linalg(self): 

        gt = np.linalg.solve(self.coef_matrix, self.d)

        pr = tdma(self.a, self.b, self.c, self.d)

        for i in range(len(self.a)): 
            self.assertAlmostEqual(gt[i], pr[i])

        print(f'Test two passed, tdma solved {pr}, linalg solved {gt}')
        
if __name__ == "__main__": 
    unittest.main()