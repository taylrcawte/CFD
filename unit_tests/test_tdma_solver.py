import unittest
import numpy as np

from modules.classes.classes1d import Tdma 
from modules.functions.functions2d import tdma_noncons, tdma_cons

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

    def test_1_compare_tdma1d_class_to_linalg(self): 

        gt = np.linalg.solve(self.coef_matrix, self.d) 
        self.tdma = Tdma(self.a, self.b, self.c, self.d)
        pr = self.tdma.solve()
        
        for i in range(self.tdma.Dim): 
            self.assertAlmostEqual(gt[i], pr[i])

    def test_2_compare_tdma2d_noncons_to_linalg(self): 

        gt = np.linalg.solve(self.coef_matrix, self.d)

        pr = tdma_noncons(self.a, self.b, self.c, self.d)

        for i in range(len(self.a)): 
            self.assertAlmostEqual(gt[i], pr[i])

    def test_3_compare_tdma2d_cons_to_linalg(self): 

        gt = np.linalg.solve(self.coef_matrix, self.d)

        pr = tdma_cons(self.a, self.b, self.c, self.d)

        for i in range(len(self.a)): 
            self.assertAlmostEqual(gt[i], pr[i])        

        
if __name__ == "__main__": 
    unittest.main()