"""
TDMA solver to solve 1D heat transfer

Author: Taylr Cawte
written on: September 1, 2020
"""
import numpy as np


def tdma_solver(A, d):

    # the thomas algorithm or the tridiagonal matrix algorithm is a simplified form of gaussian elimination that
    # can be used to solve tridaigonal systems of equations, the solution for the system may be written as
    # ax(i-1) + bx(i) + cx(i+1) = d
    a = np.diag(A, k=-1)  # k-1 diagonal of matrix A
    b = np.diag(A)  # diagonal of matrix A
    c = np.diag(A, k=1)  # k +1 diagonal of matrix A
    nf = len(d)  # number of equations

    # copy equations to dummy arrays so that changes arent mapped back to original variables
    ac = np.copy(a)
    bc = np.copy(b)
    cc = np.copy(c)
    dc = np.copy(d)

    # iterate through diagonal arrays according to TDMA algorithm and solve for x
    for it in range(1, nf):
        # matrix is decomposed into L and U triangular matrices, m is overwritten for each new value being evaluated
        m = ac[it-1]/bc[it-1]
        bc[it] = bc[it] - m*cc[it-1]
        # forward substitution
        dc[it] = dc[it] - m*dc[it-1]

    x = bc
    x[-1] = dc[-1]/bc[-1]

    for il in range(nf-2, -1, -1):
        x[il] = (dc[il]-cc[il]*x[il+1])/bc[il]  # backwards substitution starting backwards and moving to the beginning

    return x  # returns solution matrix x


# define arrays of form AX = B
matrixA = np.array([
    [300, -100, 0, 0, 0],
    [-100, 200, -100, 0, 0],
    [0, -100, 200, -100, 0],
    [0, 0, -100, 200, -100],
    [0, 0, 0, -100, 300]], dtype=float)
matrixB = np.array([20000, 0, 0, 0, 100000], dtype=float)

print("TDMA solved matrix: \n{}".format(tdma_solver(matrixA, matrixB)))

# direct solver used to check lower order matrices against TDMA method
print("Direct: {}".format(np.linalg.solve(matrixA, matrixB)))
