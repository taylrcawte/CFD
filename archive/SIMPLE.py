"""
A SIMPLE solver to solve the classic lid driven cavity flow problem

Author: Taylr Cawte
written on: 06/01/2020
"""

# import libraries
import scipy.sparse as sp
import numpy as np
import scipy.sparse.linalg as splin
import time
import matplotlib.pyplot as plt

# ------------------------------------------ FUNCTION DECLARATION --------------------------------------------------- #


def momentum_solve(x_nodes, y_nodes, matrix_a, dx, dy, index, gamma, top_vel, fe, fw, fs, fn, total_nodes, p, p_index):

    bx = np.zeros(total_nodes)
    by = np.zeros(total_nodes)

    for j in range(y_nodes):
        for i in range(x_nodes):
            if i == 0:  # west wall
                a_e = gamma * dy / dx - max(fe[index[i, j]] * dy, 0)
                a_w = 2*gamma * dy / dx + max(fw[index[i, j]] * dy, 0)
                matrix_a[index[i, j], index[i + 1, j]] = -a_e
                bx[index[i, j]] += 0
                by[index[i, j]] += 0
            elif i == x_nodes - 1:  # east wall
                a_e = 2*gamma * dy / dx - max(fe[index[i, j]] * dy, 0)
                a_w = gamma * dy / dx + max(fw[index[i, j]] * dy, 0)
                matrix_a[index[i, j], index[i - 1, j]] = -a_w
                bx[index[i, j]] += 0
                by[index[i, j]] += 0
            else:  # middle
                a_e = gamma * dy / dx - max(fe[index[i, j]] * dy, 0)
                a_w = gamma * dy / dx + max(fw[index[i, j]] * dy, 0)
                matrix_a[index[i, j], index[i + 1, j]] = -a_e
                matrix_a[index[i, j], index[i - 1, j]] = -a_w

            if j == 0:  # south wall
                a_s = 2*gamma * dx / dy + max(fs[index[i, j]] * dx, 0)
                a_n = gamma * dx / dy - max(fn[index[i, j]] * dx, 0)
                matrix_a[index[i, j], index[i, j + 1]] = -a_n
                bx[index[i, j]] += 0
                by[index[i, j]] += 0
            elif j == y_nodes - 1:  # lid
                a_s = gamma * dx / dy + max(fs[index[i, j]] * dx, 0)
                a_n = 2*gamma * dx / dy - max(fn[index[i, j]] * dx, 0)
                matrix_a[index[i, j], index[i - 1, j]] = -a_s
                bx[index[i, j]] += a_n*top_vel/0.6
                by[index[i, j]] += 0
            else:  # middle
                a_s = gamma * dx / dy + max(fs[index[i, j]] * dy, 0)
                a_n = gamma * dx / dy - max(fn[index[i, j]] * dy, 0)
                matrix_a[index[i, j], index[i, j + 1]] = -a_n
                matrix_a[index[i, j], index[i, j - 1]] = -a_s

            # pressure at faces
            pe = (p[p_index[i+1, j+1]]+p[p_index[i+2, j+1]])/2
            pw = (p[p_index[i, j+1]]+p[p_index[i+1, j+1]])/2
            pn = (p[p_index[i+1, j+1]]+p[p_index[i+1, j+2]])/2
            ps = (p[p_index[i+1, j]]+p[p_index[i+1, j+1]])/2

            # pressure change across the node
            bx[index[i, j]] += (pw - pe)*dy
            by[index[i, j]] += (ps - pn)*dx

            a_p = a_w + a_n + a_e + a_s
            matrix_a[index[i, j], index[i, j]] = a_p

    u = splin.spsolve(matrix_a.tocsr(), bx)
    v = splin.spsolve(matrix_a.tocsr(), by)

    return matrix_a, u, v


def flux_calc(x_nodes, y_nodes, dx, dy, index, p_index, rho, fe, fw, fs, fn, matrix_a, u, v, p):
    for j in range(y_nodes):
        for i in range(x_nodes):
            if i == 0:
                fw[index[i, j]] = 0
            else:
                # west flux if not at wall
                u_w = 0.5*(u[index[i, j]] + u[index[i - 1, j]]) + \
                      0.5 * (dy / matrix_a[index[i, j], index[i, j]] * ((p[p_index[i + 2, j + 1]] - p[p_index[i, j + 1]]) / 2) +
                             dy / matrix_a[index[i - 1, j], index[i - 1, j]] * ((p[p_index[i + 1, j + 1]] - p[p_index[i - 1, j + 1]]) / 2)) - \
                      0.5 * ((dy / matrix_a[index[i, j], index[i, j]] + dy / matrix_a[index[i - 1, j], index[i - 1, j]]) * (p[p_index[i + 1, j + 1]] -
                                                                                                                            p[p_index[i, j + 1]]))

                fw[index[i, j]] = rho * u_w

            if j == 0:
                fs[index[i, j]] = 0
            else:
                # south flux if not wall
                v_s = 0.5*(v[index[i, j]] + v[index[i, j - 1]]) + \
                      0.5 * (dx / matrix_a[index[i, j], index[i, j]] * ((p[p_index[i + 1, j + 2]] -  p[p_index[i + 1, j]]) / 2) +
                             dx / matrix_a[index[i, j - 1], index[i, j - 1]] * ((p[p_index[i + 1, j + 1]] - p[p_index[i + 1, j - 1]]) / 2)) - \
                      0.5 * ((dx / matrix_a[index[i, j], index[i, j]] + dx / matrix_a[index[i, j - 1], index[i, j - 1]]) * (p[p_index[i + 1, j + 1]] -
                                                                                                                            p[p_index[i + 1, j]]))
                fs[index[i, j]] = rho * v_s

    # north and east fluxes are the south and west fluxes from adjacent cell
    for j in range(y_nodes):
        for i in range(x_nodes):
            if i == x_nodes - 1:
                fe[index[i, j]] = 0
            else:
                fe[index[i, j]] = fw[index[i + 1, j]]

            if j == y_nodes - 1:
                fn[index[i, j]] = 0
            else:
                fn[index[i, j]] = fs[index[i, j + 1]]

    return fe, fw, fs, fn


def pressure_correction(x_nodes, y_nodes, dx, dy, rho, index, matrix_a, fe, fw, fn, fs, total_nodes):
    ap_cor = sp.lil_matrix((total_nodes, total_nodes))
    bp_cor = np.zeros(total_nodes)

    # bottom left corner
    a_p_ebl = rho * 2 * dy / (matrix_a[index[1, 0], index[1, 0]] + matrix_a[index[0, 0], index[0, 0]]) * dy
    a_p_nbl = rho * 2 * dx / (matrix_a[index[0, 1], index[0, 1]] + matrix_a[index[0, 0], index[0, 0]]) * dx

    a_p_pbl = a_p_ebl + a_p_nbl
    ap_cor[index[0, 0], index[0, 0]] = a_p_pbl
    ap_cor[index[0, 0], index[1, 0]] = -a_p_ebl
    ap_cor[index[0, 0], index[0, 1]] = -a_p_nbl

    # top left corner
    a_p_etl = rho * 2 * dy / (matrix_a[index[0 + 1, y_nodes - 1], index[0 + 1, y_nodes - 1]] +
                              matrix_a[index[0, y_nodes - 1], index[0, y_nodes - 1]]) * dy
    a_p_stl = rho * 2 * dx / (matrix_a[index[0, y_nodes - 1 - 1], index[0, y_nodes - 1 - 1]] +
                              matrix_a[index[0, y_nodes - 1], index[0, y_nodes - 1]]) * dx

    a_p_ptl = a_p_etl + a_p_stl
    ap_cor[index[0, y_nodes - 1], index[0, y_nodes - 1]] = a_p_ptl
    ap_cor[index[0, y_nodes - 1], index[1, y_nodes - 1]] = -a_p_etl
    ap_cor[index[0, y_nodes - 1], index[0, y_nodes - 2]] = -a_p_stl

    # bottom right corner
    a_p_wbr = rho * 2 * dy / (matrix_a[index[x_nodes - 2, 0], index[x_nodes - 2, 0]] +
                              matrix_a[index[x_nodes - 1, 0], index[x_nodes - 1, 0]]) * dy
    a_p_nbr = rho * 2 * dx / (matrix_a[index[x_nodes - 1, 1], index[x_nodes - 1, 1]] +
                              matrix_a[index[x_nodes - 1, 0], index[x_nodes - 1, 0]]) * dx

    a_p_pbr = a_p_wbr + a_p_nbr
    ap_cor[index[x_nodes - 1, 0], index[x_nodes - 1, 0]] = a_p_pbr
    ap_cor[index[x_nodes - 1, 0], index[x_nodes - 2, 0]] = -a_p_wbr
    ap_cor[index[x_nodes - 1, 0], index[x_nodes - 1, 1]] = -a_p_nbr

    # top right corner
    a_p_wtr = rho * 2 * dy / (matrix_a[index[x_nodes - 2, y_nodes - 1], index[x_nodes - 2, y_nodes - 1]] +
                              matrix_a[index[x_nodes - 1, y_nodes - 1], index[x_nodes - 1, y_nodes - 1]]) * dy
    a_p_str = rho * 2 * dx / (matrix_a[index[x_nodes - 1, y_nodes - 2], index[x_nodes - 1, y_nodes - 2]] +
                              matrix_a[index[x_nodes - 1, y_nodes - 1], index[x_nodes - 1, y_nodes - 1]]) * dx

    a_p_ptr = a_p_wtr + a_p_str
    ap_cor[index[x_nodes - 1, y_nodes - 1], index[x_nodes - 1, y_nodes - 1]] = a_p_ptr
    ap_cor[index[x_nodes - 1, y_nodes - 1], index[x_nodes - 2, y_nodes - 1]] = -a_p_wtr
    ap_cor[index[x_nodes - 1, y_nodes - 1], index[x_nodes - 1, y_nodes - 2]] = -a_p_str

    for j in range(y_nodes):
        for i in range(x_nodes):
            a_p = matrix_a[index[i, j], index[i, j]]

            if (i == 0 and j == 0) or (i == 0 and j == y_nodes-1) or (i == x_nodes-1 and j == y_nodes-1) \
                    or (i == x_nodes-1 and j == 0):  # continues past else statements
                continue

            elif i == 0 and j != 0 and j != y_nodes - 1:  # left side not in corners
                a_p_e = rho * 2 * dy / (matrix_a[index[i + 1, j], index[i + 1, j]] + a_p) * dy
                a_p_n = rho * 2 * dx / (matrix_a[index[i, j + 1], index[i, j + 1]] + a_p) * dx
                a_p_s = rho * 2 * dx / (matrix_a[index[i, j - 1], index[i, j - 1]] + a_p) * dx

                a_p_p = a_p_e + a_p_n + a_p_s

                ap_cor[index[i, j], index[i, j]] = a_p_p
                ap_cor[index[i, j], index[i + 1, j]] = -a_p_e
                ap_cor[index[i, j], index[i, j - 1]] = -a_p_s
                ap_cor[index[i, j], index[i, j + 1]] = -a_p_n

            elif i == x_nodes - 1 and j != 0 and j != y_nodes - 1:  # right side not in corners
                a_p_w = rho * 2 * dy / (matrix_a[index[i - 1, j], index[i - 1, j]] + a_p) * dy
                a_p_n = rho * 2 * dx / (matrix_a[index[i, j + 1], index[i, j + 1]] + a_p) * dx
                a_p_s = rho * 2 * dx / (matrix_a[index[i, j - 1], index[i, j - 1]] + a_p) * dx

                a_p_p = a_p_w + a_p_n + a_p_s

                ap_cor[index[i, j], index[i, j]] = a_p_p
                ap_cor[index[i, j], index[i, j - 1]] = -a_p_s
                ap_cor[index[i, j], index[i, j + 1]] = -a_p_n
                ap_cor[index[i, j], index[i - 1, j]] = -a_p_w

            elif j == 0 and i != 0 and i != x_nodes - 1:  # bottom not in corners
                a_p_w = rho * 2 * dy / (matrix_a[index[i - 1, j], index[i - 1, j]] + a_p) * dy
                a_p_e = rho * 2 * dy / (matrix_a[index[i + 1, j], index[i + 1, j]] + a_p) * dy
                a_p_n = rho * 2 * dx / (matrix_a[index[i, j + 1], index[i, j + 1]] + a_p) * dx

                a_p_p = a_p_w + a_p_e + a_p_n

                ap_cor[index[i, j], index[i, j]] = a_p_p
                ap_cor[index[i, j], index[i - 1, j]] = -a_p_w
                ap_cor[index[i, j], index[i + 1, j]] = -a_p_e
                ap_cor[index[i, j], index[i, j + 1]] = -a_p_n

            elif j == y_nodes - 1 and i != 0 and i != x_nodes - 1:  # top not in corners
                a_p_w = rho * 2 * dy / (matrix_a[index[i - 1, j], index[i - 1, j]] + a_p) * dy
                a_p_e = rho * 2 * dy / (matrix_a[index[i + 1, j], index[i + 1, j]] + a_p) * dy
                a_p_s = rho * 2 * dx / (matrix_a[index[i, j - 1], index[i, j - 1]] + a_p) * dx

                a_p_p = a_p_w + a_p_e + a_p_s

                ap_cor[index[i, j], index[i, j]] = a_p_p
                ap_cor[index[i, j], index[i - 1, j]] = -a_p_w
                ap_cor[index[i, j], index[i + 1, j]] = -a_p_e
                ap_cor[index[i, j], index[i, j - 1]] = -a_p_s

            else:  # all middle nodes
                a_p_w = rho * 2 * dy / (matrix_a[index[i - 1, j], index[i - 1, j]] + a_p) * dy
                a_p_e = rho * 2 * dy / (matrix_a[index[i + 1, j], index[i + 1, j]] + a_p) * dy
                a_p_n = rho * 2 * dx / (matrix_a[index[i, j + 1], index[i, j + 1]] + a_p) * dx
                a_p_s = rho * 2 * dx / (matrix_a[index[i, j - 1], index[i, j - 1]] + a_p) * dx

                a_p_p = a_p_w + a_p_n + a_p_s + a_p_e

                ap_cor[index[i, j], index[i, j]] = a_p_p
                ap_cor[index[i, j], index[i - 1, j]] = -a_p_w
                ap_cor[index[i, j], index[i + 1, j]] = -a_p_e
                ap_cor[index[i, j], index[i, j - 1]] = -a_p_s
                ap_cor[index[i, j], index[i, j + 1]] = -a_p_n

            bp_cor[index[i, j]] = (fw[index[i, j]] - fe[index[i, j]]) * dy + (fs[index[i, j]] - fn[index[i, j]]) * dx

    p_prime = splin.spsolve(ap_cor.tocsr(), bp_cor)
    p_ref = p_prime[index[0, 0]]

    return p_prime, p_ref


def correct_u(x_nodes, y_nodes, dy, u, index, matrix_a, p_prime, alpha_u):
    u_new = np.zeros(x_nodes*y_nodes)

    for j in range(y_nodes):
        for i in range(x_nodes):
            if i == 0:
                p_prime_e = 0.5*(p_prime[index[i, j]] + p_prime[index[i+1, j]])
                p_prime_w = 0.25*(5*p_prime[index[i, j]] - p_prime[index[i+1, j]])
            elif i == x_nodes-1:
                p_prime_e = 0.25*(5*p_prime[index[i, j]] - p_prime[index[i-1, j]])
                p_prime_w = 0.5*(p_prime[index[i, j]] + p_prime[index[i-1, j]])
            else:
                p_prime_e = 0.5*(p_prime[index[i, j]] + p_prime[index[i+1, j]])
                p_prime_w = 0.5*(p_prime[index[i, j]] + p_prime[index[i-1, j]])

            u_new[index[i, j]] = u[index[i, j]] + alpha_u * dy / matrix_a[index[i, j], index[i, j]] * (p_prime_w -
                                                                                                       p_prime_e)

    return u_new


def correct_v(x_nodes, y_nodes, dx, index, matrix_a, v, p_prime, alpha_v):
    v_new = np.zeros(x_nodes*y_nodes)

    for j in range(y_nodes):
        for i in range(x_nodes):
            if j == 0:
                p_prime_n = 0.5*(p_prime[index[i, j]] + p_prime[index[i, j+1]])
                p_prime_s = 0.25*(5*p_prime[index[i, j]] - p_prime[index[i, j+1]])
            elif j == y_nodes-1:
                p_prime_n = 0.25*(5*p_prime[index[i, j]] - p_prime[index[i, j-1]])
                p_prime_s = 0.5*(p_prime[index[i, j]] + p_prime[index[i, j-1]])
            else:
                p_prime_n = 0.5*(p_prime[index[i, j]] + p_prime[index[i, j+1]])
                p_prime_s = 0.5*(p_prime[index[i, j]] + p_prime[index[i, j-1]])

            v_new[index[i, j]] = v[index[i, j]] + alpha_v * dx / matrix_a[index[i, j], index[i, j]] * (p_prime_s -
                                                                                                       p_prime_n)
    return v_new


def correct_p(x_nodes, y_nodes, p_old, alpha_p, p_prime, p_index, index, p_ref):
    p = np.zeros((x_nodes+2)*(y_nodes*2))

    for j in range(y_nodes):
        for i in range(x_nodes):
            p[p_index[i+1, j+1]] = p_old[p_index[i+1, j+1]] + alpha_p*(p_prime[index[i, j]] - p_ref)

    return p


def extrapolate_p(x_nodes, y_nodes, p_index, p):

    for j in range(1, y_nodes+1):
        i = 0
        p[p_index[i, j]] = 0.5*(3*p[p_index[i+1, j]] - p[p_index[i+2, j]])

        i = x_nodes + 1
        p[p_index[i, j]] = 0.5*(3*p[p_index[i-1, j]] - p[p_index[i-2, j]])

    for i in range(0, y_nodes+2):
        j = 0
        p[p_index[i, j]] = 0.5*(3*p[p_index[i, j+1]] - p[p_index[i, j+2]])

        j = x_nodes + 1
        p[p_index[i, j]] = 0.5*(3*p[p_index[i, j-1]] - p[p_index[i, j-2]])

    return p


# --------------------------------------------------- MAIN ---------------------------------------------------------- #

start = time.time()  # starts process timer

# domain variables
xNodes = 200
yNodes = 200
CVL = 1
CVH = 1
mu = float(0.01)
density = float(1)
lidVel = float(1)

tNodes = xNodes * yNodes
DeltaX = CVL / xNodes
DeltaY = CVH / yNodes

Index = np.arange(tNodes).reshape((xNodes, yNodes))
pIndex = np.arange((xNodes + 2) * (yNodes + 2)).reshape(((xNodes + 2), (yNodes + 2)))
matrixA = sp.lil_matrix((tNodes, tNodes))

# relaxation factors
AlphaU = float(0.0001)
AlphaV = float(0.0001)
AlphaP = float(0.00001)

# iteration requirements
tol = float(0.00001)
maxIt = 3000
uRms = 1
vRms = 1
iteration = 0

# initial guesses and momentum
pInit = np.zeros((xNodes + 2) * (yNodes + 2))
feInit = np.zeros(tNodes)
fwInit = np.zeros(tNodes)
fnInit = np.zeros(tNodes)
fsInit = np.zeros(tNodes)

matrixA, initU, initV = momentum_solve(x_nodes=xNodes, y_nodes=yNodes, matrix_a=matrixA, total_nodes=tNodes, dx=DeltaX,
                                       dy=DeltaY, index=Index, gamma=mu, top_vel=lidVel, fe=feInit, fw=fwInit,
                                       fs=fsInit, fn=fnInit, p=pInit, p_index=pIndex)

OldU = initU
OldV = initV
OldP = pInit
OldFe = feInit
OldFw = fwInit
OldFn = fnInit
OldFs = fsInit

urmsLog = np.array(uRms)
vrmsLog = np.array(vRms)

# iterative solver
while (uRms > tol or vRms > tol) and iteration < maxIt:
    # calculate fluxes
    newFe, newFw, newFs, newFn = flux_calc(x_nodes=xNodes, y_nodes=yNodes, dx=DeltaX, dy=DeltaY, index=Index,
                                           p_index=pIndex, rho=density, fe=OldFe, fw=OldFw, fs=OldFs, fn=OldFn,
                                           matrix_a=matrixA, u=OldU, v=OldV, p=OldP)

    # Pressure Correction factor
    pPrime, pRef = pressure_correction(x_nodes=xNodes, y_nodes=yNodes, dx=DeltaX, dy=DeltaY, rho=density,
                                       index=Index, matrix_a=matrixA, fe=newFe, fw=newFw, fn=newFn, fs=newFs,
                                       total_nodes=tNodes)

    # correct u, v, p
    uCor = correct_u(x_nodes=xNodes, y_nodes=yNodes, u=OldU, dy=DeltaY, index=Index, matrix_a=matrixA, p_prime=pPrime,
                     alpha_u=AlphaU)
    vCor = correct_v(x_nodes=xNodes, y_nodes=yNodes, v=OldV, dx=DeltaX, index=Index, matrix_a=matrixA, p_prime=pPrime,
                     alpha_v=AlphaV)
    pCor = correct_p(x_nodes=xNodes, y_nodes=yNodes, p_old=OldP, alpha_p=AlphaP, p_prime=pPrime, p_index=pIndex,
                     index=Index, p_ref=pRef)

    # correct fluxes
    feCor, fwCor, fsCor, fnCor = flux_calc(x_nodes=xNodes, y_nodes=yNodes, dx=DeltaX, dy=DeltaY, index=Index,
                                           p_index=pIndex, rho=density, fe=newFe, fw=newFw, fs=newFs, fn=newFn,
                                           matrix_a=matrixA, u=uCor, v=vCor, p=pCor)

    # extrapolate pressure
    pNew = extrapolate_p(x_nodes=xNodes, y_nodes=yNodes, p_index=pIndex, p=pCor)

    # recalculate the momentum equation
    matrixA, uFin, vFin = momentum_solve(x_nodes=xNodes, y_nodes=yNodes, matrix_a=matrixA, total_nodes=tNodes,
                                         dx=DeltaX, dy=DeltaY, index=Index, p_index=pIndex, p=pNew, gamma=mu, fw=fwCor,
                                         fe=feCor, fn=fnCor, fs=fsCor, top_vel=lidVel)

    # calculate error between velocity at begin and end of iteration to estimate convergence
    uRms = (np.sum((OldU - uFin)**2)/(xNodes*yNodes))**0.5
    vRms = (np.sum((OldV - vFin)**2)/(xNodes*yNodes))**0.5

    urmsLog = np.append(urmsLog, uRms)
    vrmsLog = np.append(vrmsLog, vRms)

    # reassign values if convergence criteria are not met
    OldFe = feCor
    OldFw = fwCor
    OldFs = fsCor
    OldFn = fnCor
    OldP = np.copy(pNew)
    OldU = uFin
    OldV = vFin

    iteration += 1
    print('iteration: {} \t uRms: {:.5} \t vRms: {:.5}'.format(iteration, uRms, vRms))

end = time.time()
print("Total Run time: {} s".format(end-start))

# figure prep and plots
xPos = np.zeros(xNodes)
yPos = np.zeros(yNodes)
for w in range(xNodes):
    if w == 0:
        xPos[w] = DeltaX / 2
    else:
        xPos[w] = xPos[w - 1] + DeltaX
for w in range(yNodes):
    if w == 0:
        yPos[w] = DeltaY / 2
    else:
        yPos[w] = yPos[w - 1] + DeltaY
x, y = np.meshgrid(xPos, yPos)

U = uFin.reshape(xNodes, yNodes, order='f')
V = vFin.reshape(xNodes, yNodes, order='f')
uVec = np.sqrt(np.square(U)+np.square(V))
Diag1 = np.diag(V)
Diag2 = np.diag(np.fliplr(U))
Diag3 = np.diag(np.fliplr(uVec))

# velocity streamlines
plt.streamplot(x, y, U, V, density=3, linewidth=0.3)
plt.title("HW4Final, grid: {}, tolerance: {}, iterations: {}, alpha UV: {}, alpha P: {}".format(xNodes, tol, iteration,
                                                                                                AlphaU, AlphaP))
plt.show()

# u velocity contour
plt.contourf(x, y, U, levels=10)
plt.colorbar()
plt.title("HW4Final, grid: {}, tolerance: {}, iterations: {}, alpha UV: {}, alpha P: {}".format(xNodes, tol, iteration,
                                                                                                AlphaU, AlphaP))
plt.show()

# v velocity contour
plt.contourf(x, y, V, levels=10)
plt.colorbar()
plt.title("HW4Final, grid: {}, tolerance: {}, iterations: {}, alpha UV: {}, alpha P: {}".format(xNodes, tol, iteration,
                                                                                                AlphaU, AlphaP))
plt.show()

# diagonal plot
plt.plot(xPos, Diag1)
plt.xlabel('position along diagonal')
plt.ylabel('V velocity')
plt.show()

plt.plot(xPos, Diag2)
plt.xlabel('position along diagonal')
plt.ylabel('U velocity')
plt.show()

plt.plot(xPos, Diag3)
plt.xlabel('position along diagonal')
plt.ylabel('Total velocity')
plt.show()
