"""
A finite volume solver capable of computing upwind differencing method or centred differencing method to solve
heat transfer in a fluid of either rotating or linear velocity. The control volume has dirichlet boundary conditions.

Author: Taylr Cawte
written on: 25/11/2019

In order to run this model for different parameters toggle the inputs in the instance of the fin_val function
at the bottom of the script. Differencing scheme, velocity type, and diffusion can be changed.

"""

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as sparlin
import matplotlib.pyplot as plt
import scipy.interpolate as interp

# constants
phi_x0 = 100  # temperature at left boundary
phi_xl = 0  # temperature at right boundary
phi_xt = 100   # temperature at top
phi_xb = 0  # temp at base
CV_l = 1  # length of control volume
CV_h = 1  # height of control volume


def fin_vol(x_nodes, y_nodes, velocity_type, gamma, dif_scheme, plot):
    """
    :param x_nodes: number of x nodes
    :param y_nodes: number of y nodes
    :param velocity_type: linear velocity (ux = 2, uy = 2) or rotating velocity
    :param gamma: diffusion coefficient
    :param dif_scheme: selects the differencing scheme which is being used, either central differencing 'CD' or upwind
    differencing 'UW'
    :param plot: would you like a plot to be displayed 'yes' or 'no'
    :return: returns important parameters like phi, xindex etc...
    """

    # bring in global variables
    global phi_xt, phi_xl, phi_x0, phi_xb, CV_h, CV_l

    dx = CV_l / x_nodes  # distance between nodes in the x direction
    dy = CV_h / y_nodes   # distance between nodes in the y direction
    total_nodes = y_nodes*x_nodes  # total number of nodes

    x_index = np.zeros(x_nodes)  # initialize x nodes matrix to zero
    y_index = np.zeros(y_nodes)  # initialize y nodes matrix to zero
    matrix_index = np.arange(total_nodes).reshape(x_nodes, y_nodes)  # matrix index to map node position within CV
    matrix_a = sp.lil_matrix((total_nodes, total_nodes))  # coefficent matrix of form AX = B, "A"
    matrix_b = np.zeros(total_nodes)  # product matrix of form AX = B, "B"
    r = np.zeros(total_nodes)
    theta = np.zeros(total_nodes)
    ux = np.zeros((x_nodes, y_nodes))
    uy = np.zeros((x_nodes, y_nodes))

    # x position of node within control volume
    for i in range(x_nodes):
        if i == 0:
            x_index[i] = dx/2  # initial node position within control volume
        else:
            x_index[i] = x_index[i-1]+dx  # add dx to the last node to get next node position
    # y position of node within control volume
    for j in range(y_nodes):
        if j == 0:
            y_index[j] = dy/2  # initial y node position within control volume
        else:
            y_index[j] = y_index[j-1]+dy  # add dy to last node to get next node position

    if velocity_type == 'rotational':
        for j in range(y_nodes):
            for i in range(x_nodes):
                r[matrix_index[i, j]] = np.sqrt(np.square(x_index[i] - (CV_l/2)) + np.square(y_index[j] - (CV_h/2)))
                theta[matrix_index[i, j]] = np.arctan2((y_index[j] - CV_l / 2), (x_index[i] - CV_l / 2))
                ux[i, j] = -r[matrix_index[i, j]]*np.sin(theta[matrix_index[i, j]])
                uy[i, j] = r[matrix_index[i, j]]*np.cos(theta[matrix_index[i, j]])

    else:
        for i in range(0, x_nodes):
            for j in range(0, y_nodes):
                ux[i, j] = 2
                uy[i, j] = 2

    if dif_scheme == 'CD':

        for i in range(y_nodes):
            for j in range(x_nodes):
                if i == 0:  # northern node
                    a_n = gamma*(dx/dy)-uy[j, i]*dx/2
                    a_s = 2*gamma*(dx/dy)+uy[j, i]*dx
                    matrix_a[matrix_index[j, i], matrix_index[j, i+1]] = -a_n
                    matrix_b[matrix_index[j, i]] += a_s*phi_xb
                elif i == y_nodes-1:  # southern most node
                    a_n = 2*gamma*(dx/dy)-uy[j, i]*dx
                    a_s = gamma*(dx/dy)+uy[j, i]*dx/2
                    matrix_a[matrix_index[j, i], matrix_index[j, i-1]] = -a_s
                    matrix_b[matrix_index[j, i]] += a_n*phi_xt
                else:  # middle nodes
                    a_n = gamma*(dx/dy)-uy[j, i]*dx/2
                    a_s = gamma*(dx/dy)+uy[j, i]*dx/2
                    matrix_a[matrix_index[j, i], matrix_index[j, i+1]] = -a_n  # updates coeffs to the right
                    matrix_a[matrix_index[j, i], matrix_index[j, i-1]] = -a_s  # updates coeffs to the left

                if j == 0:
                    a_w = 2*gamma*(dy/dx)+ux[j, i]*dy
                    a_e = gamma*(dy/dx)-ux[j, i]*dy/2
                    matrix_a[matrix_index[j, i], matrix_index[j+1, i]] = -a_e  # updates row below ap
                    matrix_b[matrix_index[j, i]] += a_w*phi_x0  # updates product matrix
                elif j == x_nodes-1:
                    a_w = gamma*(dy/dx)+ux[j, i]*dy/2
                    a_e = 2*gamma*(dy/dx)-ux[j, i]*dy  # divide by 2 because at eastern most node dx is halved
                    matrix_a[matrix_index[j, i], matrix_index[j-1, i]] = -a_w  # updates row above ap
                    matrix_b[matrix_index[j, i]] += a_e*phi_xl
                else:
                    a_w = gamma*(dy/dx)+ux[j, i]*dy/2
                    a_e = gamma*(dy/dx)-ux[j, i]*dy/2
                    matrix_a[matrix_index[j, i], matrix_index[j+1, i]] = -a_e
                    matrix_a[matrix_index[j, i], matrix_index[j-1, i]] = -a_w

                a_p = a_n + a_s + a_w + a_e
                matrix_a[matrix_index[j, i], matrix_index[j, i]] = a_p

        # solves sparse matrices a and b
        phi_p = sparlin.spsolve(matrix_a.tocsc(), matrix_b).reshape(x_nodes, y_nodes)
        x, y = np.meshgrid(np.linspace(0, CV_l, x_nodes), np.linspace(0, CV_h, y_nodes))

    elif dif_scheme == 'UW':

        for i in range(y_nodes):
            for j in range(x_nodes):
                if i == 0:  # northern most node
                    a_n = gamma*(dx/(dy/2)) + max(-uy[j, i]*dx, 0)
                    a_s = gamma*(dx/dy) + max(uy[j, i]*dx, 0)
                    matrix_a[matrix_index[j, i], matrix_index[j, i+1]] = -a_n
                    matrix_b[matrix_index[j, i]] += a_s*phi_xb
                elif i == y_nodes-1:  # northern most node
                    a_n = gamma*(dx/dy) + max(-uy[j, i]*dx, 0)
                    a_s = gamma*(dx/(dy/2)) + max(uy[i, j]*dx, 0)
                    matrix_a[matrix_index[j, i], matrix_index[j, i-1]] = -a_s
                    matrix_b[matrix_index[j, i]] += a_n*phi_xt
                else:  # middle nodes
                    a_n = gamma*(dx/dy) + max(-uy[j, i]*dx, 0)
                    a_s = gamma*(dx/dy) + max(uy[j, i]*dx, 0)
                    matrix_a[matrix_index[j, i], matrix_index[j, i+1]] = -a_n
                    matrix_a[matrix_index[j, i], matrix_index[j, i-1]] = -a_s

                if j == 0:
                    a_w = gamma*(dy/(dx/2)) + max(ux[j, i]*dy, 0)
                    a_e = gamma*(dy/dx) + max(-ux[j, i]*dy, 0)
                    matrix_a[matrix_index[j, i], matrix_index[j+1, i]] = -a_e  # updates row below ap
                    matrix_b[matrix_index[j, i]] += a_w*phi_x0  # updates product matrix
                elif j == x_nodes-1:
                    a_w = gamma*(dy/dx) + max(ux[j, i]*dy, 0)
                    a_e = gamma*(dy/(dx/2)) + max(-ux[j, i]*dy, 0)
                    matrix_a[matrix_index[j, i], matrix_index[j-1, i]] = -a_w
                    matrix_b[matrix_index[j, i]] += a_e*phi_xl
                else:
                    a_w = gamma*(dy/dx) + max(ux[j, i]*dy, 0)
                    a_e = gamma*(dy/dx) + max(-ux[j, i]*dy, 0)
                    matrix_a[matrix_index[j, i], matrix_index[j+1, i]] = -a_e
                    matrix_a[matrix_index[j, i], matrix_index[j-1, i]] = -a_w

                a_p = a_n + a_s + a_w + a_e  # updates coefficient at point p (i.e. the investigated node)
                matrix_a[matrix_index[j, i], matrix_index[j, i]] = a_p  # sets node value at i,j equal to ap

        # solves sparse matrices a and b
        phi_p = sparlin.spsolve(matrix_a.tocsc(), matrix_b).reshape(x_nodes, y_nodes)
        x, y = np.meshgrid(np.linspace(0, CV_l, x_nodes), np.linspace(0, CV_h, y_nodes))
        print(phi_p)

    else:
        print('invalid input')
        exit(0)

    if plot == 'yes':
        plt.figure()
        plt.title("Mesh size: {} x {}, Method: {}, Diffusion: {}".format(x_nodes, y_nodes, dif_scheme, gamma))
        plt.xlabel('CV width')
        plt.ylabel('CV height')
        plt.contourf(y, x, phi_p, cmap='coolwarm', levels=100)
        plt.colorbar()
        # generate vector flow field plot
        slice_interval = int(x_nodes/11)

        angle = np.rad2deg(np.arctan2(-ux, -uy))

        if velocity_type == 'linear':
            scale_fact = 40
        else:
            scale_fact = 10

        skip = (slice(None, None, slice_interval), slice(None, None, slice_interval))
        quiver = plt.quiver(x[skip], y[skip],
                            ux[skip], uy[skip],
                            units='width', angles=angle[skip],
                            scale=scale_fact,
                            pivot='mid', color='black')
        plt.quiverkey(quiver, 1.25, 1.05, 2, label='2 m/s', labelcolor='black', labelpos='N', coordinates='axes')
        plt.show()

    else:
        pass

    return print('fin_vol completed Mesh size: {} x {}, Method: {}, Diffusion: {}'.format(x_nodes, y_nodes,
                                                                                          dif_scheme, gamma)), \
           phi_p, x, y, x_index, y_index, x_nodes, y_nodes, velocity_type, gamma, dif_scheme


coarseCD = fin_vol(80, 80, gamma=5, velocity_type='rotational', dif_scheme='UW', plot='no')
mediumCD = fin_vol(160, 160, gamma=5, velocity_type='rotational', dif_scheme='CD', plot='yes')
fineCD = fin_vol(320, 320, gamma=5, velocity_type='rotational', dif_scheme='CD', plot='no')

coarseUW = fin_vol(80, 80, gamma=5, velocity_type='rotational', dif_scheme='UW', plot='no')
mediumUW = fin_vol(160, 160, gamma=5, velocity_type='rotational', dif_scheme='UW', plot='yes')
fineUW = fin_vol(320, 320, gamma=5, velocity_type='rotational', dif_scheme='UW', plot='no')

# question 2/3

diag1 = np.diag(np.fliplr(coarseUW[1]))
diag2 = np.diag(np.fliplr(mediumUW[1]))
diag3 = np.diag(np.fliplr(fineUW[1]))
plt.plot(coarseUW[4], diag1, label='5x5 nodes')
plt.plot(mediumUW[4], diag2, label='10x10 nodes')
plt.plot(fineUW[4], diag3, label='100x100 nodes')
plt.title('False diffusion in various meshes using UW scheme')
plt.legend()
plt.show()

# question 4

f = interp.interp2d(mediumCD[4], mediumCD[5], mediumCD[1], kind='cubic')
f2 = interp.interp2d(fineCD[4], fineCD[5], fineCD[1], kind='cubic')
fine = f(coarseCD[4], coarseCD[5])
fine2 = f2(coarseCD[4], coarseCD[5])

error_coarse = np.abs(np.sqrt(np.sum(np.square(coarseCD[1] - fine2)/(80*80))))
error_fine = np.abs(np.sqrt(np.sum(np.square(fine - fine2)/(80*80))))
avg_convergence = np.log(error_coarse/error_fine)/np.log(2)

f_UW = interp.interp2d(mediumUW[4], mediumUW[5], mediumUW[1], kind='cubic')
f2_UW = interp.interp2d(fineUW[4], fineUW[5], fineUW[1], kind='cubic')
fine_UW = f_UW(coarseUW[4], coarseUW[5])
fine2_UW = f2_UW(coarseUW[4], coarseUW[5])

error_coarse_UW = np.abs(np.sqrt(np.sum(np.square(coarseUW[1] - fine2_UW)/(80*80))))
error_fine_UW = np.abs(np.sqrt(np.sum(np.square(fine_UW - fine2_UW)/(80*80))))
avg_convergence_UW = np.log(error_coarse_UW/error_fine_UW)/np.log(2)
print('\n')
# print('error_fine: {}, error_coarse: {}, error_fine_uw: {}, error_coarse_uw: {}'.format(error_fine,
#                                                                                         error_coarse,
#                                                                                         error_fine_UW,
#                                                                                         error_coarse_UW))
print('the order of convergence for the {} method is {}'.format(coarseCD[10], avg_convergence))
print('the order of convergence for the {} method is {}'.format(coarseUW[10], avg_convergence_UW))
