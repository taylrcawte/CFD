"""
A finite volume solver for heat transfer by diffusion and convection in a 2D control volume with
dirichlet and neumann boundary conditions

Author: Taylr Cawte
written on: 10/28/2019
"""

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as sparlin
import scipy.interpolate as interp
import matplotlib.pyplot as plotpy

print("This program illustrates heat transfer within a control volume that has the following boundary conditions "
      "\n\n\t 1. dirichlet conditions on the East/West face"
      "\n\t 2. heat flux on the North face"
      "\n\t 3. insulated boundary on the south face"
      "\n\nThe mesh sizes evaulated are 80x80 nodes, 160x160 nodes, and 320x320 nodes")

# constants
gamma = 20  # diffusion coefficient
phi_x0 = 10  # temperature at left boundary
phi_xl = 100  # temperature at right boundary
hf = 10  # convective heat transfer coefficient
phi_ext = 300  # external temperature
CV_l = 1  # length of control volume
CV_h = 1  # height of control volume

# determine whether or not inflation is required
compute_inflation = input("\nWould you like to add inflation to your grid? (enter 'yes' or 'no') ")
if compute_inflation == "yes":
    inf_fact = float(input("what is the inflation factor?(above 1)"))
else:
    inf_fact = 1


def fin_vol(x_nodes, y_nodes, inflation=compute_inflation, inflation_factor=inf_fact):
    # bring in global variables
    global gamma
    global phi_ext
    global phi_xl
    global phi_x0
    global hf
    global CV_h
    global CV_l

    dx = CV_l / x_nodes  # distance between nodes in the x direction
    dy = CV_h / y_nodes   # distance between nodes in the y direction
    total_nodes = y_nodes*x_nodes  # total number of nodes

    x_index = np.zeros(x_nodes)  # initialize x nodes matrix to zero
    y_index = np.zeros(y_nodes)  # initialize y nodes matrix to zero
    matrix_index = np.arange(total_nodes).reshape(x_nodes, y_nodes)  # matrix index to map node position within CV
    matrix_a = sp.lil_matrix((total_nodes, total_nodes))  # coefficent matrix of form AX = B, "A"
    matrix_b = np.zeros(total_nodes)  # product matrix of form AX = B, "B"

    if inflation == "no":
        for i in range(y_nodes):
            for j in range(x_nodes):
                if i == 0:  # southern most node
                    a_n = gamma*(dx/dy)
                    a_s = 0
                    s_p = 0
                    matrix_a[matrix_index[j, i], matrix_index[j, i+1]] = -a_n  # updates coef right of ap equal to an
                    # print(matrix_index[i, j], matrix_index[j, i+1])
                    matrix_b[matrix_index[j, i]] = a_s  # sets product matrix equal to as
                elif i == y_nodes-1:  # northern most node
                    a_n = 0
                    a_s = gamma*(dx/dy)
                    s_p = -hf*dx  # source term sp
                    s_u = hf*phi_ext*dx  # source term su
                    matrix_a[matrix_index[j, i], matrix_index[j, i-1]] = -a_s  # updates coef left of ap equal to as
                    matrix_b[matrix_index[j, i]] = a_n + s_u  # updates product matrix
                else:  # middle nodes
                    a_n = gamma*(dx/dy)
                    a_s = gamma*(dx/dy)
                    s_p = 0
                    matrix_a[matrix_index[j, i], matrix_index[j, i+1]] = -a_n  # updates coeffs to the right
                    matrix_a[matrix_index[j, i], matrix_index[j, i-1]] = -a_s  # updates coeffs to the left

                if j == 0:
                    a_w = gamma*(dy/(dx/2))  # divide by 2 because at western most node dx is halved
                    a_e = gamma*(dy/dx)
                    matrix_a[matrix_index[j, i], matrix_index[j+1, i]] = -a_e  # updates row below ap
                    matrix_b[matrix_index[j, i]] = a_w*phi_x0  # updates product matrix
                elif j == x_nodes-1:
                    a_w = gamma*(dy/dx)
                    a_e = gamma*(dy/(dx/2))  # divide by 2 because at eastern most node dx is halved
                    matrix_a[matrix_index[j, i], matrix_index[j-1, i]] = -a_w  # updates row above ap
                    matrix_b[matrix_index[j, i]] = a_e*phi_xl  # updates product matrix
                else:
                    a_w = gamma*(dy/dx)
                    a_e = gamma*(dy/dx)
                    matrix_a[matrix_index[j, i], matrix_index[j+1, i]] = -a_e
                    matrix_a[matrix_index[j, i], matrix_index[j-1, i]] = -a_w

                a_p = a_n + a_s + a_w + a_e - s_p  # updates coefficient at point p (i.e. the investigated node)
                matrix_a[matrix_index[j, i], matrix_index[j, i]] = a_p  # sets node value at i,j equal to ap

        # updates x position of node within control volume
        for i in range(x_nodes):
            if i == 0:
                x_index[i] = dx/2  # initial node position within control volume
            else:
                x_index[i] = x_index[i-1]+dx  # add dx to the last node to get next node position
        # updates y position of node within control volume
        for j in range(y_nodes):
            if j == 0:
                y_index[j] = dy/2  # initial y node position within control volume
            else:
                y_index[j] = y_index[j-1]+dy  # add dy to last node to get next node position

    elif inflation == "yes":

        inflate_x = np.zeros(x_nodes+1)  # initialize new array for inflated x spacing values
        inflate_y = np.zeros(y_nodes+1)  # initialize new array for inflated y spacing values
        xmap = 0  # initial position of first node in x
        ymap = 0  # initial position of first node in y

        for i in range(x_nodes+1):
            if i == 0:
                inflate_x[i] = ((1-inflation_factor)/(1-inflation_factor**(x_nodes/2)))*CV_l/2  # first node
            else:
                if xmap <= CV_l/2:
                    inflate_x[i] = inflation_factor*inflate_x[i-1]  # inflate x according to above if before halfway pt
                else:
                    inflate_x[i] = inflate_x[i-1]/inflation_factor  # deflate x after halfway point
            xmap += inflate_x[i]

        inflate_x = (1/xmap)*inflate_x  # re scales domain according to new spacing

        for i in range(y_nodes+1):
            if i == 0:
                inflate_y[i] = ((1-inflation_factor)/(1-inflation_factor**(y_nodes/2)))*CV_h/2  # first y node
            else:
                if ymap <= CV_h/2:
                    inflate_y[i] = inflation_factor*inflate_y[i-1]  # inflates y node up until halfway point
                else:
                    inflate_y[i] = inflate_y[i-1]/inflation_factor  # deflates y node after halfway
            ymap += inflate_y[i]

        inflate_y = (1/ymap)*inflate_y  # rescales domain in y direction

        # same process as no inflation except now we are using inflated x,y and not dx,dy
        for i in range(y_nodes):
            for j in range(x_nodes):
                if i == 0:
                    a_n = gamma*(inflate_x[j]/inflate_y[i+1])
                    a_s = 0
                    s_p = 0
                    matrix_a[matrix_index[j, i], matrix_index[j, i+1]] = -a_n
                    matrix_b[matrix_index[j, i]] = a_s
                elif i == y_nodes-1:
                    a_n = 0
                    a_s = gamma*(inflate_x[j]/inflate_y[i])
                    s_p = -hf*inflate_x[j]
                    s_u = hf*phi_ext*inflate_x[j]
                    matrix_a[matrix_index[j, i], matrix_index[j, i-1]] = -a_s
                    matrix_b[matrix_index[j, i]] = a_n + s_u
                else:
                    a_n = gamma*(inflate_x[j]/inflate_y[i+1])
                    a_s = gamma*(inflate_x[j]/inflate_y[i])
                    s_p = 0
                    matrix_a[matrix_index[j, i], matrix_index[j, i+1]] = -a_n
                    matrix_a[matrix_index[j, i], matrix_index[j, i-1]] = -a_s

                if j == 0:
                    a_w = gamma*(inflate_y[i]/inflate_x[j])
                    a_e = gamma*(inflate_y[i]/inflate_x[j+1])
                    matrix_a[matrix_index[j, i], matrix_index[j+1, i]] = -a_e
                    matrix_b[matrix_index[j, i]] = a_w*phi_x0
                elif j == x_nodes-1:
                    a_w = gamma*(inflate_y[i]/inflate_x[j])
                    a_e = gamma*(inflate_y[i]/inflate_x[j+1])
                    matrix_a[matrix_index[j, i], matrix_index[j-1, i]] = -a_w
                    matrix_b[matrix_index[j, i]] = a_e*phi_xl
                else:
                    a_w = gamma*(inflate_y[i]/inflate_x[j])
                    a_e = gamma*(inflate_y[i]/inflate_x[j+1])
                    matrix_a[matrix_index[j, i], matrix_index[j+1, i]] = -a_e
                    matrix_a[matrix_index[j, i], matrix_index[j-1, i]] = -a_w

                a_p = a_n + a_s + a_w + a_e - s_p
                matrix_a[matrix_index[j, i], matrix_index[j, i]] = a_p

        # updates x with inflated nodes
        for i in range(x_nodes):
            if i == 0:
                x_index[i] = inflate_x[i]
            else:
                x_index[i] = x_index[i-1]+inflate_x[i]

        # updates y with inflated nodes
        for i in range(y_nodes):
            if i == 0:
                y_index[i] = inflate_y[i]
            else:
                y_index[i] = y_index[i-1]+inflate_y[i]

    else:
        print("this is not a valid entry, please enter 'yes' or 'no'")

    # solves sparse matrices a and b
    phi_p = sparlin.spsolve(sp.csc_matrix(matrix_a), matrix_b)
    # reshape the solved matrix to match node position within the control volume
    phi_p = phi_p.reshape(x_nodes, y_nodes)

    # sets grid spacing for plots according to wether or not there is inflation
    if inflation == "no":
        x, y = np.meshgrid(np.linspace(0, CV_l, x_nodes), np.linspace(0, CV_h, y_nodes))
    else:
        x, y = np.meshgrid(x_index, y_index)

    return phi_p, x_index, y_index, x, y

# run the finite volume solver using 80, 160, and 320 nodes

fv_coarse = fin_vol(80, 80)
fv_fine = fin_vol(160, 160)
fv_finest = fin_vol(320, 320)

# create a function to interpolate between points so
# that the fine and finest meshes may be compared to
# the coarse mesh  #
f = interp.interp2d(fv_fine[1], fv_fine[2], fv_fine[0], kind='cubic')
f2 = interp.interp2d(fv_finest[1], fv_finest[2], fv_finest[0], kind='cubic')
fine = f(fv_coarse[1], fv_coarse[2])
fine2 = f2(fv_coarse[1], fv_coarse[2])

# calculate convergence according to the equation given in the assignment
error_coarse = np.abs(np.sqrt(np.sum(np.square(fv_coarse[0] - fine2))/(80*80)))
error_fine = np.abs(np.sqrt(np.sum(np.square(fine - fine2))/(80*80)))
avg_convergence = np.log(error_coarse/error_fine)/np.log(2)

print("the order of convergence is: {}".format(avg_convergence))

# plot the figures from each plotting scenario
plotpy.figure(1)
plotpy.title("coarse mesh contour, 80x80")
plotpy.xlabel('CV width')
plotpy.ylabel('CV height')
plotpy.contourf(fv_coarse[4], fv_coarse[3], fv_coarse[0])
plotpy.colorbar()
plotpy.figure(2)
plotpy.title("fine mesh contour, 160x160")
plotpy.xlabel('CV width')
plotpy.ylabel('CV height')
plotpy.contourf(fv_fine[4], fv_fine[3], fv_fine[0])
plotpy.colorbar()
plotpy.figure(3)
plotpy.title("finest mesh contour, 320x320")
plotpy.xlabel('CV width')
plotpy.ylabel('CV height')
plotpy.contourf(fv_finest[4], fv_finest[3], fv_finest[0])
plotpy.colorbar()
plotpy.show()
