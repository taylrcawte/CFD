import numpy as np 

########################################################################################
# internal node coefficient and generation term calculations 
########################################################################################
def calculate_internal_a_coef(area, k, dist): 
    coef = (area*k)/dist
    return coef 

def calculate_internal_a_w(area, k, dist):
    return calculate_internal_a_coef(area, k, dist) 

def calculate_internal_a_e(area, k, dist):
    return calculate_internal_a_coef(area, k, dist)

def calculate_internal_a_n(area, k, dist):
    return calculate_internal_a_coef(area, k, dist) 

def calculate_internal_a_s(area, k, dist):
    return calculate_internal_a_coef(area, k, dist)

def calculate_internal_a_p(a1, a2, a3, a4, s_p): 
    coef = a1 + a2 + a3 + a4 - s_p
    return coef

def calculate_internal_s_u(): 
    return 0 

def calculate_internal_s_p():
    return 0
########################################################################################
# boundary node coefficient and generatoin term calculations 
########################################################################################

def calculate_boundary_s_u(k, area, dist, bc, q): 
    coef = (q*area*dist) + ((2*k*area*bc) / dist)
    return coef

def const_temp_boundary(k, area, bc, dist): 
    return ((2*k*area*bc) / dist) 

def const_flux_boundary(q, area): 
    return q*area

def insulated_boundary(): 
    return 0

def calculate_boundary_s_p(k, area, dist):  
    return (-2*k*area) / dist 

# TODO: could make individual functions for each condition e.g. for cooling
# 
# TODO: solver function 
def tdma_noncons(A, B, C, D): 
    # inputs must be lists for now 
    if not (len(A) == len(B) == len(C) == len(D)): 
        raise ValueError(f'All vectors must be same length,\
                            provided dimensions {len(A), len(B), len(C), len(D)}')
    else:
        Dim = len(A)
        phi = np.empty(Dim)

    for i in range(1, Dim, 1):
        w = A[i] / B[i-1]
        B[i] = B[i] - w*C[i-1]
        D[i] = D[i] - w*D[i-1]

    phi[Dim-1] = D[Dim-1] / B[Dim-1]

    for i in range(Dim-2, -1, -1):
        phi[i] = (D[i]-C[i]*phi[i+1]) / B[i]

    return np.array(phi)

def tdma_cons(A, B, C, D): 
    if not (len(A) == len(B) == len(C) == len(D)): 
        raise ValueError(f'All vectors must be same length,\
                            provided dimensions {len(A), len(B), len(C), len(D)}')
    else: 
        dim = len(A) 
        c_prime = np.zeros(dim) 
        d_prime = np.zeros(dim) 
        x_hurdur = np.zeros(dim) 

    for i in range(0, dim-1, 1): 
        
        if i == 0: 
            c_prime[i] = C[i]/B[i]
        else: 
            c_prime[i] = C[i]/(B[i]-A[i]*c_prime[i-1])

    for i in range(0, dim, 1): 

        if i == 0: 
            d_prime[i] = D[i]/B[i]
        else: 
            d_prime[i] = (D[i]-A[i]*d_prime[i-1])/(B[i]-A[i]*c_prime[i-1])

    x_hurdur[dim-1] = d_prime[dim-1]

    for i in range(dim-2, -1, -1):

        x_hurdur[i] = d_prime[i] - c_prime[i]*x_hurdur[i+1]

    return x_hurdur
    


