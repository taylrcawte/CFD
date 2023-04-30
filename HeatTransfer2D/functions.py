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