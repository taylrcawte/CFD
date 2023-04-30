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

def calculate_internal_a_p(a1, a2, s_p): 
    coef = a1 + a2 - s_p
    return coef

def calculate_internal_s_u(area, dist, q, hp, T_inf): 
    return q*area*dist + hp*dist*T_inf   

def calculate_internal_s_p(hp, dist):
    return 0 - hp*dist
########################################################################################
# boundary node coefficient and generatoin term calculations 
########################################################################################

def calculate_boundary_s_u(k, area, dist, bc, T_inf, hp, q): 
    coef = (q*area*dist) + ((2*k*area*bc) / dist) + (hp*dist*T_inf)
    return coef

def calculate_boundary_s_p(k, area, hp, dist):  
    return (-hp*dist) + (-2*k*area) / dist 

# TODO: could make individual functions for each condition e.g. for cooling     