# internal node coefficient and generation term calculations 
def calculate_internal_a_coef(area, k, dx): 
    coef = (area*k)/dx
    return coef 

def calculate_internal_a_w(area, k, dx):
    return calculate_internal_a_coef(area, k, dx) 

def calculate_internal_a_e(area, k, dx):
    return calculate_internal_a_coef(area, k, dx)

def calculate_internal_a_p(a1, a2, s_p): 
    coef = a1 + a2 - s_p
    return coef

def calculate_internal_s_u(area, dx, q, hp, T_inf): 
    return q*area*dx + hp*dx*T_inf   

def calculate_internal_s_p(hp, dx):
    return 0 - hp*dx

def calculate_boundary_s_u(k, area, dx, bc, T_inf, hp, q): 
    coef = (q*area*dx) + ((2*k*area*bc) / dx) + (hp*dx*T_inf)
    return coef

def calculate_boundary_s_p(k, area, hp, dx):  
    return (-hp*dx) + (-2*k*area) / dx 

# TODO: could make individual functions for each condition e.g. for cooling     