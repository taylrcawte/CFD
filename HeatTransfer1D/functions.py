# internal node coefficient and generation term calculations 
def calculate_internal_a_coef(area, k, dx, condition): 
    
    if condition == 'case1': 
        coef = area*(k/dx)
        return coef 

    else: 
        raise ValueError('Please enter a valid condition') 

def calculate_internal_a_w(area, k, dx, condition):
    return calculate_internal_a_coef(area, k, dx, condition) 

def calculate_internal_a_e(area, k, dx, condition):
    return calculate_internal_a_coef(area, k, dx, condition)

def calculate_internal_a_p(a1, a2, s_p, condition): 
    
    if condition == 'case1': 
        coef = a1 + a2
        return coef 
    elif condition == 'case2': 
        coef = a1 + a2 + s_p
        return coef
    else: 
        raise ValueError('Please enter a valid condition') 

def calculate_internal_s_u(area, dx, q, condition): 

    if condition == 'case1': 
        return 0
    elif condition == 'case2': 
        return q*area*dx   
    else: 
        raise ValueError('Please enter a valid condition and q') 

def calculate_internal_s_p(condition):
    
    if condition == 'case1': 
        return 0 
    else: 
        raise ValueError('Please enter a valid condition')

# TODO: Create functions to solve boundary coefficients as well
def calculate_boundary_s_u(k, area, dx, bc, q, condition): 

    if condition == 'case1': 
        return (2*k*area*bc) / dx
    elif condition == 'case2': 
        coef = (q*area*dx) + ((2*k*area*bc) / dx) 
        return coef
    else: 
        raise ValueError('Please enter a valid condition')

def calculate_boundary_s_p(k, area, dx, condition):  

    if condition == 'case1': 
        return (-2*k*area) / dx 
    else: 
        raise ValueError('Please enter a valid condition')
    