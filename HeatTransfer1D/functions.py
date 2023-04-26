# internal node coefficient and generation term calculations 
def calculate_internal_a_coef(area, k, dx, condition): 
    
    if condition == 'case1': 
        coef = area*(k/dx)
        return coef 

    else: 
        raise ValueError('Please enter a valid condition') 

def calculate_internal_a_w(area, k, dx, condition):
    return calculate_internal_a_coef(area, k, dx, condition) 

def calculate_internal_a_e(area, k, dx, condition='case1'):
    return calculate_internal_a_coef(area, k, dx, condition)

def calculate_internal_a_p(a1, a2, condition): 
    
    if condition == 'case1': 
        coef = a1 + a2
        return coef 

    else: 
        raise ValueError('Please enter a valid condition') 

def calculate_internal_s_u(condition): 

    if condition == 'case1': 
        return 0 
    
    else: 
        raise ValueError('Please enter a valid condition') 

def calculate_internal_s_p(condition):
    
    if condition == 'case1': 
        return 0 
    
    else: 
        raise ValueError('Please enter a valid condition')

# TODO: Create functions to solve boundary coefficients as well
def calculate_boundary_s_u(k, area, dx, bc, condition): 

    if condition == 'case1': 
        return (2*k*area*bc) / dx
    else: 
        raise ValueError('Please enter a valid condition')

def calculate_boundary_s_p(k, area, dx, condition):  

    if condition == 'case1': 
        return (-2*k*area) / dx
    else: 
        raise ValueError('Please enter a valid condition')
    