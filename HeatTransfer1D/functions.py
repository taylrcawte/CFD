# internal node coefficient and generation term calculations 
def calculate_internal_a_coef(area, k, dx, condition): 

    
    if condition == 'case1': 
        coef = area*(k/dx)
        return coef 

    else: 
        raise ValueError('Please enter a valid condition') 

def calculate_internal_a_w(area, k, dx, condition='case1'):
    return calculate_internal_a_coef(area, k, dx, condition) 

def calculate_internal_a_e(area, k, dx, condition='case1'):
    return calculate_internal_a_coef(area, k, dx, condition)

def calculate_internal_a_p(a1, a2, condition='case1'): 
    
    if condition == 'case1': 
        coef = a1 + a2
        return coef 

    else: 
        raise ValueError('Please enter a valid condition') 

def calculate_internal_s_u(condition='case1'): 

    if condition == 'case1': 
        return 0 
    
    else: 
        raise ValueError('Please enter a valid condition') 

def calculate_internal_s_p(condition='case1'):
    
    if condition == 'case1': 
        return 0 
    
    else: 
        raise ValueError('Please enter a valid condition')

# TODO: Create functions to solve boundary coefficients as well