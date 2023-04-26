import numpy as np

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

# TODO: need to add calcualte boundary ae, aw as well

def calculate_internal_a_p(a1, a2, condition='case1'): 
    
    if condition == 'case1': 
        
        coef = a1 + a2

        return coef 

    else: 

        raise ValueError('Please enter a valid condition') 

