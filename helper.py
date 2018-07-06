import numpy as np

def get_logn_mu(m,v):
    return np.log(m**2/np.sqrt(v+m**2))
  
def get_logn_var(m,v):
  return np.sqrt(np.log(v/m**2+1))