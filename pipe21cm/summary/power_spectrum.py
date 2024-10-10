import numpy as np

import tools21cm as t2c


def calculate_1dpk(dT,box_size,kbins,norm=True):
    '''
    calculate power spectrum using tools21cm
    we should implement a our own version to reduce the dependency

    Args:
        :dT(np.array): 3D cube
        :box_size(float): size of the cube
        :kbins(np.array or int): array of k egdes or number of bins
        :norm(bool): normalize the power spectrum or not

    Returns:
        :ks(np.array): k values
        :pk(np.array): power spectrum values
    '''
    
    pk, ks = t2c.power_spectrum_1d(dT, kbins=kbins, box_dims=box_size)

    if norm:
        pk = pk*ks**3/2/np.pi**2
    

    return  ks,pk