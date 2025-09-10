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

def calculate_2dpk(lc, box_size, kbins, nu_axis=2, norm=True):
    '''
    calculate 2d Cylinder power spectrum using tools21cm

    Args:
        :lc(np.array): 3D lightcone
        :box_size(float or list): size of the cube in Mpc
        :kbins(int): number of bins for kper and kpar
        :norm(bool): normalize the power spectrum or not

    Returns:
        :kper_mid(np.array): kper values
        :kpar_mid(np.array): kpar values
        :p2d(np.array): 2D power spectrum
    '''

    p2d, kper_mid, kpar_mid = t2c.power_spectrum_2d(lc, kbins=kbins, box_dims=box_size, binning='log', nu_axis=2)

    if norm:
        pass

    return kper_mid, kpar_mid, p2d