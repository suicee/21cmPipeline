
import Pk_library as PKL
import contextlib
import numpy as np  


def caculate_icoBk(cube,box_size,kbins=None,thetas=None,norm=True,threads=1):
    '''
    calculate bispectrum from 3D cubes using Pylians3
    maybe we should implement a pure numpy version

    Args:
        :cube(np.array): 3D cube
        :boxsize(float): size of the cube
        :kbins(np.array): array of k bins
        :thetas(np.array): array of angles
        :norm(bool): normalize the bispectrum or not
        :threads(int): number of threads

    Returns:
        :bs_cube(np.array): bispectrum array
    '''

    if kbins is None:
        kF=2*np.pi/box_size
        #N=(np.arange(11)+1)*2 # for pure signal
        N=(np.arange(8)+1) #for signal with observational effects only use small ks
        kbins=kF*N

    #for each triangle, we have k1+k2=-k3, theta is the angle between vector k1 and k2 not the angle in the triangle
    if thetas is None:
        thetas = np.array([0.05, 0.1, 0.2, 0.33, 0.4, 0.5, 0.6, 0.7, 0.85, 0.95])*np.pi

    cube=np.array(cube,dtype=np.float32)

    bs_cube=np.zeros((len(kbins),len(thetas)))
    #make a meshgrid of k to store the final k values
    k3_all,k1_all = np.meshgrid(thetas,kbins)

    for idx, k_to_cal in enumerate(kbins):

        # we only consider isoceles triangles, k1=k2 is determine by the input bins, k3 is determined by the angle
        k1 = k_to_cal    
        k2 = k_to_cal  

        # compute bispectrum
        try:
            with contextlib.redirect_stdout(None):
                BBk = PKL.Bk(cube,box_size, k1, k2, thetas, None, threads)
            
            k3_all[idx]=BBk.k[2:]
            if norm:
                bs_cube[idx]=normalized_BS(BBk)
            else:
                bs_cube[idx]=BBk.B

        except ZeroDivisionError:
            #handle the ZeroDivisionError which may occur when the field is full of zeros (for example completely ionized 21cm field)
            bs_cube[idx]=0

    selection=get_k_filter(box_size,kbins,thetas)

    return k1_all[selection],k3_all[selection],bs_cube[selection]


def get_k_filter(boxsize,kbins,thetas):
    '''
    get k pairs satisfying kF<k<kmax

    Args:
        :boxsize(float): size of the cube
        :kbins(np.array): array of k bins
        :thetas(np.array): array of angles

    Returns:
        :re(np.array): array of boolean values indicating whether the k pairs satisfy the condition
    '''

    kF=2*np.pi/boxsize
    k = np.array(kbins)[:, np.newaxis]  # Reshape kbins to be a column vector

    # compute k3 for all (k, theta) pairs
    k3 = np.sqrt((k * np.sin(thetas))**2 + (k * np.cos(thetas) + k)**2)

    # Apply the condition kF<k<kmax
    re = (k3 > kF) & (k3 < kbins[-1])

    return re

def normalized_BS(BBk):
    '''
    normalize bs following Watkinson et al 2019
    '''
    bs=BBk.B
    ps=BBk.Pk
    ks=BBk.k
    
    k1,Pk1=ks[0],ps[0]
    k2,Pk2=ks[1],ps[1]
    k3s,Pk3s=ks[2:],ps[2:]
    
    normal_fac=np.sqrt((Pk1*Pk2*Pk3s)/(k1*k2*k3s))
    bs_norm=bs/normal_fac
    
    return bs_norm
