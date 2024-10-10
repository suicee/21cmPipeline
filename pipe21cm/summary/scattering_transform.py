
import torch
import numpy as np
import logging
from kymatio.numpy import HarmonicScattering3D
from kymatio.torch import HarmonicScattering3D as HarmonicScattering3D_torch

class ScatteringTransformKernel:
    def __init__(self, J, L, shape, integral_powers=[0.5, 1.0, 2.0], backend='numpy',device='cpu'):
        '''
        Class to compute scattering transform of a 3D cube

        Args:
            :J: int, number of scales
            :L: int, number of angles
            :shape: tuple, shape of the input cube
            :integral_powers: list, powers of the modulus of the wavelet transform
            :backend: str, backend to use, 'numpy' or 'torch'
            :device: str, device to use, 'cpu' or 'cuda'

        '''

        self.J = J
        self.L = L
        self.shape = shape
        self.integral_powers = integral_powers
        self.backend = backend
        self.device = device

        if backend == 'numpy' and device == 'cuda':
            logging.warning('numpy backend does not support cuda device. Switching to cpu device')

        if self.backend == 'numpy':
            self.scattering = HarmonicScattering3D(J=self.J, shape=self.shape, sigma_0=1,
                                              L=self.L, integral_powers=self.integral_powers)
            
            self.get_compact_coef = self._get_compact_coef_numpy

        elif self.backend == 'torch':
            self.scattering = HarmonicScattering3D_torch(J=self.J, shape=self.shape, sigma_0=1,
                                                    L=self.L, integral_powers=self.integral_powers)
            self.scattering.to(device)
            print('using torch backend')
            self.get_compact_coef = self._get_compact_coef_torch

    def apply_on(self, cube):
        '''
        Apply scattering transform on the input cube

        Args:
            :cube: numpy array or torch tensor, input cube

        Returns:
            :sc: numpy array or torch tensor, scattering transform of the input cube
        '''

        return self.scattering(cube)
    
    def _get_compact_coef_numpy(self, cube):
        '''
        get compact coefficients from the scattering transform following Zhao et al 2024. Numpy version
        In this function, we calculate the 0th, 1st and 2nd order scattering coefficients(average over angle) and return them as a single array
        abs_log is used to process the coefficients

        Args:
            :cube: numpy array, input cube

        Returns:
            :total_sc: numpy array, compact coefficients
        '''


        def abs_log(x):
            re=(np.sign(x))*np.log2(np.abs(x))
            re[np.isnan(re)]=0
            return re

        ndim = len(cube.shape)
        dim1 = cube.shape[0]

        #calculate 1st and 2rd st coefs
        sc = self.apply_on(cube)  

        #average over the angle dimension  
        sc = np.mean(abs_log(sc),axis=-2)

        #calculate 0th coefs
        sc0=[(np.sum(cube**i,axis=(-1,-2,-3))) for i in [2,3,4]]
        sc0= abs_log(np.array(sc0))

        #combine all coefs
        if ndim == 3:
            total_sc = np.hstack((sc0, sc.flat))
        elif ndim == 4:
            total_sc = np.hstack((sc0.T, sc.reshape(dim1,-1)))

        return total_sc
    
    def _get_compact_coef_torch(self, cube):
        '''
        get compact coefficients from the scattering transform following Zhao et al 2024. Torch version, works better on GPU
        In this function, we calculate the 0th, 1st and 2nd order scattering coefficients(average over angle) and return them as a single array
        abs_log is used to process the coefficients

        Args:
            :cube: numpy array, input cube

        Returns:
            :total_sc: numpy array, compact coefficients
        '''

        def abs_log(x):
            re=(torch.sign(x))*torch.log2(torch.abs(x))
            re[torch.isnan(re)]=0
            return re
        
        #check if cube is numpy
        if isinstance(cube, np.ndarray):
            cube = torch.tensor(cube, device=self.device).float()

        ndim = len(cube.shape)
        dim1 = cube.shape[0]

        print(cube.dtype)
        #calculate 1st and 2rd st coefs
        sc = self.apply_on(cube)  

        #average over the angle dimension  
        sc = torch.mean(abs_log(sc),dim=-2)

        #calculate 0th coefs
        sc0=torch.cat([(torch.sum(cube**i,dim=(-1,-2,-3))).unsqueeze(-1) for i in [2,3,4]],dim=-1)
        sc0= abs_log(torch.tensor(sc0))

        #combine all coefs 
        # check if the input is batched or not (should probably include more batch dims
        if ndim == 3:
            total_sc = torch.cat((sc0, sc.flat))
        elif ndim == 4:
            total_sc = torch.cat((sc0, sc.reshape(dim1,-1)),dim=1)

        return total_sc
