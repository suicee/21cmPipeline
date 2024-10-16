
import os
import numpy as np
import tools21cm as t2c


def build_physical_lightcone(file_list, 
                             redshifts, 
                             box_size):
    """
    Build a lightcone from a list of 21cm brightness temperature boxs using tools21cm.
    Slices along the line of sight are constructed wit uniform comoving distance intervals.(Default setting in tools21cm)

    Args:
        :file_list: list of str. The list of file paths to the brightness temperature maps in npy files.
                    Note that the files should have consistent name formats and should be sorted in the order of redshifts.
        :redshifts: list of float. The redshifts of the brightness temperature maps. Should be in the same order as the file_list.
        :box_size: float. The size of the box in Mpc.

    Returns:
        :lightcone: np.ndarray. The lightcone.
        :zs_lc: list of float. The redshifts of the slices in the lightcone.
    """

    z_low = redshifts[0]
    z_high = redshifts[-1]

    # use tools21cm to make the lightcone, output is the lightcone and the redshifts of the slices
    lc, zs_lc = t2c.make_lightcone(
                            file_list,
                            z_low=z_low,
                            z_high=z_high,
                            file_redshifts=redshifts,
                            los_axis=2,
                            interpolation='linear',
                            reading_function=np.load,
                            box_length_mpc=box_size,
                        )
    
    return lc, zs_lc


def build_observational_lightcone(file_list, 
                                  redshifts, 
                                  box_size,
                                  dnu=0.1,
                                  n_output_cell=None):
    """
    Build a observational lightcone from a list of 21cm brightness temperature boxs using tools21cm.
    Slices along the line of sight are constructed wit uniform frequency intervals.

    Args:
        :file_list: list of str. The list of file paths to the brightness temperature maps in npy files.
                    Note that the files should have consistent name formats and should be sorted in the order of redshifts.
        :redshifts: list of float. The redshifts of the brightness temperature maps. Should be in the same order as the file_list.
        :box_size: float. The size of the box in Mpc.
        :dnu: float. The frequency interval in MHz. Default is 100 kHz.
        :n_output_cell: int. The number of output cells in the observational lightcone. Default is set to the same as the input lightcone.
                            tools21cm will pad the slice whose angular size is smaller than the maximum angular size to match the maximum angular size.
                            Then the slices are interpolated to the same number of cells.

    Returns:
        :obs_lc: np.ndarray. The observational lightcone.
        :obs_freq: np.ndarray. The frequency axis of the observational lightcone.
    """

    lc, zs_lc = build_physical_lightcone(file_list, redshifts, box_size)    

    angular_size_deg = t2c.angular_size_comoving(box_size, redshifts)

    max_deg = np.max(angular_size_deg)

    if n_output_cell is None:
        n_output_cell = lc.shape[0]

    input_z_low   = np.min(redshifts)
    output_dnu    =  dnu
    output_dtheta = (max_deg/(n_output_cell))*60 #arcmins
    input_box_size_mpc = box_size

    obs_lc, obs_freq = t2c.physical_lightcone_to_observational(lc,
                                                                input_z_low,
                                                                output_dnu,
                                                                output_dtheta,
                                                                input_box_size_mpc=input_box_size_mpc)

    return obs_lc, obs_freq
    
    
