import numpy as np
import os
import sys
import tools21cm as t2c


class Telescope:
    def __init__(self, ncells, boxsize, zs, subarray_type="SKA1_Low_Sept2016", obs_time=100.0, total_int_time=4.0, int_time=10.0, declination=-30.0, n_jobs=16,):
        """
        This class supports:
        - Generating UV coverage maps and instrumental noise.
        - Applying telescope UV response to both simulated lightcone signals.

        Parameters
        ----------
        ncells : int
            Number of cells in the simulation box.
        boxsize : float
            Size of the simulation box in Mpc.
        zs : list or np.ndarray
            List of redshifts for which the lightcone is generated.
        subarray_type : str, optional
            Attenna Configuration to use for the simulation. Default is "SKA1_Low_Sept2016". See tools21cm.radio_telescope_layout for details.
        obs_time : float, optional
            Observation time in hours. Default is 100.0.
        total_int_time : float, optional
            Total integration time in hours. Default is 4.0.
        int_time : float, optional
            Integration time per observation in second. Default is 10.0.
        declination : float, optional
            Declination of the observation in degrees. Default is -30.0.
        n_jobs : int, optional
            Number of parallel jobs to run for the simulation. Default is 16.
        """
        allowed_subarrays = ["AA4", "AA2", "AASTAR", "AA1", "AA0.5", "SKA_Low_1", "SKA1_Low_Sept2016"]
        assert subarray_type in allowed_subarrays, f"Subarray type must be one of {allowed_subarrays}, got {subarray_type}."

        if subarray_type == "SKA1_Low_Sept2016":
            self.subarray_type = None # this is how the tools21cm library handles SKA1_Low_Sept2016
        else:
            self.subarray_type = subarray_type

        self.ncells = ncells
        self.boxsize = boxsize
        self.obs_time = obs_time
        self.zs = zs
        self.total_int_time = total_int_time
        self.int_time = int_time
        self.declination = declination
        self.n_jobs = n_jobs

    def build_lightcone_uv_map(self, save_uvmap_path):
        """
        Build the UV map for the lightcone based on the telescope configuration.
        This has to be called before applying the UV response on the lightcone signal or generating noise lightcones.

        Parameters
        ----------
        save_uvmap_path : str, optional
            Path to save the UV map. 
        
        returns
        -------
        None
        """


        uvmap = t2c.get_uv_map_lightcone(ncells=self.ncells,
                                            zs=self.zs,
                                            total_int_time=self.total_int_time,
                                            int_time=self.int_time,
                                            declination=self.declination,
                                            n_jobs=self.n_jobs,
                                            save_uvmap=save_uvmap_path,
                                            subarray_type=self.subarray_type)

        self.uv_path = save_uvmap_path
        self.uv_map = uvmap
                            
        return
    
    def apply_uv_response_on_lightcone(self, lc_signal):
        """
        apply the UV response on the lightcone signal.

        Parameters
        ----------
        lc_signal : np.ndarray
            The lightcone signal to which the UV response will be applied.
            It should be a 3D array with shape (ncells, ncells, len(zs)) where ncells is the number of cells in one dimension and zs is the list of
            redshifts for which the lightcone is generated.
        Returns
        -------
        lc_uv_applied : np.ndarray
            The lightcone signal after applying the UV response. It will have the same shape as lc_signal.
            The UV response is applied for each redshift in the lightcone signal.
        """

        assert hasattr(self, 'uv_map'), "UV map not built. Call build_lightcone_uv_map() first."

        lc_uv_applied = np.zeros_like(lc_signal)
        for i in range(len(self.zs)):
            zi = self.zs[i]
            uv_zi = self.uv_map['{:.3f}'.format(zi)]
            lc_uv_applied[...,i] = t2c.apply_uv_response_on_image(lc_signal[...,i], uv_zi)
        

        return lc_uv_applied
    
    def get_noise_lightcone(self):
        """
        Generate a noise lightcone based on the telescope configuration.

        Returns
        -------
        noise_lc : np.ndarray
            The noise lightcone generated based on the telescope configuration.
            It will be a 3D array with shape (ncells, ncells, len(zs)) where ncells is the number of cells in one dimension and zs is the list of
            redshifts for which the lightcone is generated.
        """

        assert hasattr(self, 'uv_path'), "UV map path not set. Call build_lightcone_uv_map() first."

        noise_lc = t2c.noise_lightcone(ncells=self.ncells,
                                    zs=self.zs,
                                    obs_time=self.obs_time,
                                    boxsize=self.boxsize,
                                    total_int_time=self.total_int_time,
                                    int_time=self.int_time,
                                    declination=self.declination,
                                    n_jobs=self.n_jobs,
                                    save_uvmap=self.uv_path,
                                    subarray_type=self.subarray_type)
        
        return noise_lc

