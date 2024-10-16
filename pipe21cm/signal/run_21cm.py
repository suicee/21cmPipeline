import py21cmfast as p21c
from py21cmfast import cache_tools
import numpy as np
import os

def run_coeval_bt(
    redshift=12,
    box_size=128,
    cell_dim=128,
    hii_eff_factor=30,
    ion_tvir_min=4.7,
    random_seed=42,
    save_dir=None
):
    """
    Run a coeval simulation and return the brightness temperature map.

    Args:
        :redshift: float or list of floats. The redshift(s) at which to run the simulation.
        :box_len: float. The size of the box in Mpc.
        :cell_dim: int. The number of cells along one dimension.
        :hii_eff_factor: float. The ionizing efficiency factor.
        :ion_tvir_min: float. The minimum virial temperature of halos.
        :random_seed: int. The random seed for the simulation.
        :save_dir: str. The directory to save the results to. If None, the results are not saved.

    Returns:
        :results: list of np.ndarray. The brightness temperature map(s) at the specified redshift(s).
    """
    # Define user and astro parameters based on individual arguments
    user_params = {"HII_DIM": cell_dim, "BOX_LEN": box_size}
    astro_params = p21c.AstroParams({"HII_EFF_FACTOR": hii_eff_factor, "ION_Tvir_MIN": ion_tvir_min})

    # Run the coeval simulation
    cos = p21c.run_coeval(
        redshift=redshift,
        user_params=user_params,
        astro_params=astro_params,
        random_seed=random_seed
    )

    if len(redshift) > 1:
        results = [co.brightness_temp for co in cos]
    else:
        results = [cos.brightness_temp]
        redshift = [redshift]

    
    if save_dir is not None:
        for idx, res in enumerate(results):
            np.save(os.path.join(save_dir, f"brightness_temp_{redshift[idx]:.2f}.npy"), res)
    
    return results

