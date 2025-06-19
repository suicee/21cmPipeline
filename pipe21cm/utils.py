import numpy as np
from matplotlib import pyplot as plt

def plot_lightcone(lc, los_cor, box_size, type='physical',cmap=None):
    """
    Plot the lightcone.
    
    Args:
        :lc: np.ndarray. The lightcone.
        :los_cor: float. The line of sight cordinates(in redshift or frequency).
        :box_size: float. The size of the box(in Mpc or degrees).
        :type: str. The type of the lightcone. Either 'physical' or 'observational'.
    """
    assert type in ['physical', 'observational'], 'type must be physical or observational'

    xi = np.array([los_cor for i in range(lc.shape[1])])
    yi = np.array([np.linspace(0,box_size,lc.shape[1]) for i in range(xi.shape[1])]).T
    zj = lc[lc.shape[0]//2]

    fig, axs = plt.subplots(1,1, figsize=(14, 3))
    if cmap is None:
        cmap = 'jet'
    im = axs.pcolor(xi, yi, zj, cmap=cmap)

    if type == 'physical':
        axs.set_xlabel('z', fontsize=18)
        axs.set_ylabel('L (cMpc)', fontsize=18)
    else:
        axs.set_xlabel('frequencies (MHz)', fontsize=18)
        axs.set_ylabel('L (degrees)', fontsize=18)
        axs.invert_xaxis()

    fig.subplots_adjust(bottom=0.11, right=0.91, top=0.95, left=0.06)
    cax = plt.axes([0.92, 0.15, 0.02, 0.75])
    fig.colorbar(im,cax=cax)
    plt.show()