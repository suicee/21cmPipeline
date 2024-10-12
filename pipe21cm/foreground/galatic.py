import numpy as np
from scipy import interpolate as interp
import healpy as hp
from pygdsm import GlobalSkyModel16

def generate_GSM_cube(degree, box_dim, freqs=None, existing_map_dir=None):
    '''
    Generate a cube of diffuse galactic radio emission using the GlobalSkyModel16.
    Codes modified from Shifan Zuo's(https://github.com/zuoshifan) 

    Note:
    Generating maps from GSM2016 can be time-consuming. If you plan to create multiple cubes
    from the same base map, consider generating and saving the map first, then loading it 
    for subsequent operations to improve efficiency.

    Args:
        :degree: float. The FOV of the cube in degrees.(should consider using differnt values for each frequency)
        :box_dim: int. The number of pixels along one dimension of the cube.
        :freqs: np.array. The frequencies at which to generate the GSM cube in MHz.
        :existing_map_dir: str. The directory of an existing GSM map. If provided, the map is loaded from this directory.
    
    Returns:
        :output_cube: np.ndarray. The diffuse galactic radio emission cube.
    '''

    if existing_map_dir is None:
        if freqs is None:
            raise ValueError("If existing_map_dir is None, freqs must be provided.")
        
        gsm_2016 = GlobalSkyModel16(freq_unit='MHz', interpolation='cubic')
        fg = gsm_2016.generate(freqs)

    else:
        fg = np.load(existing_map_dir)

    # get the center of the patch(should check the reason for the range, probably SKA1-Low region)
    theta0 = np.pi/12 + np.random.rand() * (5*np.pi/6) # [pi/12, 11*pi/12]
    phi0 = np.random.rand() * 2*np.pi # [0, 2*pi]
    vec0 = hp.ang2vec(theta0, phi0)
    ra0 = np.degrees(phi0 - np.pi) # degree
    dec0 = np.degrees(np.pi/2 - theta0) # degree

    # select a circular patch with radius degree, and get the ra, dec
    nside = hp.npix2nside(len(fg[0]))
    pis = hp.query_disc(nside, vec0, np.radians(degree), inclusive=True)
    theta, phi = hp.pix2ang(nside, pis)
    ra = np.degrees(phi - np.pi)
    dec = np.degrees(np.pi/2 - theta)

    # define the output cube
    output_cube = np.zeros((box_dim, box_dim, len(freqs)), dtype=np.float64) 

    # interpolate the circular patch to the target (ra, dec) grid
    for i in range(len(freqs)):
        #get target grid
        points = np.array([ra, dec]).T
        ra_low = ra0 - degree/2
        ra_high = ra0 + degree/2
        dec_low = dec0 - degree/2
        dec_high = dec0 + degree/2
        grid_ra, grid_dec = np.mgrid[ra_low:ra_high:box_dim*1j, dec_low:dec_high:box_dim*1j]

        # interpolate
        output_cube[:,:,i] = interp.griddata(points, fg[i,pis], (grid_ra, grid_dec), method='cubic')

    return output_cube*1e3
