import numpy as np
import numpy.matlib

def match_pixel(lat, lon, match_lat, match_lon, nbox=1):
    """
    Find closest pixel location to [match_lat, match_lon] in [lat, lon] map.
    Return closest pixel index and surrounding pixels for a box of total
    height/width of nbox pixels.

    Parameters:
        lat (numpy.ndarray): Matrix of latitude, map
        lon (numpy.ndarray): Matrix of longitude, map
        match_lat (float): Latitude of matched location
        match_lon (float): Longitude of matched location
        nbox (int, optional): Total size of box, number of pixels. Default is 1.

    Returns:
        ipixel (numpy.ndarray): Vector of pixel index locations in lat, lon centered on
                                closest pixel location match to match_lat, match_lon
        irow (numpy.ndarray): Vector of pixel rows
        icol (numpy.ndarray): Vector of pixel columns
    """
    if nbox is None:
        nbox = 1

    # Find closest pixel location
    distances = np.sqrt((lat - match_lat)**2 + (lon - match_lon)**2)
    ipixel = np.argmin(distances)
    d = np.amin(distances)
    

    if nbox == 1:
        return np.array([ipixel]), None, None

    # Get corresponding row/column
    r, c = np.unravel_index(ipixel, lat.shape)

    # Expand to size of box
    boxwidth = int(np.floor(nbox/2))
    r2 = np.transpose(np.matlib.repmat(np.linspace(r-boxwidth, r+boxwidth, nbox, dtype=int), nbox,1))
    c2 = np.matlib.repmat(np.linspace(c-boxwidth, c+boxwidth, nbox, dtype=int), nbox,1)
    # Wrap around the world for longitude
    c2 = np.mod(c2, lon.shape[1])

    # Get corresponding pixels
    valid_indices = (r2 >= 0) & (r2 < lat.shape[0]) & (c2 >= 0) & (c2 < lat.shape[1])
    ipixel = np.ravel_multi_index((r2[valid_indices], c2[valid_indices]), lat.shape)
    irow, icol = r2[valid_indices], c2[valid_indices]
    
    
    return d , irow, icol
