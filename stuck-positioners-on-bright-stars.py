import pylab as plt
import os
import numpy as np
from collections import Counter
from datetime import datetime, timezone

from astrometry.util.fits import fits_table
from astrometry.libkd.spherematch import match_radec
from astrometry.libkd.spherematch import tree_open, tree_build_radec, trees_match
from astrometry.util.starutil_numpy import deg2dist, arcsec_between
from astrometry.util.multiproc import multiproc

from astropy.time import Time
from astropy.table import Table

from desimodel.io import load_focalplane
from fiberassign.hardware import load_hardware
from fiberassign.hardware import FIBER_STATE_STUCK, FIBER_STATE_BROKEN, xy2radec
from fiberassign.tiles import load_tiles
from desimodel.focalplane.fieldrot import field_rotation_angle

hw = None
stuck_x = None
stuck_y = None
stuck_loc = None
starkd = None

def _match_tile(X):
    tid, tile_ra, tile_dec, tile_obstime, tile_theta, tile_obsha, match_radius = X

    loc_ra,loc_dec = xy2radec(
        hw, tile_ra, tile_dec, tile_obstime, tile_theta, tile_obsha,
        stuck_x, stuck_y, False, 0)
    kd = tree_build_radec(loc_ra, loc_dec)
    I,J,d = trees_match(starkd, kd, match_radius)
    print('Tile', tid, 'matched', len(I), 'stars')
    if len(I):
        res = tid, I, loc_ra[J], loc_dec[J], stuck_loc[J], np.rad2deg(d)*3600.
    else:
        res = None
    return res

def main():
    os.environ['DESIMODEL'] = '/global/homes/d/dstn/desimodel-data'

    global hw
    global stuck_x
    global stuck_y
    global stuck_loc
    global starkd

    hw = load_hardware()
    
    # From fiberassign/stucksky.py: find X,Y positions of stuck positioners.
    
    # (grab the hw dictionaries once -- these are python wrappers over C++ so not simple accessors)
    state = hw.state
    devtype = hw.loc_device_type
    stuck_loc = [loc for loc in hw.locations
                 if (((state[loc] & (FIBER_STATE_STUCK | FIBER_STATE_BROKEN)) == FIBER_STATE_STUCK) and
                     (devtype[loc] == 'POS'))]
    print(len(stuck_loc), 'stuck positioners')
    theta_pos = hw.loc_theta_pos
    theta_off = hw.loc_theta_offset
    phi_pos = hw.loc_phi_pos
    phi_off = hw.loc_phi_offset
    stuck_theta = [theta_pos[loc] + theta_off[loc] for loc in stuck_loc]
    stuck_phi   = [phi_pos  [loc] + phi_off  [loc] for loc in stuck_loc]
    curved_mm = hw.loc_pos_curved_mm
    theta_arm = hw.loc_theta_arm
    phi_arm   = hw.loc_phi_arm
    theta_min = hw.loc_theta_min
    theta_max = hw.loc_theta_max
    phi_min   = hw.loc_phi_min
    phi_max   = hw.loc_phi_max
    # Convert positioner angle orientations to curved focal surface X / Y (not CS5)
    # Note:  we could add some methods to the python bindings to vectorize this or make it less clunky...
    stuck_x = np.zeros(len(stuck_loc))
    stuck_y = np.zeros(len(stuck_loc))
    for iloc, (loc, theta, phi) in enumerate(zip(stuck_loc, stuck_theta, stuck_phi)):
        loc_x, loc_y = hw.thetaphi_to_xy(curved_mm[loc], theta, phi, theta_arm[loc], phi_arm[loc],
            theta_off[loc], phi_off[loc], theta_min[loc], phi_min[loc], theta_max[loc], phi_max[loc], True)
        stuck_x[iloc] = loc_x
        stuck_y[iloc] = loc_y
    
    tiles = Table.read('/global/cfs/cdirs/desi/target/surveyops/ops/tiles-main.ecsv')
    
    #tycho = fits_table('/global/cfs/cdirs/cosmo/staging/tycho2/tycho2.kd.fits')
    ##tychokd = tree_open('/global/cfs/cdirs/cosmo/staging/tycho2/tycho2.kd.fits')
    #tychokd = tree_build_radec(tycho.ra, tycho.dec)

    stars = fits_table('/global/cfs/cdirs/cosmo/data/legacysurvey/dr9/masking/gaia-mask-dr9.fits.gz')
    starkd = tree_build_radec(stars.ra, stars.dec)

    match_radius = deg2dist(20./3600.)

    stuck_loc = np.array(stuck_loc)
    
    allresults = {}
    
    tnow = datetime.now()
    tile_obstime = tnow.isoformat(timespec='seconds')
    mjd = Time(tnow).mjd

    mp = multiproc(32)

    print('Building arg lists...')
    args = []
    for i in range(len(tiles)):
        tile = tiles[i]
        tid = tile['TILEID']
        tile_ra = tile['RA']
        tile_dec = tile['DEC']
        tile_obsha = tile['DESIGNHA']
        # "fieldrot"
        tile_theta = field_rotation_angle(tile_ra, tile_dec, mjd)
        args.append((tid, tile_ra, tile_dec, tile_obstime, tile_theta, tile_obsha, match_radius))

    print('Matching in parallel...')
    res = mp.map(_match_tile, args)

    print('Organizing results...')
    T = fits_table()
    T.tileid = []
    T.loc = []
    T.petal = []
    T.device = []
    T.fiber = []
    T.pos_ra = []
    T.pos_dec = []
    T.star_ra = []
    T.star_dec = []
    T.dist_arcsec = []
    T.mask_mag = []

    loc_to_petal = hw.loc_petal
    loc_to_device = hw.loc_device
    loc_to_fiber = hw.loc_fiber

    for vals in res:
        if vals is None:
            continue
        tileid, I, pos_ra, pos_dec, pos_loc, dists = vals
        T.tileid.extend([tileid] * len(I))
        T.loc.extend(pos_loc)
        for loc in pos_loc:
            T.petal.append(loc_to_petal[loc])
            T.device.append(loc_to_device[loc])
            T.fiber.append(loc_to_fiber[loc])
        T.pos_ra.extend(pos_ra)
        T.pos_dec.extend(pos_dec)
        T.star_ra.extend(stars.ra[I])
        T.star_dec.extend(stars.dec[I])
        T.dist_arcsec.extend(dists)
        T.mask_mag.extend(stars.mask_mag[I])
    T.to_np_arrays()
    T.writeto('stuck-on-stars.fits')

if __name__ == '__main__':
    main()
    
