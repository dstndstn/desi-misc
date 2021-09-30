%matplotlib inline
import pylab as plt
import os
import numpy as np
from collections import Counter
from datetime import datetime, timezone

from astrometry.util.fits import fits_table
from astrometry.libkd.spherematch import match_radec
from astrometry.libkd.spherematch import tree_open, tree_build_radec, trees_match
from astrometry.util.starutil_numpy import deg2dist, arcsec_between

from astropy.time import Time
from astropy.table import Table

from desimodel.io import load_focalplane
from fiberassign.hardware import load_hardware
from fiberassign.hardware import FIBER_STATE_STUCK, FIBER_STATE_BROKEN, xy2radec
from fiberassign.tiles import load_tiles
from desimodel.focalplane.fieldrot import field_rotation_angle

def main():
    os.environ['DESIMODEL'] = '/global/homes/d/dstn/desimodel-data'
    
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
    
    tycho = fits_table('/global/cfs/cdirs/cosmo/staging/tycho2/tycho2.kd.fits')
    #tychokd = tree_open('/global/cfs/cdirs/cosmo/staging/tycho2/tycho2.kd.fits')
    tychokd = tree_build_radec(tycho.ra, tycho.dec)
    match_radius = deg2dist(5./3600.)
    
    stuck_loc = np.array(stuck_loc)
    
    allresults = {}
    
    tnow = datetime.now()
    tile_obstime = tnow.isoformat(timespec='seconds')
    mjd = Time(tnow).mjd
    
    inext = 2
    for i in range(len(tiles)):
        if i == inext:
            print(i, np.sum([(v is not None) for k,v in allresults.items()]))
            inext *= 2
        tile = tiles[i]
        tid = tile['TILEID']
        tile_ra = tile['RA']
        tile_dec = tile['DEC']
        tile_obsha = tile['DESIGNHA']
        # "fieldrot"
        tile_theta = field_rotation_angle(tile_ra, tile_dec, mjd)
    
        loc_ra,loc_dec = xy2radec(
            hw, tile_ra, tile_dec, tile_obstime, tile_theta, tile_obsha, stuck_x, stuck_y, False, 0)
        kd = tree_build_radec(loc_ra, loc_dec)
        I,J,d = trees_match(tychokd, kd, match_radius)
        if len(I):
            mag = tycho.mag_vt[I]
            maghp = tycho.mag_hp[I]
            magbt = tycho.mag_bt[I]
            mag[mag == 0] = maghp[mag == 0]
            mag[mag == 0] = magbt[mag == 0]
            allresults[tid] = (I, J, np.rad2deg(d) * 3600., tycho.ra[I], tycho.dec[I], loc_ra[J], loc_dec[J], stuck_loc[J], mag)
        else:
            allresults[tid] = None

    loc_to_petal = hw.loc_petal
    loc_to_device = hw.loc_device
    loc_to_fiber = hw.loc_fiber
            
    T = fits_table()
    T.tileid = []
    T.loc = []
    T.petal = []
    T.device = []
    T.fiber = []
    T.pos_ra = []
    T.pos_dec = []
    T.tycho_ra = []
    T.tycho_dec = []
    T.tycho_mag = []

    for k,v in allresults:
        if v is None:
            continue
        (I,J,D,tras,tdecs,pras,pdecs,locs,mags) = v
        tileid = k
        T.tileid.extend([tileid] * len(I))
        T.loc.extend(locs)
        for loc in locs:
            T.petal.append(loc_to_petal[loc])
            T.device.append(loc_to_device[loc])
            T.fiber.append(loc_to_fiber[loc])
        T.pos_ra.extend(pras)
        T.pos_dec.extend(pdecs)
        T.tycho_ra.extend(tras)
        T.tycho_dec.extend(tdecs)
        T.tycho_mag.extend(mags)
    T.to_np_arrays()
    T.writeto('stuck-on-stars.fits')
            
if __name__ == '__main__':
    main()
    
