import pylab as plt
import os
import numpy as np
from collections import Counter
from datetime import datetime, timezone

import matplotlib
matplotlib.use('Agg')
import pylab as plt

from astrometry.util.fits import fits_table
from astrometry.libkd.spherematch import match_radec
from astrometry.libkd.spherematch import tree_open, tree_build_radec, trees_match
#from astrometry.util.starutil_numpy import deg2dist, arcsec_between
from astrometry.util.starutil_numpy import deg2dist, radectoxyz
from astrometry.util.multiproc import multiproc

from astropy.time import Time
from astropy.table import Table

from desimodel.io import load_focalplane
from fiberassign.hardware import load_hardware
from fiberassign.hardware import FIBER_STATE_STUCK, FIBER_STATE_BROKEN, xy2radec
from fiberassign.tiles import load_tiles
from desimodel.focalplane.fieldrot import field_rotation_angle

from legacypipe.survey import radec_at_mjd

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
    print(len(tiles), 'tiles')
    # Deduplicate tiles with same RA,Dec center
    tilera = tiles['RA']
    tiledec = tiles['DEC']
    tileid = tiles['TILEID']
    rdtile = {}
    tilemap = {}
    for tid,r,d in zip(tileid, tilera, tiledec):
        key = r,d
        if key in rdtile:
            # already seen a tile with this RA,Dec; point to it
            tilemap[tid] = rdtile[key]
        else:
            rdtile[key] = tid
    del rdtile

    tnow = datetime.now()
    tile_obstime = tnow.isoformat(timespec='seconds')
    mjd = Time(tnow).mjd


    
    stars = fits_table('/global/cfs/cdirs/cosmo/data/legacysurvey/dr9/masking/gaia-mask-dr9.fits.gz')
    print(len(stars), 'stars for masking')

    print('Moving to MJD', mjd)
    ra,dec = radec_at_mjd(stars.ra, stars.dec, stars.ref_epoch.astype(float),
                          stars.pmra, stars.pmdec, stars.parallax, mjd)
    assert(np.all(np.isfinite(ra)))
    assert(np.all(np.isfinite(dec)))
    stars.ra = ra
    stars.dec = dec
    print('Building kd-tree...')

    starkd = tree_build_radec(stars.ra, stars.dec)

    match_radius = deg2dist(30./3600.)

    stuck_loc = np.array(stuck_loc)
    
    allresults = {}
    
    mp = multiproc(32)

    print('Building arg lists...')
    args = []
    for tid,tile_ra,tile_dec,tile_obsha in zip(tileid, tilera, tiledec, tiles['DESIGNHA']):
        # skip duplicate tiles
        if tid in tilemap:
            continue
        # "fieldrot"
        tile_theta = field_rotation_angle(tile_ra, tile_dec, mjd)
        args.append((tid, tile_ra, tile_dec, tile_obstime, tile_theta, tile_obsha, match_radius))

    print('Matching', len(args), 'unique tile RA,Decs in parallel...')
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

def radius_for_mag(mag):
    k0 = 15.383
    return k0 * 1.108**-mag

def arcsec_between(ra1,dec1, ra2,dec2):
    xyz1 = radectoxyz(ra1, dec1)
    xyz2 = radectoxyz(ra2, dec2)
    d2 = np.sum((xyz1 - xyz2)**2, axis=1)
    rad = np.arccos(1. - d2 / 2.)
    return 3600.*np.rad2deg(rad)


if __name__ == '__main__':
    #main()

    tiles = Table.read('/global/cfs/cdirs/desi/target/surveyops/ops/tiles-main.ecsv')
    tilera = tiles['RA']
    tiledec = tiles['DEC']
    tileid = tiles['TILEID']
    rd = list(zip(tilera, tiledec))
    print(len(tiles), 'tiles')
    print(len(set(rd)), 'unique RA,Dec centers')

    tilecenter = {}
    rdtile = {}
    tilemap = {}
    for tid,r,d in zip(tileid, tilera, tiledec):
        tilecenter[tid] = (r,d)
        key = r,d
        if key in rdtile:
            # already seen a tile with this RA,Dec; point to it
            tilemap[tid] = rdtile[key]
        else:
            rdtile[key] = tid
    del rdtile

    T = fits_table('stuck-on-stars.fits')
    print(len(T), 'matches')
    print(len(set(T.tileid)), 'tiles')

    duptile = np.isin(T.tileid, list(tilemap.keys()))
    #print(np.sum(duptile), 'duplicate tiles')
    T.cut(np.logical_not(duptile))
    print(len(T), 'matches on non-dup tiles')
    print(len(set(T.tileid)), 'tiles')

    rad = radius_for_mag(T.mask_mag)
    T.mask_radius = rad
    Ibad = np.flatnonzero(T.dist_arcsec < rad)
    print(len(Ibad), 'are too bright')
    badtiles = np.unique(T.tileid[Ibad])
    print(len(badtiles), 'tiles')

    R = 15
    dra,ddec = np.meshgrid(np.arange(-R, R+1), np.arange(-R, R+1))
    dra = dra.ravel()
    ddec = ddec.ravel()
    I = np.argsort(np.hypot(dra, ddec))
    # skip 0,0
    I = I[1:]
    dra = dra[I]
    ddec = ddec[I]

    goodshifts = {}

    for tile in badtiles:
        print()
        print('Tile', tile)
        I = np.flatnonzero(T.tileid == tile)
        bad = (T.dist_arcsec[I] < T.mask_radius[I])
        Ibad = I[bad]
        print(len(Ibad), 'bad for tile', tile)
        print('Mags:', T.mask_mag[Ibad])
        print('Mask radii:', T.mask_radius[Ibad])
        print('Dists:', T.dist_arcsec[Ibad])

        tr,td = tilecenter[tile]
        cosd = np.cos(np.deg2rad(td))
        goodshift = None
        for dr,dd in zip(dra,ddec):
            # Shift the positioners
            dnew = T.pos_dec[I] + dd/3600.
            rnew = T.pos_ra[I]  + dr/3600. / cosd
            newrad = arcsec_between(rnew, dnew, T.star_ra[I], T.star_dec[I])
            newbad = newrad < T.mask_radius[I]
            print('Shift (%+ 2i, %+ 2i)' % (dr,dd), ': bad radii',
                  ', '.join(['%.1f vs %.1f' % (nr,t)
                             for nr,t in zip(newrad[bad], T.mask_radius[Ibad])]),
                  ', total of', np.sum(newbad), 'are too close to stars')
            if not np.any(newbad):
                print('Found acceptable shift: dr,dd', dr,dd)
                goodshift = (dr,dd)
                break

        if goodshift is None:
            print('Failed to find a nudge for tile', tile)

            plt.clf()
            dr = 3600. * (T.star_ra [I] - T.pos_ra [I])*cosd
            dd = 3600. * (T.star_dec[I] - T.pos_dec[I])
            rad = T.mask_radius[I]
            plt.plot(dr, dd, 'k.')
            from matplotlib.patches import Circle
            for r,d,rr in zip(dr,dd,rad):
                plt.gca().add_artist(Circle((r,d), rr, color='b', alpha=0.2))
            plt.plot(dra, ddec, 'k.', alpha=0.1)
            plt.axis('square')
            plt.axis([-20, 20, -20, 20])
            plt.savefig('tile-%05i.png' % tile)

        goodshifts[tile] = goodshift

    tiles['NEEDS_NUDGE'] = np.zeros(len(tiles), bool)
    tiles['FOUND_NUDGE'] = np.zeros(len(tiles), bool)
    tiles['NUDGE_RA']  = np.zeros(len(tiles), np.float32)
    tiles['NUDGE_DEC'] = np.zeros(len(tiles), np.float32)
    tiles['NUDGED_RA']  = tiles['RA'].copy()
    tiles['NUDGED_DEC'] = tiles['DEC'].copy()

    for i,(tid) in enumerate(tileid):
        # map to canonical tile with this tile's RA,Dec
        tid = tilemap.get(tid, tid)

        if not tid in goodshifts:
            # this tile does not need a nudge.
            continue
        tiles['NEEDS_NUDGE'][i] = True
        nudge = goodshifts[tid]
        if nudge is None:
            # no nudge we tried worked!
            continue
        tiles['FOUND_NUDGE'][i] = True
        dr,dd = nudge
        tiles['NUDGE_RA' ][i] = dr
        tiles['NUDGE_DEC'][i] = dd
        cosd = np.cos(np.deg2rad(tiles['DEC'][i]))
        tiles['NUDGED_RA' ][i] += (dr/3600.)/cosd
        tiles['NUDGED_DEC'][i] += dd/3600.

    tiles.write('tiles-nudged.ecsv', overwrite=True)
    tiles.write('tiles-nudged.fits', overwrite=True)
