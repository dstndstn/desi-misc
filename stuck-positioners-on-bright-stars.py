'''
This is the script used to nudge DESI tile centers so that bright stars do not land on stuck
fiber positioners.
'''
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

#tiles_filename = '/global/cfs/cdirs/desi/survey/ops/surveyops/trunk/ops/tiles-main.ecsv'
# DESI-ext
tiles_filename = '/global/cfs/cdirs/desi/users/djschleg/tiling/tiles-geometry-superset-new.ecsv'

def _match_tile(X):
    k, uid, tile_ra, tile_dec, tile_obstime, tile_theta, tile_obsha, match_radius = X

    loc_ra,loc_dec = xy2radec(
        hw, tile_ra, tile_dec, tile_obstime, tile_theta, tile_obsha,
        stuck_x, stuck_y, False, 0)
    kd = tree_build_radec(loc_ra, loc_dec)
    I,J,d = trees_match(starkd, kd, match_radius)
    print(k, 'Tile', uid, 'matched', len(I), 'stars')
    if len(I):
        res = uid, I, loc_ra[J], loc_dec[J], stuck_loc[J], np.rad2deg(d)*3600.
    else:
        res = None
    return res

def find_stuck_on_stars():
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
    
    tiles = Table.read(tiles_filename)
    print(len(tiles), 'tiles')
    # Deduplicate tiles with same RA,Dec center
    tilera = tiles['RA']
    tiledec = tiles['DEC']

    tile_uid = tiles_get_unique_id(tiles)

    rdtile = {}
    tilemap = {}
    for uid,r,d in zip(tile_uid, tilera, tiledec):
        key = r,d
        if key in rdtile:
            # already seen a tile with this RA,Dec; point to it
            tilemap[uid] = rdtile[key]
        else:
            rdtile[key] = uid
    del rdtile

    t_mid = datetime(2027, 1, 1)
    tile_obstime = t_mid.isoformat(timespec='seconds')
    mjd = Time(t_mid).mjd
    print('Moving stars to MJD', mjd)

    stars = fits_table('/global/cfs/cdirs/desi/users/ioannis/SGA/2025/gaia/gaia-mask-dr3-allsky.fits')
    print(len(stars), 'stars for masking')

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
    
    mp = multiproc(128)

    print('Building arg lists...')
    args = []
    #design_ha = tiles['DESGNHA']
    # Eddie told me to do this :)
    design_ha = np.zeros(len(tiles))
    k = 0
    for uid,tile_ra,tile_dec,tile_obsha in zip(tile_uid, tilera, tiledec, design_ha):
        # skip duplicate tiles
        if uid in tilemap:
            continue
        # "fieldrot"
        tile_theta = field_rotation_angle(tile_ra, tile_dec, mjd)
        args.append((k, uid, tile_ra, tile_dec, tile_obstime, tile_theta, tile_obsha, match_radius))
        k += 1

    print('Matching', len(args), 'unique tile RA,Decs in parallel...')
    res = mp.map(_match_tile, args)

    print('Organizing results...')
    T = fits_table()
    T.tile_uniqueid = []
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
        uid, I, pos_ra, pos_dec, pos_loc, dists = vals
        T.tile_uniqueid.extend([uid] * len(I))
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

def tiles_get_unique_id(tiles):
    # in the DESI-ext file, many tileids are -1.  RA,DEC values are duplicated for programs.
    tilera = tiles['RA']
    tiledec = tiles['DEC']
    tileprog = tiles['PROGRAM']
    tile_uid = np.array(['%s_%.3f_%.3f' % (p, r, d) for p,r,d in zip(tileprog, tilera, tiledec)])
    return tile_uid
        
def nudge_tile_centers():
    # Uses stuck-on-stars.fits to nudge the tile centers
    tiles = Table.read(tiles_filename)
    tilera = tiles['RA']
    tiledec = tiles['DEC']

    tile_uid = tiles_get_unique_id(tiles)

    rd = list(zip(tilera, tiledec))
    print(len(tiles), 'tiles')
    print(len(set(rd)), 'unique RA,Dec centers')
    print(len(set(tile_uid)), 'unique tile_uids in tiles file')

    tilecenter = {}
    rdtile = {}
    tilemap = {}
    for uid,r,d in zip(tile_uid, tilera, tiledec):
        tilecenter[uid] = (r,d)
        key = r,d
        if key in rdtile:
            # already seen a tile with this RA,Dec; point to it
            tilemap[uid] = rdtile[key]
        else:
            rdtile[key] = uid
    del rdtile

    T = fits_table('stuck-on-stars.fits')
    print(len(set(T.tile_uniqueid)), 'unique tile_uniqueids in stuck-on-stars file')
    print(len(T), 'stuck-on-stars matches')

    duptile = np.isin(T.tile_uniqueid, list(tilemap.keys()))
    print(np.sum(duptile), 'duplicate tiles')
    T.cut(np.logical_not(duptile))
    print(len(T), 'matches on non-dup tiles')
    print(len(set(T.tile_uniqueid)), 'tiles')

    def dr9_based_mask_radius(mag):
        ''' in arcsec '''
        rad_dr9 = 1630. * 1.396**(-mag)
        return rad_dr9 / 8.

    def badness(rad, mag):
        # INVERSE of dr9_based_mask_radius
        mag_safe = (np.log(1630. / 8.) - np.log(rad)) / np.log(1.396)
        # Badness = flux ratio over flux at safe radius.
        badness = 10.**((mag - mag_safe)/-2.5)
        return badness

    T.mask_radius = dr9_based_mask_radius(T.mask_mag)
    T.badness = badness(np.maximum(0.5, T.dist_arcsec), T.mask_mag)

    mags = np.array([0., 10., 11., 12., 16.])
    print('Mags:', mags)
    r = dr9_based_mask_radius(mags)
    print('DR9-based mask radii:', r)
    print('Badness at r/2:', badness(r/2., mags))
    print('Badness at r:', badness(r, mags))
    print('Badness at r*2:', badness(r*2, mags))
    print('Badness at mag+1:', badness(r, mags+1.))
    print('Badness at mag-1:', badness(r, mags-1.))

    i = np.argmax(T.badness)
    print('Worst badness:', T.badness[i])
    print('Mag:', T.mask_mag[i])
    print('Safe radius:', dr9_based_mask_radius(T.mask_mag[i]))
    print('Radius:', T.dist_arcsec[i])

    # plt.hist(T.badness, range=(0,100), bins=20, log=True)
    # plt.savefig('badness.png')
    # sys.exit(0)

    Iworst = np.argsort(-T.badness)
    badtiles = []
    btset = set()
    for uid in T.tile_uniqueid[Iworst]:
        if uid in btset:
            continue
        badtiles.append(uid)
        btset.add(uid)

    T.pos_cosd = np.cos(np.deg2rad(T.pos_dec))

    goodshifts = {}
    for itile,uid in enumerate(badtiles):
        print()
        tr,td = tilecenter[uid]
        print('Tile uid', uid, 'at RA,Dec =', tr, td)
        Itile = np.flatnonzero(T.tile_uniqueid == uid)
        bad = (T.dist_arcsec[Itile] < T.mask_radius[Itile])
        Ibad = Itile[bad]
        print(len(Ibad), 'bad for tile', uid)
        print('Mags:', T.mask_mag[Ibad])
        print('Mask radii:', T.mask_radius[Ibad])
        print('Dists:', T.dist_arcsec[Ibad])

        tr,td = tilecenter[uid]
        tile_cosd = np.cos(np.deg2rad(td))

        # Compute shifts in units of 0.001 deg in RA and Dec, up to radius R (in arcsec)
        R = 15
        step = 0.001
        decsteps = int(np.ceil(R / 3600. / step))
        rasteps  = int(np.ceil(R / 3600. / tile_cosd / step))

        # dra,ddec in DEG, non-isotropic (ie, new_ra = ra + dra)
        dra,ddec = np.meshgrid(np.arange(-rasteps, rasteps+1), np.arange(-decsteps, decsteps+1))
        dra  = (dra  * step).ravel()
        ddec = (ddec * step).ravel()

        # Sort by radius
        I = np.argsort(np.hypot(dra * tile_cosd, ddec))
        ## skip 0,0
        #I = I[1:]
        dra = dra[I]
        ddec = ddec[I]

        # trim to R
        I = (np.hypot(dra * tile_cosd, ddec) <= R/3600.)
        dra = dra[I]
        ddec = ddec[I]

        leastbad = 1e12
        leastbad_max = None
        best_shift = None

        badnesses = []
        
        for dr,dd in zip(dra,ddec):
            # Shift the positioners
            dnew = T.pos_dec[Itile] + dd
            rnew = T.pos_ra[Itile]  + dr
            newrad = arcsec_between(rnew, dnew, T.star_ra[Itile], T.star_dec[Itile])
            newbad = badness(newrad, T.mask_mag[Itile])

            if dr == 0. and dd == 0.:
                print('Shift (%.4f, %.4f)' % (dr,dd), ': worst badness', max(newbad),
                      'total badness', sum(newbad))

            bad = sum(newbad)
            badnesses.append(bad)
            if bad < leastbad:
                leastbad = bad
                best_shift = (dr,dd)
                leastbad_max = max(newbad)

                # New best one is also "acceptable"?
                if max(newbad) < 1.:
                    break

        dr,dd = best_shift
        print('Shift (%.4f, %.4f)' % (dr,dd), ': worst badness', leastbad_max,
              'total badness', leastbad)
                
        goodshifts[uid] = (best_shift, leastbad, leastbad_max)

        #if itile < 10:
        #if tile in [3827]: #22427,  3427, 11815,  2647, 21647, 10191, 26112,  7112, 42039, 1848]:
        if uid in []:
            plt.clf()
            dr = 3600. * (T.star_ra [Itile] - T.pos_ra [Itile])*tile_cosd
            dd = 3600. * (T.star_dec[Itile] - T.pos_dec[Itile])
            rad = T.mask_radius[Itile]
            #plt.plot(dr, dd, 'k.')
            from matplotlib.patches import Circle
            for r,d,rr in zip(dr,dd,rad):
                plt.gca().add_artist(Circle((r,d), rr, color='k', alpha=0.2))
            #plt.plot(dra*tile_cosd*3600., ddec*3600., 'kx', alpha=0.5)

            N = len(badnesses)
            plt.scatter(dra[:N]*tile_cosd*3600., ddec[:N]*3600., c=np.log10(badnesses),
                        vmin=-1, vmax=2)
            plt.plot(dra[N:]*tile_cosd*3600., ddec[N:]*3600., 'k.')
    
            dr,dd = best_shift
            plt.plot(dr*tile_cosd*3600., dd*3600., 'o', mec='r', mfc='none', mew=3, ms=10)
            cb = plt.colorbar()
            cb.set_label('log10(badness)')
            plt.axis('square')
            plt.axis([-20, 20, -20, 20])
            plt.axhline(0., color='k', alpha=0.1)
            plt.axvline(0., color='k', alpha=0.1)
            plt.title('Tile %s: brightest mag = %.1f, max badness %.1f' % (uid, min(T.mask_mag[Itile]), leastbad_max))
            plt.savefig('tile-%05i.png' % uid)
            #sys.exit(0)

    tiles['NUDGE_SUM_BADNESS'] = np.zeros(len(tiles), np.float32)
    tiles['NUDGE_MAX_BADNESS'] = np.zeros(len(tiles), np.float32)
    tiles['NUDGE_RA']  = np.zeros(len(tiles), np.float32)
    tiles['NUDGE_DEC'] = np.zeros(len(tiles), np.float32)
    tiles['NUDGED_RA']  = tiles['RA'].copy()
    tiles['NUDGED_DEC'] = tiles['DEC'].copy()

    for i,(uid) in enumerate(tile_id):
        # map to canonical tile with this tile's RA,Dec
        uid = tilemap.get(uid, uid)

        if not uid in goodshifts:
            # this tile does not need a nudge.
            continue

        (dr,dd), bad_sum, bad_max = goodshifts[uid]

        tiles['NUDGE_SUM_BADNESS'][i] = bad_sum
        tiles['NUDGE_MAX_BADNESS'][i] = bad_max
        tiles['NUDGE_RA' ][i] = dr
        tiles['NUDGE_DEC'][i] = dd
        tiles['NUDGED_RA' ][i] += dr
        tiles['NUDGED_DEC'][i] += dd

    tiles.write('tiles-nudged.ecsv', overwrite=True)
    tiles.write('tiles-nudged.fits', overwrite=True)

    # Write out updated (drop-in-able) tiles file
    tiles['RA'][:] = tiles['NUDGED_RA']
    tiles['DEC'][:] = tiles['NUDGED_DEC']
    tiles.remove_column('NUDGE_SUM_BADNESS')
    tiles.remove_column('NUDGE_MAX_BADNESS')
    tiles.remove_column('NUDGE_RA')
    tiles.remove_column('NUDGE_DEC')
    tiles.remove_column('NUDGED_RA')
    tiles.remove_column('NUDGED_DEC')
    tiles.write('tiles-nudged-new.ecsv', overwrite=True)
    tiles.write('tiles-nudged-new.fits', overwrite=True)

if __name__ == '__main__':
    #tiles = Table.read(tiles_filename)
    #tiles.write('tiles-orig.fits', overwrite=True)
    #sys.exit(0)

    # This writes the stuck-on-stars.fits file
    find_stuck_on_stars()
    # This nudges the tile centers (using stuck-on-stars.fits)
    nudge_tile_centers()
