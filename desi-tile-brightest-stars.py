import os
import sys
import functools
from collections import Counter
import pylab as plt
import numpy as np
import fitsio
from astrometry.util.fits import fits_table, merge_tables
from astrometry.util.util import Tan
from astrometry.util.starutil_numpy import degrees_between
from astrometry.util.plotutils import plothist
from astrometry.libkd.spherematch import match_radec, tree_build_radec, tree_search_radec, tree_open
sys.path.insert(0, 'legacypipe/py')
from legacypipe.gaiacat import GaiaCatalog
from legacypipe.reference import fix_tycho, fix_gaia, merge_gaia_tycho
os.environ['GAIA_CAT_DIR'] = '/global/cfs/cdirs/cosmo/work/gaia/chunks-gaia-dr2-astrom-2/'

class CachingGaiaCatalog(GaiaCatalog):
    def __init__(self, columns=None, **kwargs):
        super().__init__(**kwargs)
        self.columns = columns

    def get_healpix_catalog(self, healpix):
        #return super().get_healpix_catalog(healpix)
        from astrometry.util.fits import fits_table
        fname = self.fnpattern % dict(hp=healpix)
        #print('Reading', fname)
        return fits_table(fname, columns=self.columns)

    def get_healpix_catalogs(self, healpixes):
        from astrometry.util.fits import merge_tables
        cats = []
        for hp in healpixes:
            cats.append(self.get_healpix_catalog(hp))
        if len(cats) == 1:
            return cats[0].copy()
        return merge_tables(cats)

    @functools.lru_cache(maxsize=4000)
    def get_healpix_tree(self, healpix):
        from astrometry.util.fits import fits_table
        fname = self.fnpattern % dict(hp=healpix)
        tab = fits_table(fname, columns=self.columns)
        kd = tree_build_radec(tab.ra, tab.dec)
        return (kd,tab)
    
    def get_healpix_rangesearch_catalogs(self, healpixes, rc, dc, rad):
        cats = []
        for hp in healpixes:
            (kd,tab) = self.get_healpix_tree(hp)
            I = tree_search_radec(kd, rc, dc, rad)
            if len(I):
                cats.append(tab[I])
        if len(cats) == 1:
            return cats[0] #.copy()
        return merge_tables(cats)
    
    def get_catalog_in_wcs(self, wcs, step=100., margin=10):
        # Grid the CCD in pixel space
        W,H = wcs.get_width(), wcs.get_height()
        xx,yy = np.meshgrid(
            np.linspace(1-margin, W+margin, 2+int((W+2*margin)/step)),
            np.linspace(1-margin, H+margin, 2+int((H+2*margin)/step)))
        # Convert to RA,Dec and then to unique healpixes
        ra,dec = wcs.pixelxy2radec(xx.ravel(), yy.ravel())
        healpixes = set()
        for r,d in zip(ra,dec):
            healpixes.add(self.healpix_for_radec(r, d))

        # Read catalog in those healpixes
        rc,dc = wcs.radec_center()
        rad = wcs.radius()
        cat = self.get_healpix_rangesearch_catalogs(healpixes, rc, dc, rad)

        if len(cat) == 0:
            return cat
        # Cut to sources actually within the CCD.
        _,xx,yy = wcs.radec2pixelxy(cat.ra, cat.dec)
        cat.x = xx
        cat.y = yy
        onccd = np.flatnonzero((xx >= 1.-margin) * (xx <= W+margin) *
                               (yy >= 1.-margin) * (yy <= H+margin))
        cat.cut(onccd)
        return cat


def main(tag):
    
    tiles = fits_table('/global/cfs/cdirs/desi//users/schlafly/tiling/tiles-%s-decorated.fits' % tag)

    I = np.flatnonzero(tiles.in_imaging)
    plt.plot(tiles.ra[I], tiles.dec[I], 'k.', alpha=0.1);

    T = fits_table('/global/cfs/cdirs/desi/users/ameisner/GFA/gfa_reduce_etc/gfa_wcs+focus.bigtan-zenith.fits')

    # Aaron's file above has all images share the boresight CRVAL and large CRPIX values.
    rel_xy = {}
    for t in T:
        wcs = Tan(0., 0., t.crpix[0], t.crpix[1],
                t.cd[0,0], t.cd[0,1], t.cd[1,0], t.cd[1,1], 
                float(t.naxis[0]), float(t.naxis[1]))
        ctype = t.extname[:5]
        cnum = int(t.extname[5])
        rel_xy[(ctype,cnum)] = (0.,0.,wcs)

    maxr = 0.
    for k,(tx,ty,wcs) in rel_xy.items():
        (gstr,gnum) = k
        h,w = wcs.shape
        print('WCS shape', w, h)
        x,y = [1,1,w,w,1],[1,h,h,1,1]
        r,d = wcs.pixelxy2radec(x, y)
        dists = degrees_between(0., 0., r, d)
        maxr = max(maxr, max(dists))

    keys = list(rel_xy.keys())
    keys.sort()

    gfa_regions = []

    Nbright = 10
    tiles_ann = fits_table()

    #wcs_subs = []

    for k in keys:
        (cstr,cnum) = k
        (tx,ty,wcs) = rel_xy[k]
        h,w = wcs.shape

        # x0, y0, x1, y1
        rois = []
        
        if cstr == 'FOCUS':
            # add the two half-chips.
            # wcs.get_subimage moves the CRPIX, but leave CRVAL unchanged, so tx,ty still work unchanged.
            # Aaron's WCS templates correct for the overscans
            #wcs_subs.append((cstr, cnum, 'a', wcs.get_subimage(0, 0, 1024, h)))
            #wcs_subs.append((cstr, cnum, 'b', wcs.get_subimage(1024, 0, 1024, h)))
    
            #all_sub_wcs[(cstr, cnum, 1)] = (tx, ty, wcs.get_subimage(50, 0, 1024, 1032))
            #all_sub_wcs[(cstr, cnum, 2)] = (tx, ty, wcs.get_subimage(1174, 0, 1024, 1032))
            # Add (negative) margin for donut size and telescope pointing uncertainty.
            # ~10" for donuts and ~10" for telescope pointing
            #margin = 100
            #wcs_subs.append((cstr, cnum, 'a_margin', wcs.get_subimage(margin, margin, 1024-2*margin, h-2*margin)))
            #wcs_subs.append((cstr, cnum, 'b_margin', wcs.get_subimage(1024+margin, margin, 1024-2*margin, h-2*margin)))
    
            # Also add a positive margin for bright-star reflections off filters
            #margin = 125
            #wcs_subs.append((cstr, cnum, 'expanded', wcs.get_subimage(-margin, -margin, w+2*margin, h+2*margin)))

            rois.append(('a', 0, 0, 1024, h))
            rois.append(('b', 1024, 0, 2048, h))
            margin = 100
            rois.append(('a_margin', margin, margin, 1024-margin, h-margin))
            rois.append(('b_margin', 1024+margin, margin, 2048-margin, h-margin))

            margin = 125
            rois.append(('expanded', -margin, -margin, w+margin, h+margin))
            
        else:
            # Guide chips include overscan pixels -- including a blank region in the middle.
            #print(cstr,cnum, 'shape', wcs.shape)
            #wcs_subs.append((cstr, cnum, 'ccd', wcs))

            rois.append(('ccd', 0, 0, w, h))
            
            # Add expanded GUIDE chips -- 25" margin / 0.2"/pix = 125 pix
            margin = 125
            #wcs_subs.append((cstr, cnum, 'expanded', wcs.get_subimage(-margin, -margin, w+2*margin, h+2*margin)))
            rois.append(('expanded', -margin, -margin, w+margin, h+margin))

        margin = 125
        expwcs = wcs.get_subimage(-margin, -margin, w+2*margin, h+2*margin)

        newrois = []
        for tag,x0,y0,x1,y1 in newrois:
            name = '%s_%i_%s' % (cstr, cnum, tag)
            arr = np.zeros(len(tiles), (np.float32, Nbright))
            tiles_ann.set('brightest_'+name, arr)
            # (the rois have zero-indexed x0,y0, and non-inclusive x1,y1!)
            newrois.append((name, arr, 1+x0, 1+y0, x1,y1))
        
        gfa_regions.append((cstr, cnum, wcs, expwcs, newrois))

        
    #Nbright = 10
    #tiles_ann = fits_table()
    # wcs_regions = []
    # for cstr,cnum,tag,wcs in wcs_subs:
    #     name = '%s_%i_%s' % (cstr, cnum, tag)
    #     arr = np.zeros(len(tiles), (np.float32, Nbright))
    #     tiles_ann.set('brightest_'+name, arr)        
    #     wcs_regions.append((name, wcs, arr))
    # print('wcs_regions:', len(wcs_regions))

    
    gaia = CachingGaiaCatalog(columns=['ra','dec','phot_g_mean_mag', 'phot_bp_mean_mag', 'phot_rp_mean_mag', 'astrometric_excess_noise',
                                       'astrometric_params_solved', 'source_id', 'pmra_error', 'pmdec_error', 'parallax_error',
                                       'ra_error', 'dec_error', 'pmra', 'pmdec', 'parallax', 'ref_epoch'])

    tyc2fn = '/global/cfs/cdirs/cosmo/staging/tycho2/tycho2.kd.fits'
    tycho_kd = tree_open(tyc2fn)
    tycho_cat = fits_table(tyc2fn)

    maxrad = maxr * 1.05
    for itile,tile in enumerate(tiles):
        if not tile.in_imaging:
            continue
        if tile.centerid % 10 == 0:
            print('tile program', tile.program, 'pass', tile.get('pass'), 'id', tile.centerid, gaia.get_healpix_tree.cache_info())
    
        I = tree_search_radec(tycho_kd, tile.ra, tile.dec, maxrad)
        tycstars = tycho_cat[I]
        fix_tycho(tycstars)

        for cstr, cname, chipwcs, bigwcs, rois in gfa_regions:
            h,w = chipwcs.shape
            chipwcs.set_crval(tile.ra, tile.dec)
            bigwcs.set_crval(tile.ra, tile.dec)
            gstars = gaia.get_catalog_in_wcs(bigwcs, step=1032, margin=0)
            fix_gaia(gstars)

            bh,bw = bigwcs.shape
            ok,x,y = bigwcs.radec2pixelxy(tycstars.ra, tycstars.dec)
            tstars = tycstars[(x >= 1) * (y >= 1) * (x <= bw) * (y <= bh)]

            print('Tile', tile.program, 'p', tile.get('pass'), tile.centerid,
                  'GFA', cstr, cname, ':', len(gstars), 'Gaia stars', len(tstars), 'Tycho-2 stars')
            
            if len(gstars) + len(tstars) == 0:
                print('No stars in tile centerid', tile.centerid, 'chip', name)
                continue

            if len(gstars)>0 and len(tstars)>0:
                print('merge_gaia_tycho')
                merge_gaia_tycho(gstars, tstars)
                print('merge_tables')
                stars = merge_tables([gstars, tstars], columns='fillzero')
                print('merged:', len(stars))
            elif len(tstars)>0:
                stars = tstars
            else:
                stars = gstars

            ok,x,y = chipwcs.radec2pixelxy(stars.ra, stars.dec)

            for name, arr, x0, y0, x1, y1 in rois:
                J = np.flatnonzero((x >= x0) * (x <= x1) * (y >= y0) * (y <= y1))
                mags = stars.mag[J]
                print('  ', len(mags), 'in name')
                K = np.argsort(mag)
                K = K[:Nbright]
                arr[itile, :len(K)] = mag[K]

    tiles.add_columns_from(tiles_ann)
    tiles.writeto('tiles-%s-decorated-brightest.fits' % tag)


if __name__ == '__main__':
    #tag = '4112-packing-20210328'
    tag = '4112-packing-20210329'
    main(tag)
    
