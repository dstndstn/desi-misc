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
from astrometry.util.resample import resample_with_wcs, NoOverlapError
from astrometry.util.multiproc import multiproc
sys.path.insert(0, 'legacypipe/py')
from legacypipe.gaiacat import GaiaCatalog
from legacypipe.reference import fix_tycho, fix_gaia, merge_gaia_tycho
from legacypipe.survey import get_git_version
sys.path.insert(0, 'desiutil/py')
from desiutil.brick import Bricks

def run_one(X):
    k, sb, bricks, version = X
    print(k, sb.brickname)

    outfn = 'skybricks/sky-%s.fits.gz' % sb.brickname
    if os.path.exists(outfn):
        print('Exists')
        return True

    I = np.flatnonzero((bricks.ra2 > sb.ra1) * (bricks.ra1 < sb.ra2) * (bricks.dec2 > sb.dec1) * (bricks.dec1 < sb.dec2))
    if len(I) == 0:
        print('No bricks overlap')
        return False

    # 3600 + 1% margin on each side
    w,h = 3672,3672
    binning = 4
    # pixscale
    cd = 1./3600.

    fullw,fullh = w*binning, h*binning
    fullcd = cd/binning

    # There are really three states: no coverage, blob, no blob.
    # Since blobs outside each brick's unique area do not appear in
    # the files, we start skyblobs as zero, but also track the
    # coverage so we can set !coverage to blob at the end.

    skyblobs = np.zeros((fullh, fullw), bool)
    covered = np.zeros((fullh, fullw), bool)

    skywcs = Tan(sb.ra, sb.dec, (fullw+1)/2., (fullh+1)/2., -fullcd, 0., 0., fullcd, float(fullw), float(fullh))

    for i in I:
        brick = bricks[i]
        #print('Blob', brickname)
        fn = '/global/cfs/cdirs/cosmo/data/legacysurvey/dr9/%s/metrics/%s/blobs-%s.fits.gz' % (brick.hemi, brick.brickname[:3], brick.brickname)
        blobs,hdr = fitsio.read(fn, header=True)
        wcs = Tan(hdr)
        blobs = (blobs > -1)
        try:
            Yo,Xo,Yi,Xi,_ = resample_with_wcs(skywcs, wcs)
        except NoOverlapError:
            continue

        skyblobs[Yo,Xo] |= blobs[Yi,Xi]
        covered[Yo,Xo] = True

    # No coverage = equivalent to there being a blob there (ie,
    # conservative for placing sky fibers)
    skyblobs[covered == False] = True

    # bin down, counting how many times 'skyblobs' is set
    subcount = np.zeros((h,w), np.uint8)
    for i in range(binning):
        for j in range(binning):
            subcount += skyblobs[i::binning, j::binning]
    if np.sum(subcount) == 0:
        print('No blobs touch skybrick')
        return False
    subwcs = Tan(sb.ra, sb.dec, (w+1)/2., (h+1)/2., -cd, 0., 0., cd, float(w), float(h))
    hdr = fitsio.FITSHDR()
    hdr.add_record(dict(name='SB_VER', value=version, comment='desi-sky-locations git version'))
    subwcs.add_to_header(hdr)
    fitsio.write(outfn, subcount, header=hdr, clobber=True)
    print('Wrote', outfn)
    return True
    
# if not os.path.exists(sbfn):
#     skybricks = Bricks(bricksize=1.0)
#     skybricks.to_table().write(sbfn)

def main():
    sbfn = 'skybricks.fits'
    SB = fits_table(sbfn)
    
    Bnorth = fits_table('/global/cfs/cdirs/cosmo/data/legacysurvey/dr9/north/survey-bricks-dr9-north.fits.gz')
    Bsouth = fits_table('/global/cfs/cdirs/cosmo/data/legacysurvey/dr9/south/survey-bricks-dr9-south.fits.gz')
    Bnorth.cut(Bnorth.survey_primary)
    Bsouth.cut(Bsouth.survey_primary)
    Bsouth.cut(Bsouth.dec > -30)
    Bnorth.hemi = np.array(['north']*len(Bnorth))
    Bsouth.hemi = np.array(['south']*len(Bsouth))
    B = merge_tables([Bnorth, Bsouth])
    
    # Rough cut the skybricks to those near bricks.
    I,J,d = match_radec(SB.ra, SB.dec, B.ra, B.dec, 1., nearest=True)
    SB.cut(I)
    # HACK -- just for debugging
    #SB = SB[np.argsort(np.hypot(SB.ra, SB.dec))]

    import argparse
    parser = argparse.ArgumentParser()
    #parser.add_argument('--brick', help='Sky brick name')
    parser.add_argument('--minra', type=float, help='Cut to a minimum RA range of sky bricks')
    parser.add_argument('--maxra', type=float, help='Cut to a maximum RA range of sky bricks')
    parser.add_argument('--threads', type=int, help='Parallelize on this many cores')
    opt = parser.parse_args()
    if opt.minra:
        SB.cut(SB.ra >= opt.minra)
    if opt.maxra:
        SB.cut(SB.ra <= opt.maxra)
    
    Inear = match_radec(SB.ra, SB.dec, B.ra, B.dec, 0.75, indexlist=True)

    version = get_git_version(os.getcwd())
    print('Version string:', version)
    
    args = []
    k = 0
    Isb = []
    for isb,(sb,inear) in enumerate(zip(SB, Inear)):
        if inear is None:
            continue
        args.append((k, sb, B[np.array(inear)], version))
        k += 1
        Isb.append(isb)
    print(len(args), 'sky bricks')
    SB.cut(np.array(Isb))

    if opt.threads:
        mp = multiproc(opt.threads)
    else:
        mp = multiproc()

    exist = mp.map(run_one, args)

    if opt.minra is None and opt.maxra is None:
        exist = np.array(exist)
        SB[exist].writeto('skybricks-exist.fits')
    return

    # # 3600 + 1% margin on each side
    # w,h = 3672,3672
    # binning = 4
    # # pixscale
    # cd = 1./3600.
    # 
    # fullw,fullh = w*binning, h*binning
    # fullcd = cd/binning
    # 
    # skyblobs = np.zeros((fullh, fullw), bool)
    # subcount = np.zeros((h,w), np.uint8)
    # for isb,sb in enumerate(SB):
    #     print('Skyblob', sb.brickname)
    #     skywcs = Tan(sb.ra, sb.dec, (fullw+1)/2., (fullh+1)/2., -fullcd, 0., 0., fullcd, float(fullw), float(fullh))
    # 
    #     skyblobs[:,:] = False
    # 
    #     # indices of bricks near this skybrick.
    #     I = np.array(Inear[isb])
    #     # cut to bricks actually inside the skybrick
    #     I = I[((B.ra2[I] > sb.ra1) * (B.ra1[I] < sb.ra2) * (B.dec2[I] > sb.dec1) * (B.dec1[I] < sb.dec2))]
    #     for i in I:
    #         brickname = B.brickname[i]
    #         print('Blob', brickname)
    #         fn = 'cosmo/data/legacysurvey/dr9/%s/metrics/%s/blobs-%s.fits.gz' % (B.hemi[i], brickname[:3], brickname)
    #         blobs,hdr = fitsio.read(fn, header=True)
    #         wcs = Tan(hdr)
    #         blobs = (blobs > -1)
    #         try:
    #             Yo,Xo,Yi,Xi,_ = resample_with_wcs(skywcs, wcs)
    #         except NoOverlapError:
    #             continue
    #         # We could have accumulated the count directly here rather than building the binary mask first
    #         # except that edge blobs appear in neighboring bricks!
    #         skyblobs[Yo,Xo] |= blobs[Yi,Xi]
    # 
    #     # bin down, counting how many times 'skyblobs' is set
    #     subcount[:,:] = 0
    #     for i in range(binning):
    #         for j in range(binning):
    #             subcount += skyblobs[i::binning, j::binning]
    #     subwcs = Tan(sb.ra, sb.dec, (w+1)/2., (h+1)/2., -cd, 0., 0., cd, float(w), float(h))
    #     
    #     hdr = fitsio.FITSHDR()
    #     subwcs.add_to_header(hdr)
    #     #fitsio.write('skybricks/sky-%s.fits.fz' % sb.brickname, blobcount, header=hdr, clobber=True,
    #     #            compress='GZIP', tiledim=(256,256))
    #     fitsio.write('skybricks/sky-%s.fits.gz' % sb.brickname, blobcount, header=hdr, clobber=True)

if __name__ == '__main__':
    main()
