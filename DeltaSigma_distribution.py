#!/bin/env python

import numpy as np
from void_basics import *
from pylab import *

def Dsigma_histogram(ax, y, mean_method=np.mean, alpha=0.32, samples=10000):
    m, CI = bootstrap_confidence_intervals(y, mean_method=mean_method, alpha=alpha, samples=samples)
    ax.hist(y, 40, range=[-2,2])
    ax.text(0.95,0.95,'$\mu=%.3f \pm %.3f$' % (mean(y), std(y)/sqrt(len(y))), ha='right', va='top', transform=ax.transAxes)
    sigma = (CI[1]-CI[0])/2 # symmetrized error bars
    ax.text(0.95,0.90,r'$\langle\rangle=%.3f \pm %.3f$' % (m, sigma), ha='right', va='top', transform=ax.transAxes)
    yl = ax.get_ylim()
    ax.plot([0,0], [0,yl[1]], 'k:')
    ax.set_ylim(yl)
    ax.set_xlabel('$\Delta\Sigma / R_v$')

if __name__ == '__main__':
    # get the lensing data HDU
    hdu = get_lensing_data()
    data = hdu[1].data
    
    # apply selection
    """
    # only main sample (everything not in LRG sample):
    lrgs = get_LRG_sample(hdu[1].data)
    selection = lrgs #(lrgs == False)
    """
    selection = np.ones(len(hdu[1].data), dtype="bool")
    print sum(selection), "voids selected"

    # get lensing quantities for selection
    radius = data.field('radius')[selection]
    dsum = data.field('dsum')[selection]
    osum = data.field('osum')[selection]
    rsum = data.field('rsum')[selection]
    wsum = data.field('wsum')[selection]
    npair = data.field('npair')[selection]

    # define binning and compute E/B modes and effective radii
    rbins = np.linspace(0.15, 2.4, 12)
    data_at_r = rebin_EBR(rbins, radius, dsum, osum, rsum, wsum, npair, rebinned=True)
    
    # show E/B mode distribution for given bin
    bin = 4
    fig = figure()
    ax = subplot(111)
    Dsigma_histogram(ax, data_at_r['E'][bin], samples=5000, mean_method=kappa_sigma)
    show()
    hdu.close()
