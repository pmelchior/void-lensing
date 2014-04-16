#!/bin/env python

import numpy as np
from void_basics import *
from pylab import *

if __name__ == '__main__':
    # get the lensing data HDU
    hdu = get_lensing_data()
    
    # apply selection
    """
    # only main sample (everything not in LRG sample):
    lrgs = get_LRG_sample(hdu[1].data)
    selection = (lrgs == False)
    """
    selection = np.ones(len(hdu[1].data), dtype="bool")
    N = sum(selection)
    print N, "voids selected"

    # define binning
    rbins = np.linspace(0.25, 2.4, 10)

    # get bootstrap profiles
    print "bootstrapping may take a while..."
    Emode, e_Emode, Bmode, e_Bmode, r_mean = stack_EBR(hdu[1].data, selection, rbins, rebinned=True, samples=5000, mean_method=np.mean)
    
    # compute chi^2 and likelihood ratio wrt to void model
    model = void_model(rbins)
    K = likelihood_ratio(Emode, e_Emode,  model, N)
    Chi2 = chi2(Emode, e_Emode,  model, N)
    S_N = SNR(Emode, e_Emode, model)

    # plot data and model
    figure()
    ax = subplot(111)
    ax.plot([0.05, 3], [0,0], 'k:')
    ax.errorbar(r_mean, Emode, yerr=e_Emode, c='k', marker='o', label='E-mode', zorder=1000)
    ax.errorbar(r_mean, Bmode, yerr=e_Bmode, c='r', marker='d', label='B-mode', zorder=999)
    # plot model: due to plotting bug, needs one extra entry to disply fully
    ax.plot(rbins, np.concatenate((model, [nan])), 'b-', drawstyle='steps-post', label='model', zorder=998)
    ax.text(0.98, 0.19, r'$N_v=' + '%d' % sum(selection) + '$', ha='right', va='center', transform=ax.transAxes)
    ax.text(0.98, 0.12, r'$N_g=' + '%.2f' % (hdu[1].data['npair'][selection].sum()/1e8) + '\cdot 10^8$', ha='right', va='center', transform=ax.transAxes)
    ax.text(0.98, 0.05, r'$\chi^2_m=%1.2f' % Chi2 + ',\ K=' + '%1.2f' % K + '$', ha='right', va='center', transform=ax.transAxes)

    # make legend with "model" label at the bottom
    handles, labels = ax.get_legend_handles_labels()
    handles.append(handles[0])
    del handles[0]
    labels.append(labels[0])
    del labels[0]
    ax.legend(handles, labels, loc='upper right', numpoints=1, frameon=True, scatterpoints=0)

    # data ranges, labels ...
    ax.set_ylim(-0.2, 0.2)
    ax.set_xlim(0.01, 2.6)
    ax.set_ylabel(r'$\Delta\Sigma/R_v\ \mathrm{[} 10^{12} h^2\, \mathrm{M_\odot Mpc^{-3}]}$')
    ax.set_xlabel(r'$r/R_v$')
    subplots_adjust(bottom=0.12, left=0.15, right=0.98, top=0.92)

    # make inset to show radius/z distribution of selection
    z = hdu[1].data["z"]
    radius = hdu[1].data["radius"]
    axi = axes([0.26, 0.65, 0.25, 0.25], axisbg='None')
    axi.scatter(z, radius, c='r', s=5,label='all voids', linewidths=0)
    axi.scatter(z[selection], radius[selection], c='k', s=10, label='selection', linewidths=0)
    axi.set_ylim(0,120)
    axi.set_xlim(0.025, 0.425)
    axi.set_xticks([0.1, 0.2, 0.3, 0.4])
    axi.set_yticks([0, 50, 100])
    axi.set_xlabel('$z$')
    axi.set_ylabel('$R_v$')

    show()
    hdu.close()
