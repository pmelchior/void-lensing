#!/bin/env python

import numpy as np

def get_lensing_data():
    """Get data for the SDSS void lensing analysis.

    The data is from SDSS DR8 r-band imaging. See section 2 of the paper for
    details. Use get_EBR() to convert to E/B-mode measurements.

    Returns:
        pyfits HDU, with the data stored in extension 1.
    """
    import pyfits
    # open shear profile catalog
    return pyfits.open("data/collated-sv03s07.fits.gz")

def kappa_sigma(y):
    """Outlier rejection with kappa-sigma clipping.

    Iteratively determines the 3-sigma region around median. When convergence
    is reached (relative change in sigma < 1e-2), returns the mean of the
    remaining samples.

    Args:
        y: list of real-values numbers
    Returns:
        The mean of y without 3-sigma outliers
    """
    mask = np.ones_like(y, dtype='bool')
    while(True):
        y = y[mask]
        mu = np.median(y)
        err = y.std()
        mask = (np.fabs(y - mu)/err < 3)
        err_ = y[mask].std()
        if np.fabs(err - err_) < 0.01*err:
            return y[mask].mean()

def get_EBR(radius, dsum, osum, rsum, wsum, npair, rebinned=True):
    """Calculated E-mode, B-mode, and associated radius for shear measurements.

    Shear measurements are stored in the FITS file in terms of the fields
    (dsum, osum, wsum, rsum, npair) and are related to E/B mode and radius:

    E-mode = dsum/wsum
    B-mode = osum/wsum
    Radius = rsum/wsum

    where wsum is the sum of weights in the original physical radius bins.
    As measurements are not always present at all radii, the number of galaxies
    npair is needed to reject npair==0 cases.

    Radius is only needed when rebinned==True, in which case a rescaling of
    each void with 1/Radius is applied.

    Returns:
        List of E-mode, B-mode, E/B-mode variance estimate, radius,
        and effective bin width.
    """
    N = radius.size
    # the binning of the shear profiles in Mpc/h
    rbins = np.exp(0.3883*np.arange(-10, 12))
    Emode = np.zeros(N * rbins.size)
    Bmode = np.zeros(N * rbins.size)
    e_modes = np.zeros(N * rbins.size)
    r_mean = np.zeros(N * rbins.size)
    e_r_mean = np.zeros(N * rbins.size)
    first = 0
    for i in xrange(N):
        valid = npair[i] > 0
        added = sum(valid)
        rescaling = 1
        if rebinned:
            rescaling = radius[i]
        if added:
            Emode[first:first+added] = dsum[i][valid]/wsum[i][valid]/rescaling
            Bmode[first:first+added] = osum[i][valid]/wsum[i][valid]/rescaling
            e_modes[first:first+added] = ((wsum[i][valid])**-1)/rescaling
            r_mean[first:first+added] =  rsum[i][valid]/npair[i][valid]/rescaling
            e_r_mean[first:first+added] = rbins[valid]/rescaling

        first += added
    return Emode[:first], Bmode[:first], e_modes[:first], r_mean[:first], e_r_mean[:first]

def rebin_EBR(rbins, radius, dsum, osum, rsum, wsum, npair, rebinned=True):
    """Bin measurements of E and B mode into given radial bins.

    Args:
        rbins: radial bins in physical units of Mpc/h (rebinned==False) or in 
            void radius units r/R_v (rebinned=True)
        radius: list of void radii
        dsu, osum, wsum, npair: see get_EBR()

    Returns:
        Dictionary with "E", "B", "R" lists, each holding k ordered entries that
        correspond to the bins provided. Each bin has a list of E/B/R entries
        found to be within the bin.
    """
    N = radius.size
    bins = rbins.size - 1
    Emode, Bmode, e_modes, r_mean, e_r_mean = get_EBR(radius, dsum, osum, rsum, wsum, npair, rebinned=rebinned)
    data_at_r = {"E": [], "B": [], "R": []}
    for k in xrange(bins):
        inbin = (r_mean >= rbins[k]) & (r_mean < rbins[k+1])
        data_at_r["E"].append(Emode[inbin])
        data_at_r["B"].append(Bmode[inbin])
        data_at_r["R"].append(r_mean[inbin])
    return data_at_r

def bootstrap_indices(N):
    """Random bootstrap index list for an array of length N"""
    return np.random.randint(N, size=N)

def jackknife_indices(N, i):
    """Random leave-one (entry i) jacknife index list for an array of length N"""
    indices = np.arange(0,N)
    indices = np.delete(indices, i)
    return indices

def bootstrap_confidence_intervals(y, mean_method=np.mean, alpha=0.32, samples=10000):
    """Boostrap confidence intervals following Efron (1987).

    Uses the bias-corrected, accelerated method (BCa) from Efron's paper
    "Better Bootstrap Confidence Intervals". This implementation is mostly
    taken from https://github.com/cgevans/scikits-bootstrap

    Args:
        y: an array of floats, or array thereof for a higher-dimensional data 
            set. In the latter case, the data is assumed delineated along 
            axis=0.         
        mean_method: the function to be applied to y, typically some sort of 
            central value, such as mean, median, or kappa_sigma.
        alpha: Symmetric percentile interval
        samples: Number of bootstrap realizations
    
    Returns:
        mean_method(y), tuple of convidence intervals
    """
    N = y.size
    # get the bootstrap and jackknife samples
    # use worker pool here
    stat = np.array([mean_method(y[bootstrap_indices(N)]) for sample in xrange(samples)])
    jstat = np.array([mean_method(y[jackknife_indices(N,i)]) for i in xrange(N)])
    # also get the statistics for the original data set
    ostat = mean_method(y)

    # bias-corrected and accelerated confidence intervals
    from scipy.stats import norm
    stat.sort(axis=0)
    # the bias correction value.
    z0 = norm.ppf((1.0*np.sum(stat < ostat, axis=0))/samples)
    jmean = np.mean(jstat, axis=0)
    # the acceleration value
    a = np.sum((jmean - jstat)**3, axis=0)/(6.0*np.sum((jmean - jstat)**2, axis=0)**1.5)
    alphas = np.array([alpha/2, 1-alpha/2])
    zs = z0 + norm.ppf(alphas).reshape(alphas.shape+(1,)*z0.ndim)
    avals = norm.cdf(z0 + zs/(1-a*zs))
    nvals = np.round((samples-1)*avals).astype('int')
    if nvals.ndim == 1:
        # All nvals are the same. Simple broadcasting
        return ostat, stat[nvals]
    else:
        # Nvals are different for each data point. Not simple broadcasting.
        # Each set of nvals along axis 0 corresponds to the data at the same
        # point in other axes.
        return ostat, stat[(nvals, np.indices(nvals.shape)[1:].squeeze())]

def stack_EBR(data, selection, rbins, rebinned = True, mean_method=np.mean, alpha=0.32, samples=10000):
    """Stack lensing E- and B-mode in radial bins and compute bootstrap errors.

    Lensing measurements are constructed from data following the description
    of get_EBR(). Errors are symmetrized confidence intervals.

    Args:
        data: the lensing data as pyfits HDU, from get_lensing_data()
        selection: a subset of the void sample
        rbins: radial bins in physical units of Mpc/h (rebinned==False) or in 
            void radius units r/R_v (rebinned=True)
        rebinned: whether rescaling with respect to the void radius should be
            applied.
        mean_method: the statistical method for the central value estimation.
            Typically, mean, median, or kappa_sigma()
        alpha: the percentile interval for the bootstrap errors
        samples: the number of bootstrap realizations

    Returns:
        E-mode, E-mode errors, B-mode, B-mode errors, mean radius in each bin
    """
    radius = data.field('radius')[selection]
    dsum = data.field('dsum')[selection]
    osum = data.field('osum')[selection]
    rsum = data.field('rsum')[selection]
    wsum = data.field('wsum')[selection]
    npair = data.field('npair')[selection]
    
    data_at_r = rebin_EBR(rbins, radius, dsum, osum, rsum, wsum, npair, rebinned=rebinned)
    bins = rbins.size - 1
    Emode = np.zeros(bins, dtype='float32')
    Bmode = np.zeros(bins, dtype='float32')
    e_Emode = np.zeros(bins, dtype='float32')
    e_Bmode = np.zeros(bins, dtype='float32')
    r_mean = np.zeros(bins, dtype='float32')
    for k in xrange(bins):
        Emode[k], CI = bootstrap_confidence_intervals(data_at_r["E"][k], mean_method=mean_method, alpha=alpha, samples=samples)
        e_Emode[k] = (CI[1]-CI[0])/2 # symmetrized error bars
        Bmode[k], CI = bootstrap_confidence_intervals(data_at_r["B"][k], mean_method=mean_method, alpha=alpha, samples=samples)
        e_Bmode[k] = (CI[1]-CI[0])/2
        r_mean[k] = data_at_r["R"][k].mean()
    return Emode, e_Emode, Bmode, e_Bmode, r_mean

def void_model(rbins=None, void_radius=None):
    """The void lensing model, following the Lavaux & Wandelt (2012) void shape.

    The radius 3D void shape of LW12 is given by 
    rho = rho_mean(0.13 + 0.70(r/R_v)**3)
    where R_v denotes the void radius. This profile is projected along the 
    line-of-sight to get the shear profile according to eqs. 1 - 2 in the paper.

    Args:
        rbins: if not None, compute void model in given list of bins
        void_radius: if not None, assume this void radius instead of rescaled
            voids.

    Returns:
        The binned model values (if rbins are provided)
    """
    # open model
    model = np.loadtxt('data/DSigma_LW12_PM.dat')
    if rbins is not None:
        # bin it like data
        from scipy.interpolate import interp1d
        model_val = np.zeros(len(rbins) - 1)
        i = 0
        if void_radius is None:
            y_inter = interp1d(model[:,0], model[:,1],copy=False, fill_value=0., bounds_error=False)
        else:
            y_inter = interp1d(model[:,0]*void_radius, model[:,1]*void_radius,copy=False, fill_value=0., bounds_error=False)
        for k in range(len(rbins)-1):
            if void_radius is None:
                mask = (model[:,0] >= rbins[k]) & (model[:,0] <= rbins[k+1])
                if sum(mask) >= 2 :
                    dr = rbins[k+1] - rbins[k]
                    x_inter = rbins[k] + dr*np.linspace(0,1,100)
                    model_val[i] = y_inter(x_inter).mean()
            else:
                mask = (model[:,0]*void_radius >= rbins[k]) & (model[:,0]*void_radius < rbins[k+1])
                dr = rbins[k+1] - rbins[k]
                x_inter = rbins[k] + dr*np.linspace(0,1,100)
                model_val[i] = y_inter(x_inter).mean()
            i += 1
        return model_val
    else:
        if void_radius is None:
            return (model[:,0], model[:,1])
        else:
            return (model[:,0]*void_radius, model[:,1]*void_radius)

def chi2(delta_sigma, e_sigma, model_val, N):
    """Inverse-variance weighted sum of squared residuals.

    Entries with e_sigma == 0 are ignored.

    To correct for the bias in the inversion of the covariance matrix,
    the factor (N-B-2)/(N-1) is applied, where B denotes the nuber of bins in
    the profile (see Hartlap et al. 2007 for details).
    
    Args:
        delta_sigma: array of measured lensing signal
        e_sigma: array of error (rms) of the measurements
        model_val: array of the values to compare with the data
        N: number of voids
    """
    has_data = e_sigma > 0
    B = sum(has_data)
    debias = (N-B-2.)/(N-1)
    return debias*(((delta_sigma[has_data] - model_val[has_data])**2)/e_sigma[has_data]**2).sum()

def deltaChi2(delta_sigma, e_sigma, model_val, N):
    """Difference of chi2 between model and null.

    Args:
        delta_sigma: measured lensing signal
        e_sigma: error (rms) of the measurements
        model_val: the values to compare with the data
        N: number of voids
    """
    model_zero = np.zeros_like(delta_sigma)
    chi2_model = chi2(delta_sigma, e_sigma, model_val, N)
    chi2_zero = chi2(delta_sigma, e_sigma, model_zero, N)
    return chi2_model - chi2_zero

def likelihood_ratio(delta_sigma, e_sigma, model_val, N):
    """Likelihood ratio for the model and the null.

    For gaussian likelihoods, the ratio is given by
    exp(-1/2 delta_chi**2)
    where delta_chi**2 is computed from deltaChi2().
    
    Args:
        delta_sigma: measured lensing signal
        e_sigma: error (rms) of the measurements
        model_val: the values to compare with the data
        N: number of voids
    """
    dchi2 = deltaChi2(delta_sigma, e_sigma, model_val, N)
    return np.exp(-(dchi2)/2)

def SNR(delta_sigma, e_sigma, model):
    """ Signal-to-noise ratio of the data given the model.

    Args:
        delta_sigma: measured lensing signal
        e_sigma: error (rms) of the measurements
        model_val: the values to compare with the data
    """
    return (delta_sigma * model).sum()/(model**2 * e_sigma**2).sum()**0.5

def get_void_catalog():
    """Get the void catalog.

    Returns:
        a data set with keywords {'ra', 'dec', 'redshift', 'radius', 'id'} for
        a total of 1031 voids
    """
    return np.genfromtxt('data/sky_positions_central.txt', dtype=None, names=True)

def get_LRG_sample(data):
    """Get the LRG sample from the set of all voids.

    The void catalog is a combination of the main and the LRG spectroscopic 
    sample from SDSS DR7. This function uses the ID column in the lensing
    catalog to determine which voids are in the LRG sample.

    Args:
        data: lensing catalog HDU, from get_lensing_data()
        
    Returns:
        a boolean mask (true for a LRG void) to be applied to the lensing data
     """
    ids = data.field('id')
    sel = np.zeros(len(ids), dtype="bool")

    voids = get_void_catalog()
    # LRG sample starts with entry 713 in sky_positions_central.txt
    lrg_ids = voids['id'][713:]
    for i in range(len(ids)):
        if ids[i] in lrg_ids:
            sel[i] = True
    return sel
    
