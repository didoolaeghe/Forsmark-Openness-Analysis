# -*- coding: utf-8 -*-

#Diane Doolaeghe: 30/05/2022


import numpy as np
import math



def return_ystat_for_xbin(n_bin, x_vector, y_vector, N_data = False, xMin = None, xMax = None):
    ''' given a x vector and a corresponding y vector, this fonction returns the bin of x vector and
    the stats of the y vector in this bin (mean and standard dev)
    n_bin is the number of bin dividing x_vector
    '''
    #we remove nans from the dataset
    x_vector = x_vector[~np.isnan(y_vector)]
    y_vector = y_vector[~np.isnan(y_vector)]

    if xMin == None:
        theMin = min(x_vector)
    else:
        theMin = xMin
    if xMax == None:
        theMax = max(x_vector)
    else:
        theMax = xMax

    x_bins = np.linspace(theMin, theMax, n_bin)
    x_digitized = np.digitize(x_vector, x_bins)

    y_mean = [np.nanmean(y_vector[x_digitized == i]) for i in range(1, len(x_bins))]
    y_std = [np.nanstd(y_vector[x_digitized == i]) for i in range(1, len(x_bins))]
    x_bin_center = [x_vector[x_digitized == i].mean() for i in range(1, len(x_bins))]
    if N_data == False:
        return x_bin_center, y_mean, y_std
    else:
        N = [y_vector[x_digitized == i].shape[0] for i in range(1, len(x_bins))]
        bin_sizes = (x_bins[1:]-x_bins[:-1])
        N = N/bin_sizes
        return x_bin_center, y_mean, y_std, N


def divide(array_A, array_B, min_denominator = 0):
    '''divide array_A by array_B, but if some values in the denominator are < min_denominator,
    the result is a Nan.
    Application: When density values are very small, we can choose to put a Nan for fop.
    '''
    ratio = []
    if array_A.shape[0] != array_B.shape[0]:
        raise ValueError('Arrays dont have the same size')
    else:
        for i in range(array_A.shape[0]):
            if array_B[i]>min_denominator:
                ratio.append(array_A[i]/array_B[i])
            else:
                ratio.append(np.nan)
    return np.array(ratio)


def half_sphere_sampling(nstep):
    '''
    samples a half sphere with same surface bins
    '''
    dphi = 90./nstep
    ds = 2*math.pi*math.cos(dphi/180*math.pi)
    valdip = np.array([i for i in np.arange(0,90,dphi)])
    valdip[1:]=valdip[1:]+dphi/2
    rvalues=[]
    for v in valdip:
        if(abs(v)<1):
            dtheta_th=ds/(math.cos((v)*math.pi/180)-math.cos((v+dphi)*math.pi/180))
        else:
            dtheta_th=ds/(math.cos((v-dphi/2)*math.pi/180)-math.cos((v+dphi/2)*math.pi/180))
        dtheta=360/max(int(360/dtheta_th+0.5),1)
        valdipd = set([i for i in np.arange(0,360,dtheta)])
        if len(valdipd)==0:
            valdipd=set([0])
        for vd in valdipd:
            rvalues.append([v,vd])
    return np.array(rvalues)


def distribution_log_bin(dataset, N_bin = 100, bin_exp = 1, max_data = -1, min_data = -1, density = True):
    '''compute dataset distribution with a logarithmic binning
    Note : bin_exp != 1 doesn't work well'''
    dataset = dataset[~np.isnan(dataset)]
    if max_data == -1:
        max_data = np.max(dataset)
    if min_data == -1:
        min_data = np.min(dataset)
    max_data = max_data + max_data/100 #to include max values
    shift = 0
    if min_data == 0:
        shift = max_data/100
        min_data = shift
        max_data = max_data+shift
    #binning creation
    n = np.linspace(0,N_bin, N_bin+1)
    ind = n/N_bin
    if bin_exp == 1:
        bins=min_data*(max_data/min_data)**ind
    else:
        bins=(min_data**(-bin_exp)+(max_data**(-bin_exp)-min_data**(-bin_exp))*ind)**(-1/bin_exp) #revoir
    bins = bins - shift
    #plt.plot(bins)
    #distribution
    bin_min = bins[:-1]
    bin_max = bins[1:]
    N = np.zeros(len(bin_min))
    X = np.zeros(len(bin_min))
    for i in range(len(bin_min)):
        index = [(dataset>=bin_min[i]) & (dataset<bin_max[i])]
        N[i] = dataset[index].shape[0]
        if N[i] != 0:
            X[i] = np.mean(dataset[index])
        else:
            X[i] = (bin_min[i] + bin_max[i])/2
    if density == True:
        N = N/dataset.shape[0]
    N = N/(bin_max-bin_min)
    integrale = np.nansum((bin_max-bin_min)*N)
    print('integrale: '+str(integrale))
    return N, X



def distribution_reg_bin(dataset, N_bin = 100, max_data = -1, min_data = -1, density = True):
    '''compute dataset distribution with a regular binning''' 
    dataset = dataset[~np.isnan(dataset)]
    if max_data == -1:
        max_data = np.max(dataset)
    if min_data == -1:
        min_data = np.min(dataset)
    max_data = max_data + max_data/100 #to include max values
    #binning creation
    bins = np.linspace(min_data, max_data, N_bin+1)
    #distribution
    bin_min = bins[:-1]
    bin_max = bins[1:]
    N = np.zeros(len(bin_min))
    X = np.zeros(len(bin_min))
    for i in range(len(bin_min)):
        index = [(dataset>=bin_min[i]) & (dataset<bin_max[i])]
        N[i] = dataset[index].shape[0]
        if N[i] != 0:
            X[i] = np.mean(dataset[index])
        else:
            X[i] = (bin_min[i] + bin_max[i])/2
    if density == True:
        N = N/dataset.shape[0]
    N = N/(bin_max-bin_min)
    integrale = np.nansum((bin_max-bin_min)*N)
    print('integrale: '+str(integrale))
    return N, X
