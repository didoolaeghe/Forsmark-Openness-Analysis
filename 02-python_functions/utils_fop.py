# -*- coding: utf-8 -*-

# Diane Doolaeghe: 03/05/2022

# Functions to calcultate fop from a fracture dataset, with open and sealed informations



import sys, os, math, numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats



def fop(data, column, **kwargs):  
    '''calculate and return the open fraction fop in bins of a parameter specified by the 'column' variable
    '''
    nbin = kwargs.get('bin_number',100)# number of bins
    bin_min = kwargs.get('min', data[column].min())
    bin_max = kwargs.get('max', data[column].max())

    bins = np.linspace(bin_min, bin_max, num = nbin, endpoint = True)
    bins_inf = bins[:-1]
    bins_sup = bins[1:]


    fop32 = []
    N_bin = []
    N_bin_norm = []
    bin_mean =[]
    for i in range(len(bins_sup)):
        data_select = data.loc[(data[column] <= bins_sup[i]) & (data[column] > bins_inf[i])].copy()

        _ndata=data_select.shape[0]
        if _ndata>0:
            #angle correction
            uncertainty = 5  
            max_f = 1/math.sin(uncertainty*math.pi/180)
            factor = 1/(np.sin(data_select['ALPHA(degrees)']*math.pi/180))
            factor[factor > max_f] = max_f
            data_select['factor']=factor


            data_select_open = data_select.loc[data_select['FRACT_INTERPRET'].isin(['Open','Partly open'])]

            N_bin.append(_ndata)
            N_bin_norm.append(_ndata/(bins_sup[i]-bins_inf[i]))#/(bins_sup[i]-bins_inf[i]))#*data.shape[0])
            bin_mean.append(data_select[column].mean())

            _fop32=data_select_open['factor'].sum()/data_select['factor'].sum()
            fop32.append(_fop32)
        else:
            bin_mean.append((bins_sup[i]+bins_inf[i])/2)
            fop32.append(np.nan)
            N_bin.append(np.nan)
            N_bin_norm.append(np.nan)

    data_save = pd.DataFrame()
    data_save[column] = np.array(bin_mean)
    data_save['min'] = bins_inf
    data_save['max'] = bins_sup
    data_save['fop32'] = fop32
    data_save['N'] = N_bin
    data_save['N_norm'] = N_bin_norm

    return data_save

def fop2D(data, column1, column2,  **kwargs):  
    '''calculate and return the open fraction fop in 2D bins made from two parameter specified by the 'column1' and 'column2' variable
    '''
    nbin1 = kwargs.get('bin_number1',50)
    nbin2 = kwargs.get('bin_number2',50)
    bin_min1 = kwargs.get('min1', data[column1].min())
    bin_max1 = kwargs.get('max1', data[column1].max())
    bin_min2 = kwargs.get('min2', data[column2].min())
    bin_max2 = kwargs.get('max2', data[column2].max())

    #angle correction
    uncertainty = 5  
    max_f = 1/math.sin(uncertainty*math.pi/180)
    factor = 1/(np.sin(data['ALPHA(degrees)']*math.pi/180))
    factor[factor > max_f] = max_f
    data['factor']=factor

    bins1 = np.linspace(bin_min1, bin_max1, num = nbin1, endpoint = True)
    bins2 = np.linspace(bin_min2, bin_max2, num = nbin2, endpoint = True)

    data_open = data.loc[data['FRACT_INTERPRET'].isin(['Open','Partly open'])]

    h1, edges1, edges2 = np.histogram2d(data[column1], data[column2], bins=(bins1, bins2), weights = data['factor'])
    h2, edges1, edges2 = np.histogram2d(data_open[column1], data_open[column2], bins=(bins1, bins2), weights = data_open['factor'])
    h = h2 / h1

    return h, edges1, edges2