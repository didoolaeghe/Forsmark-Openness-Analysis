# -*- coding: utf-8 -*-

#Diane Doolaeghe: 30/05/2022


import matplotlib as mpl
import mplstereonet
import matplotlib.pyplot as plt
import numpy as np
import math

import utils_cbar


def plot_stereonet1(dip, strike, alpha = [], title = '', cbmin = 0, cbmax = None, cbnum = 30, show_poles = False):
    '''plot stereonet with python package mplstereonet.
    The function returns the figure object.
    
    dip : numpy array of fracture dips
    strikes : numpy array of same fracture strikes
    alpha : numpy array of alpha angle of fractures, i.e. the angle between the fracture and the borehole
    cbmin and cbmax : colorbar bounds
    cbnum : number of colors in colorbar
    show_poles : choose to display the pole as black dots
    '''
    if alpha != []:
        nan_ind = np.argwhere(np.isnan(dip))
        alpha = np.delete(alpha, nan_ind)
        dip = np.delete(dip, nan_ind)
        strike = np.delete(strike, nan_ind)
        uncertainty = 5   #uncertainty angle in degrees
        max_f = 1/math.sin(uncertainty*math.pi/180)
        factor = 1/(np.sin(alpha*math.pi/180)) #factor for P32 computation
        factor = np.minimum(factor, np.ones(factor.shape)*max_f)
        factor = factor.tolist()
    else : 
        factor = None

    fig, ax = mplstereonet.subplots()    

    if cbmax == None:
        cax = ax.density_contourf(strike, dip, measurement='poles', weights = factor, method='schmidt',gridsize=25, cmap='jet', levels = cbnum)
    else:
        bounds = np.linspace(cbmin,cbmax,cbnum)
        cax = ax.density_contourf(strike, dip, measurement='poles', weights = factor, method='schmidt',gridsize=25, cmap='jet', levels = bounds, extend = 'both')

    cb = fig.colorbar(cax) #must be called before ax.grid(kind='polar')
    cb.set_label(label='$pdf(theta,\phi)$', size='20')
    cb.ax.tick_params(labelsize=15)

    # ax.set_azimuth_ticks([0,90,180,270])
    # ax.set_azimuth_ticklabels(['0', '90', '180', '270']) #does not work, shifted relative to the plot

    if show_poles == True:
        ax.pole(strike, dip, 'k+', markersize=1, markeredgewidth=1)

    ax.grid(kind='polar') #=> To work, you need to install: pip install https://github.com/joferkington/mplstereonet/zipball/master  (developer version)

    ax.set_title(title)
    ax.set_azimuth_ticks([])
    #ax.set_xlabel('Dip')
    return fig


def plot_stereonet2(dip, strike, alpha = [], title = '', res_dip = 10, res_strike = 35, cbmin = 0, cbmax = None, cbnum = 30, show_poles = False):
    '''home-made stereonet, not using the package mpl stereonets.
    The function returns the figure object.
    
    dip : numpy array of fracture dips
    strikes : numpy array of same fracture strikes
    alpha : numpy array of alpha angle of fractures, i.e. the angle between the fracture and the borehole
    res_dip : dip binning
    res_strike : strike binning
    cbmin and cbmax : colorbar bounds
    cbnum : number of colors in colorbar
    show_poles : choose to display the pole as black dots
    '''
    if alpha != []:
        nan_ind = np.argwhere(np.isnan(dip))
        alpha = np.delete(alpha, nan_ind)
        dip = np.delete(dip, nan_ind)
        strike = np.delete(strike, nan_ind)
        uncertainty = 5   #uncertainty angle in degrees
        max_f = 1/math.sin(uncertainty*math.pi/180)
        factor = 1/(np.sin(alpha*math.pi/180)) #factor for P32 computation
        factor = np.minimum(factor, np.ones(factor.shape)*max_f)
        factor = factor.tolist()
    else : 
        factor = None

    #convert strike in poles 
    dipd = (strike + 270) % 360

    #prepare bins and vectors
    dip_vector = np.linspace(0,90,res_dip)
    dip_bins = np.append(0, (dip_vector[:-1] + dip_vector[1:])/2)
    dip_bins = np.append(dip_bins, dip_vector[-1])
    dipd_vector = np.linspace(0,360,res_strike)
    dipd_bins = np.append(0, (dipd_vector[:-1] + dipd_vector[1:])/2)
    dipd_bins = np.append(dipd_bins, dipd_vector[-1])

    #sort fracture in bins, H is the fracture number in each bin
    H, dip_bins, dipd_bins = np.histogram2d(dip, dipd, bins=(dip_bins, dipd_bins), weights = factor)

    #compute bin area
    A = np.empty(H.shape)
    for i in range(len(dip_bins[:-1])):
        for j in range(len(dipd_bins[:-1])):
            A[i,j] = (np.cos(dip_bins[i]*math.pi/180)-np.cos(dip_bins[i+1]*math.pi/180))*(dipd_bins[j+1]-dipd_bins[j])*math.pi/180   #area sphere element = R²*(cos(dip1)-cos(dip2))*(dipd2-dipd1)

    #matrix formats
    H_norm = H/np.sum(H) #normalize by the tot number 
    density_m = H_norm/A #divide by bin area
    dipd_m, dip_m= np.meshgrid(dipd_vector, dip_vector) #mesh on the polar grid
      
    #create figure
    fig, ax = plt.subplots(subplot_kw=dict(projection='polar'))
    fig.suptitle(title)
    dip_m = dip_m*math.pi/180
    c_m = 2*(np.sin(dip_m/2)) # we compute the sphere string (corde) corresponding to the dip (see Lambert azimuthal equal-areal projection definition, or Terzaghi 1965)
    dipd_m = dipd_m*math.pi/180
    
    # plot
    if cbmax == None:
        cax=ax.contourf(dipd_m, c_m, density_m, levels = cbnum, cmap='jet')
    else:
        bounds = np.linspace(cbmin,cbmax,cbnum)
        cax=ax.contourf(dipd_m, c_m, density_m, levels = bounds, cmap='jet', extend = 'both')

    if show_poles == True:
        dip = dip*math.pi/180
        dipd =dipd*math.pi/180
        c_ = 2*(np.sin(dip/2))
        ax.scatter(dipd, c_, c = 'k', marker = 'x', s = 0.01)

    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    ax.set_rorigin(0)
    ax.tick_params(axis="x", labelsize=14)
    dip_label = np.array([20, 40, 60, 80])
    c_ticks = 2*(np.sin(dip_label*math.pi/(2*180)))
    ax.set_rticks(c_ticks.tolist())
    dip_label_string = ['20', '40', '60', '80']
    ax.set_yticklabels(dip_label_string, fontsize = 14)
    c_max = 2*np.sin(90*math.pi/(180*2))
    ax.set_ylim(0,c_max)
    colorbar = fig.colorbar(cax)
    colorbar.ax.tick_params(labelsize=15) 
    return fig


def plot_stereonet_fraction(dip1, strike1, dip2, strike2, alpha1 = [], alpha2 = [], title ='', res_dip = 10, res_strike = 35, dev_from_mean = False, cbmin = 0, cbmax = None, cbnum = 30):
    '''home made function to represent ratio of fracture numbers f = (dip1,strike1)/(dip2, strike2) as a function of fracture orientation in polar projection
    if dev_from_mean is true, we represent f-<f>/<f>
    dip : numpy array of fracture dips
    strikes : numpy array of same fracture strikes
    alpha : numpy array of alpha angle of fractures, i.e. the angle between the fracture and the borehole
    res_dip : dip binning
    res_strike : strike binning
    dev_from_mean : 
    cbmin and cbmax : colorbar bounds
    cbnum : number of colors in colorbar
    show_poles : choose to display the pole as black dots
    '''
    if (alpha1 != []) and (alpha2 != []):
        nan_ind1 = np.argwhere(np.isnan(dip1))
        nan_ind2 = np.argwhere(np.isnan(dip2))
        alpha1 = np.delete(alpha1, nan_ind1)
        alpha2 = np.delete(alpha2, nan_ind2)
        dip1 = np.delete(dip1, nan_ind1)
        dip2 = np.delete(dip2, nan_ind2)
        strike1 = np.delete(strike1, nan_ind1)
        strike2 = np.delete(strike2, nan_ind2)
        uncertainty = 5   #uncertainty angle in degrees
        max_f = 1/math.sin(uncertainty*math.pi/180)
        factor1 = 1/(np.sin(alpha1*math.pi/180)) #factor for P32 computation
        factor2 = 1/(np.sin(alpha2*math.pi/180)) #factor for P32 computation
        factor1 = np.minimum(factor1, np.ones(factor1.shape)*max_f)
        factor2 = np.minimum(factor2, np.ones(factor2.shape)*max_f)

        fop = np.sum(factor1)/np.sum(factor2) 

        factor1 = factor1.tolist()
        factor2 = factor2.tolist()
    else: 
        fop = dip1.shape[0]/dip2.shape[0]
        factor1 = None
        factor2 = None

    #convert strike in lower hemisphere poles
    dipd1 = (strike1 + 270) % 360 #warning : this is not the dip direction but its opposed direction, because of the 'lower hemisphere'
    dipd2 = (strike2 + 270) % 360

    #prepare bins and vectors
    dip_vector = np.linspace(0,90,res_dip)
    dip_bins = np.append(0, (dip_vector[:-1] + dip_vector[1:])/2)
    dip_bins = np.append(dip_bins, dip_vector[-1])

    dipd_vector = np.linspace(0,360,res_strike)
    dipd_bins = np.append(0, (dipd_vector[:-1] + dipd_vector[1:])/2)
    dipd_bins = np.append(dipd_bins, dipd_vector[-1])

    #sort fracture in bins
    H1, dip_bins, dipd_bins = np.histogram2d(dip1, dipd1, bins=(dip_bins, dipd_bins), weights = factor1)
    H2, dip_bins, dipd_bins = np.histogram2d(dip2, dipd2, bins=(dip_bins, dipd_bins), weights = factor2)

    # matrix formats
    dipd_m, dip_m= np.meshgrid(dipd_vector, dip_vector) 
    frac_m = H1/H2

    #create figure
    fig, ax = plt.subplots(subplot_kw=dict(projection='polar'))
    fig.suptitle(title)
    dip_m = dip_m*math.pi/180
    c_m = 2*(np.sin(dip_m/2)) # we compute the sphere string (corde) corresponding to the dip (see Lambert azimuthal equal-areal projection definition)
    dipd_m = dipd_m*math.pi/180

    if dev_from_mean == False:
        if cbmax == None:
            cax=ax.contourf(dipd_m, c_m, frac_m, levels = cbnum, cmap='jet')
        else:
            bounds = np.linspace(cbmin,cbmax,cbnum)
            cax=ax.contourf(dipd_m, c_m, frac_m, levels = bounds, cmap='jet', extend = 'both')
    else:
        frac_m = (frac_m-fop)/fop
        if cbmax == None:
            cax = ax.contourf(dipd_m, c_m, frac_m, levels = cbnum, norm=utils_cbar.MidpointNormalize(midpoint=0), cmap='coolwarm')
        else:
            bounds = np.linspace(cbmin,cbmax,cbnum)
            cax = ax.contourf(dipd_m, c_m, frac_m, levels = bounds, norm=utils_cbar.MidpointNormalize(midpoint=0), cmap='coolwarm')

    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    ax.set_rorigin(0)
    ax.tick_params(axis="x", labelsize=16)
    dip_label = np.array([20, 40, 60, 80])
    c_ticks = 2*(np.sin(dip_label*math.pi/(2*180)))
    ax.set_rticks(c_ticks.tolist())
    dip_label_string = ['20', '40', '60', '80']
    ax.set_yticklabels(dip_label_string, fontsize = 16)
    # ax.set_yticklabels([])
    # ax.set_xticklabels([])
    c_max = 2*np.sin(90*math.pi/(180*2))
    ax.set_ylim(0,c_max)
    colorbar = fig.colorbar(cax)
    colorbar.ax.tick_params(labelsize=15) 
    colorbar.set_label(label='$f_{op}$', size='20')
    return fig