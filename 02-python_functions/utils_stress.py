# -*- coding: utf-8 -*-

# Diane Doolaeghe: 03/05/2022

import numpy as np
from math import *
import sys




def stress_tensor(sigma_H,trend_H,plunge_H, sigma_h,trend_h,plunge_h, sigma_V,trend_V,plunge_V):
    '''calculate the stress tensor in x,y,z directions from the three principal components
    '''

    #principal components tensor:
    Tp = np.array([[sigma_H,0,0],[0,sigma_h,0],[0,0,sigma_V]])
    # x,y,z direction for each components
    dir_H = np.array([np.cos(plunge_H*pi/180)*np.sin(trend_H*pi/180),np.cos(plunge_H*pi/180)*np.cos(trend_H*pi/180),-np.sin(plunge_H*pi/180)])
    dir_h = np.array([np.cos(plunge_h*pi/180)*np.sin(trend_h*pi/180),np.cos(plunge_h*pi/180)*np.cos(trend_h*pi/180),-np.sin(plunge_h*pi/180)])
    #vertical stress must be pointing downward so that axis are orthonormal, we put a minus.
    dir_V = -np.array([np.cos(plunge_V*pi/180)*np.sin(trend_V*pi/180),np.cos(plunge_V*pi/180)*np.cos(trend_V*pi/180),-np.sin(plunge_V*pi/180)]) 
    #check orthogonality
    if (abs(np.dot(dir_H, dir_h) > 10**-14)):
        raise Exception("Your principal tensor is not orthogonal")
    if (abs(np.dot(dir_h, dir_V) > 10**-14)):
        print(abs(np.dot(dir_h, dir_V)))
        raise Exception("Your principal tensor is not orthogonal")
    if (abs(np.dot(dir_H, dir_V) > 10**-14)):
        raise Exception("Your principal tensor is not orthogonal")
    # normalize directions
    dir_H = dir_H/np.linalg.norm(dir_H)
    dir_h = dir_h/np.linalg.norm(dir_h)
    dir_V = dir_V/np.linalg.norm(dir_V)
    #rotation matrix
    R_mat = np.transpose(np.array([dir_H,dir_h,dir_V]))
    #stress tensor in x,y,z directions
    T = np.dot(np.dot(R_mat,Tp),np.transpose(R_mat))  #R_mat.Tp.R_mat_T
    return T


def get_fracture_normale(dip,dipd):
    dip = dip*pi/180
    dipd = dipd*pi/180
    x = np.sin(dip)*np.sin(dipd) 
    y = np.sin(dip)*np.cos(dipd) 
    z = np.cos(dip)
    normale = np.array([x,y,z])
    return normale


