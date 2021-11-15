#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 29 17:14:49 2021

@author: yentinglin
"""


import numpy as np
from numpy import log
from numba import njit

@njit() 
def loglogder(x,y):
    """ loglog derivative. of y with respect to x

        Input Parameters:
        ----------

        x : list of floats
        y : list of floats

        Output:
        ----------
        x and loglog derivative. of y with respect to x
        """    
    l=len(x)-2
    logder_y=np.zeros(l,np.float64)
    xx      =np.zeros(l,np.float64)
    for i in range(l):
        logder_y[i] = (log(y[i+2])-log(y[i]) )/( log(x[i+2])-log(x[i])  )
        xx[i]       = x[i+1]
    return [xx,logder_y]
@njit() 
def linearlogder(x,y):
    """ linear-log derivative. of y with respect to x

        Input Parameters:
        ----------

        x : list of floats
        y : list of floats

        Output:
        ----------
        x and liear-log derivative of y with respect to x
    """   
    l=len(x)-1
    logder_y=np.zeros(l,np.float64)
    xx      =np.zeros(l,np.float64)
    
    for i in range(l):
        logder_y[i] = (y[i+1]-y[i] )/( log(x[i+1])-log(x[i])  )
        xx[i]       = x[i]
    return [xx,logder_y]

@njit() 
def loglinearder(x,y):
    """ log-linear derivative. of y with respect to x

        Input Parameters:
        ----------

        x : list of floats
        y : list of floats

        Output:
        ----------
        x and log-linear derivative of y with respect to x
    """   
    l=len(x)-1
    logder_y=np.zeros(l,np.float64)
    xx=np.zeros(l,np.float64)
    
    for i in range(l):
        logder_y[i] = (log(y[i+1])-log(y[i]) )/( x[i+1]-x[i]  )
        xx[i]       = x[i]
    return [xx,logder_y]


@njit() 
def dydx(x,y):
    """ Derivative of y with respect to x

        Input Parameters:
        ----------

        x : list of floats
        y : list of floats

        Output:
        ----------
        x and derivative of y with respect to x
    """   
    a=np.zeros(int(len(x))-1,np.float64) 
    b=np.zeros(int(len(x))-1,np.float64) 
    for i in range(int(len(x))-1):
        a[i]=(y[i+1]-y[i])/(x[i+1]-x[i])
        b[i]=x[i]
    return [b,a]