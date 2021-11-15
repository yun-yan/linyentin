#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 29 00:00:56 2021

@author: yentinglin
"""

import numpy as np
from numba import njit#,float64,complex128
from numpy.linalg import inv
import cmath


@njit()
def sigma_retarded_bath(w,t1,t2,v,mu):

    delta=1e-9

    d=np.sqrt( np.power(v,2.0)+np.power((t1-t2),2.0) ) 
    dd=np.sqrt( np.power(v,2.0)+np.power((t1+t2),2.0) ) 
    sigma_bath=0.0+1.0j*0.0
    
    z=w-mu+1.0j*delta
    sigma_bath=(np.power(z,2.0)-np.power(v,2.0)-np.power(t2,2.0)+np.power(t1,2.0)-cmath.sqrt(z-d)*cmath.sqrt(z+d)*cmath.sqrt(z-dd)*cmath.sqrt(z+dd)   )  /(2.0*(z-v))


    return sigma_bath

@njit()
def boundary_g_retarded_bath(w,t1,t2,v,mu):

    delta=1e-9

    d=np.sqrt( np.power(v,2.0)+np.power((t1-t2),2.0) ) 
    dd=np.sqrt( np.power(v,2.0)+np.power((t1+t2),2.0) ) 
    sigma_bath=0.0+1.0j*0.0
    
    z=w-mu+1.0j*delta
    sigma_bath=(np.power(z,2.0)-np.power(v,2.0)-np.power(t1,2.0)+np.power(t2,2.0)-cmath.sqrt(z-d)*cmath.sqrt(z+d)*cmath.sqrt(z-dd)*cmath.sqrt(z+dd)   )  /(2.0*(z-v))


    return sigma_bath/(t2*t2)


@njit()
def g_retardation(w,beta,e,t_l,t_r,
                  t1_l,t2_l,v_l,mu_l,
                  t1_r,t2_r,v_r,mu_r):
                  
    delta=1e-10
                 
                 
    sigma_bath_l=sigma_retarded_bath(w,t1_l,t2_l,v_l,mu_l)
    sigma_bath_r=sigma_retarded_bath(w,t1_r,t2_r,v_r,mu_r)

    inv_g=np.zeros((3,3),dtype=np.complex128) 
    #print(mu_l,mu_r)
    inv_g[0,0]=w+1.0j*delta-sigma_bath_l-mu_l-v_l
    inv_g[1,1]=w+1.0j*delta-e
    inv_g[2,2]=w+1.0j*delta-sigma_bath_r-mu_r-v_r
    
    #inv_g[0,0]=0.0#t_l
    inv_g[0,1]=t_l
    inv_g[1,0]=t_l
    inv_g[1,2]=t_r
    inv_g[2,1]=t_r
    #inv_g[2,2]=0.0#t_r

    g=inv(inv_g)

    return g





def current_integrad(w,e,beta,t_l,t_r,
                     t1_l,t2_l,v_l,mu_l,
                     t1_r,t2_r,v_r,mu_r):
    sigma_l=boundary_g_retarded_bath(w,t2_l,t1_l,-v_l,mu_l)#*np.power(t1_l,2.0)
    sigma_r=boundary_g_retarded_bath(w,t2_r,t1_r,-v_r,mu_r)#*np.power(t1_r,2.0)
    
    g_ret=g_retardation(w,beta,e,t_l,t_r,
                        t1_l,t2_l,v_l,mu_l,
                        t1_r,t2_r,v_r,mu_r)

    gamma_l=2.0*np.pi*np.power(t1_l,2.0)* (-sigma_l.imag)/np.pi
    gamma_r=2.0*np.pi*np.power(t1_r,2.0)* (-sigma_r.imag)/np.pi
    
    gamma=gamma_l*g_ret[0,2]*gamma_r*np.conj(g_ret[0,2])
    
    f_r=1.0/(np.exp( beta*(w-mu_r))+1.0)
    f_l=1.0/(np.exp( beta*(w-mu_l))+1.0)
    #print('!!',gamma*(f_l-f_r))
    return np.real(gamma*(f_l-f_r))#.real

def non_interaction_current_list(NN,d,D):
    
    CURRET_LIST_TT=np.zeros(NN,dtype=np.float64)
    CURRET_LIST_TN=np.zeros(NN,dtype=np.float64)

    MU_LIST=np.zeros(NN,dtype=np.float64)
    
    for i in range(NN):


        
        #DT_R=d#*np.sin(GAMMA_R)/2.0
        T1_L=(D+d)/2.0
        T2_L=(D-d)/2.0
        V_R=0.0#d*np.cos(GAMMA_R)

        #DT_L=d#*np.sin(GAMMA_L)/2.0
        T1_R=D/2.0
        T2_R=D/2.0
        V_L=0.0#1e-10#d*np.cos(GAMMA_L)
        
        T_R=T_L=0.005
        E=0.75*d
        #print(E)
        MU=i*10*d/NN#+0.0001
        #MU=1.6*d+i*0.3*d/NN
        MU_LIST[i]=MU
        #print(MU)
        #print(MU/d)
        #U=0.1
        #T=T_R
        #T_k=np.power(T,(2.0-(4.0*U/np.pi)/(1.0+2.0*U/np.pi)  ))
        #print('T_k:',T_k)
        
        MU_R=-MU#/2.0
        MU_L=MU#/2.0
        
 
        W_I=-np.inf
        W_F=np.inf
        epsrel_acc=1.48e-11
        epsabs_acc=1.48e-11
        LIMIT=100000
        BETA=np.inf
        current_sol_real_TT=quad(current_integrad, W_I, W_F, args=(E,BETA,T_L,T_R,
                                                                   T1_L,T2_L,V_L,MU_L,
                                                                   T1_R,T2_R,V_R,MU_R), limit=LIMIT,epsabs=epsabs_acc, epsrel=epsrel_acc)
        current_sol_real_TN=quad(current_integrad, W_I, W_F, args=(E,BETA,T_L,T_R,
                                                                   T2_L,T1_L,V_L,MU_L,
                                                                   T2_R,T1_R,V_R,MU_R), limit=LIMIT,epsabs=epsabs_acc, epsrel=epsrel_acc)
         
                                                                     
        CURRET_LIST_TT[i]=current_sol_real_TT[0].real/(4.0*np.pi*T_L*T_L)                                             
        CURRET_LIST_TN[i]=current_sol_real_TN[0].real/(4.0*np.pi*T_L*T_L)

    return [MU_LIST,CURRET_LIST_TT,CURRET_LIST_TN]





