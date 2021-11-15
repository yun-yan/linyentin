#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 26 00:09:53 2021

@author: yentinglin
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 25 08:53:38 2021

@author: yentinglin
"""


import numpy as np
from tool.tool_current_noninteracting import current_integrad
from scipy.integrate import quad
import matplotlib.pyplot as plt
from tool.tool_general import loglogder

#%%
def non_interaction_current_list(NN,E,d,D):
    
    CURRET_LIST_A=np.zeros(NN,dtype=np.float64)
    CURRET_LIST_B=np.zeros(NN,dtype=np.float64)
    CURRET_LIST_C=np.zeros(NN,dtype=np.float64)
    CURRET_LIST_D=np.zeros(NN,dtype=np.float64)

    MU_LIST=np.zeros(NN,dtype=np.float64)
    
    for i in range(NN):

        print(i)
        

        V_R=0.0#d*np.cos(GAMMA_R)
        V_L=0.0#1e-10#d*np.cos(GAMMA_L)
        
        T_R=T_L=0.0018
        GAMMA=8.0*T_R*T_R
        
        print(E)
        MU=99.0*GAMMA+i*0.07*d/NN
        #MU=99.0*GAMMA+i*7.0*GAMMA/NN#+0.0001
        
        MU_LIST[i]=MU
        #print(MU)
        #print(MU/d)
        #U=0.1
        #T=T_R
        #T_k=np.power(T,(2.0-(4.0*U/np.pi)/(1.0+2.0*U/np.pi)  ))
        #print('T_k:',T_k)
        T1_L_A=(D+d)/2.0
        T2_L_A=(D-d)/2.0
        T1_R_A=(D+d)/2.0
        T2_R_A=(D-d)/2.0
        
        T1_L_B=(D-d)/2.0
        T2_L_B=(D+d)/2.0
        T1_R_B=(D+d)/2.0
        T2_R_B=(D-d)/2.0
        
        T1_L_C=(D+d)/2.0
        T2_L_C=(D-d)/2.0
        T1_R_C=(D-d)/2.0
        T2_R_C=(D+d)/2.0        
 
        T1_L_D=(D-d)/2.0
        T2_L_D=(D+d)/2.0
        T1_R_D=(D-d)/2.0
        T2_R_D=(D+d)/2.0
        
        MU_R=-MU#/2.0
        MU_L=MU#/2.0
        
 
        W_I=-MU
        W_F=MU
        epsrel_acc=1.48e-12
        epsabs_acc=1.48e-12
        LIMIT=500000
        BETA=np.inf
        current_sol_a=quad(current_integrad, W_I, W_F, args=(E,BETA,T_L,T_R,
                                                                   T1_L_A,T2_L_A,V_L,MU_L,
                                                                   T1_R_A,T2_R_A,V_R,MU_R), limit=LIMIT,epsabs=epsabs_acc, epsrel=epsrel_acc)
        current_sol_b=quad(current_integrad, W_I, W_F, args=(E,BETA,T_L,T_R,
                                                                   T1_L_B,T2_L_B,V_L,MU_L,
                                                                   T1_R_B,T2_R_B,V_R,MU_R), limit=LIMIT,epsabs=epsabs_acc, epsrel=epsrel_acc)
         
        current_sol_c=quad(current_integrad, W_I, W_F, args=(E,BETA,T_L,T_R,
                                                                   T1_L_C,T2_L_C,V_L,MU_L,
                                                                   T1_R_C,T2_R_C,V_R,MU_R), limit=LIMIT,epsabs=epsabs_acc, epsrel=epsrel_acc)
         
        current_sol_d=quad(current_integrad, W_I, W_F, args=(E,BETA,T_L,T_R,
                                                                   T1_L_D,T2_L_D,V_L,MU_L,
                                                                   T1_R_D,T2_R_D,V_R,MU_R), limit=LIMIT,epsabs=epsabs_acc, epsrel=epsrel_acc)
         
                                                                     
        CURRET_LIST_A[i]=current_sol_a[0].real/GAMMA                                            
        CURRET_LIST_B[i]=current_sol_b[0].real/GAMMA
        CURRET_LIST_C[i]=current_sol_c[0].real/GAMMA
        CURRET_LIST_D[i]=current_sol_d[0].real/GAMMA

    return [MU_LIST,CURRET_LIST_A,CURRET_LIST_B,CURRET_LIST_C,CURRET_LIST_D]






#%% 

NN=80


D=1.0
T=0.0018
GAMMA=8.0*T*T
d=100.0*GAMMA#*D
#E=0.75*d
E1=-1.0*GAMMA
E2=4.0*GAMMA
     
a=non_interaction_current_list(NN,E1,d,D)
b=non_interaction_current_list(NN,E2,d,D)

U=0.001
DT=d/2.0
#%%
NINI_a=np.load('data_paper/current_NINI_d=0.002592_U=0.001_E=-2.592e-05.npy')
NITI_a=np.load('data_paper/current_NITI_d=0.002592_U=0.001_E=-2.592e-05.npy')
TINI_a=np.load('data_paper/current_TINI_d=0.002592_U=0.001_E=-2.592e-05.npy')
TITI_a=np.load('data_paper/current_TITI_d=0.002592_U=0.001_E=-2.592e-05.npy')



NINI_b=np.load('data_paper/current_NINI_d=0.002592_U=0.001_E=0.00010368.npy')
NITI_b=np.load('data_paper/current_NITI_d=0.002592_U=0.001_E=0.00010368.npy')
TINI_b=np.load('data_paper/current_TINI_d=0.002592_U=0.001_E=0.00010368.npy')
TITI_b=np.load('data_paper/current_TITI_d=0.002592_U=0.001_E=0.00010368.npy')


#%%


T=0.0018
N=40
D=1.0
T=0.0018
U_MAX=0.5
N=6
NN=40
G=8.0*T*T
d=100.0*G#*D

exponent_TITI_list=np.zeros(NN,dtype=np.float64)
exponent_NITI_list=np.zeros(NN,dtype=np.float64)
exponent_TINI_list=np.zeros(NN,dtype=np.float64)
exponent_NINI_list=np.zeros(NN,dtype=np.float64)

exponent_analytic_list=np.zeros(NN,dtype=np.float64)
u_list=np.zeros(NN,dtype=np.float64)

mu_list=np.zeros(N,dtype=np.float64)


    

#%%
fig,ax=plt.subplots()
#ax.set_xscale('log')
#ax.set_yscale('log')
plt.xticks(fontsize=13)
plt.yticks(fontsize=13)
#plt.xlim([99,106])
plt.xlim([99,106])
plt.ylim([0.0,0.1])
my_y_ticks = [0.0,0.05,0.1]
plt.yticks(my_y_ticks)
ax.set_xticklabels(['',100,'',102,'',104,'',106])
#my_x_ticks = [100,102,104,106]
#plt.xticks(my_x_ticks)


#my_x_ticks = [98,'' ,100,'' ,102, '',104]
#plt.xticks(my_x_ticks)
#plt.plot(D_LIST,scaling_d,'.-',color='blue',zorder=5)#   
#plt.plot(b[0]/d,b[1],'-',color='blue',label=r"$U=0.0$",zorder=5)#
#plt.plot(b[0]/d,b[2],'-',color='red',label=r"$U=0.0$ TN",zorder=5)#
#plt.plot(b[0]/d,b[2],'-',color='blue',label=r"$U=0.0$",zorder=5)#
factor=2.0*np.pi
plt.plot(a[0]/GAMMA,a[1]/factor,'-',color='tab:green',label=r"$NI+NI$",zorder=5)#
plt.plot(a[0]/GAMMA,a[2]/factor,'-',color='tab:red',label=r"$TI+NI$",zorder=5)#
plt.plot(a[0]/GAMMA,a[3]/factor,'-',color='tab:blue',label=r"$NI+TI$",zorder=5)#
plt.plot(a[0]/GAMMA,a[4]/factor,'--',color='k',label=r"$TI+TI$",zorder=5)#

plt.plot(NINI_a[1]/GAMMA,NINI_a[0]/factor,':',color='tab:green',zorder=5)#
plt.plot(TINI_a[1]/GAMMA,TINI_a[0]/factor,':',color='tab:red',zorder=5)#
plt.plot(NITI_a[1]/GAMMA,NITI_a[0]/factor,':',color='tab:blue',zorder=5)#
plt.plot(TITI_a[1]/GAMMA,TITI_a[0]/factor,':',color='k',zorder=5)#



plt.plot(b[0]/GAMMA,b[1]/factor,'-',color='tab:green',zorder=5)#
plt.plot(b[0]/GAMMA,b[2]/factor,'-',color='tab:red',zorder=5)#
plt.plot(b[0]/GAMMA,b[3]/factor,'-',color='tab:blue',zorder=5)#
plt.plot(b[0]/GAMMA,b[4]/factor,'--',color='k',zorder=5)#

plt.plot(NINI_b[1]/GAMMA,NINI_b[0]/factor,':',color='tab:green',zorder=5)#
plt.plot(TINI_b[1]/GAMMA,TINI_b[0]/factor,':',color='tab:red',zorder=5)#
plt.plot(NITI_b[1]/GAMMA,NITI_b[0]/factor,':',color='tab:blue',zorder=5)#
plt.plot(TITI_b[1]/GAMMA,TITI_b[0]/factor,':',color='k',zorder=5)#
#plt.plot(b[0]/d,b[2],'-',color='blue',label=r"$U=0.0$",zorder=5)#

plt.text(104.9, 0.1/factor, '$\epsilon=4\Gamma$',fontsize=16.5)
plt.text(99.3, 0.15/factor, '$\epsilon=-\Gamma$',fontsize=16.5)

plt.legend(loc='best',fontsize=12.5)
plt.xlabel(r'$V/(2\Gamma)$',fontsize=22)
plt.ylabel(r"$I/\Gamma$",fontsize=22)
plt.tight_layout()

#plt.savefig('plot/current_frg=%g.png'%(d),format='png',dpi=300)
#plt.savefig('plot/current_SSH_d=%g.png'%(d),format='png',dpi=300)

#plt.savefig('plot/IRLM_RGflow_tau_SSH_gap=0.0015.pdf',format='pdf',dpi=300)
plt.show()  

