# -*- coding: utf-8 -*-
"""
Created on Tue Apr 13 13:48:25 2021

@author: BVH
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

plt.rcParams.update({'font.size': 28})

H = 18.1
J = 19.6

exps = np.array([1,10,30,60,90,120,150,300,600,800,1200])
log_exps = np.log(exps)

no_var_no_dead = np.array([0.197,0.585,0.731,0.832,0.874,0.899,0.911,0.937,0.953,0.962,0.963])
log_no_var_no_dead = np.log(no_var_no_dead)
var_dead = np.array([0.068,0.426,0.633,0.825,0.836,0.863,0.879,0.918,0.943,0.948,0.952])
log_var_dead = np.log(var_dead)
var_no_dead = np.array([0.179,0.608,0.747,0.822,0.882,0.893,0.907,0.935,0.953,0.956,0.959])
log_var_no_dead = np.log(var_no_dead)
no_var_dead = np.array([0.055,0.458,0.683,0.848,0.871,0.900,0.908,0.940,0.953,0.962,0.969])
log_no_var_dead = np.log(no_var_dead)

'''
plt.figure()
plt.plot(exps,no_var_no_dead,'r-')
plt.plot(exps,no_var_no_dead,'r^',label='0-sig var, 0% dead')

plt.plot(exps,var_dead,'b-')
plt.plot(exps,var_dead,'b^',label='3-sig var, 25% dead')

plt.plot(exps,var_no_dead,'g-')
plt.plot(exps,var_no_dead,'g^',label='3-sig var, 0% dead')

plt.plot(exps,no_var_dead,'m-')
plt.plot(exps,no_var_dead,'m^',label='0-sig var, 25% dead')

plt.legend(loc='best')
plt.xlabel('Exposure time / s')
plt.ylabel('R value')
'''

def func(x,a,b,c,d):
    return (a*np.log((b*x) + c)) + d

def straight_func(x,m,c):
    return m*x + c

#no_var_no_dead (NN)
NNpars,NNcov = curve_fit(func,exps,no_var_no_dead)
logNNpars,logNNcov = curve_fit(straight_func,log_exps,log_no_var_no_dead)
#var_dead (YY)
YYpars,YYcov = curve_fit(func,exps,var_dead)
logYYpars,logYYcov = curve_fit(straight_func,log_exps,log_var_dead)
#var_no_dead (YN)
YNpars,YNcov = curve_fit(func,exps,var_no_dead)
logYNpars,logYNcov = curve_fit(straight_func,log_exps,log_var_no_dead)
#no_var_dead (NY)
NYpars,NYcov = curve_fit(func,exps,no_var_dead)
logNYpars,logNYcov = curve_fit(straight_func,log_exps,log_no_var_dead)

x = np.linspace(1,1200,100000)

plt.figure()
plt.plot(x,func(x,*NNpars),'r-',label='0-sig var, 0% dead')
plt.plot(exps,no_var_no_dead,'r^')
plt.plot(x,func(x,*YYpars),'b-',label='3-sig var, 25% dead')
plt.plot(exps,var_dead,'b^')
plt.plot(x,func(x,*YNpars),'g-',label='3-sig var, 0% dead')
plt.plot(exps,var_no_dead,'g^')
plt.plot(x,func(x,*NYpars),'m-',label='0-sig var, 25% dead')
plt.plot(exps,no_var_dead,'m^')
plt.legend(loc='best')
plt.xlabel('Exposure time / s')
plt.ylabel('R value')


plt.figure()
plt.loglog(x,func(x,*NNpars),'r-',label='0-sig var, 0% dead')
plt.loglog(exps,no_var_no_dead,'r^')
plt.loglog(x,func(x,*YYpars),'b-',label='3-sig var, 25% dead')
plt.loglog(exps,var_dead,'b^')
plt.loglog(x,func(x,*YNpars),'g-',label='3-sig var, 0% dead')
plt.loglog(exps,var_no_dead,'g^')
plt.loglog(x,func(x,*NYpars),'m-',label='0-sig var, 25% dead')
plt.loglog(exps,no_var_dead,'m^')
plt.legend(loc='best')
plt.xlabel('Exposure time / s')
plt.ylabel('R value')


'''
plt.figure()
plt.plot(np.log(x),straight_func(np.log(x),*logNNpars),'r-',label='0-sig var, 0% dead')
plt.plot(log_exps,log_no_var_no_dead,'r^')
plt.plot(np.log(x),straight_func(np.log(x),*logYYpars),'b-',label='3-sig var, 25% dead')
plt.plot(log_exps,log_var_dead,'b^')
plt.plot(np.log(x),straight_func(np.log(x),*logYNpars),'g-',label='3-sig var, 0% dead')
plt.plot(log_exps,log_var_no_dead,'g^')
plt.plot(np.log(x),straight_func(np.log(x),*logNYpars),'m-',label='0-sig var, 25% dead')
plt.plot(log_exps,log_no_var_dead,'m^')
plt.legend(loc='best')
plt.xlabel('Exposure time / s')
plt.ylabel('R value')
'''














