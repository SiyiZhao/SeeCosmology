import numpy as np
from scipy.misc import derivative

import astropy.units as u
from colossus.cosmology import cosmology

import matplotlib.pyplot as plt
plt.rcParams['text.usetex'] = False

cosmo = cosmology.setCosmology('planck15')
h = cosmo.h
H0 = cosmo.H0 
OmM, OmL = cosmo.Om0, cosmo.Ode0
Mpc_per_h = u.def_unit('Mpc/h', u.Mpc / h)

def GrowthRate(z):
    a = 1/(1+z)
    
    def D(a):
        z = 1/a - 1
        return cosmo.growthFactor(z) 
    
    def logarithmic_derivative(a):
        # 使用 scipy 的 derivative 函数来计算导数
        dD_da = derivative(D, a, dx=1e-6)
        # 计算对数导数
        dlnD_dlna = dD_da * a / D(a)
        return dlnD_dlna    
    
    return logarithmic_derivative(a)

def f(z):
    return GrowthRate(z)
