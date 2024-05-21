import numpy as np
from scipy.misc import derivative

import astropy.units as u
from astropy import constants as const
from colossus.cosmology import cosmology

import matplotlib.pyplot as plt
plt.rcParams['text.usetex'] = False

cosmo = cosmology.setCosmology('planck15')
h = cosmo.h
H0 = cosmo.H0 * u.km/u.s/u.Mpc
OmM, OmL = cosmo.Om0, cosmo.Ode0
Mpc_per_h = u.def_unit('Mpc/h', u.Mpc / h)

def z2a(z):
    return 1/(1+z)
def a2z(a):
    return 1/a - 1

# Growth Rate f(z)=dlnD/dlna
def GrowthRate(z):
    a = 1/(1+z)
    
    def D(a):
        z = 1/a - 1
        return cosmo.growthFactor(z) 
    
    def logarithmic_derivative(a):
        dD_da = derivative(D, a, dx=1e-6)
        dlnD_dlna = dD_da * a / D(a)
        return dlnD_dlna    
    
    return logarithmic_derivative(a)

def f(z):
    return GrowthRate(z)

def x2s(x, vx, z):
    '''input x in cMpc/h, vx in physical peculiar km/s, output s in cMpc/h'''
    dist = vx *u.km/u.s * (1+z) / (cosmo.Hz(z)*u.km/u.s/u.Mpc)
    return x + dist.to(Mpc_per_h).value

def periodic_boundary(szs, boxL):
    szs = np.where(szs < 0, szs + boxL, szs)
    szs = np.where(szs > boxL, szs - boxL, szs)
    return szs