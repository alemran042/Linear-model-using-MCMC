"""
############################################
# Program for estimating parameters from a multiple linear
# model using a Markov Chain Monte Carlo (MCMC) Technique.
# Prepared by A. Emran, NASA Jet Propulsion Laboratory.
# Email: al.emran@jpl.nasa.gov; Ph: +1 (334)-400-9371.
############################################
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import pandas as pd
np.set_printoptions(suppress=True)
np.seterr(divide='ignore', invalid='ignore')
import emcee
import corner
plt.rc('font', family='serif')
plt.rc('mathtext', fontset='cm')
from pysptools.abundance_maps.amaps import UCLS, NNLS, FCLS

### Load Data ####
##################
data = np.loadtxt('data.dat')
data = np.transpose(data)

y = np.array(data[0], dtype='float64') # Y
yerr = np.array(data[1], dtype='float64') # error of Y
x = np.array(data[[2, 4, 6, 8, 10]],  dtype='float64').T
N = x.shape[1]

def model(theta, x):
	''' Model defined'''
	k, f = np.array(theta[:N]), np.array(theta[N:])
	y = np.sum((k, x), axis = 1) + f
	return y

def lnlike(theta, x, y, yerr):
	'''Fuction represents how good the model is'''
	LnLike = -0.5*np.sum(((y-model(theta, x))/yerr)**2)
	return LnLike

def lnprior(theta):
	if all(0. < t < 1. for t in theta) and 0. < f < max(y):
		return 0.0
	return -np.inf

def lnprob(theta, x, y, yerr):
	lp = lnprior(theta)
	if not np.isfinite(lp):
		return -np.inf
	return lp + lnlike(theta, x, y, yerr)

######## Set-up the initials ###########
from scipy.optimize import nnls
result = nnls(x, y)
initial = np.concatenate([result[0], np.array([result[1]])])


nwalkers = 240 #  of walker
niter = 2000  # of iteration
ndim = len(initial)
step = 1e-5 ## Define step size Stp size
p0 = [np.array(initial) + step * np.random.randn(ndim) for i in range(nwalkers)] 

########## Run the MCMC ############
#####################################

def main(p0,nwalkers,niter,ndim,lnprob,data):
	sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=data)

	print("Running burn-in...")
	p0, _, _ = sampler.run_mcmc(p0, 100)
	sampler.reset()

	print("Running production...")
	pos, prob, state = sampler.run_mcmc(p0, niter)
	return sampler, pos, prob, state

############ Extract result ############
########################################

sampler, pos, prob, state = main(p0, nwalkers, niter, ndim, lnprob, (x, y, yerr))
samples = sampler.flatchain

theta_max  = samples[np.argmax(sampler.flatlnprobability)]
#best_fit_model = model(theta_max, x)


print('###############################')
print('Theta max (parameters): ', np.around(theta_max, 5))
print('###############################')


####### Corner plot (median and 16-84% confidence) ###########
##############################################################

labels = [r'Cfm', r'Ce',r'Ccp',r'Crw', r'Ce (Outer)', r'Offset']
fig = corner.corner(samples, show_titles=True, labels=labels, plot_datapoints=True, quantiles=[0.16, 0.5, 0.84])
fig.savefig('Corner_plot_mcmc_NEW.png', dpi = 500)
#plt.show()
#print(best_fit_model)
