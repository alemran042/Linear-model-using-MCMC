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


### Load Data ####
##################
data = np.loadtxt('data.dat')
data = np.transpose(data)

y = np.array(data[0], dtype='float64') # Y
yerr = np.array(data[1], dtype='float64') # error of Y
x1 = np.array(data[2], dtype='float64')
x2 = np.array(data[4], dtype='float64')
x3 = np.array(data[6], dtype='float64')
x4 = np.array(data[8], dtype='float64')
x5 = np.array(data[10], dtype='float64')


x = x1, x2, x3, x4, x5


def model(theta, x1, x2, x3, x4, x5):
	''' Model defined'''
	a, b, c, d, e, f = theta
	y = a*x1 + b*x2 + c*x3 + d*x4 + e*x5 + f
	return y

def lnlike(theta, x, y, yerr):
	'''Fuction represents how good the model is'''
	LnLike = -0.5*np.sum(((y-model(theta, x1, x2, x3, x4, x5))/yerr)**2)
	return LnLike

def lnprior(theta):
	a, b, c, d, e, f = theta
	if 0. < a < 1. and 0. < b < 1.0 and 0. < c < 1. and 0. < d < 1. and 0. < e < 1. and 0. < f < max(y):
		return 0.0
	else:
		return -np.inf

def lnprob(theta, x, y, yerr):
	lp = lnprior(theta)
	if not np.isfinite(lp):
		return -np.inf
	return lp + lnlike(theta, x, y, yerr)

######## Set-up the initials ###########
########################################

data = (x, y, yerr) # Define data array
nwalkers = 240 #  of walker
niter = 5000  # of iteration
initial = np.array([.2, .4, .3, .2, .2, 5.]) # Set initial value
ndim = len(initial)
step = .1 ## Define step size Stp size
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

sampler, pos, prob, state = main(p0, nwalkers, niter, ndim, lnprob, data)
samples = sampler.flatchain

theta_max  = samples[np.argmax(sampler.flatlnprobability)]
best_fit_model = model(theta_max, x1, x2, x3, x4, x5)


print('###############################')
print('Theta max (parameters): ', np.around(theta_max, 5))
print('###############################')


####### Corner plot (median and 16-84% confidence) ###########
##############################################################

labels = [r'Cfm', r'Ce',r'Ccp',r'Crw', r'Ce (Outer)', r'Offset']
fig = corner.corner(samples, show_titles=True, labels=labels, plot_datapoints=True, quantiles=[0.16, 0.5, 0.84])
fig.savefig('Corner_plot_mcmc.png', dpi = 500)
plt.show()
#print(best_fit_model)
