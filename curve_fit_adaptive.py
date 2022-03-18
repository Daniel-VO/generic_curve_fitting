"""
Created 18. March 2022 by Daniel Van Opdenbosch, Technical University of Munich

This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version. It is distributed without any warranty or implied warranty of merchantability or fitness for a particular purpose. See the GNU general public license for more details: <http://www.gnu.org/licenses/>
"""

import glob
import lmfit
import numpy
import os
import matplotlib as mpl
import matplotlib.pyplot as plt
import scipy
from scipy import stats

# ~ x=numpy.linspace(1,100,num=1000)
# ~ y=(numpy.sin(x/5/numpy.pi))**2+1*(5**2/((x-70)**2+5**2))

x,y=numpy.genfromtxt('PTMOS_FTIR.txt',unpack=True)
args=numpy.where((x>600)&(x<900))
x=x[args]
y=y[args]

def basis(baseline):
	return baseline*numpy.ones(len(x))
def fitfunc(x,height,center,width):
	# ~ return height/(width*(2*numpy.pi)**0.5)*numpy.exp(-1./2*((x-center)/width)**2)					#Gauss Dichte
	# ~ return -height/2*(1+scipy.special.erf((x-center)/(width*2**0.5)))								#Gauss Kumulativ
	return height*(width**2/((x-center)**2+width**2))												#Lorentz Dichte
	# ~ return height/(x*width*(2*numpy.pi)**0.5)*numpy.exp(-(numpy.log(x)-center)**2/(2*width**2))		#Log-normal

Fehler=0.95	#Anpassen
tau,i,params=0,0,lmfit.Parameters()
params.add('baseline',0)
while tau<Fehler or numpy.isnan(tau):
	i+=1
	for n in range(i):
		params.add('height'+str(n),1,min=0)
		params.add('center'+str(n),(max(x)+min(x))*(n+1)/(i+2),min=min(x),max=max(x))
		params.add('width'+str(n),1,min=0)
	def multi_fitfunc(params):
		prm=params.valuesdict()
		global func
		func=basis(prm['baseline'])
		for n in range(i):
			func+=fitfunc(x,prm['height'+str(n)],prm['center'+str(n)],prm['width'+str(n)])
		global res
		res=func-y
		return res
	result=lmfit.minimize(multi_fitfunc,params,method='least_squares')
	tau=stats.kendalltau(func,y)[0]
	print('i:',i,'tau:',tau)
	prm=result.params.valuesdict()
	####
	# ~ result.params.pretty_print()
	# ~ plt.plot(x,y)
	# ~ plt.plot(x,func)
	# ~ for n in range(i):
		# ~ plt.plot(x,fitfunc(x,prm['height'+str(n)],prm['center'+str(n)],prm['width'+str(n)]))
	# ~ plt.show()
	####

result.params.pretty_print()
prm=result.params.valuesdict()
plt.plot(x,y)
plt.plot(x,func)
for n in range(i):
	plt.plot(x,fitfunc(x,prm['height'+str(n)],prm['center'+str(n)],prm['width'+str(n)]))
plt.show()
