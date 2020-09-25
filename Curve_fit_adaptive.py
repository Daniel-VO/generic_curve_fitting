"""
Created 29. April 2020 by Daniel Van Opdenbosch, Technical University of Munich

This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version. It is distributed without any warranty or implied warranty of merchantability or fitness for a particular purpose. See the GNU general public license for more details: <http://www.gnu.org/licenses/>
"""

import glob
import lmfit
import numpy
import os
import matplotlib as mpl
import matplotlib.pyplot as plt
import scipy

x=numpy.linspace(1,100,num=1000)
y=(numpy.sin(x/5/numpy.pi))**2+1*(5**2/((x-70)**2+5**2))

def basis(baseline):
	return baseline*numpy.ones(len(x))
def fitfunc(x,height,center,width):
	return height/(width*(2*numpy.pi)**0.5)*numpy.exp(-1./2*((x-center)/width)**2)	#Gauss Dichte
	# ~ return -height/2*(1+scipy.special.erf((x-center)/(width*2**0.5)))				#Gauss Kumulativ
	# ~ return height*(width**2/((x-center)**2+width**2))								#Lorentz Dichte

Fehler=0.9995	#Anpassen
Rquadrat,i,params=0,0,lmfit.Parameters()
params.add('baseline',0,min=0)
while Rquadrat<Fehler:
	i+=1
	for n in range(i):
		params.add('height'+str(n),(max(y)+min(y))/2,min=min(y))
		params.add('center'+str(n),(max(x)+min(x))/2,min=min(x),max=max(x))
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
	Rquadrat=1-numpy.var(res)/numpy.var(y)
	print('i:',i,'Rquadrat:',Rquadrat)
	prm=result.params.valuesdict()
	####
	plt.plot(x,y)
	plt.plot(x,func)
	for n in range(i):
		plt.plot(x,fitfunc(x,prm['height'+str(n)],prm['center'+str(n)],prm['width'+str(n)]))
	plt.show()
	####

result.params.pretty_print()
prm=result.params.valuesdict()
plt.plot(x,y)
plt.plot(x,func)
for n in range(i):
	plt.plot(x,fitfunc(x,prm['height'+str(n)],prm['center'+str(n)],prm['width'+str(n)]))
plt.show()
