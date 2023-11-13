"""
Created 13. November 2023 by Daniel Van Opdenbosch, Technical University of Munich

This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version. It is distributed without any warranty or implied warranty of merchantability or fitness for a particular purpose. See the GNU general public license for more details: <http://www.gnu.org/licenses/>
"""

import glob
import lmfit as lm
import numpy as np
import os
import matplotlib as mpl
import matplotlib.pyplot as plt
import scipy
from scipy import stats

os.system('mv results.log results.alt')
log=open('results.log','a')

for f in glob.glob('*.csv'):
	filename=os.path.splitext(f)[0]
	x,y=np.genfromtxt(f,delimiter=',',unpack=True,skip_header=1,usecols=(0,2),encoding='iso-8859-1')
	# ~ args=np.where((x>100)&(x<800))
	# ~ x=x[args];y=y[args]
	# ~ y-=y[0]

	def basis(baseline):
		return baseline*np.ones(len(x))
	def fitfunc(x,height,center,width):
		# ~ return height/(width*(2*np.pi)**0.5)*np.exp(-1./2*((x-center)/width)**2)					#Gauss Dichte
		return -height/2*(1+scipy.special.erf((x-center)/(width*2**0.5)))								#Gauss Kumulativ
		# ~ return height*(width**2/((x-center)**2+width**2))											#Lorentz Dichte
		# ~ return height/(x*width*(2*np.pi)**0.5)*np.exp(-(np.log(x)-center)**2/(2*width**2))			#Log-normal

	Kriterium=1e-5	#Anpassen
	Konvergenz=Kriterium+1
	tau,i,params=0,0,lm.Parameters()
	params.add('baseline',0)
	while Konvergenz>Kriterium:
		i+=1
		for n in range(i):
			params.add('height'+str(n),1,min=0,max=2*max(abs(y)))
			params.add('center'+str(n),(max(x)+min(x))*(n+1)/(i+2))
			params.add('width'+str(n),1,min=0,max=2*max(abs(x)))
		def multi_fitfunc(params):
			prm=params.valuesdict()
			global func
			func=basis(prm['baseline'])
			for n in range(i):
				func+=fitfunc(x,prm['height'+str(n)],prm['center'+str(n)],prm['width'+str(n)])
			global res
			res=func-y
			return res
		result=lm.minimize(multi_fitfunc,params,method='least_squares')
		Konvergenz=np.var(res)/np.var(y)
		print(filename,'i:',i,'Konvergenz:',Konvergenz)

		####
		# ~ prm=result.params.valuesdict()
		# ~ result.params.pretty_print()
		# ~ plt.close('all')
		# ~ plt.plot(x,y)
		# ~ plt.plot(x,func)
		# ~ for n in range(i):
			# ~ plt.plot(x,fitfunc(x,prm['height'+str(n)],prm['center'+str(n)],prm['width'+str(n)]))
		# ~ plt.show()
		####

	log.write(filename+': '+str(result.params.valuesdict())+'\n')

	plt.close('all')
	prm=result.params.valuesdict()
	plt.plot(x,y)
	plt.plot(x,func,linestyle='dashed')
	for n in range(i):
		plt.plot(x,fitfunc(x,prm['height'+str(n)],prm['center'+str(n)],prm['width'+str(n)]))
	plt.savefig(filename+'.png',dpi=300)
