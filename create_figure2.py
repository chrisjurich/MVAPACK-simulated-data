import pickle
import numpy as np
import pandas as pd
from copy import deepcopy
import matplotlib.pyplot as plt
from sklearn.neighbors import KernelDensity

data = pickle.load(open('data/eics-ideal.pickle', 'rb'))
def f(x, y):
    return np.sin(np.sqrt(x ** 2 + y ** 2))

x = np.linspace(-6, 6, 30)
y = np.linspace(-6, 6, 30)

fig = plt.figure()
#ax = plt.axes(projection='3d')
TARGET_MZ=218.1286854139034
mz_bins = np.zeros((1500))
rt_bins = np.zeros((3000))

np.random.seed( 100 )
np.random.shuffle( data )

max_it, max_mz = 0,0
TARGET_EIC=None

for d in data[0:250]:
	if d.max_it() > 7.5*(10**7):
		continue
	if np.isclose(d.mzs()[0],TARGET_MZ):
		#print('here')
		TARGET_EIC = d
	#d.remove_baseline(0.075)
	d.add_noise( 500000 )
	mzs, rts, its = d.mzs(), d.rts(), d.its()
		
	if np.max(mzs) > 600:
		continue	
	#if np.max(rts)-np.min(rts) > 100:
	#	continue
	rts /= 60
	#if np.isclose(d.mzs()[0],TARGET_MZ):
	#	ax.plot( mzs, rts,d.its(), c='r')
	#else:
	#	ax.plot( mzs, rts,d.its(), c='b')
#fig = plt.figure()
#ax = plt.axes()
print(max_mz)
max_mz = TARGET_EIC.mzs()[0]
print(TARGET_EIC.__dict__)
df = pd.DataFrame()
df['rt'] = TARGET_EIC.rts()
df['mz'] = TARGET_EIC.mzs()
df['cts'] = TARGET_EIC.its()
df.to_csv('demo-eic.csv',index=False)
#print( df )
#exit( 0 )
#for d in data[0:250]:
#	if np.isclose(d.mzs()[0],max_mz):
#d = TARGET_EIC
plt.plot(TARGET_EIC.rts(),TARGET_EIC.its(),linewidth=3,c='brown')
plt.axis('off')
plt.xlim(5,15)
#plt.xlabel('retention time (minutes)')
#plt.ylabel('intensity (counts)')
#plt.title(f'extracted ion chromatogram, mz=[{max_mz-0.01:.3f},{max_mz+0.01:.3f}]')
#plt.show()
#break
#exit( 0 )
	#print( d.__dict__ )
#ax.contour3D(X, Y, Z, 50, cmap='binary')
#min_it,max_it = ax.get_zlim()
#mz_min, mz_max = ax.get_xlim()
#min_rt, max_rt = ax.get_ylim()
## add the mzs
#mz_min = int(mz_min)
#mz_max = int(mz_max+0.5)
#new_mz = np.arange(mz_min,600,0.5)
#new_its = list(map(lambda nm: mz_bins[int(nm/0.5)], new_mz ))
#new_its = np.array( new_its )
#new_its /= np.max( new_its )
#new_its *= max_it
#new_rts = [25]*len(new_mz)
#ax.plot( new_mz, new_rts, new_its, c='k' )
## add the mts
#min_rt = int(min_rt)
#max_rt = int(max_rt+0.5)
#new_rt = np.arange(min_rt,25,0.1)
#new_its = list(map(lambda nr: rt_bins[int(nr/0.1)], new_rt))
#new_mz = [600]*len(new_its)
#new_its /= np.max( new_its )
#new_its *= max_it
#ax.plot( new_mz, new_rt, new_its, c='g' )

ax.set_ylim(0,25)
ax.set_xlabel('mz ratio (daltons)')
ax.set_ylabel('retention time (minutes)')
ax.set_zlabel('intensity (counts)')
plt.gca().invert_yaxis()
ax.set_xlim(600,100)
plt.show()
