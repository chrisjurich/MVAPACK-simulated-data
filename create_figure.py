import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KernelDensity
from mpl_toolkits.mplot3d import Axes3D

data = pickle.load(open('data/eics-ideal.pickle', 'rb'))

def f(x, y):
    return np.sin(np.sqrt(x ** 2 + y ** 2))

x = np.linspace(-6, 6, 30)
y = np.linspace(-6, 6, 30)

fig = plt.figure()

ax = fig.add_subplot(projection='3d')

mz_bins = np.zeros((1500))
rt_bins = np.zeros((3000))

for d in data[0:250]:
	if d.max_it() > 7.5*(10**7):
		continue
	d.remove_baseline(0.075)
	d.add_noise( 50000 )
	mzs, rts, its = d.mzs(), d.rts(), d.its()
	if np.max(mzs) > 600:
		continue	
	if np.max(rts)-np.min(rts) > 100:
		continue
	rts /= 60
	for m,r,it in zip(mzs,rts, its): 
		mz_bins[int(m/0.5)] += it
		rt_bins[int(r/0.1)] += it
	ax.plot( mzs, rts,d.its(), c='brown')
	#print( d.__dict__ )
#ax.contour3D(X, Y, Z, 50, cmap='binary')
min_it,max_it = ax.get_zlim()
mz_min, mz_max = ax.get_xlim()
min_rt, max_rt = ax.get_ylim()
# add the mzs
mz_min = int(mz_min)
mz_max = int(mz_max+0.5)
new_mz = np.arange(mz_min,600,0.5)
new_its = list(map(lambda nm: mz_bins[int(nm/0.5)], new_mz ))
new_its = np.array( new_its )
new_its /= np.max( new_its )
new_its *= max_it
new_rts = [25]*len(new_mz)
#ax.plot( new_mz, new_rts, new_its, c='k' )
# add the mts
min_rt = int(min_rt)
max_rt = int(max_rt+0.5)
new_rt = np.arange(min_rt,25,0.1)
new_its = list(map(lambda nr: rt_bins[int(nr/0.1)], new_rt))
new_mz = [600]*len(new_its)
new_its /= np.max( new_its )
new_its *= max_it
#ax.plot( new_mz, new_rt, new_its, c='g' )

ax.set_ylim(0,25)
#ax.set_xlabel('mz ratio (daltons)')
#ax.set_ylabel('retention time (minutes)')
#ax.set_zlabel('intensity (counts)')
#plt.gca().invert_yaxis()
#ax.xaxis.set_tick_params(labelbottom=False,tick1On=False,tick2On=False)
#ax.yaxis.set_tick_params(labelbottom=False,tick1On=False,tick2On=False)
#ax.zaxis.set_tick_params(labelbottom=False,tick1On=False,tick2On=False)
#ax.yaxis.set_tick_params(labelbottom=False,)
#ax.zaxis.set_tick_params(labelbottom=False,)
#ax.set_tick_params(axis='y',labelleft=False)
#ax.set_tick_params(axis='z',labelleft=False)
ax.set_xlim(600,100)



for axis in [ax.xaxis, ax.yaxis, ax.zaxis]:
    axis.set_ticklabels([])
    axis._axinfo['axisline']['linewidth'] = 1
    axis._axinfo['axisline']['color'] = (0, 0, 0)
    axis._axinfo['grid']['linewidth'] = 0.5
    axis._axinfo['grid']['linestyle'] = "-"
    axis._axinfo['grid']['color'] = (0, 0, 0)
    axis._axinfo['tick']['inward_factor'] = 0.0
    axis._axinfo['tick']['outward_factor'] = 0.0
    axis.set_pane_color((0.95, 0.95, 0.95))

plt.savefig('final_image.png',dpi=3000)

