import pickle
import numpy as np
import pandas as pd
from copy import deepcopy
import matplotlib.pyplot as plt


np.random.seed( 100 )
data = pickle.load(open('data/eics-ideal.pickle', 'rb'))
eic = np.random.choice(data)

eic.to_mins()

#fig = plt.figure()
fig, ax = plt.subplots(nrows=1,ncols=3)
peak_it = np.argmax( eic.its() )
max_it = np.max( eic.its() )
rts = eic.rts()
ax[0].plot(rts[peak_it-30:peak_it+30],eic.its()[peak_it-30:peak_it+30], c='b')
eic.add_noise( 0.05*max_it )
ax[1].plot(rts[peak_it-30:peak_it+30],eic.its()[peak_it-30:peak_it+30], c='b')
ax[1].tick_params(left=False,labelleft=False)
eic.add_noise( 0.1*max_it )
ax[2].plot(rts[peak_it-30:peak_it+30],eic.its()[peak_it-30:peak_it+30], c='b')
ax[2].tick_params(left=False,labelleft=False)
ax[0].set_ylabel('intensity (counts)')
ax[1].set_xlabel('retention time (minutes)')
ax[0].set_title('0% noise')
ax[1].set_title('5% noise')
ax[2].set_title('10% noise')
plt.show()
