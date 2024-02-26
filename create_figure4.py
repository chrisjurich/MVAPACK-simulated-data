import pickle
import matplotlib.pyplot as plt
from collections import defaultdict

data = pickle.load(open('./data/eics-ideal.pickle','rb'))

dd = defaultdict( list )


for d in data:
	dd[d.name].append( d )

for name, chems in dd.items():
	if len(chems) == 3:
		break

ax = plt.axes(projection='3d')
for cc in chems:
	mask = (cc.rts()<240)&(cc.rts()>180)
	ax.plot(cc.mzs()[mask],cc.rts()[mask]/60, cc.its()[mask],c='b')

ax.set_xlabel('mz ratio (da)')
ax.set_ylabel('retention time (mins)')
ax.set_zlabel('intensity')
ax.set_xlim(175,185)
ax.set_ylim(180/60,240/60)
ax.set_title(cc.name)
plt.show()
#print(chems)
#plt.plot(data[0].rts(),data[0].its())
#plt.show()

#print(data[0].__dict__)
