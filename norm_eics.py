import pickle
from collections import defaultdict

data = pickle.load(open('./data/eics-cpy2.pickle','rb'))

eic_holder = defaultdict( list )

for dd in data:
	eic_holder[ dd.name ].append( dd )

new_eics = []
for cname, eic_set in eic_holder.items():
	print([p.max_it() for p in eic_set])
	c_peak = max([p.max_it() for p in eic_set])
	for ee in eic_set:
		ee.its_ /= c_peak
		new_eics.append( ee )

	eic_holder[cname] = new_eics
	#print([p.max_it() for p in new_eics])
	#print(c_peak)
#	break
exit( 0 )
pickle.dump(new_eics, open('./data/eics-cpy2.pickle','wb'))
