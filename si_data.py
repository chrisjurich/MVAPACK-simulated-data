import pickle
import numpy as np
from utils import *



eics = pickle.load(open("data/eics-cpy2.pickle", "rb"))

mzs  = list()

features = list()

for ee in eics:
    features.append(( ee.rts_[np.argmax(ee.its_)], ee.mzs()[0] ))


features = sorted(features, key=lambda pr: pr[-1])

diffs = list()
for fidx, ff in enumerate( features ):
    
    left_idx, right_idx = fidx-1, fidx+1

    while left_idx >= 0 and abs(ff[0]-features[left_idx][0]) < 60:
        left_idx -= 1
    
    while right_idx < len(features) and abs(ff[0]-features[right_idx][0]) < 60:
        right_idx += 1

    
    differences = list()
    for idx in range(left_idx, right_idx+1):
        if idx == fidx:
            continue
        
        if idx >= len(features):
            continue

        differences.append( abs(features[idx][1] - ff[1]) )

    diffs.append(min(differences))

diffs = np.array(diffs)

print(np.mean(diffs))
exit( 0 )
print(eics[0].__dict__)
print(len(eics))
exit( 0 )


for ee in eics:
    print(ee.__dict__)

    break
