import utils
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

np.random.seed( 100 )

def fc( v1, v2 ):
	v1_avg, v2_avg = np.mean(v1), np.mean(v2)
	return max(v1_avg,v2_avg)/min(v1_avg,v2_avg)


def get_sig():	
	m1, m2 = np.random.normal(loc=1, scale=0.1, size=10), np.random.normal(loc=3, scale=0.1, size=10)
	
	if np.random.choice([True,False]):
		return np.concatenate((m1,m2))
	else:
		return np.concatenate((m2,m1))

# 1. we need to make a contrived dataset

df = pd.DataFrame()

toBeRemoved = set(open('validation.txt','r').read().splitlines())

features = []

df = pd.read_csv('simulated/0_missing/0.00_noise/labelled_summary.csv')
df = df[ list(filter(lambda c: c.find('U') == -1, df.columns))]
features = list(df.columns)
keep = []
for f in features:
	vals = df[f]
	goodFc = fc(vals[0:10], vals[10:]) > 2
	goodCv = utils.log_cv(vals[0:10]) <= 0.175 and utils.log_cv(vals[10:]) <= 0.175
	if goodCv and goodFc:
		keep.append( f )
df = df[keep]
#print( df )
#print(len(df.columns))
#exit( 0)
#for f in features:
#	df[f] = np.log(df[f])

#for idx in range( 10 ):
#	repname = f'f{idx}'
#	features.append( repname )
#	#df[repname] = get_sig()
newDf = pd.DataFrame()
newDf['target'] = ['r1']*10+['r2']*10
for c in df.columns:
	newDf[c] = np.log2(df[c])
	#newDf[c] = df[c]
df = newDf
print( df )
# 2. make the pca model
pca = PCA(n_components=2)
x = df.loc[:, keep].values
y = df.loc[:, ['target']].values
principalComponents = pca.fit_transform(x)
principalDf = pd.DataFrame(data = principalComponents
             , columns = ['principal component 1', 'principal component 2'])
print( principalDf )


finalDf = pd.concat([principalDf, df[['target']]], axis = 1)

fig = plt.figure(figsize = (8,8))
ax = fig.add_subplot(1,1,1) 
ax.set_xlabel('Principal Component 1', fontsize = 15)
ax.set_ylabel('Principal Component 2', fontsize = 15)
ax.set_title('2 component PCA', fontsize = 20)
targets = ['r1', 'r2']
colors = ['r', 'g']
for target, color in zip(targets,colors):
    indicesToKeep = finalDf['target'] == target
    ax.scatter(finalDf.loc[indicesToKeep, 'principal component 1']
               , finalDf.loc[indicesToKeep, 'principal component 2']
               , c = color
               , s = 50)
ax.legend(targets)
ax.grid()

plt.show()
