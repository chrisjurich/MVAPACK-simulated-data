import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def cv( values ):
	return np.std( values )/ np.mean( values)	

def get_mults_good( base_val, cv, n, mean_var=0.005 ):
	mults = np.random.normal(loc=base_val, scale=cv*base_val*0.90, size=n )
	allowed_mean_var = base_val*mean_var
	mult_mean, mult_sd = np.mean(mults), np.std(mults)
	
	while True:
		if np.abs(mult_mean-base_val) < allowed_mean_var and (mult_sd/mult_mean) < cv:
			break
		mults = np.random.normal(loc=base_val, scale=cv*base_val*0.90, size=n )
		mult_mean, mult_sd = np.mean(mults), np.std(mults)
	return mults

def get_mults_bad( base_val, cv, n, mean_var=0.005 ):
	mults = np.random.normal(loc=base_val, scale=cv*base_val*1.10, size=n )
	allowed_mean_var = base_val*mean_var
	mult_mean, mult_sd = np.mean(mults), np.std(mults)
	while True:
		if np.abs(mult_mean-base_val) < allowed_mean_var and (mult_sd/mult_mean) > cv:
			break
		mults = np.random.normal(loc=base_val, scale=cv*base_val*1.10, size=n )
		mult_mean, mult_sd = np.mean(mults), np.std(mults)
	return mults


def rand_float(lower, upper):
	span = upper-lower
	return lower + np.random.rand()*span

def rand_bool():
	return np.random.choice([True,False])


def main():
	np.random.seed( 100 )
	
	good, bad = [], []
	
	# make the good values first
	for idx in range(1000):
		big_value = rand_float(15, 25)
		small_value = big_value - rand_float( 2.2, 4 )
		if rand_bool():
			big_mults, small_mults = (get_mults_good(big_value, 0.15, 10 ),get_mults_good(small_value, 0.15, 10 ))
		else:
			small_mults, big_mults = (get_mults_good(big_value, 0.15, 10 ),get_mults_good(small_value, 0.15, 10 ))
		
		assert abs(np.mean(big_mults)-np.mean(small_mults)) >= 2 and cv(small_mults) <= 0.15 and cv(big_mults) <= 0.15
	
		good.append(np.concatenate((big_mults,small_mults)))
	
	
	
	for idx in range(3000):
		big_value = rand_float(15, 25)
		small_value = big_value - rand_float( 0, 1.7 )
		if rand_bool():
			big_mults, small_mults = (get_mults_bad(big_value, 0.30, 10 ),get_mults_bad(small_value, 0.30, 10 ))
		else:
			small_mults, big_mults = (get_mults_bad(big_value, 0.30, 10 ),get_mults_bad(small_value, 0.30, 10 ))
	
		assert abs(np.mean(big_mults)-np.mean(small_mults)) <= 2 and cv(small_mults) >= 0.30 and cv(big_mults) >= 0.30
	
		bad.append(np.concatenate((big_mults,small_mults)))
	
	rawData = []
	
	for gg in good:
		rawData.append( ['good'] + gg.tolist())
	
	for bb in bad:
		rawData.append( ['bad'] + bb.tolist())
	
	
	df = pd.DataFrame(data=rawData, columns=['flag']+list(range(20)))
	for colname in df.columns:
		if colname == 'flag':
			continue
		df[colname] = 2**df[colname]
	
	df.to_csv('multipliers.csv',index=False)


if __name__ == '__main__':
	main()
