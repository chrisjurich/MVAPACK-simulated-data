import pickle
import numpy as np
import pandas as pd
from utils import *
from eic import ClassEIC
from multiset import Multiset
from typing import Dict
import matplotlib.pyplot as plt
from condition_manager import ConditionManager
from collections import defaultdict




eics = pickle.load(open("data/eics-cpy2.pickle", "rb"))

eic_mapper = dict()

for ee in eics:
    eic_mapper[f"{ee.mz_:.4f}"] = ee.name



csv="/Users/chrisjurich/projects/simulated-lcms/simulated-integral/0_missing_0.00_noise/labelled_significance.csv"


df = pd.read_csv(csv)



peak_data = defaultdict(list)

for i, row in df.iterrows():
    rawt, rawmz = row['feature'].split('_')
    t, mz = float(rawt), float(rawmz)
    peak_data['mz'].append( rawmz )
    peak_data['significant'].append( row['significant'] )
    peak_data['chemical'].append( eic_mapper[rawmz] )

#print(len(mapper))

#print(sum(list(mapper.values())))



p_df = pd.DataFrame(peak_data)

p_df.to_csv('labelled_peak_list.csv',index=False)




peak_data = defaultdict(list)

csv="/Users/chrisjurich/projects/simulated-lcms/simulated-integral/0_missing_0.00_noise/unlabelled_significance.csv"


df = pd.read_csv(csv)
print(df)

existing = set()

for i, row in df.iterrows():
    rawt, rawmz = row['feature'].split('_')
    if rawmz in existing:
        continue
    existing.add( rawmz )
    t, mz = float(rawt), float(rawmz)
    peak_data['mz'].append( rawmz )
    peak_data['significant'].append( row['significant'] )
    peak_data['chemical'].append( eic_mapper[rawmz] )

del peak_data['chemical']


p_df = pd.DataFrame(peak_data)

p_df.to_csv('unlabelled_peak_list.csv',index=False)
