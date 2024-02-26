"""Driver script to create all of the mzML files with the statistially significant peaks.


Author: Chris Jurich <cjurich2@huskers.unl.edu>
Date: 2022-04-17
"""
import pickle
import numpy as np
from utils import *
from eic import ClassEIC
from multiset import Multiset
from typing import Dict
import matplotlib.pyplot as plt
from condition_manager import ConditionManager

RNG_SEED = 100

def report_significant( mapper  : Dict[str,bool]) -> None:
	"""Generates a basic table with counts of significant vs non-significant compounds."""
	values = np.array(list(mapper.values()))
	significant = np.sum(values)
	not_significant = np.sum(values == False)
	total = significant + not_significant
	print(f"Significant:     {significant: 4d}\t{significant/total:4.2%}")
	print(f"Not-Significant: {not_significant: 4d}\t{not_significant/total:4.2%}")
	print(f"RNG seed: {RNG_SEED}")

def main():
    #"""Driver function that reads in idealized files, creates"""
    np.random.seed( 100 )
    eics = pickle.load(open("data/eics-cpy2.pickle", "rb"))
    rts = eics[0].rts()
    # probably need to fix the EIC namedtuple
    mapper = determine_significant(eics, 0.35)
    #print(sum(list(mapper.values())))
    #print(mapper)
    report_significant( mapper );exit( 0 )
    replicates = {1: 10, 2: 10}
    for noise in [0, 0.05, 0.1]:
    	for pct_missing in [0, 0.10, 0.20]:
    		dirname = f"simulated-integral/{int(pct_missing*100)}_missing_{noise:.2f}_noise"
    		safe_mkdir(dirname)
    		print(dirname)
    		cm = ConditionManager(
    			eics=eics,
    			mapper=mapper,
    			replicates=replicates,
    			pct_missing=pct_missing,
    			noise=noise,
    		)
    		cm.generate_condition()
    		cm.export_mzMLs(dirname)
    

if __name__ == "__main__":
    main()
