import sys
import pymzml
import numpy as np
from pathlib import Path

def increasing_only( vals ):
	for idx in range(len(vals)-1):
		if vals[idx] > vals[idx+1]:
			return False
	return True

paths = Path('./simulated' ).rglob('*mzML')

for pp in paths:
	run = pymzml.run.Reader(str(pp))
	for spec in run:
		assert increasing_only( spec.mz )
		#print(spec.mz)
		#print('-'*100)
		#for k,v in spec.__dict__.items():
		#	print(k,v)
