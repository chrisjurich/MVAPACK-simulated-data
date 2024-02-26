import numpy as np
from utils import *
from copy import deepcopy
from multiset import Multiset
import matplotlib.pyplot as plt
from collections import defaultdict

# TODO need to generate this stuff
def estimate_width(y):
	peak = np.argmax(y)
	cutoff = y[peak]/2
	left, right = peak-1, peak+1
	while left > 0 and y[left] > cutoff:
		left -= 1
	while right < len(y) and y[right] > cutoff:
		right += 1
	left -= abs( peak - left )
	left = max( 0, left )
	right += abs(peak-right)
	right = min( right, len(y))
	return (left,right)



class ClassEIC:
	def __init__(self, eic=None):
		self.mz_ = None
		self.rts_ = None
		self.its_ = None
		self.name = None
		self.curr_mult = 0
		
		if eic:
			self.mz_ = eic.mzs[0]
			self.rts_ = deepcopy(eic.rts)
			self.its_ = deepcopy(eic.its)
			self.name = eic.name
	
	def clone( self ):
		cpy = ClassEIC()
		cpy.mz_= np.copy( self.mz_ )
		cpy.rts_= np.copy( self.rts_ )
		cpy.its_= np.copy( self.its_ )
		cpy.name = self.name

	def multiplier(self, mult):
		highest = np.max(self.its_)
		c = highest*mult/self.integral()
		self.its_ *= c
		self.curr_mult = mult

	def undo_multiplier( self ):
		self.its_ /= self.curr_mult

	def max_it(self):
		return np.max(self.its_)
	
	def max_rt(self):
		it = np.argmax(self.its_)
		return self.rts_[it]
	
	def mz(self):
		return self.mz_
	
	def key(self):
		return (self.max_rt(), self.mz(), self.max_it(), self.name)
	
	def rts(self):
		return self.rts_
	
	def its(self):
		return self.its_
	
	def mzs(self):
		return np.repeat(self.mz(), len(self.its_))
	
	def get_baseline_(self):
		counter = defaultdict(int)
		for val in self.its_:
			counter[round(val)] += 1
		values, mults = np.array(list(counter.keys())), np.array(list(counter.values()))
		it = np.argmax(mults)
		return values[it]
	
	def remove_baseline( self, variance ):
		assert variance > 0 and variance < 1
		base = self.get_baseline_()
		mask = np.abs((self.its()-base)/base) > variance
		self.its_ = self.its_[ mask ]
		self.rts_ = self.rts_[ mask ]
	
	def to_mins( self ):
		self.rts_ /= 60

	def add_noise(self, noise_amt):
		bline = self.get_baseline_()
		noise = np.array(
		    list(map(lambda idx: np.random.rand() * noise_amt + bline, self.its_))
		)
		self.its_ = np.maximum(self.its_, noise)

	def integral(self) -> float:
		(l_bound, r_bound) = estimate_width( self.its_ )
		return np.trapz(self.its_[l_bound:r_bound], self.rts_[l_bound:r_bound])

