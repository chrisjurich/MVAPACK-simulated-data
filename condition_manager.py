"""Holds the ConditionManager() class which creates the mzML files for each condition.
"""
from utils import *
import pandas as pd
import pickle
from copy import deepcopy
from itertools import product
from collections import defaultdict
import matplotlib.pyplot as plt
from typing import Dict, List, Set, Tuple


class ConditionManager: 
	
	def __init__(self, **kwargs):
	
		self.eics_ = kwargs.get("eics", None)
		self.mapper_ = kwargs.get("mapper", None)
		self.pct_missing_ = kwargs.get("pct_missing", 0)
		self.noise_ = kwargs.get("noise", 0)
		self.replicate_info_ = kwargs.get("replicates", None)
		self.replicates_ = dict()
		self.multipliers_ = dict()
		self.scans_ = dict()
		self.l_summary_ = dict()
		self.ul_summary_ = dict()
		self.keeper_ = dict()
		self.fc_mapper_ = dict()
		self.mult_holder_ = dict() 
		self.good_mults_ = list()
		self.bad_mults_ = list()
		
		if self.eics_ is None:
			raise TypeError("Must supply eics!")
		
		if self.mapper_ is None:
			raise TypeError("Must supply mapper!")
		
		if self.replicate_info_ is None:
			raise TypeError("Must supply mapper!")

		df = pd.read_csv('multipliers.csv')
		for i, row in df.iterrows():
			raw_vals = row.to_list()
			mult_type, mults = raw_vals[0], raw_vals[1:]
			# TODO make this not hardcoded... integrate the mults.py script
			assert len(mults) == 20
			if mult_type == 'good':
				self.good_mults_.append( mults )
			else:
				self.bad_mults_.append( mults )
		
	
	def chemical_names(self):
		return sorted(list(self.mapper_.keys()))

	def setup_removable_(self):
		removable = []
		for chem in self.chemical_names():
			if not self.mapper_[chem]:
				continue
			for group, n_reps in self.replicate_info_.items():
				candidates = []
				for r_idx in range(n_reps):
					candidates.append((group,r_idx,chem))
				np.random.shuffle( candidates )
				removable.append(candidates.pop())
				removable.append(candidates.pop())
	

		num_sig = sum(self.mapper_.values())
		num_to_remove = int(self.pct_missing_ * num_sig*sum(self.replicate_info_.values()))
		np.random.shuffle(removable)
		return set(removable[0:num_to_remove])

	def setup_keepers_(self):
		removable = self.setup_removable_()
		
		for group, n_reps in self.replicate_info_.items():
			for r_idx in range(n_reps):
				for chem in self.chemical_names():
					if not self.mapper_[chem]:
						self.keeper_[(group, r_idx, chem)] = True
					else:
						self.keeper_[(group, r_idx, chem)] = not ((group, r_idx, chem) in removable)
	
	def setup_fc_mapper_( self ):
		n_groups = len(self.replicate_info_.keys())
		sig_chems = sum(self.mapper_.values())
		g_ids = list(map(lambda idx: idx%n_groups, range(sig_chems)))
		np.random.shuffle( g_ids )
		
		for chem, is_sig in self.mapper_.items():
			if not is_sig:
				continue
			self.fc_mapper_[chem] = (g_ids.pop(), 2 + np.random.random()*2)
	
	def get_random_group_idx( self ):
		return np.random.randint(0, len(self.replicate_info_.keys()))


	def get_mults(self, chem):
		if self.mapper_[chem]: 
			return self.good_mults_.pop()
		else:
			return self.bad_mults_.pop()

	def setup_multipliers_(self):
		# ok now we loop through each chemical
		for chem in self.chemical_names():
			# if its significant
			mults = self.get_mults( chem )
			for idx, (group, n_reps) in enumerate(self.replicate_info_.items()):
				for r_idx in range(n_reps):
					assert ((group-1)*10+r_idx) <= 20 and ((group-1)*10+r_idx) >= 0
					self.multipliers_[(group, r_idx, chem)] = mults[(group-1)*10+r_idx]


	def remove_peaks_(self):
		num_sig = sum(self.mapper_.values())
		num_to_remove = self.pct_missing_ * num_sig
		
		if not num_to_remove:
			return
		
		cnames = self.chemical_names()
		for key_name, eics in self.replicates_.items():
			np.random.shuffle(cnames)
			remove = set(cnames[0:num_to_remove])
			self.replicates_[key_name] = list(
				filter(lambda eic: eic.name not in remove, eics)
			)
		
	def convert_to_scans_(self):
		for rep_name, eics in self.replicates_.items():
			self.scans_[rep_name] = eics_to_scans(eics)
		rns = list(self.replicates_.keys())
		for rn in rns:
			del self.replicates_[rn]
	
	def apply_multipliers_(self):
		for (g_idx, r_idx), eics in self.replicates_.items():
			self.replicates_[(g_idx, r_idx)] = list(
				map(
					lambda eic: apply_eic_multiplier(
						eic, self.multipliers_[(g_idx, r_idx, eic.name)]
					),
					eics,
				)
			)
	
	def generate_condition(self):
		# self.setup_replicates_( )
		#self.setup_fc_mapper_()
		self.setup_multipliers_()
		self.setup_keepers_()
		# self.apply_multipliers_( )
		# self.remove_peaks_( )
		# self.add_noise_( )
		# self.create_summary_( )
		# self.convert_to_scans_( )
	
	def labelled_peaks_from_eics_(self, eics, g_id, r_idx):
		df = pd.DataFrame(eics, columns=["rt", "mz", "cts", "name"])
		#df.sort_values(by=["name", "cts"], ascending=True, inplace=True)
		#df.reset_index(drop=True, inplace=True)
		#df.drop_duplicates(subset="name", keep="first", inplace=True)
		result = dict()
		for i, row in df.iterrows():
			key, value =  (g_id, r_idx, row["name"]), (row["rt"], row["mz"], row["cts"])
			if key not in result:
				result[key] = value
			else:
				existing = result[key]
				if value[-1] > existing[-1]:
					result[key] = value
		return result
	
	def unlabelled_peaks_from_eics_(self, eics, g_id, r_idx):
		def rename_row(row):
			return f"{row['name']}_{int(row['mz'])}"
		
		df = pd.DataFrame(eics, columns=["rt", "mz", "cts", "name"])
		df["name"] = df.apply(rename_row, axis=1)
		assert len(df["name"].unique()) == len(df)
		
		result = dict()
		for i, row in df.iterrows():
			result[(g_id, r_idx, row["name"])] = (row["rt"], row["mz"], row["cts"])
		return result
	
	def catalog_eics_(self, eics, g_id, r_idx):
		# basically we are looking at both
		# labelled and un-labelled peaks
		# format = (name, rt, mz, it )
		e_keys = list(map(lambda e: e.key(), eics))
		l_peaks = self.labelled_peaks_from_eics_(e_keys, g_id, r_idx)
		ul_peaks = self.unlabelled_peaks_from_eics_(e_keys, g_id, r_idx)
		
		for lk, lv in l_peaks.items():
			assert lk not in self.l_summary_  # LOL
			self.l_summary_[lk] = lv
		
		for ulk, ulv in ul_peaks.items():
			assert ulk not in self.ul_summary_  # LOL
			self.ul_summary_[ulk] = ulv
		
	def create_summary_(self):
		# come up with a list of all the
		raw = []
		for eics in self.replicates_.values():
			raw.extend(list(map(lambda e: e.key(), eics)))
	
	def create_scans_(self, g_id, r_idx):
		eics = list(map(deepcopy, self.eics_))
		# filter out bad ones
		eics = list(filter(lambda e: self.keeper_[(g_id, r_idx, e.name)], eics))
		# add the multipliers
		self.eics_[0].integral()
		_ = list(
			map(lambda e: e.multiplier(self.multipliers_[(g_id, r_idx, e.name)]), eics)
		)
		# TODO add noise
		_ = list(map(lambda e: e.add_noise(e.max_it() * self.noise_), eics))
		
		self.catalog_eics_(eics, g_id, r_idx)
		
		return eics_to_scans(eics)
	
	def get_rep_ids_(self):
		rep_ids = []
		for g_id, r_idx in self.replicate_info_.items():
			rep_ids.extend(list(product([g_id], range(r_idx))))
		return rep_ids
	
	def summarize_df_(self, sdict : Dict[Tuple[int,int,str],Tuple[float,float,float]]): 
		chems = set(map(lambda key: key[-1], sdict.keys()))
		rep_ids = self.get_rep_ids_()
		data = {}
		for (g_id, r_id) in rep_ids:
			repname = f"G{g_id}-R{r_id}"
			data[repname] = list(
				map(lambda chem: sdict.get((g_id, r_id, chem), None), chems)
			)
		df = pd.DataFrame(data)
		df["chems"] = chems
		df.set_index("chems", inplace=True)
		df = df.transpose()
		colnames = list(df.columns)
		mapper = dict()
		sig_mapper = dict()
		for c in colnames:
			raw_col_vals = df[c].to_list()
			rts, mzs, its = [], [], []
			for rv in df[c]:
				if rv is None:
					its.append(0)
				else:
					(r, m, i) = rv
					rts.append(r)
					mzs.append(m)
					its.append(i)
			rts, mzs, its = np.array(rts), np.array(mzs), np.array(its)
			rts /= 60  # seconds to minutes
			mapper[c] = f"{np.mean(rts):.2f}_{np.mean(mzs):.4f}"
			c_cleaned = c	
			if c.find('_') != -1:
				c_cleaned = c_cleaned.split('_')[0]
			sig_mapper[f"{np.mean(rts):.2f}_{np.mean(mzs):.4f}"] = self.mapper_[c_cleaned]
			df[c] = its
		sig_df = pd.DataFrame()
		sig_df['feature'] = list(sig_mapper.keys())
		sig_df['significant'] = list(sig_mapper.values())
		return df.rename(columns=mapper), sig_df

	def save_summaries_(self, dirname):
		(ldf, l_sdf) = self.summarize_df_(self.l_summary_)
		ldf.to_csv(f"{dirname}/labelled_summary.csv")
		l_sdf.to_csv(f"{dirname}/labelled_significance.csv", index=False)
		(uldf, ul_sdf) = self.summarize_df_(self.ul_summary_)
		ul_sdf.to_csv(f"{dirname}/unlabelled_significance.csv", index=False)
		uldf.to_csv(f"{dirname}/unlabelled_summary.csv")
	
	def export_mzMLs(self, dirname):
		for g_id, num_rs in self.replicate_info_.items():
			for r_idx in range(num_rs):
				repname = f"G{g_id}-R{r_idx}.mzML"
				print(repname)
				scans = self.create_scans_(g_id, r_idx)
				export_scans(scans, f"{dirname}/{repname}")
		
		self.save_summaries_(dirname)
