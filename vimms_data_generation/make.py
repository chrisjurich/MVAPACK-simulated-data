#!/usr/bin/env python3
# coding: utf-8 import json import sys, os import shutil import pandas as pd
import json
import pickle
import numpy as np
import pandas as pd
from glob import glob
from pathlib import Path
from itertools import product
from vimms.Chemicals import ChemicalCreator
from collections import namedtuple, defaultdict
from vimms.MassSpec import IndependentMassSpectrometer, Scan
from vimms.Controller import SimpleMs1Controller
from vimms.Common import *
from vimms.MzmlWriter import MzmlWriter
import vimms
from scipy.interpolate import interp1d
import argparse


# TODO need to document what each of the replicates does and doesn't have
# TODO need to set it up so that we're using the same multipliers for the
# significant compounds
from copy import deepcopy
import matplotlib.pyplot as plt
DATA = json.load(open("multipliers.json", "r")) 
GOOD_MULTS = DATA["good"]*10
BAD_MULTS = DATA["bad"]*10

def cv(vals):
    return np.std(vals) / np.mean(vals) 

def random_bool( ):
    return bool( np.random.randint(0, 2) )

def get_max_mz( chem ):
    return max( [pr[0] for pr in chem.isotopes] )

class Params:
    rt_lower = 0
    rt_upper = 2000
    mz_lower = 0
    mz_upper = 2005
    pct_missing = [0, 10, 20]
    noise_level = [0, 0.01, 0.1]


def safe_mkdir(dirname):
    if not os.path.exists(dirname):
        os.mkdir(dirname)


def remove_x_pct(df: pd.DataFrame, x: int):
    """Function that randomly removes x pct of entries from the supplied list of vals. Makes a copy of the inputted vals"""
    assert x >= 0 and x <= 100, f"x must be a percentage on the range 0 < x < 100"
    # print(float(chems.size)*(1.-float(x)/100.))
    num_cols, num_rows = len(df.columns), len(df[df.columns[0]])
    n_chems = num_cols * num_rows
    n_remove = int(n_chems * x / 100.0 + 0.5)
    rm_per_col = np.zeros(num_cols, np.int32)
    for idx in range(n_remove):
        rm_per_col[idx % num_cols] += 1

    np.random.shuffle(rm_per_col)
    row_idxs = np.arange(0, num_rows, 1)

    result = pd.DataFrame()
    for c_idx, col in enumerate(df.columns):

        new_col = list(map(lambda x: deepcopy(x), df[col]))
        row_idxs = np.sort(row_idxs)
        np.random.shuffle(row_idxs)

        col_remove = rm_per_col[c_idx]

        for cr_idx in range(col_remove):
            new_col[row_idxs[cr_idx]] = None

        result[col] = new_col

    return result




def setup_dirs(params: Params, basedir="mvapack-data"):
    # layout of the directories are:
    # basedir / pct_missing / noise / group
    # maybe should return the directories too?
    for edir in glob(f"{basedir}/*"):
        shutil.rmtree(edir)

    for pct in params.pct_missing:
        for noise in params.noise_level:
            dirname = "/".join([basedir, f"{pct}_missing", f"{noise:.2f}_noise"])
            if not os.path.isdir(dirname):
                Path(dirname).mkdir(parents=True, exist_ok=True)


def load_dbs():
    base_dir = os.path.abspath("example_data")
    ps = load_obj(Path(base_dir, "peak_sampler_mz_rt_int_19_beers_fullscan.p"))
    hmdb = load_obj(Path(base_dir, "hmdb_compounds.p"))
    out_dir = Path(base_dir, "results", "MS1_single")
    # the list of ROI sources created in the previous notebook '01. Download Data.ipynb'
    ROI_Sources = [str(Path(base_dir, "DsDA", "DsDA_Beer", "beer_t10_simulator_files"))]

    return (ROI_Sources, ps, hmdb)


def make_dataset(ROI_Sources, ps, hmdb, n_chems):
    params = Params()
    mz_range = [(params.mz_lower, params.mz_upper)]
    rt_range = [(params.rt_lower, params.rt_upper)]
    min_ms1_intensity = 1.75e5  # TODO make this into a global

    # m/z and RT range of chemicals

    # the number of chemicals in the sample
    # n_chems = 2000

    # maximum MS level (we do not generate fragmentation peaks when this value is 1)
    ms_level = 1
    
    chems = ChemicalCreator(ps, ROI_Sources, hmdb)
    reps = chems.sample(mz_range, rt_range, min_ms1_intensity, int(n_chems), ms_level)
    # out_dir = 'mvapack-data'
    # out_dir = 'demo'
    # save_obj(dataset, Path(out_dir, 'dataset.p'))
    
   
    bad = set(list(open('bad_chems', 'r').read().splitlines()))
    reps = list(filter(lambda s: str(s.formula) not in bad and get_max_mz( s ) < 1800, reps ))
    for r in reps:
        if r.rt <= params.rt_lower:
            r.rt = params.rt_lower + 20
        elif r.rt >= params.rt_upper:
            r.rt = params.rt_upper - 20
    return reps[0:n_chems]


def merge_chems(cdict, chems):

    for cd_key, chem in zip(cdict.keys(), chems):
        cdict[cd_key].append(chem)


def get_fc(is_sig):

    if is_sig:
        return np.random.random() * 2 + 2
    else:
        return np.random.random() * 1


def get_mults(is_sig):
    if is_sig:
        return GOOD_MULTS.pop(), GOOD_MULTS.pop()
    else:
        return BAD_MULTS.pop(), BAD_MULTS.pop()


def get_t_offset(is_sig):
    if is_sig:
        return (np.random.random() * 10) - 5
    else:
        return (np.random.random() * 60) - 30


def get_t_base(is_sig):
    if is_sig:
        return (np.random.random() * 30) - 15
    else:
        return (np.random.random() * 60) - 30


def get_chemical_lists(chems, NUM_REPS=10):

    SIGNIFICANT = int(len(chems) * 0.40)
    keys = list(map(lambda idx: f"rep_{idx}", range(NUM_REPS)))
    values = list(map(lambda idx: deepcopy([]), range(NUM_REPS)))
    clean_dict = lambda: deepcopy(dict(zip(keys, values)))
    group1, group2 = clean_dict(), clean_dict()
    np.random.shuffle(chems)
    g1_sig, g1_isig, g2_sig, g2_isig = None, None, None, None

    for idx, chem in enumerate(chems):
        # first figure out the fold change.... between 2 and 6?
        is_sig = idx < SIGNIFICANT
        if idx == (SIGNIFICANT - 1):
            g1_sig = pd.DataFrame(group1)
            g2_sig = pd.DataFrame(group2)
            group1, group2 = clean_dict(), clean_dict()

        fc = get_fc(is_sig)
        greater, lesser = (
            list(map(lambda idx: deepcopy(chem), range(NUM_REPS))),
            list(map(lambda idx: deepcopy(chem), range(NUM_REPS))),
        )
        mults1, mults2 = get_mults(is_sig)

        greater_base, lesser_base = get_t_base(is_sig), get_t_base(is_sig)

        for ii, r_key in enumerate(keys):
            greater[ii].max_intensity *= mults1[ii]
            greater[ii].max_intensity *= fc
            greater[ii].rt += greater_base + get_t_offset(is_sig)
            lesser[ii].max_intensity *= mults2[ii]
            lesser[ii].rt += lesser_base + get_t_offset(is_sig)
            # for some reason we can actually get negative peaks... that's a problem
            greater[ii].rt = max(greater[ii].rt, 10)
            lesser[ii].rt = max(lesser[ii].rt, 10)

        g1_greater = bool(np.random.randint(0, 2))

        merge_chems(group1 if g1_greater else group2, greater)
        merge_chems(group2 if g1_greater else group1, lesser)

    return g1_sig, pd.DataFrame(group1), g2_sig, pd.DataFrame(group2), keys


def write_summary(g1, g2, dirname, peak_values):

    unique_formulas = set()
    summary_info = dict()
    for idx, c in enumerate(g1.columns):
        summary_info[f"G0-R{idx}"] = deepcopy(g1[c])

    for idx, c in enumerate(g2.columns):
        summary_info[f"G1-R{idx}"] = deepcopy(g2[c])

    for k, v in summary_info.items():
        pvs = peak_values[k]
        for chem in v:
            print(str(chem.formula) in pvs)
        break
    print(peak_values.keys())

    for clist in summary_info.values():
        _ = list(
            map(
                lambda c: unique_formulas.add(str(c.formula)),
                list(filter(lambda chem: chem is not None, clist)),
            )
        )
    summary = dict(chemical=deepcopy(list(unique_formulas)))

    for rname, rchems in summary_info.items():
        rchem_list = []
        rep_mapper = dict()
        for rc in rchems:
            if rc is None:
                continue
            rep_mapper[str(rc.formula)] = deepcopy(rc)

        for uf in unique_formulas:
            rchem_list.append(rep_mapper.get(uf, None))

        summary[rname] = rchem_list

    summary = pd.DataFrame(summary)
    summary.set_index("chemical", inplace=True)
    summary = summary.transpose()
    mapper = dict()
    for col in summary.columns:
        sum_col = summary[col]
        for c in sum_col:
            print(c)
        # print( sum_col[0] )
        # print( sum_col[0].__dict__ )
        exit(0)
        times = np.array(
            list(map(lambda chem: chem.rt if chem is not None else None, sum_col))
        )
        mzs = np.array(
            list(
                map(
                    lambda chem: chem.isotopes[0][0] if chem is not None else None,
                    sum_col,
                )
            )
        )
        times = times[times != None]
        mzs = mzs[mzs != None]
        avg_mz, avg_rt = np.mean(mzs), np.mean(times)
        mapper[col] = f"{avg_rt:.2f}_{avg_mz:.4f}"

    summary.rename(mapper=mapper, inplace=True, axis=1)
    for col in summary.columns:
        summary[col] = summary.apply(
            lambda row: row[col].max_intensity if row[col] else 0, axis=1
        )

    summary.to_csv(f"{dirname}/summary.csv")
    summary.to_pickle(f"{dirname}/summary.pickle")
    print(f"{dirname}/summary.csv")


def increase_resolution(controller):
    new_scans = []
    old_len = len(controller.scans[1])
    
    for idx, scan in enumerate(controller.scans[1]):
        # loop through each scan
        new_mz, new_it = [], []
        interp_mz = []
        num_points = len(scan.mzs)

        # need to check that there are points
        if num_points:
            new_mz.append(scan.mzs[0] - 0.1)
            new_it.append(0)

        for pt_idx, (m, i) in enumerate(zip(scan.mzs, scan.intensities)):
            interp_mz.append(np.linspace(m - 0.021, m + 0.021, 10))

            new_mz.extend([m - 0.02, m, m + 0.02])
            new_it.extend([0, i, 0])

            if pt_idx > 0 and pt_idx < num_points - 1:
                interp_mz.append(np.arange(m + 0.25, scan.mzs[pt_idx + 1] - 0.25, 0.2))

        if num_points:
            new_mz.append(scan.mzs[-1] + 0.1)
            new_it.append(0)

        if not len(scan.mzs):
            new_scans.append(deepcopy( scan ))
            continue

        interp_mz = np.concatenate(interp_mz)
        interp_mz = sorted(interp_mz)

        interp_func = interp1d(new_mz, new_it, kind="linear")
        new_x = np.linspace(np.min(new_mz), np.max(new_mz), 1000)
        interp_it = interp_func(interp_mz)
        interp_it[interp_it < 0] = 0

        new_scan = deepcopy(scan)
        new_scan.mzs = interp_mz
        new_scan.intensities = interp_it
        new_scans.append(new_scan)

    controller.scans[1] = new_scans

    assert len(controller.scans[1]) == old_len 

    return controller


def valid_mzs(mzs):
    for idx in range(1, len(mzs)):
        assert mzs[idx] >= mzs[idx - 1], f"{mzs[idx-1]}, {mzs[idx]}"


def add_noise(controller, noise):
    if noise == 0:
        return controller
    
    def max_it( scan ):
        if len(scan.intensities):
            return np.max( scan.intensities )
        else:
            return 0

    old_len = len(controller.scans[1])
    
    max_it = np.max(
        max(controller.scans[1], key=max_it ).intensities
    )
    max_noise = float(max_it) * (float(noise) / 100.0)
    new_scans = []

    for idx, scan in enumerate(controller.scans[1]):
        valid_mzs(scan.mzs)
        scan_noise = list(
            map(lambda it: np.random.rand() * max_noise, range(len(scan.mzs)))
        )
        new_its = []
        for ct, nz in zip(scan.intensities, scan_noise):
            if ct > 0:
                new_its.append(ct)
            else:
                new_its.append(nz)

        new_scan = deepcopy(scan)
        assert len(scan.mzs) == len(scan.intensities)
        new_scan.intensities = new_its
        new_scans.append(new_scan)

    controller.scans[1] = new_scans
    assert len(controller.scans[1]) == old_len
    
    return controller


def make_chem_dict(
    group1_sig_orig, group1_isig_orig, group2_sig_orig, group2_isig_orig, rep_keys
):
    # shuffle up and deal!
    group1_sig_orig = group1_sig_orig.sample(frac=1).reset_index(drop=True)
    group2_sig_orig = group2_sig_orig.sample(frac=1).reset_index(drop=True)
    group1_isig_orig = group1_isig_orig.sample(frac=1).reset_index(drop=True)
    group2_isig_orig = group2_isig_orig.sample(frac=1).reset_index(drop=True)

    chem_dict = dict()
    for g_idx, (grp_sig, grp_isig) in enumerate(
        [(group1_sig_orig, group1_isig_orig), (group2_sig_orig, group2_isig_orig)]
    ):
        for idx, r_key in enumerate(rep_keys):
            # what do we care about now? need to locate
            # the part where noise is actually added
            chems_sig, chems_isig = (
                deepcopy(grp_sig[r_key]).to_list(),
                deepcopy(grp_isig[r_key]).to_list(),
            )
            chems_sig.extend(chems_isig)
            chem_dict[f"G{g_idx}-R{idx}"] = deepcopy(chems_sig)
    return chem_dict


def num_zeros(df):
    ct = 0

    for c in df.columns:
        ct += sum(list(map(lambda ch: ch.max_intensity == 0, df[c])))

    return ct


def adjust_groups(group1_sig, group2_sig, peak_values):
    # print( peak_values.keys() )
    # print( group1_sig )

    def col_to_key(colname):
        return str(int(colname.split("_")[-1]) + 1)

    for g1_col in group1_sig.columns:
        key_name = f"G0-R{col_to_key(g1_col)}"
        c_dict = peak_values[key_name]
        print(c_dict)
        exit(0)
        # print( [t.formula for t in group1_sig[ g1_col ]] )
        for t_chem in group1_sig[g1_col]:
            print(t_chem.formula in c_dict)
        #            if t_chem.formula in c_dict:
        #                print( t_chem.formula )
        #                print( c_dict[t_chem.formula] )
        exit(0)
        print(key_name)
        assert key_name in peak_values
        # need to convert the g1_col to the repname
        print(g1_col)

    for g2_col in group2_sig.columns:
        key_name = f"G1-R{col_to_key(g2_col)}"
        print(key_name)
        assert key_name in peak_values  # need to convert the g1_col to the repname
        print(g1_col)

    exit(0)

def make_master_replicates( chem_dict, POSITIVE, ps, params, sig ):
    replicates = dict() 
    for rep_name, chems in chem_dict.items():
        print( rep_name )
        filt_chems = list(filter(lambda c: c is not None, chems))
        filt_chems = sorted( filt_chems, key=lambda c: str(c.formula))
        
        mass_spec = IndependentMassSpectrometer(
            POSITIVE, filt_chems, ps, sig, None, True
        )
        controller = SimpleMs1Controller(mass_spec, params.mz_upper)
        controller.run(params.rt_lower, params.rt_upper, False)
        replicates[ rep_name ] = controller             

    return replicates

        
def validate_cd( chem_dict ):
    counter = defaultdict( int )
    num_reps = len(chem_dict.keys())
    for rep_chems in chem_dict.values():
        for c in rep_chems:
            counter[ str(c.formula) ] += 1
    for cname, v in counter.items():
        assert v == num_reps, cname

def validate_reps( replicates ):
    print( replicates.keys() )
    counter = defaultdict( int )
    num_reps = len(replicates.keys())
    
    for rep_chems in replicates.values():
        for c in rep_chems.mass_spec.chemicals:
            counter[ str(c.formula) ] += 1
    for cname, v in counter.items():
        assert v == num_reps, cname

def roughly_equal( a, b, tol=0.001 ):
    return abs( a - b ) <= tol

class RtMapper:

    def __init__( self, fset ):
        self.rt_to_idx_ = list( )
        self.idx_to_rt_ = dict( )
        
        self.rts_ = list( fset )

        for idx, rt in enumerate( self.rts_ ):
            self.rt_to_idx_.append((rt, idx))
            self.idx_to_rt_[ idx ] = rt

        self.length_ = len( self.rts_ )

    def rt_to_idx(self, nrt ):
        assert isinstance(nrt, float)

        for (rt, idx) in self.rt_to_idx_:
            if roughly_equal(rt, nrt):
                return idx
        else:
            raise TypeError('no match')
    
    def rts( self ):
        return self.rts_

    def idx_to_rt( self, idx ):
        return self.idx_to_rt_[ idx ]

    def length( self ):
        return self.length_


def get_all_rts( controller ):
    rts = [ sc.rt for sc in controller.scans[1] ]
    rts = sorted( rts )
    return RtMapper( rts )

def determine_significant( eics ):
    eicnames = list(set(list(map(lambda e: e.name, eics))))
    np.random.shuffle( eicnames )
    mapper = dict( )
    for idx, name in enumerate( eicnames ):
        mapper[ name ] = idx < 800
    return mapper


def get_multiplier_mapper( sigmapper ):
    result = dict( )
    for chem, is_sig in sigmapper.items( ):
        mults1, mults2 = get_mults(is_sig)
        mults1 = np.array( mults1 )
        mults2 = np.array( mults2 )
        fc = get_fc( is_sig )
        if random_bool( ):
            result[ chem ] = np.concatenate((
                    mults1*fc, mults2
                ))
        else:
            result[ chem ] = np.concatenate((
                    mults2*fc, mults1
                ))
    return result


def get_removal_mapper( sigmapper, pct_missing ):
    num_sig = sum(sigmapper.values())
    num_to_remove = int(pct_missing/100*num_sig)
    chemnames = list(sigmapper.keys())
    np.random.shuffle( chemnames )
    
    result = dict( )
    for idx, cname in enumerate( chemnames ):
        result[ cname ] = idx < num_to_remove
    
    return result

EIC = namedtuple('EIC', 'name mzs rts its')

def export_summary( summary, target_dir ):
    all_chems = set()
    for rep, chems in summary.items():
        for chemname in chems.keys():
            all_chems.add( chemname )
    repnames = summary.keys() 
    data = dict( )
    data['rep'] = repnames
    for ac in all_chems:
        col_vals = []
        for r in repnames:
            if ac in summary[r]:
                col_vals.append( summary[r][ac] )
            else:
                col_vals.append( None )
        data[ ac ] = col_vals
    
    df = pd.DataFrame( data )
    mapper = dict( )
    for col in df.columns:
        if col == 'rep':
            continue

        rts, mzs, its = [], [], []
        for row  in df[col]:
            if row is None:
                its.append( 0 )
                continue
            (rt, mz, ct ) = row
            rts.append( rt )
            mzs.append( mz )
            its.append( ct )
        rts = np.array( rts )
        mzs = np.array( mzs )
        df[col] = its
        mapper[ col ] = f"{np.mean(rts):.2f}_{np.mean(mzs):.4f}" 
    df.rename( mapper, axis=1, inplace=True)
    df.to_csv( f"{target_dir}/summary.csv", index=False )

def make_plot( controller ):
    scan = controller.scans[1][100]

    plt.plot(scan.mzs, scan.intensities, c='k')
    plt.ylabel('intensity (A.U.)')
    plt.xlabel('mz (daltons)')
    plt.show()

def create_condition( controller, pct_missing, noise, eics, mapper, sigmapper, basedir='mvapack-data' ):
    target_dir = f"{basedir}/{int(pct_missing)}_missing/{noise:.2f}_noise"
    mult_mapper = get_multiplier_mapper( sigmapper )
    pickle.dump(mult_mapper, open('mult_mapper.pickle', 'wb'))
    exit( 0 )
    assert os.path.isdir( target_dir ), target_dir
    allchemnames = list(map(lambda pr: pr.name, eics ))
    summary = dict( )
    for r_idx in range( 20 ):
        print( r_idx )
        features = dict( )
        should_remove = get_removal_mapper( sigmapper, pct_missing )
        group = "G0" if r_idx < 10 else "G1"
        rep = f"R{r_idx%10}"
        full_name = group + '-' + rep + '.mzML'
        local_eics =  [deepcopy(ee) for ee in eics ]
        
        finalized_eics = []
        for le in local_eics:
            rts, its, name = le.rts, le.its, le.name
            mult = mult_mapper[ name ][ r_idx ]
            its *= mult

            if should_remove[ name ]:
                continue

            finalized_eics.append(
                    EIC(name=name, mzs=le.mzs, rts=rts, its=its)
                    )
        controller.scans = eics_to_scans( finalized_eics ) 
        #controller.update_scans( finalized_eics, mapper )
        #make_plot( controller )
        #controller = increase_resolution( controller )
        #controller = add_noise( controller, noise )
        #make_plot( controller )
        controller.write_mzML( target_dir, target_dir + '/' + full_name )
        
        for acn in allchemnames:
            if should_remove[acn]:
                continue
            filtered = list(filter( lambda pr: pr.name == acn, finalized_eics ))
            max_mz, max_it, max_rt = 0, 0, 0 
            for feic in filtered:
                mzs, rts, its = feic.mzs, feic.rts, feic.its

                max_idx = np.argmax( its )
                if its[ max_idx ] > max_it:
                    max_it = its[ max_idx ]
                    max_mz = mzs[ max_idx ]
                    max_rt = rts[ max_idx ]

            features[ acn ] = ( max_rt, max_mz, max_it )

        summary[ full_name ] = features
    
    export_summary( summary, target_dir )

def setup_params( ):
    parser = argparse.ArgumentParser( )
    #parser.add_argument('--mode', required=True, type=str )
    parser.add_argument('--pct_missing', required=False, type=int, default=0 )
    parser.add_argument('--noise', required=False, type=float, default=0 )
    parser.add_argument('--idx', required=False, type=int, default=0 )

    return parser.parse_args( )

def eics_to_scans( eics ):
    eics = sorted( eics, key=lambda e: e.mzs[0] )
    scans = []
    avg_dur = np.mean(np.array([abs(t1 - t2) for t1, t2 in zip( eics[0].rts[0:-1], eics[0].rts[1:])]))
    rt_len = len( eics[0].rts )
    mzs = list(map(lambda idx: [],range(rt_len)))
    its = list(map(lambda idx: [],range(rt_len)))
    
    for idx in range( rt_len ):
        for e in eics:
            mzs[idx].append( e.mzs[idx] )
            its[idx].append( e.its[idx] )

    for idx,rt in enumerate( eics[0].rts ):
        if idx != rt_len-1:
            duration = eics[0].rts[idx+1]-rt
        else:
            duration = avg_dur
        assert duration > 0
        scans.append(Scan(
            idx, np.array(mzs[idx]), np.array(its[idx]), 1, rt, duration
            ))

    return {1: scans}

def main( params ):

    np.random.seed( 100 )
    params = Params()
    #params.rt_upper = 120 # TODO this is just for debugging
    #dgs = setup_dirs(params)
    (ROI_Sources, ps, hmdb) = load_dbs()
    # can probably make the smallest peaks a bit smaller here
    if False: 
        set_log_level_debug()
        (ROI_Sources, ps, hmdb) = load_dbs()
        orig_chems = make_dataset(ROI_Sources, ps, hmdb, 10000)
        pickle.dump(orig_chems, open('chems.pickle', 'wb'))
    else:
        orig_chems = pickle.load(open('chems.pickle', 'rb'))
    #print(len(orig_chems))
    orig_chems = list(filter(lambda oc: oc.rt > 60, orig_chems ))
    sig = False
    mass_spec = IndependentMassSpectrometer(
            POSITIVE, orig_chems, ps, sig, None, True
    )
    controller = SimpleMs1Controller(mass_spec, params.mz_upper)
    #controller.run(params.rt_lower, params.rt_upper, False)
    ### ok so at this point we have all of the data points.
    ### Basically want to take the peak_recorder and then
    ### return a list of tuples (chem name (mzs, rts, its )).
    mapper = get_all_rts( controller )
    ##eics = controller.create_eics( mapper )
    ##pickle.dump(eics, open('eics.pickle', 'wb'))
    ##exit( 0 )
    eics = pickle.load(open('eics-ideal.pickle', 'rb'))
    controller.scans = eics_to_scans( eics )
    #controller.write_mzML('idk', 'test.mzML')
    #print( scans )
    sigmapper = determine_significant( eics )
    
    for pct_missing in params.pct_missing:
        for noise in params.noise_level:
            create_condition( controller, pct_missing, 0.0025, eics, mapper, sigmapper )


if __name__ == "__main__":
    main( setup_params() )
