"""A variety of utility functions for generating the synthetic data. Also defines the EIC and Scan namedtuple()'s

Author: Chris Jurich <cjurich2@huskers.unl.edu>
Date: 2024-02-26
"""
import os
import numpy as np
from typing import List, Dict
from pathlib import Path
from copy import deepcopy
import matplotlib.pyplot as plt
from collections import namedtuple
from psims.mzml.writer import MzMLWriter

np.random.seed(100)

EIC = namedtuple("EIC", "name mzs rts its")
EIC.__doc__ = "Data holder that defines a single extracted ion chromatogram. Works with ClassEIC objects. Has a str() name as well as arrays for mzs, rts and its."

Scan = namedtuple("Scan", "id mz_array intensity_array rt")
Scan.__doc__ = "Data holder that defines a single MS Scan. Has a single int() id, a float() rt, and an mz_array and intensity_array."


def is_close(a:float, b:float, tol:float) -> bool:
    """Are two numbers within the supplied tolerance?"""
    return abs(a - b) <= tol


def same_mz(a:float, b:float, tol:float=0.01) -> bool:
    """Are the two mz values equivalent within a default tolerance of 0.01?"""
    return is_close(a, b, tol)


def safe_mkdir(dirname:str) -> None:
    """Utility function to safely make a directory."""
    if not os.path.isdir(dirname):
        Path(dirname).mkdir(parents=True, exist_ok=True)


def clone_eic(eic:EIC) -> EIC:
    """Function to create a deep copy of an EIC namedtuple()"""
    return EIC(
        name=deepcopy(eic.name),
        mzs=deepcopy(eic.mzs),
        rts=deepcopy(eic.rts),
        its=deepcopy(eic.its),
    )


def apply_eic_multiplier(eic:"ClassEIC", mult:float) -> "ClassEIC":
    """Apply an amplitude multiplier to the supplied EIC's intensities. i.e. it_new = it_old*mult"""
    result = eic.clone()
    result.multiplier(mult)
    return result


def cv(vals:List[float]) -> float:
    """Get the coeffiecient of variation of the supplied sequence of values."""
    return np.std(vals) / np.mean(vals)


def log_cv(vals:List[float]) -> float:
    """Get the base2 log of coeffiecient of variation of the supplied sequence of values."""
    return cv(np.log2(vals))


def log2FC( g1_vals:List[float], g2_vals:List[float] ) -> float:
    """Get the base2 log of fold change between two grousp of equal length values."""
    return np.abs(np.mean(np.log2(g1_vals)) - np.mean(np.log2(g2_vals)))

def generate_mults(num:int, is_sig:bool, sig_cutoff:float) -> List[float]:
    """Driver function for creating group multipliers. 

    Args:
        num: number of values to create
        is_sig: Should the values be significant? I.e. close in value to eachother
        sig_cutoff: The cv cutoff for significance.

    Returns:
        The List[float] of multipliers.
    """
    if is_sig:
        while True:
            candidates = np.random.normal(loc=1, scale=sig_cutoff * 0.75, size=num)
            if cv(candidates) < sig_cutoff:
                return candidates
    else:
        while True:
            candidates = np.random.normal(loc=1, scale=sig_cutoff * 2, size=num)
            if cv(candidates) > sig_cutoff * 2:
                return candidates


def determine_significant(eics:List[EIC], pct:float) -> Dict[str, List[bool]]:
    """Give a list of EIC namedtuple()'s and a percentage that should be significant, determine which 
    chemicals will be significant."""
    assert pct >= 0 and pct <= 1
    # first, get the chem names
    names = set()
    _ = list(map(lambda eic: names.add(eic.name), eics))
    num_chems = len(names)
    cutoff = int(pct * num_chems)
    masks = list(map(lambda idx: idx < cutoff, range(num_chems)))
    np.random.shuffle(masks)
    
    return dict(zip(list(names), masks))


def random_bool():
    return bool(np.random.randint(0, 2))


def eics_to_scans(eics):
    data = {"rts": [], "mzs": [], "its": []}

    eics = sorted( eics, key=lambda e: -e.mz(), reverse=True )

    for rt in eics[0].rts():
        data["rts"].append(rt)
        data["mzs"].append([])
        data["its"].append([])
    
    for eic in eics:
        for idx, (mz, it) in enumerate(zip(eic.mzs(), eic.its())):
            data["mzs"][idx].append(mz)
            data["its"][idx].append(it)
    
    scans = []
    
    for idx, (rt, mzs, its) in enumerate(zip(data["rts"], data["mzs"], data["its"])):
        scans.append(
            Scan(
                id=idx,
                mz_array=mzs,
                intensity_array=np.array(its).astype(np.int64),
                rt=rt/60, # to minutes
            )
        )
    
    return scans


def write_info(out):
    # check file contains what kind of spectra
    # has_ms1_spectrum = 1 in self.scans
    # has_msn_spectrum = 1 in self.scans and len(self.scans) > 1
    file_contents = ["profile spectrum"]
    # if has_ms1_spectrum:
    file_contents.append("MS1 spectrum")
    # if has_msn_spectrum:
    out.file_description(file_contents=file_contents, source_files=[])
    out.sample_list(samples=[])
    out.software_list(software_list={"id": "VMS", "version": "1.0.0"})
    out.scan_settings_list(scan_settings=[])
    out.instrument_configuration_list(
        instrument_configurations={"id": "VMS", "component_list": []}
    )
    out.data_processing_list({"id": "VMS"})


def export_scans(scans, pathname):

    with MzMLWriter(open(pathname, "wb"), close=True) as out:
        # Add default controlled vocabularies
        out.controlled_vocabularies()
        # Open the run and spectrum list sections
        write_info(out)
        with out.run(id="my_analysis"):
            spectrum_count = len(
                scans
            )  # + sum([len(products) for _, products in scans])
            with out.spectrum_list(count=spectrum_count):
                for scan in scans:
                    # Write Precursor scan
                    out.write_spectrum(
                        scan.mz_array,
                        scan.intensity_array,
                        scan_start_time=scan.rt,
                        id=scan.id,
                        params=[
                            "MS1 Spectrum",
                            {"ms level": 1},
                            {"total ion current": sum(scan.intensity_array)},
                            {"lowest observed m/z": np.min(scan.mz_array)},
                            {"highest observed m/z": np.max(scan.mz_array)},
                        ],
                    )
                    
                    # Write MSn scans
                    # for prod in products:
                    #    out.write_spectrum(
                    #        prod.mz_array, prod.intensity_array,
                    #        id=prod.id, params=[
                    #            "MSn Spectrum",
                    #            {"ms level": 2},
                    #            {"total ion current": sum(prod.intensity_array)}
                    #         ],
                    #         # Include precursor information
                    #         precursor_information={
                    #            "mz": prod.precursor_mz,
                    #            "intensity": prod.precursor_intensity,
                    #            "charge": prod.precursor_charge,
                    #            "scan_id": prod.precursor_scan_id,
                    #            "activation": ["beam-type collisional dissociation", {"collision energy": 25}],
                    #            "isolation_window": [prod.precursor_mz - 1, prod.precursor_mz, prod.precursor_mz + 1]
                        #         })
