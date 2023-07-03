import os 
from glob import glob
from pprint import pprint
from pyteomics import mzxml
from masserstein import Spectrum
import pandas as pd
from copy import deepcopy
import numpy as np
from tqdm import tqdm 
from collections import Counter, OrderedDict
from masserstein.model_selection import get_composition
import matplotlib.pyplot as plt
import functools
from itertools import cycle, combinations_with_replacement, combinations
from typing import Dict, List


def load_mzxml(path, huge_tree=False):
  """
  Loads spectrum from mzXML file.
  _____
    Parameters:
    path: str
        Path to mzXML file.
    huge_tree: bool
        Whether we need to pass huge_tree parameter further to mzmxl.read().
        This option is passed to the lxml parser and defines whether security checks for XML tree depth and node size should be disabled.
        Default is False. Enable this option for trusted files to avoid XMLSyntaxError exceptions
  """
  if huge_tree==True:
    data = list(mzxml.read(path, huge_tree=True))
  else:
    data = list(mzxml.read(path))
  assert len(data) == 1
  data = data[0]  
  
  mz = data['m/z array']
  i = data['intensity array']
  
  nonzero = i > 0
  
  confs = list(zip(mz, i))
  s = Spectrum('', empty=True, label=os.path.split(path)[-1])
  print(f"{s.label:16} loaded: {len(confs)} ({100*nonzero.mean():.2f}% non-zero) datapoints in {path}")
  s.set_confs(confs)
  return s

def restrict(spectrum, mz_min, mz_max):
  mz, i = np.array(spectrum.confs).T
  ix = (mz_min <= mz) & (mz <= mz_max)
  spectrum = deepcopy(spectrum)
  spectrum.confs = list(zip(mz[ix], i[ix]))
  return spectrum


def correct_baseline(spectrum, base=1000):
  mz, i = np.array(spectrum.confs).T
  i -= (i.min() + base)
  i[np.where(i<=0)] = abs(0.001*i.mean())
  i += 0.001*i.mean()
  spectrum = deepcopy(spectrum)
  spectrum.confs = list(zip(mz, i))
  return spectrum

def centroided(spectrum, max_width=1, peak_height_fraction=0.5):
  """[0] - only centroided peak intensity needed"""
  centroided, _ = spectrum.centroid(max_width=max_width, peak_height_fraction=peak_height_fraction)
  return Spectrum(confs=centroided, label=spectrum.label)

def reduce(s):
  mz = np.array(s.confs)[:, 0]
  gb = (mz - (mz%1).mean() - .5 )//1
  gb = pd.DataFrame(s.confs).groupby(gb)
  mz = gb.mean()[0]
  i  = gb.sum()[1]
  confs = list(zip(mz.values, i.values))
  s = deepcopy(s)
  s.confs = confs
  return s

def remove_low_signal(spectrum, signal_proportion = 0.001):
    signal_thr = signal_proportion * spectrum.get_modal_peak()[1]
    filtered_confs = []
    for mz, i in spectrum.confs:
        if i > signal_thr:
            filtered_confs.append((mz, i))
    new_spectrum = deepcopy(spectrum)
    new_spectrum.confs = filtered_confs
    return new_spectrum

def _normalize(self):
  self = deepcopy(self)
  self.normalize()
  return self

def _gaussian_smoothing(self, sd=0.01, new_mz=0.01):
  self = deepcopy(self)
  self.gaussian_smoothing(sd, new_mz)
  return self

# Spectrum._normalize = _normalize
# Spectrum._gaussian_smoothing = _gaussian_smoothing

def plot(empirical_spectrum, query_spectra, proportions, threshold=1e-2, legend=True):
  colors = cycle(['royalblue']+list('grcmyk'))
  model = zip(proportions, query_spectra)
  model = sorted(model, reverse=True, key=lambda p_q: p_q[0])

  res = []
  for p, s in model:
    if p == 0:
      break
    (p*reduce(s)).plot(show=False, linewidth=3, color=next(colors), alpha=0.9)
    print(f"{100*p:.2f}%: {s.label}")
    res.append(s)
    if p < threshold:
      break

  empirical_spectrum.plot(show=False)
  if legend==True:
    plt.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')
  plt.ylabel('Signal intensity')
  plt.xlabel('m/z')

  return res

def plot_sum_regressed(empirical_spectrum, query_spectra, proportions, threshold=1e-2, legend=True):

  colors = cycle(['royalblue']+list('grcmyk'))
  model = zip(proportions, query_spectra)
  model = sorted(model, reverse=True, key=lambda p_q: p_q[0])
  
  #sum spectra 
  sum_spectra = []
  for p, s in model:
    if p == 0:
      break
    sum_spectra.append(p*reduce(s))
    if p < threshold:
      break

  #process sum of spectras
  sum_spectra = np.sum(sum_spectra)
  sum_spectra = reduce(sum_spectra)
  sum_spectra.label = "Sum of regressed spectras"

  #plot sum of expected spectra
  empirical_spectrum.plot(show=False)
  sum_spectra.plot(show=False, linewidth=3, color=next(colors), alpha=0.9)
  if legend==True:
    plt.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')
    #plt.legend(loc=0, ncol=1)
  plt.ylabel('Signal intensity')
  plt.xlabel('m/z')

  return sum_spectra

# Generating theorethical spectra
# multiplication 
Counter.__mul__ = lambda self, x: sum([self]*x, Counter())

def formula_from_dict(d):
  return ''.join(f"{k}{v}" for k, v in d.items())

# Example

# bt = Counter(C=36, H=60, S=2)
# bt_16 = Counter(C=40, H=68, S=2)
# tt = Counter(C=6, H=2, S=2)
# dtt = Counter(C=8, H=2, S=3)
# end_groups = dict(
#   Stannyl = Counter(C=3, H=9, Sn=1),
#   Br = Counter(Br=1),
#   H = Counter(H=1),
#   Methyl = Counter(C=1, H=3),
#   Phenyl = Counter(C=6, H=5),
#   # Orthothyl = Counter(C=7, H=7),
# )

# #we assume there will be only one adduct
# adducts = dict(
#     H = Counter(H=1),
#     Na = Counter(Na=1),
#     K = Counter(K=1),
# )

def get_possible_compounds(heavier_monomer:Dict, lighter_monomer:Dict, end_groups:Dict, min_mz, max_mz, max_count_diff, adducts=None , verbose=False):

  possible_compounds=[]
  A, a, = heavier_monomer.items(), 
  B, b = lighter_monomer.items()

  def add_lighter_end_adduct(a_count, start):
    #core polymer
    for b_count in range(start, a_count+max_count_diff+1):
      core = a*a_count + b*b_count
      #endgroups
      for (end_group_name1, end_group1), (end_group_name2, end_group2) in combinations_with_replacement(end_groups.items(), 2):
        if adducts: #adducts
          for adduct_name, adduct in adducts.items():
            f = core + end_group1 + end_group2 + adduct
            if end_group_name1 == end_group_name2:
              label = f"{a_count}{A}+{b_count}{B}+2{end_group_name1}+{adduct_name}"
            else:
              label = f"{a_count}{A}+{b_count}{B}+{end_group_name1}+{end_group_name2}+{adduct_name}"
            #generating spectrum 
            s = Spectrum(formula=formula_from_dict(f), label=label)
            if min_mz <= s.confs[0][0] <= max_mz:
              s.normalize()
              possible_compounds.append(s)
        else: #without adducts
          f = core + end_group1 + end_group2
          if end_group_name1 == end_group_name2:
            label = f"{a_count}{A}+{b_count}{B}+2{end_group_name1}"
          else:
            label = f"{a_count}{A}+{b_count}{B}+{end_group_name1}+{end_group_name2}"
          #generating spectrum 
          s = Spectrum(formula=formula_from_dict(f), label=label)
          if min_mz <= s.confs[0][0] <= max_mz:
            s.normalize()
            possible_compounds.append(s)

  a_count = 0
  start = 1
  add_lighter_end_adduct(a_count, start) # Only lighter monomers: B
  # at least one A
  while True:
    a_count += 1
    if Spectrum(formula=formula_from_dict(a*a_count)).confs[0][0] > max_mz:
      break
    if a_count-max_count_diff >= 0: start = a_count-max_count_diff
    else: start=0
    add_lighter_end_adduct(a_count, start)

  possible_compounds = sorted(possible_compounds, key=lambda s: s.confs[0][0])
  if verbose==True:
    for s in possible_compounds:
      print(f"{s.label:25} {s.formula:30} {s.confs[0][0]}")

  print(f"\n Found {len(possible_compounds)} expected spectras")
  return possible_compounds

# homocoupling
def homocoupling_frequency(df):
  df = df.T
  c = df.columns[0]
  df = df[df[c]>0]
  df = df.reset_index()
  df = df.rename(columns={"index": "polymer"})
  homocoupling = []
  for polymer in df["polymer"]:
    polymer = polymer.split("+")
    a_count = int(polymer[0][:(len(polymer[0]) - 2)]) 
    b_count = int(polymer[1][:(len(polymer[1]) - 2)])
    homocoupling.append(b_count-a_count)
  return homocoupling

def homocoupling_frequency_prob(df):
  df = df.T
  c = df.columns[0]
  df = df[df[c]>0]
  df = df.reset_index()
  df = df.rename(columns={"index": "polymer"})
  homocoupling = {}
  for prob, polymer in zip(df[c], df["polymer"]):
    polymer = polymer.split("+")
    a_count = int(polymer[0][:(len(polymer[0]) - 2)]) 
    b_count = int(polymer[1][:(len(polymer[1]) - 2)])
    diff = b_count-a_count
    if diff in homocoupling: homocoupling[diff] += prob
    else: homocoupling[diff] = prob
  return homocoupling

def monomer_frequency_prob(df):
  df = df.T
  c = df.columns[0]
  df = df[df[c]>0]
  df = df.reset_index()
  df = df.rename(columns={"index": "polymer"})
  a_freq, b_freq = {}, {}
  for prob, polymer in zip(df[c], df["polymer"]):
    polymer = polymer.split("+")
    a_count = int(polymer[0][:(len(polymer[0]) - 2)]) 
    b_count = int(polymer[1][:(len(polymer[1]) - 2)])
    if a_count in a_freq: a_freq[a_count] += prob
    else: a_freq[a_count] = prob
    if b_count in b_freq: b_freq[b_count] += prob
    else: b_freq[b_count] = prob
  return a_freq, b_freq

def estimate_homocoupling(df):
  df = df.T
  c = df.columns[0]
  df = df[df[c]>0]
  df = df.reset_index()
  df = df.rename(columns={"index": "polymer"})
  all_homocoupling = 0
  for prob, polymer in zip(df[c], df["polymer"]):
    polymer = polymer.split("+")
    a_count = int(polymer[0][:(len(polymer[0]) - 2)]) 
    b_count = int(polymer[1][:(len(polymer[1]) - 2)])
    all_homocoupling += prob * abs(b_count-a_count)
  return all_homocoupling

def estimate_minimal_homocoupling(df, a_end_groups=None, b_end_groups=None, other_end_groups=None):

  if not a_end_groups: a_end_groups = ["Br", "H"]
  if not b_end_groups: b_end_groups = ["Stannyl", "Methyl"]
  if not other_end_groups: other_end_groups = ["Phenyl", "Orthothyl"]

  df = df.T
  c = df.columns.item()
  df = df[df[c]>0]
  df = df.reset_index()
  df = df.rename(columns={"index": "polymer"})

  min_homocoupling = []
  min_homocoupling_prob = 0
  for prob, polymer in zip(df[c], df["polymer"]):
      polymer = polymer.split("+")
      a_count = int(polymer[0][:(len(polymer[0]) - 2)]) 
      b_count = int(polymer[1][:(len(polymer[1]) - 2)])
      if len(polymer)==3:
        end1, end2 = polymer[2][1:], polymer[2][1:]
      if len(polymer)==4:
        end1, end2 = polymer[2], polymer[3] 
      # if bt or tt count is 0
      if a_count == 0: 
        min_h = b_count - 1
        min_homocoupling.append(min_h)
        min_homocoupling_prob += prob*min_h
      elif b_count == 0:
        min_h = a_count - 1
        min_homocoupling.append(min_h)
        min_homocoupling_prob += prob*min_h
      else:
        if end1 in other_end_groups or end2 in other_end_groups: 
          if (end1 in other_end_groups and end2 in a_end_groups) or (end2 in other_end_groups and end1 in a_end_groups):
            # at one end bt 
            if (a_count - 1) - b_count >= 0: 
              min_h = (a_count - 1) - b_count
              min_homocoupling.append(min_h)
              min_homocoupling_prob += prob*min_h
            elif b_count - (a_count - 1) == 1:
              min_h = 0
              min_homocoupling.append(min_h)
              min_homocoupling_prob += prob*min_h
            else:
              min_h = b_count - (a_count - 1)
              min_homocoupling.append(min_h)
              min_homocoupling_prob += prob*min_h
          elif (end1 in other_end_groups and end2 in b_end_groups) or (end2 in other_end_groups and end1 in b_end_groups):
            # at one end tt 
            if (b_count - 1) - a_count >= 0: 
              min_h = (b_count - 1) - a_count
              min_homocoupling.append(min_h)
              min_homocoupling_prob += prob*min_h
            elif a_count - (b_count - 1) == 1:
              min_h = 0
              min_homocoupling.append(min_h)
              min_homocoupling_prob += prob*min_h
            else:
              min_h = a_count - (b_count - 1)
              min_homocoupling.append(min_h)
              min_homocoupling_prob += prob*min_h
          else: #if end1 in other_end_groups and end2 in other_end_groups:
            # bt or tt at the end
            if abs(b_count - a_count) > 1:
              min_h = abs(b_count - a_count) - 1
              min_homocoupling.append(min_h)
              min_homocoupling_prob += prob*min_h
            else:
              min_h = 0
              min_homocoupling.append(min_h)
              min_homocoupling_prob += prob*min_h
        if end1 in a_end_groups and end2 in a_end_groups:
          # bt at the ends: a_count - 2
          if (a_count - 2) - b_count >= 0: # still more bts
            min_h = (a_count - 2) - b_count + 1
            min_homocoupling.append(min_h)
            min_homocoupling_prob += prob*min_h
          else: #more tts
            if b_count - (a_count - 2) == 1:
              min_h = 0
              min_homocoupling.append(min_h)
              min_homocoupling_prob += prob*min_h
            else:
              min_h = b_count - (a_count - 2) - 1
              min_homocoupling.append(min_h)
              min_homocoupling_prob += prob*min_h
        if end1 in b_end_groups and end2 in b_end_groups:
          # tt at the ends: b_count - 2
          if (b_count - 2) - a_count >= 0: # still more tts
            min_h = (b_count - 2) - a_count + 1
            min_homocoupling.append(min_h)
            min_homocoupling_prob += prob*min_h
          else: #more bts
            if a_count - (b_count - 2) == 1:
              min_h = 0
              min_homocoupling.append(min_h)
              min_homocoupling_prob += prob*min_h
            else:
              min_h = a_count - (b_count - 2) - 1
              min_homocoupling.append(min_h)   
              min_homocoupling_prob += prob*min_h
        if (end1 in a_end_groups and end2 in b_end_groups) or (end2 in a_end_groups and end1 in b_end_groups):
          # one end bt, other tt
          min_h = abs(a_count - b_count)
          min_homocoupling.append(min_h)
          min_homocoupling_prob += prob*min_h
  
  return min_homocoupling, min_homocoupling_prob, df[c]