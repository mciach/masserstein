import os 
from pyteomics import mzxml
from spectrum import Spectrum
import pandas as pd
from copy import deepcopy
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.lines import Line2D
from itertools import cycle, combinations_with_replacement
from warnings import warn
from typing import Dict, List, Tuple

############################################################################################################################################

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
  if huge_tree==True: data = list(mzxml.read(path, huge_tree=True))
  else: data = list(mzxml.read(path))
  
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

def restrict(spectrum:Spectrum, mz_min:float, mz_max:float):
  mz, i = np.array(spectrum.confs).T
  ix = (mz_min <= mz) & (mz <= mz_max)
  spectrum = deepcopy(spectrum)
  spectrum.confs = list(zip(mz[ix], i[ix]))
  return spectrum

def correct_baseline(spectrum:Spectrum, base=1000):
  mz, i = np.array(spectrum.confs).T
  i -= (i.min() + base)
  i[np.where(i<=0)] = abs(0.001*i.mean())
  i += 0.001*i.mean()
  spectrum = deepcopy(spectrum)
  spectrum.confs = list(zip(mz, i))
  return spectrum

def centroided(spectrum:Spectrum, max_width=1, peak_height_fraction=0.5):
  """[0] - only centroided peak intensity needed"""
  centroided, _ = spectrum.centroid(max_width=max_width, peak_height_fraction=peak_height_fraction)
  return Spectrum(confs=centroided, label=spectrum.label)

def reduce(spectrum:Spectrum):
  mz = np.array(spectrum.confs)[:, 0]
  gb = (mz - (mz%1).mean() - .5 )//1
  gb = pd.DataFrame(s.confs).groupby(gb)
  mz = gb.mean()[0]
  i  = gb.sum()[1]
  confs = list(zip(mz.values, i.values))
  spectrum = deepcopy(spectrum)
  spectrum.confs = confs
  return spectrum

def remove_low_signal(spectrum:Spectrum, signal_proportion = 0.001):
    signal_thr = signal_proportion * spectrum.get_modal_peak()[1]
    filtered_confs = []
    for mz, i in spectrum.confs:
        if i > signal_thr:
            filtered_confs.append((mz, i))
    new_spectrum = deepcopy(spectrum)
    new_spectrum.confs = filtered_confs
    return new_spectrum

############################################################################################################################################
#plots
def plot(empirical_spectrum:Spectrum, query_spectra:List[Spectrum], proportions:List[float], threshold=1e-2, legend=True, verbose=True):
  
  colors = cycle(['royalblue']+list('grcmy')+['orange', 'fuchsia', 'b'])
  model = zip(proportions, query_spectra)
  model = sorted(model, reverse=True, key=lambda p_q: p_q[0])

  empirical_spectrum.plot(color= "k", linewidth=2, linestyle="-", alpha=0.7, show=False) #experimental - wider and black

  for p, s in model:
    if p < threshold or p == 0: break
    (p*reduce(s)).plot(color=next(colors), linewidth=2, linestyle=(0, (2, 1, 2, 1)), alpha=0.9, show=False)
    if verbose: print(f"{100*p:.2f}%: {s.label}")

  if legend==True:
    plt.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')
  plt.ylabel('Signal intensity')
  plt.xlabel('m/z')

def plot_sum_regressed(empirical_spectrum:Spectrum, query_spectra:List[Spectrum], proportions:List[float], threshold=1e-2, legend=True):

  model = zip(proportions, query_spectra)
  model = sorted(model, reverse=True, key=lambda p_q: p_q[0])
  
  #sum spectra 
  sum_spectra = []
  for p, s in model:
    if p < threshold or p == 0: break
    sum_spectra.append(p*reduce(s))

  #process sum of spectra
  sum_spectra = np.sum(sum_spectra)
  sum_spectra = reduce(sum_spectra)
  sum_spectra.label = "Sum of regressed spectra"

  #plot sum of expected spectra
  empirical_spectrum.plot(color= "k", linewidth=2, linestyle="-", alpha=0.7, show=False) #experimental - wider (or not) black
  sum_spectra.plot(color = "dodgerblue", linewidth=2, linestyle=(0, (2, 1, 2, 1)), alpha=0.9, show=False) # theorethical - thin, dashed, blue
  if legend==True:
    plt.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')
    #plt.legend(loc=0, ncol=1)
  plt.ylabel('Signal intensity')
  plt.xlabel('m/z')

############################################################################################################################################

# Generating theorethical spectra
  
class MCounter(Counter):
    """This is a slight extention of the ``Collections.Counter`` class
    to also allow multiplication with integers."""

    def __mul__(self, other):
      if not isinstance(other, int):
          raise TypeError("Non-int factor")
      return MCounter({k: other * v for k, v in self.items()})

def formula_from_dict(d):
  return ''.join(f"{k}{v}" for k, v in d.items())

def get_possible_compounds(heavier_monomer:Tuple, lighter_monomer:Tuple, end_groups:Dict, min_mz:float, max_mz:float, max_count_diff:int, adducts=None, verbose=False):

  """
  _____
  Parameters:
    heavier_monomer: Tuple:(str, Counter Object)
        list of reference spectra (masserstein.Spectrum objects)
    end_groups_costs: Dict
        dictionary, where keys are end group configurations 
        in the same convention as generated by get_possible_compounds function,
        so for example "2Phenyl", "H+Phenyl", 
        while values are costs of this end groups configuration. 
  _____
  Returns:
    possible_compounds: List[Spectrum]
      list with all reference spectra (masserstein.Spectrum objects) of polymers comprised of given monomers and end groups (and adducts (optional)), 
      that can be present in given interval of experimental spectrum.
  """

  possible_compounds=[]
  A, a, = heavier_monomer 
  B, b = lighter_monomer

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
            if min_mz <= s.confs[0][0] <= max_mz: #whether theorethical spectrum is in interval of interest
              s.normalize()
              possible_compounds.append(s)
            # if min_mz <= s.confs[0][0] and s.confs[-1][0] <= max_mz: #whether theorethical spectrum is in interval of interest
            #   s.normalize()
            #   possible_compounds.append(s)
        else: #without adducts
          f = core + end_group1 + end_group2
          if end_group_name1 == end_group_name2:
            label = f"{a_count}{A}+{b_count}{B}+2{end_group_name1}"
          else:
            label = f"{a_count}{A}+{b_count}{B}+{end_group_name1}+{end_group_name2}"
          #generating spectrum 
          s = Spectrum(formula=formula_from_dict(f), label=label)
          if min_mz <= s.confs[0][0] <= max_mz: #whether theorethical spectrum is in interval of interest
            s.normalize()
            possible_compounds.append(s)
          # if min_mz <= s.confs[0][0] and s.confs[-1][0] <= max_mz: #whether theorethical spectrum is in interval of interest
          #   s.normalize()
          #   possible_compounds.append(s)

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

  print(f"\n Found {len(possible_compounds)} expected spectra")
  return possible_compounds

############################################################################################################################################

def generate_costs_by_end_group(reference_spectra: List[Spectrum], end_groups_costs:Dict) -> List[float]:
    """
    Generates list with costs of each spectum from given lists of reference spectra
    based on provided costs of end groups. If there are labels of spectra missing it raises exception.
    If labels are dont follow naming convention from get_possible_compounds, 
    this function return costs = 0 or other incorrect costs for these spectra.
    _____
    Parameters:
    refence_spectra: List[Spectrum]
        list of reference spectra (masserstein.Spectrum objects)
    end_groups_costs: Dict
        dictionary, where keys are end group configurations 
        in the same convention as generated by get_possible_compounds function,
        so for example "2Phenyl", "H+Phenyl", 
        while values are costs of this end groups configuration. 
    _____
    Returns:
    costs: List[float]
        list of costs values corresponding to given list of reference spectra.
    """
    costs = []
    incorrect_labels = False
    for spectrum in reference_spectra:
      if spectrum.label: label = spectrum.label.split("+")
      else: raise Exception("""There is a spectrum without label. 
                            Remember reference spectra must be labeled using the same convention as produces by get_possible_compounds function.
                            """)
      if len(label)==3:
        costs.append(end_groups_costs[label[-1]])
      elif len(label)==4:
        costs.append(end_groups_costs[f"{label[-2]}+{label[-1]}"])
      elif len(label)==1:
        costs.append(0)
        incorrect_labels = True
      else:
        costs.append(0)

    if incorrect_labels:
      warn("""Some labels don't follow required naming convention, thus they were assign costs = 0.
      Check whether label naming convention follows the one from polymers.get_possible_compounds function.
      """)
    return costs

############################################################################################################################################

# Average cost of 2 phenyls
def average_2Phenyl_cost(heavier_monomer:Tuple, lighter_monomer:Tuple, end_groups:Dict, min_mz:float, max_mz:float, max_count_diff:int, verbose=False):
    """Helper function to estimate average cost of polymers with 2 Phenyl endgroups based on the average Wasserstain distance 
    between mBT+(n+1)TT+H+Methyl and mBT+nTT+2Phenyl spectra.
    _____
        Parameters:
    _____
        Returns:
    """
    
    A, a, = heavier_monomer 
    B, b = lighter_monomer

    phenyl_2_costs = []
    reference_spectra = []
    a_count = 0
    while True:
        a_count += 1
        if Spectrum(formula=formula_from_dict(a*a_count)).confs[0][0] > max_mz:
            break
        for b_count in range(1, a_count+max_count_diff+1):
            ph2 = Spectrum(formula=formula_from_dict(a*a_count+b*b_count+end_groups["Phenyl"]*2), 
                           label = f"{a_count}{A}+{b_count}{B}+2Phenyl")
            h_me = Spectrum(formula=formula_from_dict(a*a_count+b*(b_count+1)+end_groups["H"]+end_groups["Methyl"]),
                           label = f"{a_count}{A}+{b_count+1}{B}+H+Methyl")
            if (min_mz <= ph2.confs[0][0] <= max_mz) and (min_mz <= h_me.confs[0][0] <= max_mz):
                ph2.normalize()
                h_me.normalize()
                reference_spectra.append((ph2, h_me))
                phenyl_2_costs.append(ph2.WSDistance(h_me)) # Wasserstein disstance as cost
    
    if verbose:         
        for (ph2, h_me) in reference_spectra:
            print(f"{ph2.label:25} {ph2.formula:30} {ph2.confs[0][0]}")
            print(f"{h_me.label:25} {h_me.formula:30} {h_me.confs[0][0]}")
            print("\n")
            
    return np.mean(phenyl_2_costs), reference_spectra

############################################################################################################################################
# Homocoupling measures

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

def monomer_difference_frequency(df):
  df = df.T
  c = df.columns[0]
  df = df[df[c]>0]
  df = df.reset_index()
  df = df.rename(columns={"index": "polymer"})
  monomer_difference = {}
  for prob, polymer in zip(df[c], df["polymer"]):
    polymer = polymer.split("+")
    a_count = int(polymer[0][:(len(polymer[0]) - 2)]) 
    b_count = int(polymer[1][:(len(polymer[1]) - 2)])
    diff = b_count-a_count
    if diff in monomer_difference: monomer_difference[diff] += prob
    else: monomer_difference[diff] = prob
  return monomer_difference

def monomer_frequency(df):
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
  """HC simple"""
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

def homocoupling_proportion(df):
  """HP - the sum of proportions for which |m-n| > 1."""
  df = df.T
  c = df.columns[0]
  df = df[df[c]>0]
  df = df.reset_index()
  df = df.rename(columns={"index": "polymer"})
  hc_proportion = 0
  for prob, polymer in zip(df[c], df["polymer"]):
    polymer = polymer.split("+")
    a_count = int(polymer[0][:(len(polymer[0]) - 2)]) 
    b_count = int(polymer[1][:(len(polymer[1]) - 2)])
    if abs(b_count - a_count) > 1:
      hc_proportion += prob
  return hc_proportion

def estimate_constrained_homocoupling(df, a_end_groups=None, b_end_groups=None, other_end_groups=None):
  """HC constrained - estimates homocoupling level assuming that certain polymer endgroups determine which monomer is present at the end of the chain.
  _____
      Parameters:
  _____
      Returns:
  """
  if not a_end_groups: a_end_groups = ["Br", "Methyl"] #BT end groups
  if not b_end_groups: b_end_groups = ["Stannyl"] #TT end groups
  if not other_end_groups: other_end_groups = ["H", "Phenyl", "Orthothyl"] #non determinating

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

############################################################################################################################################
# Parse and visualize annotations

def load_centroided_spectrum(centroided_spectrum_path:str, spectrum_label:str = ""):
  """
  Creates Spectrum object from csv file where the first line contains m/z values and second contains corresponding intensities.
  _____
      Parameters:
      centroided_spectrum_path: str
          path to the csv file with centroided spectrum, 
          where the first line contains m/z values and second contains corresponding intensities.
      spectrum_label: str
          name of the spectrum, default="".
  _____
      Returns: Spectrum object

  """
  # Load spectra
  spectrum = open(centroided_spectrum_path)
  spectrum_mz = next(spectrum)
  spectrum_mz = [l.strip() for l in spectrum_mz.split(',') if l.strip()]
  spectrum_intsy = next(spectrum)
  spectrum_intsy = [l.strip() for l in spectrum_intsy.split(',') if l.strip()]
  spectrum_confs = list(zip(map(float, spectrum_mz), map(float, spectrum_intsy)))

  polymer_spectrum = Spectrum(confs=spectrum_confs, label=spectrum_label)
  polymer_spectrum.normalize()
  return polymer_spectrum

def parse_annotation_results(annotation_file_path:str, 
                             polymer_info_path:str, 
                             centroided_spectra_paths:List[str], 
                             centroided_spectra_labels:List[str], 
                             expert_annotations=None):
  
  """Function parses annotation results and returns them in the format needed for plotting with ``plot_annotations_hc`` or ``plot_annotations_endgroups``
  _____
      Parameters:
      annotation_file_path: str
          path to the csv file with Masserstein annotation results - estimated proportions of reference spectra (columns) in experimental spectrum (rows),
      polymer_info_path: str
          path to the csv file with names of reference spectra with the smallest mass present in their isothopic envelopes.
      centroided_spectra_paths: List[str]
          list consisting of paths to centroided
  _____
      Returns: Spectrum object
  """
  
  # Parse annotation results
  annot_file = open(annotation_file_path)
  annotated_polymers = next(annot_file)
  annotated_polymers = [l.strip() for l in annotated_polymers.split(',') if l.strip()]
  parsed_annotations = [l.split('+') for l in annotated_polymers]
  endgroups = [frozenset(l[2:]) for l in parsed_annotations]
  bt_count = [int(l[0][:-2]) for l in parsed_annotations]
  tt_count = [int(l[1][:-2]) for l in parsed_annotations]

  proportions_all = annot_file.readlines()
  proportions_all = [l.strip().split(',') for l in proportions_all if l.strip()]
  proportions_all = {l[0]: list(map(float, l[1:])) for l in proportions_all}

  # Encode the engroups as integers:
  all_different_endgroups = list(set(endgroups))
  all_different_endgroups_parsed = ['+'.join(l) for l in all_different_endgroups]
  endgroup_integer_coding = [all_different_endgroups.index(egr) for egr in endgroups]

  # Load polymer info
  polymer_info_file = open(polymer_info_path)
  polymer_names = next(polymer_info_file)
  polymer_names = [l.strip() for l in polymer_names.split(',')]
  polymer_masses = next(polymer_info_file)
  polymer_masses = [float(l.strip()) for l in polymer_masses.split(',')]
  name_to_mass = {n: float(m) for n, m in zip(polymer_names, polymer_masses)}
  masses = [name_to_mass[n] for n in annotated_polymers]

  # Load centroided spectra
  polymer_spectra = {label: load_centroided_spectrum(path, label) for path, label in zip(centroided_spectra_paths, centroided_spectra_labels)}
     
  # Encode HC type as integers
  hc_types = [0 if abs(tt-bt)<2 else (tt-1)-bt if tt>bt else tt-(bt-1) for bt, tt in zip(bt_count, tt_count)] # -1, 0, 1 - are the same category - no evidence of homocoupling
  
  if expert_annotations: # Encode expert annotations
      #Check if in expert annotations the assign proportions == 1 other proportions 0
      expert_proportions = {name:[1 if p in expert_annotations[name] else 0 for p in annotated_polymers] for name in expert_annotations}
      return proportions_all, polymer_spectra, endgroup_integer_coding, all_different_endgroups_parsed, bt_count, tt_count, masses, hc_types, expert_proportions
  else: return proportions_all, polymer_spectra, endgroup_integer_coding, all_different_endgroups_parsed, bt_count, tt_count, masses, hc_types

def plot_annotations_hc(proportions, 
                     polymer_spectrum, 
                     endgroup_integer_coding, 
                     all_different_endgroups_parsed, 
                     bt_count, 
                     tt_count, 
                     masses,
                     hc_types,
                     cmap = 'tab20',
                     proportion_threshold = 0.005,
                     group_width_for_top_peak=25,
                     group_width = 20,
                     vertical_separation = 0.001,
                     figsize = (12, 7),
                     endgroup_alt_names=None,
                     ):
  """  
  1. A vector of proportions `proportions` estimated with masserstein
  2. A `masserstein.Spectrum` object `polymer_spectrum` with the experimental spectrum  
  3. A vector of polymer endgroups coded as integers `endgroup_integer_coding` 
  4. A vector `all_different_endgroups_parsed` containing the names of endgroups, 
    in order corresponding to the encoding in `endgroup_integer_coding`, 
    so that endgroup encoded as 0 corresponds to `all_different_endgroups_parsed[0]`
  5. Vectors `bt_count`, `tt_count`, and `masses` containing the numbers of bt subunits, tt subunits, 
    and the monoisotopic mass for the polymers, in the same order as `endgroup_integer_coding`
  6. Float `proportion_threshold` 
    - polymers with proportions above threshold will be visualized. Default 0.005.
  7. Float `group_width_for_top_peak`
    - distance from the mass of reference polymer within which we search for the closest top peak from experimental spectrum. 
    Default 25.
  8. Float `group_width`
    - polymers within this distance will be treated as a single group and plotted together. Default 20.
  9. Float `vertical_separation`
    - parameter that controls the verical distance between points in the same group. Default 0.001.
  """

  def hc_types_map(x):
      """Takes tt - bt difference and returns string with the description of homocoupling type.
      1.  -1, 0, 1 - are the same category - no evidence of homocoupling
      2. >= 2 TT homocoupling
      3. <= 2 BT homocoupling
      """
      if x <= -2:
        return f"At least {abs(x)} BT-HC"
      elif x >= 2:
        return f"At least {abs(x)} TT-HC"
      else:
        return f"No evidence of HC"

  # encoding type for cm.tab10/cm.tab20, cm.twilight, cm.coolwarm
  if cmap == "twilight":
    cmap = cm.get_cmap(cmap)
    hc_types_coding = np.array(hc_types)*40 + 255
  elif cmap == "coolwarm":
    cmap = cm.get_cmap(cmap)
    arr = np.array(hc_types)
    hc_types_coding = ((arr - arr.min()) * (1/(arr.max() - arr.min()) * 255)).astype('uint8')
  elif cmap == "tab20":
    cmap == cm.get_cmap(cmap)
    all_different_hc_types = list(set(hc_types))
    all_different_hc_types.sort() # bt-hc, no-hc, tt-hc
    hc_types_coding = [all_different_hc_types.index(hc) for hc in hc_types] 
  else:
    print(f"Unrecognized cmap: {cmap}. Available cmaps: twilight, coolwarm, tab20. Selecting default tab20.")
    cmap == cm.get_cmap(cmap)
    all_different_hc_types = list(set(hc_types))
    all_different_hc_types.sort() # bt-hc, no-hc, tt-hc
    hc_types_coding = [all_different_hc_types.index(hc) for hc in hc_types] 

  # Select the information about the polymers to show on the spectrum:
  is_over_threshold = [p > proportion_threshold for p in proportions]
  props_to_plot = [p for p, t in zip(proportions, is_over_threshold) if t]
  endgroup_coding_to_plot = [p for p, t in zip(endgroup_integer_coding, is_over_threshold) if t]
  endgroups_to_plot = [all_different_endgroups_parsed[egr] for egr in endgroup_coding_to_plot]
  endgroups_to_plot = [egr.split('+') for egr in endgroups_to_plot]
  if endgroup_alt_names:
      endgroups_to_plot = [f"2{endgroup_alt_names[egr[0][1:]]}" if len(egr)==1 else f"{endgroup_alt_names[egr[0]]},{endgroup_alt_names[egr[1]]}" for egr in endgroups_to_plot]
  else:
      endgroups_to_plot = [f"2{egr[0][1:]},{egr[0][1:]}" if len(egr)==1 else f"{egr[0]},{egr[1]}" for egr in endgroups_to_plot]
  bt_count_to_plot = [p for p, t in zip(bt_count, is_over_threshold) if t]
  tt_count_to_plot = [p for p, t in zip(tt_count, is_over_threshold) if t]
  masses_to_plot = [p for p, t in zip(masses, is_over_threshold) if t]
  hc_types_to_plot = [p for p, t in zip(hc_types, is_over_threshold) if t]
  hc_types_encoding_to_plot = [p for p, t in zip(hc_types_coding, is_over_threshold) if t]

  # Group polymers with similar masses, and calculate the horizontal and vertical positions of the visualization of bt/tt counts on the spectrum.
  vertical_positions = []
  horizontal_positions = []
  group_index = [-1]
  top_peak_mz = [0]
  for ms, prev_ms in zip(masses_to_plot, [0] + masses_to_plot[:-1]):
      experimental_nbh = [cf for cf in polymer_spectrum.confs if abs(ms-cf[0]) <= group_width_for_top_peak] 
      top_peak = max(experimental_nbh, key=lambda x: x[1]) # finding hifhest peak in the group
      if abs(top_peak_mz[-1] - top_peak[0]) <= group_width:
          vertical_positions.append(vertical_positions[-1] + vertical_separation)
          horizontal_positions.append(horizontal_positions[-1])
          group_index.append(group_index[-1])
          top_peak_mz.append(top_peak_mz[-1])
      else:
          vertical_positions.append(top_peak[1] + vertical_separation)
          horizontal_positions.append(top_peak[0])
          group_index.append(group_index[-1] + 1)
          top_peak_mz.append(top_peak[0])
  group_index = group_index[1:]
  top_peak_mz = top_peak_mz[1:]

  # We'll sort polymers in groups with respect to their estimated proportions.
  all_groups = set(group_index)
  for group in all_groups:
      group_coords = [i for i, g in enumerate(group_index) if g == group]
      assert group_coords[-1] - group_coords[0] == len(group_coords) - 1, 'non-consecutive group'
      props_in_group = [(i, props_to_plot[i]) for i in group_coords]
      sorted_props = sorted(props_in_group, key = lambda x : x[1])
      sorted_positions = sorted(vertical_positions[g] for g in group_coords)
      for (g, p), vp in zip(sorted_props, sorted_positions): 
          vertical_positions[g] = vp

  fig, ax = plt.subplots(figsize=figsize)
  polymer_spectrum.plot(show=False, color='k')
  legend_handles = []
  for bt, tt, h, v, e, p, hc in zip(bt_count_to_plot, 
                                tt_count_to_plot, 
                                horizontal_positions, 
                                vertical_positions,
                                endgroups_to_plot, 
                                props_to_plot,
                                hc_types_encoding_to_plot):
      ax.text(h, v, 
              s=f"{bt}BT+{tt}TT\n{e}",
              c=cmap(hc), 
              rotation=90,
              horizontalalignment='center',
              verticalalignment='top',
              fontsize=7)  # adjust the fonts size as needed

  hc_types_colors = list(set((hc_type, cmap(hc_code)) for hc_type, hc_code in zip(hc_types_to_plot, hc_types_encoding_to_plot)))
  hc_types_colors.sort()
  hc_types_colors = list(map(lambda x: (hc_types_map(x[0]), x[1]), hc_types_colors))

  legend_handles = [Line2D([0],
                          [0], 
                          linewidth=3, 
                          c=hc_color, 
                          label=hc_label) for hc_label, hc_color in hc_types_colors]
  legend1 = ax.legend(handles = legend_handles, 
                      title = 'Homocoupling', 
                      bbox_to_anchor=(1., 1.), 
                      loc='upper right',
                      ncol=2)

  plt.title(f'Numbers of BT / TT subunits for {polymer_spectrum.label} spectrum')

  max_prop = max(vertical_positions)
  ax.set_ylim(0 - 0.01*max_prop, 1.1*max_prop)
  ax.set_xlim(round(polymer_spectrum.confs[0][0])-40, round(polymer_spectrum.confs[-1][0])+20)
  plt.tight_layout()

##########################################################################################################################
##########################################################################################################################

def plot_annotations_endgroups(proportions, 
                     polymer_spectrum, 
                     endgroup_integer_coding, 
                     all_different_endgroups_parsed, 
                     bt_count, 
                     tt_count, 
                     masses,
                     proportion_threshold = 0.005,
                     group_width = 20,
                     vertical_separation = 0.001,
                     figsize=(10,4),
                     ):
  """  
  1. A vector of proportions `proportions` estimated with masserstein
  2. A `masserstein.Spectrum` object `polymer_spectrum` with the experimental spectrum  
  3. A vector of polymer endgroups coded as integers `endgroup_integer_coding` 
  4. A vector `all_different_endgroups_parsed` containing the names of endgroups, 
    in order corresponding to the encoding in `endgroup_integer_coding`, 
    so that endgroup encoded as 0 corresponds to `all_different_endgroups_parsed[0]`
  5. Vectors `bt_count`, `tt_count`, and `masses` containing the numbers of bt subunits, tt subunits, 
    and the monoisotopic mass for the polymers, in the same order as `endgroup_integer_coding`
  6. Float `proportion_threshold` 
    - polymers with proportions above threshold will be visualized. Default 0.005.
  7. Float `group_width`
    - polymers within this distance will be treated as a single group and plotted together. Default 20.
  8. Float `vertical_separation`
    - parameter that controls the verical distance between points in the same group. Default 0.001.
  9. Tuple with the figure's width and height
  """

  # Select the information about the polymers to show on the spectrum:
  is_over_threshold = [p > proportion_threshold for p in proportions]
  props_to_plot = [p for p, t in zip(proportions, is_over_threshold) if t]
  endgroup_coding_to_plot = [p for p, t in zip(endgroup_integer_coding, is_over_threshold) if t]
  endgroups_to_plot = [all_different_endgroups_parsed[egr] for egr in endgroup_coding_to_plot]
  bt_count_to_plot = [p for p, t in zip(bt_count, is_over_threshold) if t]
  tt_count_to_plot = [p for p, t in zip(tt_count, is_over_threshold) if t]
  masses_to_plot = [p for p, t in zip(masses, is_over_threshold) if t]

  # Group polymers with similar masses, and calculate the horizontal and vertical positions of the visualization of bt/tt counts on the spectrum.
  vertical_positions = []
  horizontal_positions = []
  group_index = [0]
  for ms, prev_ms in zip(masses_to_plot, [0]+masses_to_plot[:-1]):
      experimental_nbh = [cf for cf in polymer_spectrum.confs if abs(ms-cf[0]) < group_width]
      top_peak = max(experimental_nbh, key=lambda x: x[1])
      if abs(ms - prev_ms) < group_width:
          vertical_positions.append(vertical_positions[-1] + vertical_separation)
          horizontal_positions.append(horizontal_positions[-1])
          group_index.append(group_index[-1])
      else:
          vertical_positions.append(top_peak[1] + vertical_separation)
          horizontal_positions.append(top_peak[0])
          group_index.append(group_index[-1] + 1)
  group_index = group_index[1:]

  # We'll sort polymers in groups with respect to their estimated proportions.
  all_groups = set(group_index)
  for group in all_groups:
      group_coords = [i for i, g in enumerate(group_index) if g == group]
      assert group_coords[-1] - group_coords[0] == len(group_coords) - 1, 'non-consecutive group'
      props_in_group = [(i, props_to_plot[i]) for i in group_coords]
      sorted_props = sorted(props_in_group, key = lambda x : x[1])
      sorted_positions = sorted(vertical_positions[g] for g in group_coords)
      for (g, p), vp in zip(sorted_props, sorted_positions): 
          vertical_positions[g] = vp
          
  #plt.figure()

  fig, ax = plt.subplots(figsize=figsize)
  polymer_spectrum.plot(show=False, color='k')
  legend_handles = []
  for bt, tt, h, v, e, p in zip(bt_count_to_plot, 
                                tt_count_to_plot, 
                                horizontal_positions, 
                                vertical_positions, 
                                endgroup_coding_to_plot, 
                                props_to_plot):
      ax.text(h, v, s=str(bt) + '/' + str(tt), c=cm.tab20(e), 
              horizontalalignment='center',
              verticalalignment='top',
            fontsize=9)  # adjust the fonts size as needed


  endgroup_colors = set((all_different_endgroups_parsed[e], cm.tab20(e)) for e in endgroup_coding_to_plot)
  legend_handles = [Line2D([0],
                          [0], 
                          linewidth=3, 
                          c=end_color, 
                          label=end_label) for end_label, end_color in endgroup_colors]
  legend1 = ax.legend(handles = legend_handles, 
                      title = 'Endgroups', bbox_to_anchor=(1, 1.), loc='upper left')

  plt.title(f'Numbers of BT / TT subunits for {polymer_spectrum.label} spectrum')

  max_prop = max(vertical_positions)
  ax.set_ylim(0 - 0.01*max_prop, 1.1*max_prop)

  plt.tight_layout()

############################################################################################################################################