from __future__ import division
from .peptides import get_protein_formula
import math
import IsoSpecPy
from pprint import pprint
import numpy as np
from math import exp
from scipy.stats import norm, uniform, gamma
import random
import heapq
import re
from collections import Counter
import numpy.random as rd
from scipy.ndimage import gaussian_filter
from copy import deepcopy

try:
    xrange
except NameError:
    xrange = range

class Spectrum:
    def __init__(self, formula, threshold=0.001, intensity = 1.0,
                         empty = False, charge=1, adduct=None, label = None):
        ### TODO: zrobic zeby mozna bylo od razu tworzyc widmo na dwa sposoby:
        ### albo przez formule albo przez liste confs. 
        self.label = label
        self.confs = []
        if isinstance(formula, dict):
            formula = ''.join(str(k)+str(v) for k, v in formula.items())
        if label is None and formula != "":
            self.label = formula
        elif label is None:
            self.label = "Unknown"

        if not empty:
            parsed = re.findall('([A-Z][a-z]*)([0-9]*)', formula)
            formula = Counter()
            for e, n in parsed:
                n = int(n) if n else 1
                formula[e] += n
            #formula = [(x[0], 0 if x[1] == '' else int(x[1])) for x in formula]
            #formula = Counter(dict(formula))
            if adduct:
                formula[adduct] += charge
            formula = ''.join(x+str(formula[x]) for x in formula if formula[x] > 0)
            self.isospec = IsoSpecPy.IsoThreshold(formula = formula, threshold
                                                  = threshold, absolute =
                                                  False, get_confs = False)
            self.confs = [(x[0]/abs(charge), intensity*x[1]) for x in
                          zip(self.isospec.masses, self.isospec.probs)]
            # ^^^ even if charge is negative, m/zs should be positive
            self.sort_confs()
            self.formula = formula
        # empty or not charge is must have!
        self.charge = charge
        self.is_this_noise = [False]*len(self.confs)

    @staticmethod
    def new_from_fasta(fasta, threshold=0.001, intensity = 1.0, empty = False,
                       charge=1, label = None):
        return Spectrum(get_protein_formula(fasta), threshold=threshold,
                        intensity=intensity, empty=empty, charge=charge,
                        label=label)

    @staticmethod
    def new_from_csv(filename, delimiter=","):
        spectrum = Spectrum("", empty=True, label=filename)

        with open(filename, "r") as infile:
            header = next(infile)
            for line in infile:
                if line[0] == '#':
                    continue
                line = line.strip()
                line = line.split(delimiter)
                spectrum.confs.append(tuple(map(float, line)))
        spectrum.sort_confs()
        spectrum.merge_confs()
        return spectrum


    @staticmethod
    def new_from_masses_int(masses, intensities):
        ret = Spectrum("", 0.0, empty = True)
        assert len(masses) == len(intensities)
        ret.confs = zip(masses, intensities)
        ret.sort_confs()
        ret.normalize()
        return ret


    @staticmethod
    def new_random(domain = (0.0, 1.0), peaks = 10):
        ret = Spectrum("", 0.0, empty = True)
        ret.confs = []
        for _ in xrange(peaks):
            ret.confs.append((random.uniform(*domain), random.uniform(0.0, 1.0)))
        ret.sort_confs()
        ret.normalize()
        return ret

    def average_mass(self):
        """
        Returns the average mass.
        """
        norm = float(sum(x[1] for x in self.confs))
        return sum(x[0]*x[1]/norm for x in self.confs)

    def copy(self):
        isospec = self.isospec
        self.isospec = None
        ret = deepcopy(self)
        ret.isospec = isospec
        self.isospec = isospec
        return ret

    def get_modal_peak(self):
        """
        Returns the peak with the highest intensity.
        """
        return max(self.confs, key=lambda x: x[1])

    def sort_confs(self):
        self.confs.sort(key = lambda x: x[0])

    def merge_confs(self):
        cmass = self.confs[0][0]
        cprob = 0.0
        ret = []
        for mass, prob in self.confs + [(-1, 0)]:
            if mass != cmass:
                ret.append((cmass, cprob))
                cmass = mass
                cprob = 0.0
            cprob += prob
        self.confs = [x for x in ret if x[1] > 1e-12]

    def set_confs(self, confs):
        self.confs = confs
        self.sort_confs()
        self.merge_confs()

    def __add__(self, other):
        res = Spectrum("", 0.0, empty=True)
        res.confs = self.confs + other.confs
        res.sort_confs()
        res.merge_confs()
        res.label = self.label + ' + ' + other.label
        return res

    def __mul__(self, number):
        res = Spectrum("", 0.0, empty=True)
        res.confs = [(x[0], number*x[1]) for x in self.confs]
        res.sort_confs()
        res.merge_confs()
        res.label = self.label
        return res

    def __len__(self):
        return len(self.confs)

    @staticmethod
    def ScalarProduct(spectra, weights):
        ret = Spectrum("", 0.0, empty = True)
        Q = [(spectra[i].confs[0], i, 0) for i in xrange(len(spectra))]
        heapq.heapify(Q)
        while Q != []:
            conf, spectre_no, conf_idx = heapq.heappop(Q)
            ret.confs.append((conf[0], conf[1] * weights[spectre_no]))
            conf_idx += 1
            if conf_idx < len(spectra[spectre_no]):
                heapq.heappush(Q, (spectra[spectre_no].confs[conf_idx], spectre_no, conf_idx))
        ret.merge_confs()
        return ret

    def normalize(self, target_value = 1.0):
        x = target_value/math.fsum(v[1] for v in self.confs)
        self.confs = [(v[0], v[1]*x) for v in self.confs]

    def WSDistanceMoves(self, other):
        try:
            ii = 0
            leftoverprob = other.confs[0][1]
            for mass, prob in self.confs:
                while leftoverprob <= prob:
                    yield (other.confs[ii][0], mass, leftoverprob)
                    prob -= leftoverprob
                    ii += 1
                    leftoverprob = other.confs[ii][1]
                yield (other.confs[ii][0], mass, prob)
                leftoverprob -= prob
        except IndexError:
            raise StopIteration()

    def WSDistance(self, other):
        if not np.isclose(sum(x[1] for x in self.confs), 1.):
            raise ValueError('Self is not normalized.')
        if not np.isclose(sum(x[1] for x in other.confs), 1.):
            raise ValueError('Other is not normalized.')
        return math.fsum(abs(x[0]-x[1])*x[2] for x in self.WSDistanceMoves(other))

    def explained_intensity(self,other):
        """
        Returns the amount of mutual intensity between self and other,
        defined as sum of minima of intensities, mass-wise.
        """
        e = 0
        for i in xrange(len(self.confs)):
            e += min(self.confs[i][1],other.confs[i][1])
        return e

    def bin_to_nominal(self, nb_of_digits=0):
        """
        Rounds mass values to a given number of decimal digits.
        Works in situ, returns None.
        The masses are multiplied by the charge prior to rounding,
        and divided by the charge again after rounding.
        The default nb_of_digits is zero, meaning that the m/z values
        will correspond to nominal mass of peaks.
        """
        xcoord, ycoord = zip(*self.confs)
        xcoord = map(lambda x: x*self.charge, xcoord)
        xcoord = (round(x, nb_of_digits) for x in xcoord)
        xcoord = map(lambda x: x/self.charge, xcoord)
        self.confs = list(zip(xcoord, ycoord))
        self.sort_confs()
        self.merge_confs()

    def coarse_bin(self, nb_of_digits):
        """
        Rounds the m/z to a given number of decimal digits
        """
        self.confs = [(round(x[0], nb_of_digits), x[1]) for x in self.confs]
        self.merge_confs()

    def add_chemical_noise(self, nb_of_noise_peaks, noise_fraction):
        """
        Adds additional peaks with uniform distribution in the m/z domain
        and gamma distribution in the intensity domain. The spectrum does NOT need
        to be normalized. Accordingly, the method does not normalize the intensity afterwards!
        noise_fraction controls the amount of noise signal in the spectrum.
        nb_of_noise_peaks controls the number of peaks added.

        Return: list
            A boolean list indicating if a given peak corresponds to noise
        """
        span = min(x[0] for x in self.confs), max(x[0] for x in self.confs)
        span_increase = 1.2  # increase the mass range by a factor of 1.2
        span = [span_increase*x + (1-span_increase)*sum(span)/2 for x in span]
        noisex = uniform.rvs(loc=span[0], scale=span[1]-span[0], size=nb_of_noise_peaks)
        noisey = gamma.rvs(a=2, scale=2, size=nb_of_noise_peaks)
        noisey /= sum(noisey)
        signal = sum(x[1] for x in self.confs)
        noisey *=  signal*noise_fraction /(1-noise_fraction)
        noise = [(x, y) for x,y in zip(noisex, noisey)]
        self.confs += noise
        self.sort_confs()
        self.merge_confs()
        return [True if mz in noisex else False for mz in [x[0] for x in self.confs]]

    def add_gaussian_noise(self, sd):
        """
        Adds gaussian noise to each peak, simulating
        electronic noise.
        """
        noised = rd.normal([y for x,y in self.confs], sd)
        # noised = noised - min(noised)
        self.confs = [(x[0], y) for x, y in zip(self.confs, noised) if y > 0]
    
    def distort_intensity(self, N, gain, sd):
        """
        Distorts the intensity measurement in a mutiplicative noise model - i.e.
        assumes that each ion yields a random amount of signal.
        Assumes the molecule is composed of one element, so it's
        an approximation for normal molecules.
        The resulting spectrum is not normalized.
        Works in situ (modifies self).
        N: int
            number of ions
        gain: float
            mean amount of signal of one ion
        sd: float
            standard deviation of one ion's signal

        Return: np.array
            The applied deviations.
        """
        p = np.array([x[1] for x in self.confs])
        assert np.isclose(sum(p), 1), 'Spectrum needs to be normalized prior to distortion'
        X = [(x[0], N*gain*x[1]) for x in self.confs]  # average signal
        peakSD = np.sqrt(N*sd**2*p + N*gain**2*p*(1-p))
        U = rd.normal(0, 1, len(X))
        U *= peakSD
        X = [(x[0], max(x[1] + u, 0.)) for x, u in zip(X, U)]
        self.confs = X
        return U

    def distort_mz(self, mean, sd):
        """
        Distorts the m/z measurement by a normally distributed
        random variable with given mean and standard deviation.
        Use non-zero mean to approximate calibration error.
        Returns the applied shift.
        """
        N = rd.normal(mean, sd, len(self.confs))
        self.confs = [(x[0] + u, x[1]) for x, u in zip(self.confs, N)]
        self.sort_confs()
        self.merge_confs()
        return N

    @staticmethod
    def sample_multinomial(reference, N, gain, sd):
        """
        Samples a spectrum of N molecules based on peak probabilities
        from the reference spectrum. Simulates both isotope composition
        and amplifier randomness.
        The returned spectrum is not normalized.
        N: int
            number of ions in the spectrum
        gain: float
            The gain of the amplifier, i.e. average signal from one ion
        sd: float
            Standard deviation of one ion's signal
        """
        p = [x[1] for x in reference.confs]
        assert np.isclose(sum(p), 1), 'Spectrum needs to be normalized prior to sampling'
        U = rd.multinomial(N, p)
        U = rd.normal(U*gain, np.sqrt(U*sd**2))
        retSp = Spectrum('', empty=True, label='Sampled ' + reference.label)
        retSp.set_confs([(x[0], max(u, 0.)) for x, u in zip(reference.confs, U)])
        return retSp

    def find_peaks(self):
        """
        Returns a list of local maxima.
        Each maximum is reported as a tuple of m/z and intensity.
        The last and final configuration is never reported as a maximum.
        Note that this function should only be applied to profile spectra - the result
        does not make sense for centroided spectrum.
        Applying a gaussian or Savitzky-Golay filter prior to peak picking
        is advised, in order to avoid detection of noise.
        """
        diffs = [n[1]-p[1] for n,p in zip(self.confs[1:], self.confs[:-1])]
        is_max = [nd <0 and pd > 0 for nd, pd in zip(diffs[1:], diffs[:-1])]
        peaks = [x for x, p in zip(self.confs[1:-1], is_max) if p]
        return peaks
        

    def fuzzify_peaks(self, sd, step):
        """
        Applies a gaussian filter to the peaks, effectively broadening them
        and simulating low resolution. Works in place, modifying self.
        The parameter step gives the distance between samples in m/z axis.
        Note that after the filtering, the area below curve is equal to 1,
        instead of the sum of 'peak' intensities!
        """
        new_mass = np.arange(self.confs[0][0] - 4*sd, self.confs[-1][0] + 4*sd, step)
        new_intensity = np.zeros(len(new_mass))
        lb = new_mass[0]
        for x, y in self.confs:
            xrnb = int((x-lb)//step)  # x's index in new_mass
            xr = lb + step*xrnb
            lnb = int((xr-x+4*sd)//step)   # nb of steps left of x to add gauss
            xlb = xr - step*lnb
            xrb = xr + step*lnb
            xv = np.array([xlb + i*step for i in range(2*lnb + 2)])
            nv = y*norm.pdf(xv, x, sd)
            new_intensity[(xrnb-lnb):(xrnb+lnb+2)] += nv
        self.confs = [(x, y) for x, y in zip(new_mass, new_intensity)]

    def cut_smallest_peaks(self, removed_proportion=0.001):
        """
        Removes smallest peaks until the total removed intensity amounts
        to the given proportion of the total ion current in the spectrum.
        """
        self.confs.sort(key = lambda x: x[1], reverse=True)
        threshold  = removed_proportion*sum(x[1] for x in self.confs)
        removed = 0
        while len(self.confs)>0 and removed + self.confs[-1][1] <= threshold:
            removed += self.confs.pop()[1]
        self.confs.sort(key = lambda x: x[0])

    def filter_peaks(self, list_of_others, margin):
        """
        Removes peaks which do not match any isotopic envelope from
        the list_of_others, with a given mass margin for matching.
        Works in situ (modifies self).
        Assumes that list_of_others contains proper Spectrum objects
        (i.e. with default sorting of confs).
        _____
        Parameters:
            list_of_others: list
                A list of Spectrum objects.
            margin: float
                The isotopic envelopes of target spectra are widened by this margin.
        _____
        Returns: None
        """
        bounds = [(s.confs[0][0] - margin, s.confs[-1][0] + margin) for s in list_of_others]
        bounds.sort(key = lambda x: x[0])  # sort by lower bound
        merged_bounds = []
        c_low, c_up = bounds[0]
        for b in bounds:
            if b[0] <= c:
                pass # to be finished
        

    @staticmethod
    def filter_against_theoretical(experimental, theoreticals, margin=0.15):
        """
        Remove trash empirical spectra fragments and leave only interesting.

        Removes from experimental spectrum fragments outside theoretical peaks
        +- margin.

        experimental: empirical spectrum,
        theoreticals: one instance of theoretical or iterable of instances of
                    theoretical spectra,
        margin: m/z radius within empirical spectrum should be left.
        """
        try:
            th_confs = []
            for theoretical_spectrum in theoreticals:
                th_confs.extend(theoretical_spectrum.confs)
            theoretical = Spectrum("", empty=True)
            theoretical.confs = th_confs
            theoretical.sort_confs()
            theoretical.merge_confs()
        except TypeError:
            theoretical = theoreticals
        experimental_confs = experimental.confs
        theoretical_masses = [i[0] for i in theoretical.confs]

        result_confs = []
        index = 0
        for mz, abund in experimental_confs:
            while (index + 1 < len(theoretical_masses) and
                   theoretical_masses[index + 1] < mz):
                index += 1
            if abs(mz - theoretical_masses[index]) <= margin or (
                    index + 1 < len(theoretical_masses) and
                    abs(mz - theoretical_masses[index + 1]) <= margin):
                result_confs.append((mz, abund))
        new_spectrum = Spectrum("", empty=True,
                                label=experimental.label if
                                hasattr(experimental, 'label') else None)
        new_spectrum.confs = result_confs
        return new_spectrum

    def plot(self, show = True, profile=False, **plot_kwargs):
        import matplotlib.pyplot as plt
        if show:
            plt.clf()
        if profile:
            plt.plot([x[0] for x in self.confs], [x[1] for x in self.confs], linestyle='-', label=self.label, **plot_kwargs)
        else:
            plt.vlines([x[0] for x in self.confs], [0], [x[1] for x in self.confs], label = self.label, linewidth=1, **plot_kwargs)
        if show:
            plt.show()

    @staticmethod
    def plot_all(spectra, show=True, profile=False, cmap=None):
        import matplotlib.pyplot as plt
        import matplotlib.cm as cm
        import numpy as np
        if not cmap:
            colors = cm.rainbow(np.linspace(0, 1, len(spectra)))
            colors =  [[0, 0, 0, 0.8]] + [list(x[:3]) + [0.6] for x in colors]
        else:
            try:
                colors = [[0, 0, 0, 0.8]] + [cmap(x, alpha=1) for x in range(len(spectra))]
            except:
                colors = cmap
        if show:
            plt.clf()
        i = 0
        for spectre in spectra:
            spectre.plot(show = False, profile=profile, color = colors[i])
            i += 1
        #plt.legend(loc=9, bbox_to_anchor=(0.5, -0.1), ncol=len(spectra))  # legend below plot
        plt.legend(loc=0, ncol=1)
        if show: plt.show()






