import math
import IsoSpecPy
import numpy as np
from scipy.stats import norm, uniform, gamma
import random
import heapq
import re
from collections import Counter
import numpy.random as rd
from scipy.signal import argrelmax
from .peptides import get_protein_formula
from warnings import warn
from copy import deepcopy
from .base_spectrum import BaseSpectrum


class Spectrum(BaseSpectrum):

    def __init__(self, formula='', threshold=0.001, total_prob=None,
                 charge=1, adduct=None, confs=None, label=None, **other):
        """Initialize a Spectrum class.

        Initialization can be done either by simulating a spectrum of an ion
        with a given formula and charge, or setting a peak list.

        The initialized spectrum is not normalised. In order to do this use
        normalize method.

        Parameters
        ----------

        formula: str
            The chemical formula of the molecule. If empty, then `confs`
            cannot be None. If the formula is a valid chemical formula then
            spectrum peaks (`confs`) are simulated.
        threshold: float
            Lower threshold on the intensity of simulated peaks. Used when
            `formula` is not an empty string, ignored when `total_prob` is not
            None.
        total_prob: float
            Lower bound on the total probability of simulated peaks, i.e.
            fraction of all potential signal, which will be simulated. Used
            when `formula` is not an empty string. When not None, then
            `threshold` value is ignored.
        charge: int
            A charge of the ion.
        adduct: str
            The ionizing element. When not None, then formula is updated
            with `charge` number of adduct atoms.
        confs: list
            A list of tuples of mz and intensity. Confs contains peaks of an
            initialized spectrum. If not None, then `formula` needs be an empty
            string.
        label: str
            An optional spectrum label.
        """
        ### TODO2: seprarate subclasses for centroid & profile spectra

        if formula != '' and confs is not None:
            raise ValueError(
                "Formula and confs cannot be set at the same time!")

        # BaseSpectrum.__init__ stores confs (or leaves the spectrum empty).
        BaseSpectrum.__init__(self, confs=confs, label=label, **other)

        self.formula = formula
        self.threshold = threshold
        self.total_prob = total_prob
        self.charge = charge
        self.adduct = adduct

        if label is None:
            self.label = formula
        else:
            self.label = label

        if formula != '':
            self.set_confs(
                self.confs_from_formula(
                    formula, threshold, total_prob, charge, adduct))

    @staticmethod
    def confs_from_formula(formula, threshold=0.001, total_prob=None,
                           charge=1, adduct=None):
        """Simulate and return spectrum peaks for given formula.

        Parameters as in __init__ method. `formula` must be a nonempty string.
        """
        parsed = re.findall('([A-Z][a-z]*)([0-9]*)', formula)
        formula = Counter()
        for e, n in parsed:
            n = int(n) if n else 1
            formula[e] += n
        if adduct:
            formula[adduct] += charge
        assert all(v >= 0 for v in formula.values())
        formula = ''.join(x+str(formula[x]) for x in formula if formula[x])
        if total_prob is not None:
            isospec = IsoSpecPy.IsoTotalProb(formula=formula,
                                             prob_to_cover=total_prob,
                                             get_minimal_pset=True,
                                             get_confs=False)
        else:
            isospec = IsoSpecPy.IsoThreshold(formula=formula,
                                             threshold=threshold,
                                             absolute=False,
                                             get_confs=False)
        confs = [(x[0]/abs(charge), x[1]) for x in
                 zip(isospec.masses, isospec.probs)]
        return confs

    @staticmethod
    def new_from_fasta(fasta, threshold=0.001, total_prob=None, intensity=1.0,
                       empty=False, charge=1, label=None):
        return Spectrum(get_protein_formula(fasta), threshold=threshold,
                        total_prob=total_prob, intensity=intensity,
                        empty=empty, charge=charge, label=label)

    def average_mass(self):
        """
        Returns the average mass.
        """
        norm = float(sum(x[1] for x in self.confs))
        return sum(x[0]*x[1]/norm for x in self.confs)
    
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
        xcoord = list(map(lambda x: x*self.charge, xcoord))
        xcoord = [xcoord[0] + round(x-xcoord[0], nb_of_digits) for x in xcoord]
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

    def resample(self, target_mz, mz_distance_threshold=0.05):
        """
        Returns a resampled spectrum with intensity values approximated
        at points given by a sorted iterable target_mz.
        The approximation is performed by a piecewise linear interpolation
        of the spectrum intensities. The spectrum needs to be in profile mode
        in order for this procedure to work properly.
        The spectrum is interpolated only if two target mz values closest to a
        given target mz are closer than the specified threshold
        This is done in order to interpolate the intensity only within peaks, not between them.
        If the surrounding mz values are further away than the threshold,
        it is assumed that the given target mz corresponds to the background and
        there is no intensity at that point.
        A rule-of-thumb is to set threshold as twice the distance between
        neighboring m/z measurements.
        Large thresholds may lead to non-zero resampled intensity in the background,
        low thresholds might cause bad interpolation due to missing intensity values.
        """
        mz = [mz for mz, intsy in self.confs]
        intsy = [intsy for mz, intsy in self.confs]
        x = target_mz[0]
        for m in target_mz:
            assert m >= x, "The target_mz list is not sorted!"
            x = m
        lenx = len(target_mz)
        lent = len(mz)
        qi = 0  # query (x) index
        ti = 0  # target index - the first index s.t. mz[ti] >= x[qi]
        y = [0.]*lenx  # resampled intensities
        y0, y1 = intsy[0], intsy[0]  # intensities of target spectrum around the point target_mz[qi]
        x0, x1 = mz[0], mz[0]  # mz around the point target_mz[qi]
        # before mz starts, the intensity is zero:
        while target_mz[qi] < mz[0]:
            qi += 1
        # interpolating:
        while ti < lent-1:
            ti += 1
            y0 = y1
            y1 = intsy[ti]
            x0 = x1
            x1 = mz[ti]
            while qi < lenx and target_mz[qi] <= mz[ti]:
                # note: maybe in this case set one of the values to zero to get a better interpolation of edges
                if x1-x0 < mz_distance_threshold:
                    y[qi] = y1 + (target_mz[qi]-x1)*(y0-y1)/(x0-x1)
                qi += 1
        return Spectrum(confs = list(zip(target_mz, y)))


    def fuzzify_peaks(self, sd, step):
        """
        LEGACY FUNCTION. USE SELF.GAUSSIAN_SMOOTING INSTEAD.   
        Applies a gaussian filter to the peaks, effectively broadening them
        and simulating low resolution. Works in place, modifying self.
        The parameter step gives the distance between samples in m/z axis.
        After the filtering, the area below curve (not the sum of intensities!)
        is equal to the sum of the input peak intensities.
        """
        new_mass = np.arange(self.confs[0][0] - 4*sd, self.confs[-1][0] + 4*sd, step)
        A = new_mass[:,np.newaxis] - np.array([m for m,i in self.confs])
        # we don't need to evaluate gaussians to far from their mean,
        # from our perspective 4 standard deviations from the mean is the same
        # as the infinity; this allows to avoid overflow as well:
        A[np.abs(A) > 4*sd] = np.inf
        A **= 2
        A /= (-2*sd**2)
        A = np.exp(A)
        new_intensity = A @ np.array([i for m,i in self.confs])  # matrix multiplication
        new_intensity /= (np.sqrt(2*np.pi)*sd)
        self.set_confs(list(zip(new_mass, new_intensity)))


    def gaussian_smoothing(self, sd=0.01, new_mz=0.01):
        """
        Applies a gaussian filter to the mass spectrum in order to smooth
        it out and decrease the electronic noise.
        Technically, each intensity measurement is replaced by a Gaussian weighted average
        of the neighbouring intensities.  
        As a consequence, the resolution gets decreased.
        Parameter sd (float) controls the width of the gaussian filter.
        Parameter new_mz (float or np.array) is the mass axis of the resulting smoothed spectrum.
        Setting it to float generates an equally spaced mass axis with new_mz being the step length.
        Setting it to np.array sets it as the resulting mass axis.  
        Note that after filtering, the area below curve (not the sum of intensities!)
        is equal to the area of the original spectrum in profile mode,
        or the sum of the input peak intensities in centroid mode.
        """
        if isinstance(new_mz, float):
            new_mz = np.arange(self.confs[0][0] - 4*sd, self.confs[-1][0] + 4*sd, new_mz)
        assert np.all(new_mz[1:] >= new_mz[:-1]), 'The new mz axis needs to be sorted!'
        smooth_intensity = np.zeros(new_mz.shape)
        for mz, intsy in self.confs:
            # smooth_intensity += intsy*np.exp(-(mz - new_mz)**2)**(1/(2*sd**2))
            lpid, rpid = np.searchsorted(new_mz, (mz - 4*sd, mz + 4*sd))
            peak_mz = new_mz[lpid:rpid]
            smooth_intensity[lpid:rpid] += intsy*np.exp(-(mz - peak_mz)**2)**(1/(2*sd**2))
        smooth_intensity /= np.sqrt(2*np.pi)*sd
        self.set_confs(list(zip(new_mz, smooth_intensity)))

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
            if b[0] <= c_low:
                pass # to be finished


if __name__=="__main__":
    import matplotlib.pyplot as plt
    from copy import deepcopy
    S = Spectrum(formula="C2H5OH", threshold=0.01)

    S.add_chemical_noise(4, 0.2)
    S.plot()

    sd = 0.01
    C = deepcopy(S)
    C = C*(np.sqrt(2*np.pi)*sd)**-1
    S1 = deepcopy(S)
    S.gaussian_smoothing(0.01,  0.001)
    S1.fuzzify_peaks(0.01, 0.001)
    S.plot(profile=True, show=False)
    S1.plot(profile=True, show=False)
    C.plot(show=False)
    plt.show()

    T = Spectrum(confs=[(1, 1)])
    T.gaussian_smoothing(0.01, np.array([0.96, 0.97, 0.98, 0.99, 1, 1.01, 1.02]))
    T.plot(profile=True, show=False)
    T = Spectrum(confs=[(1, 1)])
    T.gaussian_smoothing(0.01, 0.001)
    T.plot(profile=True)

    target_mz = np.linspace(45, 56, num=100)
    R = S.resample(target_mz)
    plt.subplot(221)
    S.plot(show=False, profile=True)
    plt.subplot(222)
    R.plot(show=False, profile=True)
    plt.subplot(223)
    S.plot(show=False, profile=True)
    plt.plot([mz for mz, intsy in S.confs],
                   [intsy for mz, intsy in S.confs],
                   'r.')
    plt.subplot(224)
    R.plot(show=False, profile=True)
    plt.plot([mz for mz, intsy in R.confs],
                   [intsy for mz, intsy in R.confs],
                   'r.')
    plt.tight_layout()
    plt.show()
