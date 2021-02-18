import math
import IsoSpecPy
import IsoSpecPy.Distributions
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

class Spectrum:
    def __init__(self, formula='', threshold=0.001, total_prob=None,
                 charge=1, adduct=None, confs=None, label=None, isospec=None, **other):
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
        isospec: IsoSpecPy.IsoDistribution
            An IsoSpecPy object representing the spectrum. This method takes
            ownership of the passed object, and it should not be used any further.
        """
        ### TODO2: seprarate subclasses for centroid & profile spectra
        self.formula = formula
        self.empty = False

        if label is None:
            self.label = formula
        else:
            self.label = label

        self.charge = charge

        if formula != '' and confs is not None:
            raise ValueError(
                "Formula and confs cannot be set at the same time!")
        elif confs is not None:
            self.set_confs(confs)
        elif formula != '':
            self.set_confs(
                self.confs_from_formula(
                    formula, threshold, total_prob, charge, adduct))
        elif isospec is not None:
            self.set_isospec(isospec)
        else:
            self.empty = True
            self.confs = []

    @property
    def confs(self):
        return list(zip(self._masses, self._probs))

    @confs.setter
    def confs(self, new_confs):
        self.set_masses_probs([nc[0] for nc in new_confs], [nc[1] for nc in new_confs])
        #raise Exception("Changing of the spectrum through the spectrum.confs accessor is hard-deprecated")

    def set_isospec(self, iso_obj):
        self._isospec = iso_obj
        self._isospec.sort_by_mass()
        self._masses = self._isospec.np_masses()
        self._probs = self._isospec.np_probs()

    def set_masses_probs(self, masses, probs):
        self.set_isospec(IsoSpecPy.IsoDistribution(masses = masses, probs = probs))

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

    @staticmethod
    def new_from_csv(filename, delimiter=","):
        spectrum = Spectrum(label=filename)

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
    def new_random(domain=(0.0, 1.0), peaks=10):
        ret = Spectrum()
        confs = []
        for _ in range(peaks):
            confs.append((random.uniform(*domain), random.uniform(0.0, 1.0)))
        ret.set_confs(confs)
        return ret

    def average_mass(self):
        """
        Returns the average mass.
        """
        return self._isospec.empiric_average_mass()

    # def copy(self):
    #     isospec = self.isospec
    #     self.isospec = None
    #     ret = deepcopy(self)
    #     ret.isospec = isospec
    #     self.isospec = isospec
    #     return ret

    def get_modal_peak(self):
        """
        Returns the peak with the highest intensity.
        """
        idx = np.argmax(self._probs)
        return (self._masses[idx], self._probs[idx])

    def sort_confs(self):
        """
        Sorts configurations by their mass.
        """
        self._isospec.sort_by_mass()

    def merge_confs(self):
        """
        Merges configurations with an identical mass, summing their intensities.
        """
        cmass = self._masses[0]
        cprob = 0.0
        ret = []
        for mass, prob in self.confs + [(-1, 0)]:
            if mass != cmass:
                ret.append((cmass, cprob))
                cmass = mass
                cprob = 0.0
            cprob += prob
        ### TODO3: for profile spectra, set a margin of max. 5 zero intensities
        ### around any observed intensity to preserve peak shape
        ### For centroid spectra, remove all zero intensities.
        #self.confs = [x for x in ret if x[1] > 1e-12]
        self.confs = ret

    def set_confs(self, confs):
        self.confs = confs
        self.merge_confs()

    def __add__(self, other):
        res = Spectrum(isospec=self._isospec + other._isospec)
        res.sort_confs()
        res.merge_confs()
        res.label = self.label + ' + ' + other.label
        return res

    def __mul__(self, number):
        new_iso = IsoSpecPy.IsoDistribution(masses = self._masses, probs = self._probs) # for lack of a straight copy() method...
        new_iso.scale(number)
        res = Spectrum(isospec = new_iso)
        res.label = self.label
        return res

    def __rmul__(self, number):
        # Here * is commutative
        return self * number

    def __len__(self):
        return len(self._isospec)

    @staticmethod
    def ScalarProduct(spectra, weights):
        new_iso = IsoSpecPy.IsoDistribution.LinearCombination([s._isospec for s in spectra], weights)
        ret = Spectrum(isospec = new_iso)
        ret.sort_confs()
        ret.merge_confs()
        return ret

    def normalize(self, target_value = 1.0):
        """
        Normalize the intensity values so that they sum up to the target value.
        """
        self._isospec.normalize()

    def WSDistanceMoves(self, other):
        """
        Return the optimal transport plan between self and other.
        """
        try:
            ii = 0
            leftoverprob = other._probs[0]
            for mass, prob in self._isospec:
                while leftoverprob <= prob:
                    yield (other._masses[ii], mass, leftoverprob)
                    prob -= leftoverprob
                    ii += 1
                    leftoverprob = other._probs[ii]
                yield (other._masses[ii], mass, prob)
                leftoverprob -= prob
        except IndexError:
            return

    def WSDistance(self, other):
        return self._isospec.wassersteinDistance(other._isospec)

    def explained_intensity(self,other):
        """
        Returns the amount of mutual intensity between self and other,
        defined as sum of minima of intensities, mass-wise.
        """
        e = 0
        for i in range(len(self._isospec)):
            e += min(self._probs[i],other._probs[i])
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
        self.set_isospec(self._isospec.binned((0.1**nb_of_digits) / self.charge))

    def coarse_bin(self, nb_of_digits):
        """
        Rounds the m/z to a given number of decimal digits
        """
        self.set_isospec(self._isospec.binned(0.1**nb_of_digits))
        self.merge_confs()

    def add_chemical_noise(self, nb_of_noise_peaks, noise_fraction):
        """
        Adds additional peaks with uniform distribution in the m/z domain
        and gamma distribution in the intensity domain. The spectrum does NOT need
        to be normalized. Accordingly, the method does not normalize the intensity afterwards!
        noise_fraction controls the amount of noise signal in the spectrum.
        nb_of_noise_peaks controls the number of peaks added.
        """
        span = min(self._masses), max(self._masses)
        span_increase = 1.2  # increase the mass range by a factor of 1.2
        span = [span_increase*x + (1-span_increase)*sum(span)/2 for x in span]
        noisex = uniform.rvs(loc=span[0], scale=span[1]-span[0], size=nb_of_noise_peaks)
        noisey = gamma.rvs(a=2, scale=2, size=nb_of_noise_peaks)
        noisey /= sum(noisey)
        signal = sum(self._probs)
        noisey *=  signal*noise_fraction /(1-noise_fraction)
        noise_iso = IsoSpecPy.IsoDistribution(masses = noisex, probs = noisey)
        self.set_iso(self._isospec + noise_iso)
        self.sort_confs()
        self.merge_confs()

    def add_gaussian_noise(self, sd):
        """
        Adds gaussian noise to each peak, simulating
        electronic noise.
        """
        noised = rd.normal(self._probs, sd)
        # noised = noised - min(noised)
        self.set_masses_probs(self._masses, noised)

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
        assert np.isclose(sum(self._probs), 1), 'Spectrum needs to be normalized prior to distortion'
        X = [(x[0], N*gain*x[1]) for x in self.confs]  # average signal
        peakSD = np.sqrt(N*sd**2*self._probs + N*gain**2*self._probs*(1-self._probs))
        U = rd.normal(0.0, peakSD)
        self.set_masses_probs(self._masses, [max(0.0, p) for p in zip(self._probs,U)])
        return U

    def distort_mz(self, mean, sd):
        """
        Distorts the m/z measurement by a normally distributed
        random variable with given mean and standard deviation.
        Use non-zero mean to approximate calibration error.
        Returns the applied shift.
        """
        N = rd.normal(mean, sd, len(self._isospec))
        self.set_masses_probs(N + self._masses, self._probs)
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
        assert np.isclose(sum(self._probs), 1), 'Spectrum needs to be normalized prior to sampling'
        U = rd.multinomial(N, self._probs)
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
        is advised in order to avoid detection of noise.
        """
        diffs = [n[1]-p[1] for n,p in zip(self.confs[1:], self.confs[:-1])]
        is_max = [nd <0 and pd > 0 for nd, pd in zip(diffs[1:], diffs[:-1])]
        peaks = [x for x, p in zip(self.confs[1:-1], is_max) if p]
        return peaks
    
    def trim_negative_intensities(self):
        """
        Detects negative intensity measurements and sets them to 0.
        """
        self.confs = [(mz, intsy if intsy >= 0 else 0.) for mz, intsy in self._isospec]

    def centroid(self, max_width, peak_height_fraction=0.5):
        """Return confs of a centroided spectrum.

        The function identifies local maxima of intensity and integrates peaks in the regions
        delimited by peak_height_fraction of the apex intensity.
        By default, for each peak the function will integrate the region delimited by the full width at half maximum.
        If the detected region is wider than max_width, the peak is considered as noise and discarded.
        Small values of max_width tend to miss peaks, while large ones increase computational complexity
        and may lead to false positives.

        Note that this function should only be applied to profile spectra - the result
        does not make sense for centroided spectrum.
        Applying a gaussian or Savitzky-Golay filter prior to peak picking
        is advised in order to avoid detection of noise.

        Returns
        -----------------
            A tuple of two peak lists that can be used to construct a new Spectrum object.
            The first list contains configurations of centroids (i.e. centers of mass and areas of peaks).
            The second list contains configurations of peak apices corresponding to the centroids
            (i.e. locations and heights of the local maxima of intensity.)
        """
        ### TODO: change max_width to be in ppm?
        # Validate the input:
        if any(intsy < 0 for intsy in self._probs):
            warn("""
                 The spectrum contains negative intensities! 
                 It is advised to use Spectrum.trim_negative_intensities() before any processing
                 (unless you know what you're doing).
                 """)

        # Transpose the confs list to get an array of masses and an array of intensities:
        mz, intsy = self._masses, self._probs

        # Find the local maxima of intensity:
        peak_indices = argrelmax(intsy)[0]

        peak_mz = []
        peak_intensity = []
        centroid_mz = []
        centroid_intensity = []
        max_dist = max_width/2.
        n = len(mz)
        for p in peak_indices:
            current_mz = mz[p]
            current_intsy = intsy[p]
            # Compute peak centroids:
            target_intsy = peak_height_fraction*current_intsy
            right_shift = 1
            left_shift = 1
            # Get the mz points bounding the peak fragment to integrate.
            # First, go to the right from the detected apex until one of the four conditions are met:
            # 1. we exceed the mz range of the spectrum
            # 2. we exceed the maximum distance from the apex given by max_dist
            # 3. the intensity exceeds the apex intensity (meaning that we've reached another peak) 
            # 4. we go below the threshold intensity (the desired stopping condition)
            # Note: in step 3, an alternative is to check if the intensity simply starts to increase w.r.t. the previous inspected point.
            # Such an approach may give less false positive peaks, but is very sensitive to electronic noise and to overlapping peaks.
            # When we check if the intensity has not exceeded the apex intensity, and we encounter a cluster of overlapping peaks,
            # then we will effectively consider the highest one as the true apex of the cluster and integrate the whole cluster only once.
            while p + right_shift < n-1 and mz[p+right_shift] - mz[p] < max_dist and intsy[p+right_shift] <= current_intsy and intsy[p+right_shift] > target_intsy:
                right_shift += 1
            # Get the mz values of points around left mz value of the peak boundary (which will be interpolated):
            rx1, rx2 = mz[p+right_shift-1], mz[p+right_shift] 
            ry1, ry2 = intsy[p+right_shift-1], intsy[p+right_shift]
            if not ry1 >= target_intsy >= ry2:
                # warn('Failed to find the right boundary of the peak at %f (probably found an overlapping peak)' % current_mz)
                continue
            # Find the left boundary of the peak: 
            while p - left_shift > 1 and mz[p] - mz[p-left_shift] < max_dist and intsy[p-left_shift] <= current_intsy and intsy[p-left_shift] > target_intsy:
                left_shift += 1
            lx1, lx2 = mz[p-left_shift], mz[p-left_shift+1]  
            ly1, ly2 = intsy[p-left_shift], intsy[p-left_shift+1]
            if not ly1 <= target_intsy <= ly2:
                # warn('Failed to find the left boundary of the peak at %f (probably found an overlapping peak)' % current_mz)
                continue
            # Interpolate the mz values actually corresponding to peak_height_fraction*current_intsy:
            lx = (target_intsy-ly1)*(lx2-lx1)/(ly2-ly1) + lx1
            if not lx1 <= lx <= lx2:
                raise RuntimeError('Failed to interpolate the left boundary mz value of the peak at %f' % current_mz)
            rx = (target_intsy-ry1)*(rx2-rx1)/(ry2-ry1) + rx1
            if not rx1 <= rx <= rx2:
                raise RuntimeError('Failed to interpolate the right boundary mz value of the peak at %f' % current_mz)
            # Join the interpolated boundary with the actual measurements:
            x = np.hstack((lx, mz[(p-left_shift+1):(p+right_shift)], rx))
            y = np.hstack((target_intsy, intsy[(p-left_shift+1):(p+right_shift)], target_intsy))
            # Integrate the area:
            cint = np.trapz(y, x)
            cmz = np.trapz(y*x, x)/cint
            if cmz not in centroid_mz:  # intensity errors may introduce artificial peaks
                centroid_mz.append(cmz)
                centroid_intensity.append(cint)
                # Store the apex data:
                peak_mz.append(current_mz)
                peak_intensity.append(current_intsy)
        return(list(zip(centroid_mz, centroid_intensity)), list(zip(peak_mz, peak_intensity)))

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
        mz = self._masses
        intsy = self._probs
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
        ret = Spectrum()
        ret.set_masses_probs(target_mz, y)
        return ret


    def fuzzify_peaks(self, sd, step):
        """
        Applies a gaussian filter to the peaks, effectively broadening them
        and simulating low resolution. Works in place, modifying self.
        The parameter step gives the distance between samples in m/z axis.
        After the filtering, the area below curve (not the sum of intensities!)
        is equal to the sum of the input peak intensities.
        """
        gauss_spec = IsoSpecPy.Distributions.Gaussian(stdev=sd, bin_width=step, precision=0.9999)
        self.set_isospec(self._isospec * gauss_spec)
        self.sort_confs()
        self.merge_confs()

    def cut_smallest_peaks(self, removed_proportion=0.001):
        """
        Removes smallest peaks until the total removed intensity amounts
        to the given proportion of the total ion current in the spectrum.
        """
        self._isospec.sort_by_probs()
        threshold  = removed_proportion*sum(self._probs)
        removed = 0
        for ii in range(len(self._isospec)):
            removed += self._probs[ii]
            if removed > threshold:
                break
        self.set_masses_probs(masses = self._masses[ii:], probs = self.probs[ii:])

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
        raise NotImplemented()


    @staticmethod
    def filter_against_theoretical(experimental, theoreticals, margin=0.15):
        """
        Remove signal from the empirical spectra which is far from theoretical.

        This method removes peaks from experimental spectrum which are outside
        theoretical peaks +/- margin.

        Parameters
        ----------
        experimental
            Empirical spectrum.
        theoreticals:
            One instance of theoretical or iterable of instances of theoretical
            spectra.
        margin
            m/z radius within empirical spectrum should be left.

        Returns
        -------
        Spectrum
            An empirical spectrum with filtered out peaks.
        """
        try:
            th_confs = []
            for theoretical_spectrum in theoreticals:
                th_confs.extend(theoretical_spectrum.confs)
            theoretical = Spectrum()
            theoretical.set_confs(th_confs)
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
        new_spectrum = Spectrum(label=experimental.label)
        new_spectrum.confs = result_confs
        return new_spectrum

    def plot(self, show = True, profile=False, linewidth=1, **plot_kwargs):
        """
        Plots the spectrum.
        The keyword argument show is retained for backwards compatibility.
        """
        import matplotlib.pyplot as plt
        if profile:
            plt.plot(self._masses, self._probs,
                     linestyle='-', linewidth=linewidth, label=self.label, **plot_kwargs)
        else:
            plt.vlines(self._masses, [0],
                       self._probs, label = self.label,
                       linewidth=linewidth, **plot_kwargs)
        if show:
            plt.show()

    @staticmethod
    def plot_all(spectra, show=True, profile=False, cmap=None, **plot_kwargs):
        """
        Shows the supplied list of spectra on a single plot. 
        """
        import matplotlib.pyplot as plt
        import matplotlib.cm as cm
        if not cmap:
            colors = cm.rainbow(np.linspace(0, 1, len(spectra)))
            colors =  [[0, 0, 0, 0.8]] + [list(x[:3]) + [0.6] for x in colors]
        else:
            try:
                colors = [[0, 0, 0, 0.8]] + [cmap(x, alpha=1) for x in range(len(spectra))]
            except:
                colors = cmap
        i = 0
        for spectre in spectra:
            spectre.plot(show=False, profile=profile, color = colors[i],
                         **plot_kwargs)
            i += 1
        #plt.legend(loc=9, bbox_to_anchor=(0.5, -0.1), ncol=len(spectra))  # legend below plot
        plt.legend(loc=0, ncol=1)
        if show:
            plt.show()


if __name__=="__main__":
    import matplotlib.pyplot as plt
    from copy import deepcopy
    S = Spectrum(formula="C2H5OH", threshold=0.01)
    
    S.add_chemical_noise(4, 0.2)
    S.plot()

    sd = 0.01
    C = deepcopy(S)
    C = C*(np.sqrt(2*np.pi)*sd)**-1
    S.fuzzify_peaks(0.01, 0.0001)
    S.plot(profile=True, show=False)
    C.plot(show=False)
    plt.show()
    
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
