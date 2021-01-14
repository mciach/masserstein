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

class Spectrum:
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
        else:
            self.empty = True
            self.confs = []

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
        norm = float(sum(x[1] for x in self.confs))
        return sum(x[0]*x[1]/norm for x in self.confs)

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
        return max(self.confs, key=lambda x: x[1])

    def sort_confs(self):
        """
        Sorts configurations by their mass.
        """
        self.confs.sort(key = lambda x: x[0])

    def merge_confs(self):
        """
        Merges configurations with an identical mass, summing their intensities.
        """
        cmass = self.confs[0][0]
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
        self.sort_confs()
        self.merge_confs()

    def __add__(self, other):
        res = Spectrum()
        res.confs = self.confs + other.confs
        res.sort_confs()
        res.merge_confs()
        res.label = self.label + ' + ' + other.label
        return res

    def __mul__(self, number):
        res = Spectrum()
        res.set_confs([(x[0], number*x[1]) for x in self.confs])
        res.label = self.label
        return res

    def __rmul__(self, number):
        # Here * is commutative
        return self * number

    def __len__(self):
        return len(self.confs)

    @staticmethod
    def ScalarProduct(spectra, weights):
        ret = Spectrum()
        Q = [(spectra[i].confs[0], i, 0) for i in range(len(spectra))]
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
        """
        Normalize the intensity values so that they sum up to the target value.
        """
        x = target_value/math.fsum(v[1] for v in self.confs)
        self.confs = [(v[0], v[1]*x) for v in self.confs]

    def WSDistanceMoves(self, other):
        """
        Return the optimal transport plan between self and other.
        """
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
            return

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
        for i in range(len(self.confs)):
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
        xcoord = (xcoord[0] + round(x-xcoord[0], nb_of_digits) for x in xcoord)
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
        is advised in order to avoid detection of noise.
        """
        diffs = [n[1]-p[1] for n,p in zip(self.confs[1:], self.confs[:-1])]
        is_max = [nd <0 and pd > 0 for nd, pd in zip(diffs[1:], diffs[:-1])]
        peaks = [x for x, p in zip(self.confs[1:-1], is_max) if p]
        return peaks

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

        # Transpose the confs list to get an array of masses and an array of intensities:
        mz, intsy = np.array(self.confs).T
        
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
            if intsy[p+right_shift] > target_intsy:
                continue
            while p - left_shift > 1 and mz[p] - mz[p-left_shift] < max_dist and intsy[p-left_shift] <= current_intsy and intsy[p-left_shift] > target_intsy:
                left_shift += 1
            if intsy[p-left_shift] > target_intsy:
                continue
            # Get the mz value actually corresponding to peak_height_fraction*current_intsy:
            lx1, lx2 = mz[p-left_shift], mz[p-left_shift+1]  # x coordinates of points around left mz value we're looking for
            ly1, ly2 = intsy[p-left_shift], intsy[p-left_shift+1]
            assert ly1 <= target_intsy <= ly2
            rx1, rx2 = mz[p+right_shift-1], mz[p+right_shift] 
            ry1, ry2 = intsy[p+right_shift-1], intsy[p+right_shift]
            assert ry1 >= target_intsy >= ry2
            lx = (target_intsy-ly1)*(lx2-lx1)/(ly2-ly1) + lx1
            assert lx1 <= lx <= lx2
            rx = (target_intsy-ry1)*(rx2-rx1)/(ry2-ry1) + rx1
            assert rx1 <= rx <= rx2
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
            plt.plot([x[0] for x in self.confs], [x[1] for x in self.confs],
                     linestyle='-', linewidth=linewidth, label=self.label, **plot_kwargs)
        else:
            plt.vlines([x[0] for x in self.confs], [0],
                       [x[1] for x in self.confs], label = self.label,
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
