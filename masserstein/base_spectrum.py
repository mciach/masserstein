import math
import numpy as np
from scipy.stats import uniform, gamma
import random
import heapq
from collections import Counter
import numpy.random as rd
from scipy.signal import argrelmax
from warnings import warn
from copy import deepcopy

class BaseSpectrum:
    
    def __init__(self, confs=None, label=None, **other):
        """Initialize a BaseSpectrum class.

        Initialization can be done by setting a peak list.

        The initialized spectrum is not normalised. In order to do this, use
        normalize method.

        Parameters
        ----------

        confs: list
            A list of tuples:
            - the first element corresponds to the position on the horizontal axis,
            - the second element corresponds to the vertical axis (intensity).

        label: str
            An optional spectrum label.

        """
        self.label = label

        if confs is not None:
            self.set_confs(confs)
            self.empty = False
        else:
            self.empty = True
            self.confs = []


    @classmethod
    def new_from_csv(cls, filename, delimiter=","):
        """
        Creates a new spectrum from csv file.
        File should contain two columne.
        The first column corresponds to the horizontal axis.
        The second column corresponds to the vertical axis (intensity).
        """
        spectrum = cls(label=filename)

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

    @classmethod
    def new_random(cls, domain=(0.0, 1.0), peaks=10):
        """
        Creates a new random spectrum.
        """
        ret = cls()
        confs = []
        for _ in range(peaks):
            confs.append((random.uniform(*domain), random.uniform(0.0, 1.0)))
        ret.set_confs(confs)
        return ret


    def copy(self):
        """
        Returns a (deep) copy of self.
        """
        return deepcopy(self)


    def get_modal_peak(self):
        """
        Returns the peak with the highest intensity.
        """
        return max(self.confs, key=lambda x: x[1])

    def sort_confs(self):
        """
        Sorts configurations by their first coordinate.
        Works in place.
        """
        self.confs.sort(key = lambda x: x[0])

    def merge_confs(self):
        """
        Merges configurations with an identical locations on horizontal axis, summing their intensities.
        Works in place.
        """
        if self.confs:
            cmass = self.confs[0][0]
            cprob = 0.0
            ret = []
            for mass, prob in self.confs + [(-1, 0)]:
                if mass != cmass:
                    ret.append((cmass, cprob))
                    cmass = mass
                    cprob = 0.0
                cprob += prob
            self.confs = ret

    def set_confs(self, confs):
        """
        Sets .confs, i.e. a list containing 2-tuples with the first element corresponding to
        the position on the horizontal axis, and the second element corresponding to intensity.
        Works in place.
        """
        self.confs = confs
        if len(self.confs) > 0:
            self.empty = False
            self.sort_confs()
            self.merge_confs()
        else:
            self.empty = True

    def __add__(self, other):
        res = self.__class__()
        res.confs = self.confs + other.confs
        res.sort_confs()
        res.merge_confs()
        res.label = self.label + ' + ' + other.label
        return res

    def __mul__(self, number):
        res = self.__class__()
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
        """
        Calculates the scalar product of spectra with specified weights.
        """
        ret = spectra[0].__class__()
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
        Normalizes the intensity values so that they sum up to the target value.
        Works in place.
        """
        x = target_value/math.fsum(v[1] for v in self.confs)
        self.confs = [(v[0], v[1]*x) for v in self.confs]

    def WSDistanceMoves(self, other):
        """
        Returns the optimal transport plan between self and other.
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
        """
        Calculates the Wasserstein distance between self and other.
        """
        if not np.isclose(sum(x[1] for x in self.confs), 1.):
            raise ValueError('Self is not normalized.')
        if not np.isclose(sum(x[1] for x in other.confs), 1.):
            raise ValueError('Other is not normalized.')
        return math.fsum(abs(x[0]-x[1])*x[2] for x in self.WSDistanceMoves(other))

    def explained_intensity(self,other):
        """
        Returns the amount of mutual intensity between self and other,
        defined as sum of minima of intensities, pointwise.
        """
        e = 0
        for i in range(len(self.confs)):
            e += min(self.confs[i][1],other.confs[i][1])
        return e

    def add_chemical_noise(self, nb_of_noise_peaks, noise_fraction, span=1.2):
        """
        Adds additional peaks that simulate chemical noise.

        The method adds additional peaks with uniform distribution on the horizontal axis
         and gamma distribution in the intensity domain. The spectrum
        does NOT need to be normalized. Accordingly, the method does not
        normalize the intensity afterwards! Works in place.

        Parameters
        ----------
        nb_of_noise_peaks : int
            The number of added peaks.
        noise_fraction : float
            The amount of noise signal in the spectrum, >= 0 and <= 1.
        span: float or 2-tuple of floats
           If float, then `span` specifies a factor by which the range along the horizontal axis is
           increased. If 2-tuple, then `span` specifies range along the horizontal axis, which is
           noised.

        """
        if isinstance(span, (float, int)):
            span_increase = span
            prev_span = (min(x[0] for x in self.confs),
                         max(x[0] for x in self.confs))
            span_move = 0.5 * (span_increase - 1) * (prev_span[1] - prev_span[0])
            span = (max(prev_span[0] - span_move, 0),
                    prev_span[1] + span_move)
        noisex = uniform.rvs(loc=span[0], scale=span[1]-span[0],
                             size=nb_of_noise_peaks)
        noisey = gamma.rvs(a=2, scale=2, size=nb_of_noise_peaks)
        noisey /= sum(noisey)
        signal = sum(x[1] for x in self.confs)
        noisey *=  signal * noise_fraction / (1 - noise_fraction)
        noise = [(x, y) for x,y in zip(noisex, noisey)]
        self.confs.extend(noise)
        self.sort_confs()
        self.merge_confs()

    def add_gaussian_noise(self, sd):
        """
        Adds gaussian noise to each peak, simulating
        electronic noise.
        Works in place.
        """
        noised = rd.normal([y for x,y in self.confs], sd)
        # noised = noised - min(noised)
        self.confs = [(x[0], y) for x, y in zip(self.confs, noised) if y > 0]


    def find_peaks(self):
        """
        Returns a list of local maxima.
        Each maximum is reported as a tuple of location on the horizontal axis and intensity.
        The first and the final configuration is never reported as a maximum.
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
        Works in place.
        """
        self.confs = [(mz, intsy if intsy >= 0 else 0.) for mz, intsy in self.confs]

    def centroid(self, max_width, peak_height_fraction=0.5):
        """Return confs of a centroided spectrum (i.e. peak list).

        The function identifies local maxima of intensity and integrates peaks in the regions
        delimited by peak_height_fraction of the apex intensity.
        By default, for each peak the function will integrate the region delimited by the full width at half maximum (FWHM).
        If the detected region is wider than max_width, the peak is considered as noise and discarded.
        Small values of max_width tend to miss peaks, while large ones increase computational complexity
        and may lead to false positives.

        Note that this function should only be applied to profile spectra - the result
        does not make sense for peak lists.
        Applying a gaussian or Savitzky-Golay filter prior to peak picking
        is advised in order to avoid detection of noise.

        Returns
        -----------------
            A tuple of two lists that can be used to construct a new BaseSpectrum object.
            The first list contains configurations of centroids (i.e. locations and areas of peaks).
            The second list contains configurations of peak apices corresponding to the centroids
            (i.e. locations and heights of the local maxima of intensity.)
        """
        # Validate the input:
        if any(intsy < 0 for mz, intsy in self.confs):
            warn("""
                 The spectrum contains negative intensities!
                 It is advised to use BaseSpectrum.trim_negative_intensities() before any processing
                 (unless you know what you're doing).
                 """)

        # Transpose the confs list to get an array of points on the horizontal axis and an array of intensities:
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
            # Get the points bounding the peak fragment to integrate.
            # First, go to the right from the detected apex until one of the four conditions are met:
            # 1. we exceed the horizontal range of the spectrum
            # 2. we exceed the maximum distance from the apex given by max_dist
            # 3. the intensity exceeds the apex intensity (meaning that we've reached another peak)
            # 4. we go below the threshold intensity (the desired stopping condition)
            # Note: in step 3, an alternative is to check if the intensity simply starts to increase w.r.t. the previous inspected point.
            # Such an approach may give less false positive peaks, but is very sensitive to electronic noise and to overlapping peaks.
            # When we check if the intensity has not exceeded the apex intensity, and we encounter a cluster of overlapping peaks,
            # then we will effectively consider the highest one as the true apex of the cluster and integrate the whole cluster only once.
            while p + right_shift < n-1 and mz[p+right_shift] - mz[p] < max_dist and intsy[p+right_shift] <= current_intsy and intsy[p+right_shift] > target_intsy:
                right_shift += 1
            # Get the values of points around left mz value of the peak boundary (which will be interpolated):
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
            # Interpolate the values on the horizontal axis actually corresponding to peak_height_fraction*current_intsy:
            lx = (target_intsy-ly1)*(lx2-lx1)/(ly2-ly1) + lx1
            if not lx1 <= lx <= lx2:
                raise RuntimeError('Failed to interpolate the left boundary value of the peak at %f' % current_mz)
            rx = (target_intsy-ry1)*(rx2-rx1)/(ry2-ry1) + rx1
            if not rx1 <= rx <= rx2:
                raise RuntimeError('Failed to interpolate the right boundary value of the peak at %f' % current_mz)
            # Join the interpolated boundary with the actual measurements:
            x = np.hstack((lx, mz[(p-left_shift+1):(p+right_shift)], rx))
            y = np.hstack((target_intsy, intsy[(p-left_shift+1):(p+right_shift)], target_intsy))
            # Integrate the area:
            cint = np.trapezoid(y, x)
            cmz = np.trapezoid(y*x, x)/cint
            if cmz not in centroid_mz:  # intensity errors may introduce artificial peaks
                centroid_mz.append(cmz)
                centroid_intensity.append(cint)
                # Store the apex data:
                peak_mz.append(current_mz)
                peak_intensity.append(current_intsy)
        return(list(zip(centroid_mz, centroid_intensity)), list(zip(peak_mz, peak_intensity)))


    def cut_smallest_peaks(self, removed_proportion=0.001):
        """
        Removes smallest peaks until the total removed intensity amounts
        to the given proportion of the total intensity in the spectrum.
        Works in place.
        """
        self.confs.sort(key = lambda x: x[1], reverse=True)
        threshold  = removed_proportion*sum(x[1] for x in self.confs)
        removed = 0
        while len(self.confs)>0 and removed + self.confs[-1][1] <= threshold:
            removed += self.confs.pop()[1]
        self.confs.sort(key = lambda x: x[0])


    def filter_against_other(self, others, margin=0.15):
        """
        Remove signal from the spectrum which is far from other spectra signal.

        This method removes peaks whose location is outside the
        area of any peak of other spectra +/- margin. The method does not
        modify self spectrum and returns a new instance of the filtered
        spectrum.

        Parameters
        ----------
        self
            Spectrum to be filtered.
        others:
            One instance of the spectrum against self is filtered or iterable of
            instances of other spectra.
        margin
            horizontal radius within signal that should be left.

        Returns
        -------
        (Base)Spectrum
            A new spectrum with filtered peaks.

        """
        try:
            other_confs = []
            for other_spectrum in others:
                other_confs.extend(other_spectrum.confs)
            other = self.__class__(confs=other_confs)
        except TypeError:
            other = others
        other_masses = [i[0] for i in other.confs]


        result_confs = []
        index = 0
        for mz, abund in self.confs:
            while (index + 1 < len(other_masses) and
                   other_masses[index + 1] < mz):
                index += 1
            if abs(mz - other_masses[index]) <= margin or (
                    index + 1 < len(other_masses) and
                    abs(mz - other_masses[index + 1]) <= margin):
                result_confs.append((mz, abund))

        result_spectrum = self.__class__(confs=result_confs, label=self.label)
        return result_spectrum

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

