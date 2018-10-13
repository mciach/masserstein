from __future__ import division
from peptides import get_protein_formula
import math
from IsoSpecPy import IsoSpecPy
#from Users.wandaniemyska.Documents.PROGRAMY.Wassetrstein.IsoSpec.IsoSpecPy.IsoSpecPy import IsoSpecPy
from pprint import pprint
import numpy as np
from math import exp
from scipy.stats import norm
import random
import heapq
import re
from collections import Counter

try: 
    xrange
except NameError:
    xrange = range

class Spectrum:
    def __init__(self, formula, threshold=0.001, intensity = 1.0, empty = False, charge=1, adduct=None, label = None):
        self.label = label
        self.confs = []
        if label is None and formula != "":
            self.label = formula
        elif label is None:
            self.label = "Unknown"

        if not empty:
            formula = re.findall('([A-Z][a-z]*)([0-9]*)', formula)
            formula = [(x[0], int(x[1])) for x in formula]
            formula = Counter(dict(formula))
            if adduct:
                formula[adduct] += charge
            formula = ''.join(x+str(formula[x]) for x in formula)
            masses, lprobs, _unused = IsoSpecPy.IsoSpec.IsoFromFormula(formula = formula, cutoff = threshold, method = "threshold_relative").getConfs()
            probs = map(exp, lprobs)
            self.confs = [(x[0]/charge, intensity*x[1]) for x in zip(masses, probs)]
            self.sort_confs()

    @staticmethod
    def new_from_fasta(fasta, threshold=0.001, intensity = 1.0, empty = False, charge=1, label = None):
        return Spectrum(get_protein_formula(fasta), threshold, intensity, empty, charge, label)

    @staticmethod
    def new_from_csv(filename):
        spectrum = Spectrum("", empty=True, label=filename)

        with open(filename, "r") as infile:
            header = next(infile)
            for line in infile:
                if line[0] == '#':
                    continue
                line = line.strip()
                if ',' in line:
                    line = line.split(',')
                else:
                    line = line.split()
                spectrum.confs.append(tuple(map(float, line)))
        return spectrum


    @staticmethod
    def new_random(domain = (0.0, 1.0), peaks = 10):
        ret = Spectrum("", 0.0, empty = True)
        ret.confs = []
        for _ in xrange(peaks):
            ret.confs.append((random.uniform(*domain), random.uniform(0.0, 1.0)))
        ret.sort_confs()
        ret.normalize()
        return ret

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
        self.confs = ret

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
        return math.fsum(abs(x[0]-x[1])*x[2] for x in self.WSDistanceMoves(other))

    def WSVortex(self, other, penalty):
        """
        Compute WSDistance with moving intensity from other spectrum to thrash with penalty.
        Return: distance value and a list of intensities transported from peaks of other spectrum.
        """
        sp1 = self
        sp2 = other
        F = np.array([x+(0,) for x in sp1.confs])
        F[:,1] = np.cumsum(F[:,1])
        assert abs(F[-1, 1] -1) < 1e-10, "Spectrum 1 not normalized!"
        H = np.array([x+(1,) for x in sp2.confs])
        H[:,1] = np.cumsum(H[:,1])
        assert abs(H[-1, 1] -1) < 1e-10, "Spectrum 2 not normalized!"
        S = np.vstack((F, H))
        S = S[np.argsort(S[:,0])]
        uniq_masses = np.unique(S[:,0])
        uniq_masses.sort()
        interval_lengths = np.abs(uniq_masses[1:] - uniq_masses[:-1])
        uniq_masses = len(uniq_masses)
        # order of vars: transport to thrash, dummy cdf vars
        varnum = len(sp2.confs) + uniq_masses - 1
        # target
        c = np.zeros(varnum)
        c[:len(sp2.confs)]  = penalty
        c[len(sp2.confs):] = interval_lengths
        # inequality constraints
        A_ub = np.zeros((2*uniq_masses-1, varnum))
        b_ub = np.zeros(2*uniq_masses-1)
        # constraint for amount of thrash
        A_ub[-1, :len(sp2.confs)] = 1
        b_ub[-1] = 1
        # constraints for dummy cdfs
        cpref = [0., 0.]  # current prefix sums of sp1 and sp2 respectively
        cmass = S[0, 0]
        t_id = 0
        g_id = 0
        for mass, prefsum, sp_id in S:
            if mass > cmass:
                A_ub[2*t_id, :g_id] = 1- cpref[0] 
                A_ub[2*t_id, g_id:len(sp2.confs)] = -cpref[0]
                A_ub[2*t_id+1, :g_id] = cpref[0] - 1
                A_ub[2*t_id+1, g_id:len(sp2.confs)] = cpref[0]
                A_ub[2*t_id, len(sp2.confs) + t_id] = -1
                A_ub[2*t_id+1, len(sp2.confs) + t_id] = -1
                b_ub[2*t_id] = cpref[1] - cpref[0]
                b_ub[2*t_id+1] = cpref[0]-cpref[1]
                t_id += 1
            if sp_id == 1:
                g_id += 1
            cpref[int(sp_id)] = prefsum
        res = linprog(c, A_ub = A_ub, b_ub = b_ub)
        for g, h in zip(res.x[:len(sp2.confs)], sp2.confs):
            assert g <= h[1], "Overloaded spectrum 2"
        return (res.fun, res.x[:len(sp2.confs)])
    
    def explained_intensity(self,other):
        e = 0
        for i in xrange(len(self.confs)):
            e += min(self.confs[i][1],other.confs[i][1])
        return e

    def bin(self, precision, kernel_cdf = None):
        if kernel_cdf is not None:
            assert kernel_cdf(0) == 0.5  # assume that distribuation is symmetric
            i = 0
            while kernel_cdf(precision*i) <= 0.99: i+=1
            output = []
            for mz,prob in self.confs:
                first_bin_start = precision*np.round(mz/precision) - (i+0.5)*precision
                for k in xrange(2*i+1):
                    current_bin_start = first_bin_start + k*precision
                    mass = (kernel_cdf(current_bin_start + precision - mz) - kernel_cdf(current_bin_start - mz))*prob
                    output.append((current_bin_start + 0.5*precision, mass))
            self.confs = output
        else:
            self.confs = [(precision*np.round(x[0]/precision), x[1]) for x in self.confs]
        self.merge_confs()

    def interfere_calibration(self, m, scale=0):
    # move self.confs along ox by m value
    # if scale!=0, then move self.confs along ox by random values from normal distribution(m,scale) 
        n = len(self.confs)
        if scale == 0: move = [m]*n
        else: move = norm.rvs(m, scale, n)
        output = []
        for i in xrange(n):
            output.append((self.confs[i][0] + move[i], self.confs[i][1]))
        self.confs = output

    def interfere_y(self, m=0, scale=0):
    # change self.confs oy by random values from normal distribution(m,scale) 
        output = [(x[0], math.exp(math.log(x[1]) + norm.rvs(m, scale))) for x in self.confs]
        self.confs = output

    def interfere_ignore_picks(self, part=0):
    # remove randomly some picks - but no more than "part*all"
    # return removed total intensity (as a fraction of total intensity)
        s = sum([x[1] for x in self.confs])
        n = int(part*len(self.confs))
        to_remove = sorted(random.sample(xrange(len(self.confs)),n))
        to_remove.append(len(self.confs)+1)
        s_removed = 0
        output = []
        j = 0
        for i in xrange(len(self.confs)):
            if to_remove[j] == i:
                if (s_removed + self.confs[i][1])/s <= part: s_removed += self.confs[i][1]
                else: output.append(self.confs[i])
                j += 1
            else: output.append(self.confs[i])
        self.confs = output
        return s_removed/s

    def cut_smallest_peaks(self,intensity=0.001):
        self.confs.sort(key = lambda x: x[1])
        self.confs.reverse()
        removed = 0
        while len(self.confs)>0 and removed + self.confs[-1][1] <= intensity: removed += self.confs.pop()[1]
        self.confs.sort(key = lambda x: x[0])


    def plot(self, show = True, color = 'k'):
        import matplotlib.pyplot as plt
        if show:
            plt.clf()
        plt.vlines([x[0] for x in self.confs], [0], [x[1] for x in self.confs], color = color, label = self.label, linewidth=2)
        if show:
            plt.show()
    
    @staticmethod
    def plot_all(spectra, show=True):
        import matplotlib.pyplot as plt
        import matplotlib.cm as cm
        import numpy as np
        colors = cm.rainbow(np.linspace(0, 1, len(spectra)))
        colors =  [[0, 0, 0, 0.4]] + [list(x[:3]) + [0.4] for x in colors]
        plt.clf()
        i = 0
        for spectre in spectra:
            spectre.plot(show = False, color = colors[i])
            i += 1
        plt.legend(loc=9, bbox_to_anchor=(0.5, -0.1), ncol=len(spectra))
        if show: plt.show()
            
def random_formula(mass):
#N-14u (14,0067 u +- 0,0001 u), O-16u (15,999), C-12u (12,0107 u +- 0,0008 u), H-1u (1,00794 u +- 0,00001 u), S - 32u (32,065 u +- 0,005 u)
    masy = {"N":14, "O":16, "C":12, "H":1, "S":32}
    worek = []
    for i in range(10):
        worek.append("N")
        worek.append("C")
        worek.append("H")
    for i in range(5): worek.append("O")
    worek.append("S")

    wylosowane = {"N":0, "O":0, "C":0, "H":0, "S":0}
    wmass = 0
    while wmass < mass:
        c = random.choice(worek)
        if wmass + masy[c] <= mass:
            wylosowane[c] += 1
            wmass += masy[c]
        else:
            worek = list(filter(lambda a: a != c, worek))

    formula = ""
    kolejne = ["C","O","H","N","S"]
    for c in kolejne:
        if wylosowane[c]>0: formula += c + str(wylosowane[c])

    return formula, wylosowane


def deconvolve(observed, theoreticals):
    pass

if __name__=="__main__":

    a = "C1625H2345N623S3"
    b = "C1625H2346N623S3"
    c = "C1625H2346N623S4"
    #C = Spectrum("C1234H3245N234S764", 0.001, 1.3)
    A = Spectrum(a, 0.1, 0.5)
    B = Spectrum(b, 0.1, 0.5)
    C = Spectrum(c, 0.1, 0.5)
    Spectrum.plot_all([A, B, C])

    # print "AAA"


    C.normalize()
    for i in xrange(11):
        A = Spectrum(a, 0.001, 10.0-i)
        B = Spectrum(b, 0.001, float(i))
        S = A+B
        S.normalize()
        
        #print i, S.WSDistance(C)


    A.normalize()
    B.normalize()

    #print A.WSDistance(B)
    pprint(list(A.WSDistanceMoves(B)))


    from scipy.optimize import linprog

