import math
from IsoSpecPy import IsoSpecPy
#from Users.wandaniemyska.Documents.PROGRAMY.Wassetrstein.IsoSpec.IsoSpecPy.IsoSpecPy import IsoSpecPy
from pprint import pprint
import numpy as np
from math import exp
from scipy.stats import norm
import random
import heapq

try: 
    xrange
except NameError:
    xrange = range

class Spectrum:
    def __init__(self, formula, threshold=0.001, intensity = 1.0, empty = False, charge=1, label = None):
        self.label = label
        self.confs = []
        if label is None and formula != "":
            self.label = formula
        elif label is None:
            self.label = "Unknown"

        if not empty:
            masses, lprobs, _unused = IsoSpecPy.IsoSpec.IsoFromFormula(formula = formula, cutoff = threshold, method = "threshold_relative").getConfs()
            probs = list(map(exp, lprobs))
            self.confs = [(x[0]/charge, intensity*x[1]) for x in zip(masses, probs)]
            self.sort_confs()

    @staticmethod
    def new_random(domain = (0.0, 1.0), peaks = 10):
        ret = Spectrum("", 0.0, empty = True)
        ret.confs = []
        for _ in range(peaks):
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
        return res

    def __len__(self):
        return len(self.confs)

    @staticmethod
    def ScalarProduct(spectra, weights):
        ret = Spectrum("", 0.0, empty = True)
        Q = [(spectra[i].confs[0], i) for i in range(len(spectra))]
        P = [0] * len(spectra)
        heapq.heapify(Q)
        while Q != []:
            _, i = heapq.heappop(Q)
            ret.confs.append((spectra[i].confs[P[i]][0], spectra[i].confs[P[i]][1]*weights[i]))
            P[i] += 1
            if P[i] < len(spectra[i]):
                heapq.heappush(Q, (spectra[i].confs[0], i))
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
    
    def explained_intensity(self,other):
        e = 0
        for i in range(len(self.confs)):
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
                for k in range(2*i+1):
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
        for i in range(n):
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
        to_remove = sorted(random.sample(range(len(self.confs)),n))
        to_remove.append(len(self.confs)+1)
        s_removed = 0
        output = []
        j = 0
        for i in range(len(self.confs)):
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
        plt.vlines([x[0] for x in self.confs], [0], [x[1] for x in self.confs], color = color, label = self.label, linewidth=3)
        if show:
            plt.show()
    
    @staticmethod
    def plot_all(spectra):
        import matplotlib.pyplot as plt
        import matplotlib.cm as cm
        import numpy as np
        colors = cm.rainbow(np.linspace(0, 1, 4))
        colors = [list(x[:3]) + [0.3] for x in colors]
        plt.clf()
        i = 0
        for spectre in spectra:
            spectre.plot(show = False, color = colors[i])
            i += 1
        plt.legend()
        plt.show()
            
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
            worek = list([a for a in worek if a != c])

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
    for i in range(11):
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