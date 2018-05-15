from src.wasserstein import Spectrum
from src.FastIPMDeconv2 import FastIPMDeconv
from getopt import getopt
import sys

doc="""NAME:
    WSDistance.py

USAGE:
    python WSDistance.py [OPTIONS] FILE1 FILE2

EXAMPLES:
    python WSDistance.py examples/ethanol.txt examples/acetic_acid.txt
    python WSDistance.py -t 1. -s examples/ethanol.txt examples/propane.txt
    
DESCRIPTION:
    Computes the Wasserstein distance between two spectra.
    The two spectra need to be supplied as peak lists in separate files (FILE1, FILE2).
    Each file should be composed of two tab-separated columns.
    The first column corresponds to m/z values, the second one to peak intensity (ion current).
    A header can be present, starting with a hash sign #. 
    Any line starting with # will be ommitted. 
    The supplied spectra do not need to have normalized peak intensities. 
    The normalization will be performed by the program.
    
RETURNS:
    By default, the Wasserstein distance is printed to stdout, together with the program configuration.
    Optionally, mass transport can be printed to stdout as well. The mass transport is 
    represented as a table with three columns, corresponding to the origin m/z, the target m/z,
    and the amount of transported intensity.

OPTIONS:
    -s 
        Print the optimal mass transport scheme. 
    -t: float
        The total intensity that is to remain in a spectrum after denoising. 
        Default: 0.99, which means that peaks corresponding to at most 0.01 of the intensity  
        will be removed. 
        Setting this parameter to 1 disables denoising.  
    -h
        Print this message and exit.
"""

thr = 0.99
norm = True
print_transport = False

opts, args = getopt(sys.argv[1:], 'hst:')
if not args:
    print doc
    quit()

for opt, arg in opts:
    if opt == '-t':
        thr = float(arg)
        if not 0 <= thr <= 1:
            raise ValueError("Improper threshold value: %f" % thr)
    if opt == '-h':
        print doc
        quit()
    if opt == '-s':
        print_transport = True

sp1, sp2 = args

print "Spectrum 1:", sp1
print "Spectrum 2:", sp2
print "Intensity cutoff:", thr

sp1 = open(sp1).readlines()
sp2 = open(sp2).readlines()

sp1 = [l.strip() for l in sp1]
sp2 = [l.strip() for l in sp2]

sp1 = [map(float, l.split()) for l in sp1 if l and l[0] != '#']
sp2 = [map(float, l.split()) for l in sp2 if l and l[0] != '#']

if norm:
    sum1 = sum(l[1] for l in sp1)
    sp1 = [(l[0], l[1]/sum1) for l in sp1]
    sum2 = sum(l[1] for l in sp2)
    sp2 = [(l[0], l[1]/sum2) for l in sp2]

if thr < 1:
    order1 = sorted([x for x in enumerate((l[1] for l in sp1))], key=lambda y: y[1])  # ordering of intensities
    cmsm1 = reduce(lambda x,y: x + [x[-1] + y], (l[1] for l in order1), [0])[1:]  # cumsum of ordered intensities
    to_remove1 = [o[0] for o, c in zip(order1, cmsm1) if c < 1-thr]  # indices of peaks below threshold
    sp1 = [l for i, l in enumerate(sp1) if i not in to_remove1]  # denoised spectrum

    order2 = sorted([x for x in enumerate((l[1] for l in sp2))], key=lambda y: y[1])
    cmsm2 = reduce(lambda x, y: x + [x[-1] + y], (l[1] for l in order2), [0])[1:]
    to_remove2 = [o[0] for o, c in zip(order2, cmsm2) if c < 1-thr]
    sp2 = [l for i, l in enumerate(sp2) if i not in to_remove2]

Spectrum1 = Spectrum("", empty=True)
Spectrum1.set_confs(sp1)
Spectrum1.normalize()
Spectrum2 = Spectrum("", empty=True)
Spectrum2.set_confs(sp2)
Spectrum2.normalize()

W = Spectrum1.WSDistance(Spectrum2)

print
print "Wasserstein distance:"
print W

if print_transport:
    mvs = list(Spectrum1.WSDistanceMoves(Spectrum2))
    print "Optimal transport scheme:"
    for m in mvs:
        if m[2] > 1e-6:
            print '\t'.join(map(str, m))

