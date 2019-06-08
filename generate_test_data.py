from masserstein import Spectrum, peptides, estimate_proportions
from copy import deepcopy
import scipy as sp
import numpy as np
from matplotlib import pyplot as plt
from copy import deepcopy

myoglobin_fasta ="""GLSDGEWQLVLNVWGKVEADIPGHGQEVLIRLFKGHPETLEKFDKFKHLKSEDEMKASE
DLKKHGATVLTALGGILKKKGHHEAEIKPLAQSHATKHKIPVKYLEFISECIIQVLQSKH
PGDFGADAQGAMNKALELFRKDMASNYKELGFQG"""
haemoglobinB_fasta = """VHLTPEEKSAVTALWGKVNVDEVGGEALGRLLVVYPWTQRFFESFGDLSTPDAVMGNPK
VKAHGKKVLGAFSDGLAHLDNLKGTFATLSELHCDKLHVDPENFRLLGNVLVCVLAHHFG
KEFTPPVQAAYQKVVAGVANALAHKYH"""
haemoglobinA_fasta = """VLSPADKTNVKAAWGKVGAHAGEYGAEALERMFLSFPTTKTYFPHFDLSHGSAQVKGHG
KKVADALTNAVAHVDDMPNALSALSDLHAHKLRVDPVNFKLLSHCLLVTLAAHLPAEFTP
AVHASLDKFLASVSTVLTSKYR"""

# Obtaining chemical formulas
haemoglobinA_formula = peptides.get_protein_formula(haemoglobinA_fasta)
haemoglobinB_formula = peptides.get_protein_formula(haemoglobinB_fasta)
myoglobin_formula = peptides.get_protein_formula(myoglobin_fasta)

# Generatin individual spectra
hA19 = Spectrum(haemoglobinA_formula, charge=19, adduct='H', label='hA 19+')
hA20 = Spectrum(haemoglobinA_formula, charge=20, adduct='H', label='hA 20+')
hA21 = Spectrum(haemoglobinA_formula, charge=21, adduct='H', label='hA 21+')
hB20 = Spectrum(haemoglobinB_formula, charge=20, adduct='H', label='hB 20+')
hB21 = Spectrum(haemoglobinB_formula, charge=21, adduct='H', label='hB 21+')
hB22 = Spectrum(haemoglobinB_formula, charge=22, adduct='H', label='hB 22+')
m21  = Spectrum(myoglobin_formula, charge=21, adduct='H', label='myo 21+')
m22  = Spectrum(myoglobin_formula, charge=22, adduct='H', label='myo 22+')
m23  = Spectrum(myoglobin_formula, charge=23, adduct='H', label='myo 23+')
m24  = Spectrum(myoglobin_formula, charge=24, adduct='H', label='myo 24+')
spectra = [hA19, hA20, hA21, hB20, hB21, hB22, m21, m22, m23, m24]
k = len(spectra)

# TIC normalization:
for s in spectra:
    s.normalize()

# Visualization:
#Spectrum.plot_all(spectra)

# Wasserstein distance matrix:
wM = np.zeros((k,k))
for i in range(k):
    for j in range(k):
        wM[i,j] = spectra[i].WSDistance(spectra[j])

# Setting example proportions
proportions = [1, 2, 1.2, 0.5, 0.9, 0.6, 0.2, 0.3, 0.4, 0.]
proportions = [p/sum(proportions) for p in proportions]

# Generating a convolved spectrum
convolved = Spectrum('', empty=True, label='Convolved')
for s, p in zip(spectra, proportions):
    convolved += s*p

##convolved.plot()

##est1 = estimate_proportions(convolved, spectra, MTD=0.001)
##print('True:', 'Estimate w/o noise:', sep='\t')
##for e, p in zip(est1['proportions'], proportions):
##    print(p, e, sep='\t')

# Distorting intensity - simulating finite number of molecules
sampled = Spectrum.sample_multinomial(convolved, 1e05, 1, 0.01)
sampled.add_chemical_noise(1000, 0.1)
sampled.normalize()
plt.subplot(211)
convolved.plot(show=False)
plt.subplot(212)
sampled.plot(show=False)
plt.show()

est2 = estimate_proportions(sampled, spectra, MTD=0.01)
print('True:', 'Estimate after sampling and adding noise peaks:', sep='\t')
for e, p in zip(est2['proportions'], proportions):
    print(p, e, sep='\t')

# Simulating finite resolution
lowres  = deepcopy(sampled)
sampled.coarse_bin(3)
lowres.fuzzify_peaks(0.02, 0.001)
sampled.plot(show=False)
lowres.plot(show=False, profile=True)
plt.show()

# Simulating electronic noise
lowres.add_gaussian_noise(0.001)
lowres.normalize()
plt.subplot(211)
sampled.plot(show=False)
lowres.plot(show=False, profile=True)
plt.subplot(212)
lowres.plot(show=False, profile=True)
plt.show()


est3 = estimate_proportions(lowres, spectra, MTD=0.01)
print('True:', 'Estimate with full noise:', sep='\t')
for e, p in zip(est3['proportions'], proportions):
    print(p, e/sum(est3['proportions']), sep='\t')

# Saving the files

##with open('low_res_spectrum.txt', 'w') as h:
##    for x, y in lowres.confs:
##        h.write(str(x)+'\t'+str(y)+'\n')
##
##with open('molecules.txt', 'w') as h:
##    h.write('#True proportions:')
##    h.write(','.join(map(str, [round(p, 4) for p in proportions])) + '\n')
##    h.write(haemoglobinA_formula+'+H19 \n')
##    h.write(haemoglobinA_formula+'+H20 \n')
##    h.write(haemoglobinA_formula+'+H21 \n')
##    h.write(haemoglobinB_formula+'+H20 \n')
##    h.write(haemoglobinB_formula+'+H21 \n')
##    h.write(haemoglobinB_formula+'+H22 \n')
##    h.write(myoglobin_formula+'+H21 \n')
##    h.write(myoglobin_formula+'+H22 \n')
##    h.write(myoglobin_formula+'+H23 \n')
##    h.write(myoglobin_formula+'+H24 \n')
##print('Spectrum and molecules saved')
