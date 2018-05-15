# Wasserstein MS
This repository contains software tools which allow to:
1. Compare two spectra using the Wasserstein distance (WSDist.py),
2. Deconvolve a spectrum to obtain proportions of overlapping isotopic distributions (WSDeconvolve.py).

If you encounter any difficulties during installation or usage of these programs, or if you have any suggestions regarding their functionality, please send an email to m_ciach@student.uw.edu.pl. 

# Installation

To be able to use the software provided in this repository, you will need to have a working Python 2.7 distribution installed on your computer. You will also need to have the following packages installed (availiable on PIP):

* IsoSpecPy
* Numpy
* Scipy

After installing the requirements, simply download or clone this repository. On Ubuntu linux, this can be done by typing into the command line:

```
git clone https://github.com/mciach/wassersteinms.git
```

This will create a folder `wassersteinms` in your current working directory. The folder contains the programs and some example data. 

# Usage

This repository contains two programs: WSDist.py and WSDeconvolve.py. At this moment, both programs are commandline applications, and no graphical interface is provided. Example data for both programs is availiable in the `examples` folder.

The WSDist.py application allows to compute the Wasserstein distance between two spectra. Intuitively, the Wasserstein distance is the total distance that the ion current needs to travel from one spectrum into the other. The spectra need to be supplied as peak lists in text files. The basic usage is as follows:

```
python WSDist.py spectrum1.txt spectrum2.txt
```

This will perform a basic normalization and denoising of the spectra and print the distance into the command line. Additional options include fine-tuning the denoising and printing the transport scheme. More details are availiable in the help message of the application, which can be obtained by typing `python WSDist.py -h`. 

The WSDeconvolve.py allows to obtain proportions of overlapping isotopic distributions of several compounds. The user needs to supply a file with a peak list of the spectrum and a file listing the elemental compositions of the molecules that are to be deconvolved. The program will automatically compute the theoretical isotopic envelopes of the supplied molecules, using the IsoSpec algorithm. The basic usage is similar to WSDeconvolve.py:

```
python WSDeconvolve.py spectrum.txt molecule_list.txt
```

This will print the program configuration, followed by a table containing the molecules and the corresponding proportions of their isotopic envelopes. Typing `python WSDeconvolve.py` will print the help message with description of additional options.

Note that when specifying the molecules in the molecule list, after each element a number needs to be present. For example, S1N2 is the correct way to indicate one sulfur and one nitrogen atom, while SN2 will be interpreted as two atoms of tin.  

