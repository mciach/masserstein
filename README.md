# Contents of this repository
This repository contains software tools which allow to compare spectra using the Wasserstein distance and estimate relative abundances of molecules from the spectrum by minimizing the Wasserstein distance. 

The tools are distributed as a Python3 package called `masserstein`. Basic functionality is also available as a set of commandline applications: `WSDistance` to compute the Wasserstein distance and `WSDeconv` to estimate proportions. 

If you encounter any difficulties during installation or usage of these programs, or if you have any suggestions regarding their functionality, please post a GitHub issue or send an email to m.ciach@mimuw.edu.pl. 

# Installation

To be able to use the software provided in this repository, you will need to have a working Python3 distribution installed on your computer. You will also need to have the following packages installed (availiable on PIP):

* `IsoSpecPy`
* `numpy`
* `scipy`
* `PuLP`

After installing these packages, download or clone this repository. In the commandline, this can be done by typing:

```
git clone https://github.com/mciach/wassersteinms.git
```

The above command will create a folder `wassersteinms` in your current working directory. The folder contains the setup file and some example data. Finally, install the package by running the `setup.py` file:

```
python3 setup.py install --user
```

This will install the `masserstein` package for the current user, as well as the applications `WSDistance` and `WSDeconv`.


# Commandline usage

The commandline applications are a (moderately) convenient way to use the basic functionality of the package. They do not require knowledge of any programming language, but require a basic familiarity with the commandline. At this moment, no graphical interface is provided.

The `WSDist` application allows to compute the Wasserstein distance between two spectra. The spectra need to be supplied in a peaklist format: a text file with two tab-separated columns, with m/z values in the first column and signal intensity in the second one. Additionally, comments may be included in the file by starting a line with a hash `#` symbol. Example files are provided with the package, e.g. `examples/small_molecule_spectrum.txt`. 

The basic usage of the application is as follows:

```
WSDistance spectrum1.txt spectrum2.txt
```

This will perform a basic normalization and denoising of both spectra and print the distance to the command line. Additional options include fine-tuning the denoising procedure and printing the optimal signal transport scheme. More details and usage examples are available in the help message of the application, which can be obtained by typing `python WSDist.py -h`. This application does not support the transport distance limit ('the vortex'). 

The `WSDeconvolve` application allows to obtain proportions of a set of compounds (the 'query' molecules) by minimizing the Wasserstein distance between the observed spectrum and a linear combination of calculated theoretical spectra of the compouns. The user needs to supply a spectrum in the peaklist format and a file with a list of the elemental compositions of the query molecules. Each query molecular formula should consist of the neutral part, followed by a charge sign, followed by the adduct formula. For example, `C685H1071N187O194S3 + H19` represents a 19+ protonated human haemoglobin A.  If the adduct and charge signs are ommited, it is assumed that the formula corresponds to an `[M]+` ion. The theoretical isotopic envelopes of the supplied molecules are computed using the `IsoSpec` algorithm. 

The basic usage is as follows:

```
WSDeconvolve [OPTIONS] spectrum.txt molecule_list.txt [OUTPUT_FILENAME]
```

If `[OPTIONS]` and `[OUTPUT_FILENAME]` are ommited, this will print the program configuration, followed by a table containing the molecules and the corresponding proportions of their isotopic envelopes. The program also reports the optimal distance including the cost of the 'vortex', as well as the optimal distance after removing signal from the vortex (i.e. optimal transport distance after removing signal detected as noise).

More details, such as options to fine-tune the program and save the detailed results in a file, are described in the help message obtainable by running  `WSDeconvolve`. The help message also shows several example runs, such as

```
WSDeconv examples/protein_spectrum.txt examples/protein_molecule_list.txt Proteins_example
``` 

# Programmatic usage

A more advanced functionality can be accessed using the `masserstein` package in the Python programming language. The package includes functions to create theoretical or experimental spectrum objects and a minimalistic spectrum processing toolbox which allows, among others, to smooth and peak-pick the spectra, as well as to distort them by simulating realistic noise. This package can be used to estimate proportions using experimentally measured query spectra instead of the ones simulated by IsoSpec.

Examples of use of the package functions are shown in the tutorials available on this repository.  


# Citing 

If you use tools from this package, please cite the following article:  

Ciach, M. A., Miasojedow, B., Skoraczynski, G., Majewski, S., Startek, M., Valkenborg, D., & Gambin, A. (2020). Masserstein: linear regression of mass spectra by optimal transport. Rapid Communications in Mass Spectrometry.


