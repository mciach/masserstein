# Contents of this repository
This repository contains software tools which allow to compare spectra using the Wasserstein distance and estimate relative abundances of molecules from the spectrum by minimizing the Wasserstein distance. 

The tools are distributed as a Python3 package called `masserstein`. The package contains functions for generating spectra from chemical formulas, basic processing (centroiding, smoothing), and fitting combinations of reference spectra of analytes to a spectrum of their mixture.  

Example uses of the package are shown in the tutorials available on this repository. The `Tutorials/` directory contains Jupyter notebooks that show:   
1. The basic functionality of the package (`Tutorials/Package presentation.ipynb`);
2. An application of the package to study structural defects in alternating co-polymers (`Tutorials/PBTTT_polymer_analysis/PBTTT_analysis.ipynb`);
3. An application of the package to analyze Mass Spectrometry Imaging data (`Tutorials/Analysis of Mass Spectrometry Imaging data.ipynb`).  

If you encounter any difficulties during installation or usage of these programs, or if you have any suggestions regarding their functionality, please post a GitHub issue or send an email to michal.ciach@um.edu.mt. 

# Installation

To be able to use the software provided in this repository, you will need to have a working Python3 distribution installed on your computer.  
The simplest way to install `masserstein` is to use the `pip` tool in the command-line: 

```
pip install masserstein
```

This will install the latest stable release of the package.  
To get the development version, clone this repository. In the commandline, this can be done by typing:

```
git clone https://github.com/mciach/masserstein.git
```

The above command will create a folder `masserstein` in your current working directory. The folder contains the setup file and some example data. Finally, install the package by running the `setup.py` file:

```
python3 setup.py install --user
```

This will install the `masserstein` package for the current user.  
You will also need to have the following packages installed (all availiable via pip):

* `IsoSpecPy`
* `numpy`
* `scipy`
* `PuLP`


# Citing 

If you use tools from this package, please cite the following article:  

Ciach, M. A., Miasojedow, B., Skoraczynski, G., Majewski, S., Startek, M., Valkenborg, D., & Gambin, A. (2020). Masserstein: linear regression of mass spectra by optimal transport. Rapid Communications in Mass Spectrometry.


