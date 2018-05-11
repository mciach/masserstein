


doc="""
WSDistance.py

USAGE:
    python WSDistance.py [OPTIONS] FILE1 FILE2

EXAMPLE:
    python WSDistance.py spectrum1.txt spectrum2.txt
    
DESCRIPTION:
    Computes the Wasserstein distance between two spectra.
    The two spectra need to be supplied as peak lists in separate files (FILE1, FILE2).
    Each file should be composed of two tab-separated columns.
    The first column corresponds to m/z values, the second one to peak intensity.
    The intensities need not be normalized. The normalization will be performed by the program.

OPTIONS:
    -n
        suppress normalization of spectra
    -t: float
        set intensity cutoff threshold
