from collections import Counter
import re


aminoacids = {
'A' : 'C3H5N1O1',
'C' : 'C3H5N1O1S1',
'D' : 'C4H5N1O3',
'E' : 'C5H7N1O3',
'F' : 'C9H9N1O1',
'G' : 'C2H3N1O1',
'H' : 'C6H7N3O1',
'I' : 'C6H11N1O1',
'K' : 'C6H12N2O1',
'L' : 'C6H11N1O1',
'M' : 'C5H9N1O1S1',
'N' : 'C4H6N2O2',
'O' : 'C12H21N3O3',
'P' : 'C5H7N1O1',
'Q' : 'C5H8N2O2',
'R' : 'C6H12N4O1',
'S' : 'C3H5N1O2',
'T' : 'C4H7N1O2',
'U' : 'C3H5N1O1Se1',
'V' : 'C5H9N1O1',
'W' : 'C11H10N2O1',
'Y' : 'C9H9N1O2',
' ' : '',
'\n' : ''
}

daminoacids = {}

# From now we are able to model all changes which happen to peptides!
# Notation of changes is a modX proposed by pyteomics.

# Variable modifications of peptides
# N -> D # deamidation
# Q -> E # deamidation
# Q -> pyroglutamic acid # deamidation
# M -> methionine sulfoxide # oxidation
modifications = [("deaN", Counter({"H": -1, "N": 1, "O": 1})),
                 ("deaeQ", Counter({"H": -1, "N": -1, "O": 1})),
                 ("deapQ", Counter({"H": -3, "N": -1})),
                 ("oxM", Counter({"O": 1})),
                 ("carC", Counter({"C": 2, "H": 4, "N": 2, "O": 1}))]

for symbol, formula in aminoacids.items():
    symbols = re.findall("\D+", formula)
    atom_counts = [int(x) for x in re.findall("\d+", formula)]
    C = Counter(dict(zip(symbols, atom_counts)))
    daminoacids[symbol] = C

def get_protein_counter(seq, add_water=True):
    cnts = [(k, sum(char == k for char in seq)) for k in aminoacids]
    cnts = Counter(dict(k for k in cnts if k[1] > 0))
    return aacnt_to_elecnt(cnts, add_water=add_water)

def get_protein_formula(seq, add_water=True):
    c = get_protein_counter(seq, add_water)
    # Apply modifications
    for mod_desc, mod_val in modifications:
        mod_cnt = seq.count(mod_desc)
        for _ in range(mod_cnt):
            # Oh yeah, Counter supports + but does not support *.
            # :-)
            c += mod_val
    return ''.join(sym+str(count) for sym,count in sorted(c.items()))

def aacnt_to_elecnt(cnts, add_water = True):
    if add_water:
        ret = Counter({"H":2, "O":1})
    else:
        ret = Counter()
    for aa, cnt in cnts.items():
        for ele, ecnt in daminoacids[aa].items():
            ret[ele] += ecnt*cnt
    return ret

if __name__ == "__main__":
    import sys
    print(get_protein_formula(sys.argv[1]))
