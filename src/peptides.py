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

for symbol, formula in aminoacids.items():
    symbols = re.findall("\D+", formula)
    atom_counts = [int(x) for x in re.findall("\d+", formula)]
    C = Counter(dict(zip(symbols, atom_counts)))
    daminoacids[symbol] = C


def get_protein_formula(seq):
    global daminoacids
    c = Counter()
    for aminoacid in seq:
        c += daminoacids[aminoacid]


    return ''.join(sym+str(count) for sym,count in c.items())

