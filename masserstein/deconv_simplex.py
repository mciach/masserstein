import numpy as np
from time import time
from masserstein import Spectrum
import pulp as lp
from warnings import warn


def intensity_generator(confs, mzaxis):
        """
        Generates intensities from spectrum sp, represented as a confs list, over m/z values from mzaxis.
        Assumes mzaxis and confs are sorted and returns consecutive intensities.
        """
        i = 0
        confs.append((-1, -1))
        mzaxis = iter(mzaxis)
        cm = next(mzaxis)
        for m, i in confs:
            while cm < m or m == -1:
                yield 0.
                cm = next(mzaxis)
            if cm == m:
                yield i
                cm = next(mzaxis)
        confs.pop()  # ten pop nie dziala


def dualdeconv2(exp_sp, thr_sps, penalty, quiet=True):
    """
    Different formulation, maybe faster
    exp_sp: experimental spectrum
    thr_sp: list of theoretical spectra
    penalty: denoising penalty
    """
    start = time()
    exp_confs = exp_sp.confs.copy()
    thr_confs = [thr_sp.confs.copy() for thr_sp in thr_sps]
    # Normalization check:
    assert np.isclose(sum(x[1] for x in exp_confs) , 1), 'Experimental spectrum not normalized'
    for i, thrcnf in enumerate(thr_confs):
        assert np.isclose(sum(x[1] for x in thrcnf), 1), 'Theoretical spectrum %i not normalized' % i
    global_mass_axis = set(x[0] for x in exp_confs)
    global_mass_axis.update(x[0] for s in thr_confs for x in s)
    global_mass_axis = sorted(global_mass_axis)
    if not quiet:
        print("Global mass axis computed")
    n = len(global_mass_axis)
    k = len(thr_confs)
    
    interval_lengths = [snext - scur for scur, snext in zip(global_mass_axis[:-1], global_mass_axis[1:])]
    if not quiet:
        print("Interval lengths computed")
    # linear program:
    program = lp.LpProblem('Dual L1 regression sparse', lp.LpMaximize)
    if not quiet:
        print("Linear program initialized")
    # variables:
    lpVars = []
    for i in range(n):
        lpVars.append(lp.LpVariable('Z%i' % (i+1), None, penalty, lp.LpContinuous))
##        # in case one would like to explicitly forbid non-experimental abyss:
##        if V[i] > 0:
##            lpVars.append(lp.LpVariable('W%i' % (i+1), None, penalty, lp.LpContinuous))
##        else:
##            lpVars.append(lp.LpVariable('W%i' % (i+1), None, None, lp.LpContinuous))
    if not quiet:
        print("Variables created")
    # objective function:
    exp_vec = intensity_generator(exp_confs, global_mass_axis)  # generator of experimental intensity observations
    program += lp.lpSum(v*x for v, x in zip(exp_vec, lpVars)), 'Dual objective'
    # constraints:
    for j in range(k):
        thr_vec = intensity_generator(thr_confs[j], global_mass_axis)
        program += lp.lpSum(v*x for v, x in zip(thr_vec, lpVars) if v > 0.) <= 0, 'P%i' % (j+1)
    if not quiet:
        print('tsk tsk')
    for i in range(n-1):
        program += lpVars[i]-lpVars[i+1] <= interval_lengths[i], 'EpsPlus %i' % (i+1)
        program += lpVars[i] - lpVars[i+1] >=  -interval_lengths[i], 'EpsMinus %i' % (i+1)
    if not quiet:
        print("Constraints written")
    program.writeLP('WassersteinL1.lp')
    if not quiet:
        print("Starting solver")
    program.solve()
    end = time()
    if not quiet:
        print("Solver finished.")
        print("Status:", lp.LpStatus[program.status])
        print("Optimal value:", lp.value(program.objective))
        print("Time:", end - start)
    constraints = program.constraints
    probs = [round(constraints['P%i' % i].pi, 12) for i in range(1, k+1)]
    exp_vec = list(intensity_generator(exp_confs, global_mass_axis))
    # 'if' clause below is to restrict returned abyss to experimental confs
    abyss = [round(x.dj, 12) for i, x in enumerate(lpVars) if exp_vec[i] > 0.]
    # note: accounting for number of summands in checking of result correctness,
    # because summation of many small numbers introduces numerical errors
    if not np.isclose(sum(probs)+sum(abyss), 1., atol=len(abyss)*1e-03):
        warn("""Proportions of signal and noise sum to %f instead of 1.
This may indicate improper results.
Please check the deconvolution results and consider reporting this warning to the authors.
                            """ % (sum(probs)+sum(abyss)))
            
    return {"probs": probs, "trash": abyss, "fun": lp.value(program.objective)}


def estimate_proportions(spectrum, query, MTD=0.1, MDC=1e-8, MMD=-1, verbose=False):
    """
    Returns estimated proportions of molecules from query in spectrum.
    Performs initial filtering of formulas and experimental spectrum to speed
    up the computations.
    _____
    Parameters:
    
    spectrum: Spectrum object
        The experimental (subject) spectrum.
    query: list of Spectrum objects
        A list of theoretical (query) spectra.
    MTD: Maximum Transport Distance, float
        Ion current will be transported up to this distance when estimating
        molecule proportions.
    MDC: Minimum Detectable Current, float
        If the isotopic envelope of an ion encompasses less than
        this amount of the total ion current, it is assumed that this ion
        is absent in the spectrum.
    MMD: Maximum Mode Distance, float
        If there is no experimental peak within this distance from the
        highest peak of an isotopic envelope of a molecule,
        it is assumed that this molecule is absent in the spectrum.
        Setting this value to -1 disables filtering.
    TSC: Theoretical Spectrum Coverage, float in [0, 1]
        The peak intensities in any theoretical spectrum will sum up to this value. 
        Setting this value to 1 means that all theoretical peaks are computed,
        which is in general undesirable.
    max_threads: int
        Maximum numbers of subprocesses to spawn during deconvolution.
    verbose: bool
        Print diagnistic messages?
    _____
    Returns: dict
        A dictionary with entry 'proportions', storing a list of proportions of query spectra,
        and 'noise', storing a list of intensities that could not be
        explained by the supplied formulas. The intensities correspond
        to the m/z values of experimental spectrum.
    """
    try:
        exp_confs = spectrum.confs
    except:
        print("Could not retrieve the confs list. Is the supplied spectrum an object of class Spectrum?")
        raise
    assert np.isclose(sum(x[1] for x in exp_confs), 1.), 'The experimental spectrum is not normalized.'
    assert all(x[0] >= 0. for x in exp_confs), 'Found experimental peaks with negative masses!'
    vortex = [0.]*len(exp_confs)  # unxplained signal
    k = len(query)
    proportions = [0.]*k

    for i, q in enumerate(query):
        assert np.isclose(sum(x[1] for x in q.confs), 1.), 'Theoretical spectrum %i is not normalized' %i
        assert all(x[0] >= 0 for x in q.confs), 'Theoretical spectrum %i has negative masses!' % i
        
    # Initial filtering of formulas
    envelope_bounds = []
    filtered = []
    for i in range(k):
        s = query[i]
        mode = s.get_modal_peak()[0]
        mn = s.confs[0][0]
        mx = s.confs[-1][0] 
        matching_current = MDC==0. or sum(x[1] for x in exp_confs if x[0] >= mn - MTD and x[0] <= mx + MTD) >= MDC
        matching_mode = MMD==-1 or min(abs(mode - x[0]) for x in exp_confs) <= MMD
        if matching_mode and matching_current:
            envelope_bounds.append((mn, mx, i))
        else:
            envelope_bounds.append((-1, -1, i))
            filtered.append(i)

    envelope_bounds.sort(key=lambda x: x[0])  # sorting by lower bounds
    if verbose:
        print("Removed theoretical spectra due to no matching experimental peaks:", filtered)
        print('Envelope bounds:', envelope_bounds)
    
    # Computing chunks
    chunkIDs = [0]*k  # Grouping of theoretical spectra
    # Note: order of chunkIDs corresponds to order of query, not the envelope bounds
    # chunk_bounds = mass intervals matching chunks, accounting for mass transport
    # order of chunk_bounds corresponds to increasing chunk ID,
    # so that chunk_bounds[0] is the interval for chunk nr 0
    chunk_bounds = []
    current_chunk = 0
    first_present = 0
    while envelope_bounds[first_present][0] == -1 and first_present < k-1:
        _, _, sp_id = envelope_bounds[first_present]
        chunkIDs[sp_id] = -1
        first_present += 1
    prev_mn, prev_mx, prev_id = envelope_bounds[first_present]
    for i in range(first_present, k):
        mn, mx, sp_id = envelope_bounds[i]
        if mn - prev_mx > 2*MTD:
            current_chunk += 1
            chunk_bounds.append( (prev_mn-MTD, prev_mx+MTD) )
            prev_mn = mn  # get lower bound of new chunk
        prev_mx = mx  # update the lower bound of current chunk
        chunkIDs[sp_id] = current_chunk
    chunk_bounds.append( (prev_mn-MTD, prev_mx+MTD) )
    nb_of_chunks = len(chunk_bounds)
    if verbose:
        print('Number of chunks: %i' % nb_of_chunks)
        print("ChunkIDs:", chunkIDs)
        print("Chunk bounds:", chunk_bounds)

    # Splitting the experimental spectrum into chunks
    exp_conf_chunks = []  # list of indices of experimental confs matching chunks
    current_chunk = 0
    matching_confs = []  # experimental confs matching current chunk
    cur_bound = chunk_bounds[current_chunk]
    for conf_id, cur_conf in enumerate(exp_confs):
        while cur_bound[1] < cur_conf[0] and current_chunk < nb_of_chunks-1:
            exp_conf_chunks.append(matching_confs)
            matching_confs = []
            current_chunk += 1
            cur_bound = chunk_bounds[current_chunk]
        if cur_bound[0] <= cur_conf[0] <= cur_bound[1]:
            matching_confs.append(conf_id)
        else:
            # experimental peaks outside chunks go straight to vortex
            vortex[conf_id] = cur_conf[1]
    exp_conf_chunks.append(matching_confs)
    chunk_TICs = [sum(exp_confs[i][1] for i in chunk_list) for chunk_list in exp_conf_chunks]
    if verbose:
        # print('Trash after filtering:', vortex)
        print("Ion currents in chunks:", chunk_TICs)

    # Deconvolving chunks:
    for current_chunk_ID, conf_IDs in enumerate(exp_conf_chunks):
        if verbose:
            print("Deconvolving chunk %i" % current_chunk_ID)
        if chunk_TICs[current_chunk_ID] < 1e-16:
            # nothing to deconvolve, pushing remaining signal to vortex
            if verbose:
                print('Chunk %i is almost empty - skipping deconvolution' % current_chunk_ID)
            for i in conf_IDs:
                vortex[i] = exp_confs[i][1]
        else:
            chunkSp = Spectrum('', empty=True)
            # Note: conf_IDs are monotonic w.r.t. conf mass,
            # so constructing a spectrum will not change the order
            # of confs supplied in the list below:
            chunkSp.set_confs([exp_confs[i] for i in conf_IDs])
            chunkSp.normalize()
            theoretical_spectra_IDs = [i for i, c in enumerate(chunkIDs) if c == current_chunk_ID]
            thrSp = [query[i] for i in theoretical_spectra_IDs]
            dec = dualdeconv2(chunkSp, thrSp, MTD, quiet=True)
            for i, p in enumerate(dec['probs']):
                original_thr_spectrum_ID = theoretical_spectra_IDs[i]
                proportions[original_thr_spectrum_ID] = p*chunk_TICs[current_chunk_ID]
            for i, p in enumerate(dec['trash']):
                original_conf_id = conf_IDs[i]
                vortex[original_conf_id] = p*chunk_TICs[current_chunk_ID]       

    if not np.isclose(sum(proportions)+sum(vortex), 1., atol=len(vortex)*1e-03):
        warn("""Proportions of signal and noise sum to %f instead of 1.
This may indicate improper results.
Please check the deconvolution results and consider reporting this warning to the authors.
                        """ % (sum(proportions)+sum(vortex)))    
    return {'proportions': proportions, 'noise': vortex}
            

if __name__=="__main__":
    exper = [(1., 1/6.), (2., 3/6.), (3., 2/6.)]
    thr1 = [(1., 1/2.), (2.,1/2.)]
    thr2 = [(2., 1/2.), (3., 1/2.)]

    exper = [(1, 0.25), (3, 0.5), (6, 0.25)]
    thr1 = [(1., 1.), (3, 0.)]
    thr2 = [(3, 0.5), (4, 0.5)]
    thr = [thr1, thr2]

    exper = [(1.1, 1/3), (2.2, 5/12), (3.1, 1/4)]

    
    exper = [(0, 1/4), (1.1, 1/6), (2.2, 5/24), (3.1, 1/8), (4, 1/4), (60, .1) ]
    ##thr1 = [(1, 1/2), (2, 1/2)]
    ##thr2 = [(2, 1/4), (3, 3/4)]
    thr1 = [(0.1, 1./2), (1.0, 1./2)]
    thr2 = [(3., 1/4), (4.2, 3/4.)]
    thr3 = [(0.5, 1/4.), (1.2, 3./4)]
    thr4 = [(20., 1.)]
    thr = [thr1, thr2, thr3, thr4]

    experSp = Spectrum('', empty=True)
    experSp.set_confs(exper)
    experSp.normalize()
    thrSp = [Spectrum('', empty=True) for _ in range(len(thr))]
    for i in range(len(thr)):
        thrSp[i].set_confs(thr[i])
        thrSp[i].normalize()
    sol2 = dualdeconv2(experSp, thrSp, .2)
    print('sum:', sum(sol2['probs']+sol2['trash']))
    test = estimate_proportions(experSp, thrSp, MTD=.2, MMD=0.21)


    # Other tests:
##    experSp2 = Spectrum('', empty=True)
##    fr = exper[:-1].copy()
##    experSp2.set_confs(fr)
##    experSp2.normalize()
##    sol22 = dualdeconv2(experSp2, thrSp[:2], .2)
##    print('sum:', sum(sol22['probs']+sol22['trash']))
##
##    global_mass_axis = set(x[0] for x in experSp2.confs)
##    global_mass_axis.update(x[0] for s in thrSp for x in s.confs)
##    global_mass_axis = sorted(global_mass_axis)
##
##    thr_conf_iters = [list(intensity_generator(t.confs, global_mass_axis)) for t in thrSp]
##    thr_conf_iters = [[(m, i) for m, i in zip(global_mass_axis, cnflist) if i>0] for cnflist in thr_conf_iters]
##
##    exper2 = [(0, 1/4), (1.1, 1/6), (2.2, 5/24), (3.1, 1/8), (4, 1/4)]
##    sp2 = Spectrum('', empty=True)
##    sp2.set_confs(exper2)
##    sp2.normalize()
##    noise = Spectrum('', empty=True)
##    noise.set_confs([exper2[2]])
##    noise.normalize()
##    sp2.WSDistance(thrSp[0]*0.35 + thrSp[1]*0.5 + noise*0.125)
##    dualdeconv2(sp2, thrSp[:2], 1.)
##
##    thr1 = [(1., 1.)]
##    thr2 = [(2., 1.)]
##    exper = [(1., 0.4), (1.5, 0.2), (2, 0.4)]
##    thr = [thr1, thr2]
##    experSp = Spectrum('', empty=True)
##    experSp.set_confs(exper)
##    experSp.normalize()
##    thrSp = [Spectrum('', empty=True) for _ in range(len(thr))]
##    for i in range(len(thr)):
##        thrSp[i].set_confs(thr[i])
##        thrSp[i].normalize()
##    sol2 = dualdeconv2(experSp, thrSp, .5)
