import numpy as np
from time import time
from masserstein import Spectrum
from masserstein import NMRSpectrum
import pulp as lp
from warnings import warn
import tempfile
from tqdm import tqdm
from pulp.apis import LpSolverDefault
from masserstein import misc



def intensity_generator(confs, mzaxis):
        """
        Generates intensities from spectrum represented as a confs list,
        over m/z values from mzaxis.
        Assumes mzaxis and confs are sorted and returns consecutive intensities.
        """
        mzaxis_id = 0
        mzaxis_len = len(mzaxis)
        for mz, intsy in confs:
            while mzaxis[mzaxis_id] < mz:
                yield 0.
                mzaxis_id += 1
                if mzaxis_id == mzaxis_len:
                    return
            if mzaxis[mzaxis_id] == mz:
                yield intsy
                mzaxis_id += 1
                if mzaxis_id == mzaxis_len:
                    return
        for i in range(mzaxis_id, mzaxis_len):
                yield 0.


def dualdeconv2(exp_sp, thr_sps, penalty, quiet=True, solver=LpSolverDefault):
        """
        This function solves linear program describing optimal transport of signal between the experimental spectrum
        and the list of theoretical (reference) spectra. Additionally, an auxiliary point is introduced in order to
        remove noise from the experimental spectrum, as described by Ciach et al., 2020. 
        _____
        Parameters:
            exp_sp: Spectrum object
                Experimental spectrum.
            thr_sp: list of Spectrum objects
                List of theoretical (reference) spectra.
            penalty: float
                Denoising penalty.
            solver: 
                Which solver should be used. In case of problems with the default solver,
                pulp.GUROBI() is recommended (note that it requires obtaining a licence).
                To see all solvers available at your machine execute: pulp.listSolvers(onlyAvailable=True).
        _____
        Returns: dict
            Dictionary with the following entries:
            - probs: List containing proportions of consecutive theoretical (reference) spectra in the experimental
            spectrum. Note that they do not have to sum up to 1, because some part of the signal can be noise.
            - trash: Amount of noise in the consecutive m/z points of the experimental spectrum.
            - fun: Optimal value of the objective function.
            - status: Status of the linear program.
        """
        start = time()
        exp_confs = exp_sp.confs.copy()
        thr_confs = [thr_sp.confs.copy() for thr_sp in thr_sps]

        # Normalization check:
        assert np.isclose(sum(x[1] for x in exp_confs) , 1), 'Experimental spectrum not normalized'
        for i, thrcnf in enumerate(thr_confs):
                assert np.isclose(sum(x[1] for x in thrcnf), 1), 'Theoretical spectrum %i not normalized' % i

        # Computing a common mass axis for all spectra
        exp_confs = [(m, i) for m, i in exp_confs]
        thr_confs = [[(m, i) for m, i in cfs] for cfs in thr_confs]
        global_mass_axis = set(x[0] for x in exp_confs)
        global_mass_axis.update(x[0] for s in thr_confs for x in s)
        global_mass_axis = sorted(global_mass_axis)
        if not quiet:
                print("Global mass axis computed")
        n = len(global_mass_axis)
        k = len(thr_confs)

        # Computing lengths of intervals between mz measurements (l_i variables)
        interval_lengths = [global_mass_axis[i+1] - global_mass_axis[i] for i in range(n-1)]
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
        program += lp.lpSum(v*x for v, x in zip(exp_vec, lpVars)), 'Dual_objective'
        # constraints:
        for j in range(k):
                thr_vec = intensity_generator(thr_confs[j], global_mass_axis)
                program += lp.lpSum(v*x for v, x in zip(thr_vec, lpVars) if v > 0.) <= 0, 'P%i' % (j+1)
        if not quiet:
                print('tsk tsk')
        ##    for i in range(n-1):
        ##        program += lpVars[i]-lpVars[i+1] <= interval_lengths[i], 'EpsPlus %i' % (i+1)
        ##        program += lpVars[i] - lpVars[i+1] >=  -interval_lengths[i], 'EpsMinus %i' % (i+1)
        for i in range(n-1):
                program +=  lpVars[i] - lpVars[i+1]  <=  interval_lengths[i], 'EpsPlus_%i' % (i+1)
                program +=  lpVars[i] - lpVars[i+1]  >= -interval_lengths[i], 'EpsMinus_%i' % (i+1)
        if not quiet:
                print("Constraints written")
        #program.writeLP('WassersteinL1.lp')
        if not quiet:
                print("Starting solver")
        LpSolverDefault.msg = not quiet
        program.solve(solver = solver)
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
                warn("""In dualdeconv2:
                Proportions of signal and noise sum to %f instead of 1.
                This may indicate improper results.
                Please check the deconvolution results and consider reporting this warning to the authors.
                                    """ % (sum(probs)+sum(abyss)))

        return {"probs": probs, "trash": abyss, "fun": lp.value(program.objective), 'status': program.status}


def dualdeconv2_alternative(exp_sp, thr_sps, penalty, quiet=True, solver=LpSolverDefault):

        """
        Alternative version of dualdeconv2 - using .pi instead of .dj to extract optimal values of variables.
        Slower, but .pi is better documented than .dj in pulp. Gives the same results as dualdeconv2.
        This function solves linear program describing optimal transport of signal between the experimental spectrum
        and the list of theoretical (reference) spectra. Additionally, an auxiliary point is introduced in order to
        remove noise from the experimental spectrum, as described by Ciach et al., 2020. 
        _____
        Parameters:
            exp_sp: Spectrum object
                Experimental spectrum.
            thr_sp: list of Spectrum objects
                List of theoretical (reference) spectra.
            penalty: float
                Denoising penalty.
            solver: 
                Which solver should be used. In case of problems with the default solver,
                pulp.GUROBI() is recommended (note that it requires obtaining a licence).
                To see all solvers available at your machine execute: pulp.listSolvers(onlyAvailable=True).
        _____
        Returns: dict
            Dictionary with the following entries:
            - probs: List containing proportions of consecutive theoretical (reference) spectra in the experimental
            spectrum. Note that they do not have to sum up to 1, because some part of the signal can be noise.
            - trash: Amount of noise in the consecutive m/z points of the experimental spectrum.
            - fun: Optimal value of the objective function.
            - status: Status of the linear program.
        """

        start = time()
        exp_confs = exp_sp.confs.copy()
        thr_confs = [thr_sp.confs.copy() for thr_sp in thr_sps]

        # Normalization check:
        assert np.isclose(sum(x[1] for x in exp_confs) , 1), 'Experimental spectrum not normalized'
        for i, thrcnf in enumerate(thr_confs):
                assert np.isclose(sum(x[1] for x in thrcnf), 1), 'Theoretical spectrum %i not normalized' % i

        # Computing a common mass axis for all spectra
        exp_confs = [(m, i) for m, i in exp_confs]
        thr_confs = [[(m, i) for m, i in cfs] for cfs in thr_confs]
        global_mass_axis = set(x[0] for x in exp_confs)
        global_mass_axis.update(x[0] for s in thr_confs for x in s)
        global_mass_axis = sorted(global_mass_axis)
        if not quiet:
                print("Global mass axis computed")
        n = len(global_mass_axis)
        k = len(thr_confs)

        # Computing lengths of intervals between mz measurements (l_i variables)
        interval_lengths = [global_mass_axis[i+1] - global_mass_axis[i] for i in range(n-1)]
        if not quiet:
                print("Interval lengths computed")

        # linear program:
        program = lp.LpProblem('Dual L1 regression sparse', lp.LpMaximize)
        if not quiet:
                print("Linear program initialized")
        # variables:
        lpVars = []
        for i in range(n):
                lpVars.append(lp.LpVariable('Z%i' % (i+1), None, None, lp.LpContinuous))
        ##        # in case one would like to explicitly forbid non-experimental abyss:
        ##        if V[i] > 0:
        ##            lpVars.append(lp.LpVariable('W%i' % (i+1), None, penalty, lp.LpContinuous))
        ##        else:
        ##            lpVars.append(lp.LpVariable('W%i' % (i+1), None, None, lp.LpContinuous))
        if not quiet:
                print("Variables created")
        # objective function:
        exp_vec = intensity_generator(exp_confs, global_mass_axis)  # generator of experimental intensity observations
        program += lp.lpSum(v*x for v, x in zip(exp_vec, lpVars)), 'Dual_objective'
        # constraints:
        for j in range(k):
                thr_vec = intensity_generator(thr_confs[j], global_mass_axis)
                program += lp.lpSum(v*x for v, x in zip(thr_vec, lpVars) if v > 0.) <= 0, 'P%i' % (j+1)
        if not quiet:
                print('tsk tsk')
        for i in range(n):
                program += lpVars[i] <= penalty, 'g%i' % (i+1)
        for i in range(n-1):
                program +=  lpVars[i] - lpVars[i+1]  <=  interval_lengths[i], 'EpsPlus_%i' % (i+1)
                program +=  lpVars[i] - lpVars[i+1]  >= -interval_lengths[i], 'EpsMinus_%i' % (i+1)
        if not quiet:
                print("Constraints written")
        #program.writeLP('WassersteinL1.lp')
        if not quiet:
                print("Starting solver")
        LpSolverDefault.msg = not quiet
        program.solve(solver = solver)
        end = time()
        if not quiet:
                print("Solver finished.")
                print("Status:", lp.LpStatus[program.status])
                print("Optimal value:", lp.value(program.objective))
                print("Time:", end - start)
        constraints = program.constraints
        probs = [round(constraints['P%i' % i].pi, 12) for i in range(1, k+1)]
        exp_vec = list(intensity_generator(exp_confs, global_mass_axis))
        abyss = [round(constraints['g%i' % i].pi, 12) for i in range(1, n+1) if exp_vec[i-1] > 0.]
        # note: accounting for number of summands in checking of result correctness,
        # because summation of many small numbers introduces numerical errors
        if not np.isclose(sum(probs)+sum(abyss), 1., atol=len(abyss)*1e-03):
                warn("""In dualdeconv2_alternative:
                Proportions of signal and noise sum to %f instead of 1.
                This may indicate improper results.
                Please check the deconvolution results and consider reporting this warning to the authors.
                                    """ % (sum(probs)+sum(abyss)))

        return {"probs": probs, "trash": abyss, "fun": lp.value(program.objective), 'status': program.status}



def dualdeconv3(exp_sp, thr_sps, penalty, penalty_th, quiet=True, solver=LpSolverDefault):

        """
        This function solves linear program describing optimal transport of signal between 
        the experimental spectrum and the list of theoretical (reference) spectra. 
        Two auxiliary points are introduced in order to remove noise from the experimental spectrum
        and from the combination of theoretical (reference) spectra, as described by Domżał et al., 2022. 
        Transport of signal between the two auxiliary points is explicitly forbidden.
        Mathematically, this formulation is equivalent to the one implemented in dualdeconv4
        and both give the same results up to roundoff errors.
        _____
        Parameters:
            exp_sp: Spectrum object
                Experimental spectrum.
            thr_sp: Spectrum object
                List of theoretical (reference) spectra.
            penalty: float
                Denoising penalty for the experimental spectrum.
            penalty_th: float
                Denoising penalty for the theoretical (reference) spectra.
            solver: 
                Which solver should be used. In case of problems with the default solver,
                pulp.GUROBI() is recommended (note that it requires obtaining a licence).
                To see all solvers available at your machine execute: pulp.listSolvers(onlyAvailable=True).
        _____
        Returns: dict
            Dictionary with the following entries:
            - probs: List containing proportions of consecutive theoretical (reference) spectra in the experimental
            spectrum. Note that they do not have to sum up to 1, because some part of the signal can be noise.
            - noise_in_theoretical: Proportion of noise present in the combination of theoretical
            (reference) spectra.
            - trash: Amount of noise in the consecutive m/z points of the experimental spectrum.
            - theoretical trash: Amount of noise present in the combination of theoretical (reference)
            spectra in consecutive m/z points from global mass axis.
            - fun: Optimal value of the objective function.
            - status: Status of the linear program.
            - global mass axis: All the m/z values from the experimental spectrum and from the theoretical 
            (reference) spectra in a sorted list. 
        """

        start = time()
        exp_confs = exp_sp.confs.copy()
        thr_confs = [thr_sp.confs.copy() for thr_sp in thr_sps]

        # Normalization check:
        assert np.isclose(sum(x[1] for x in exp_confs) , 1), 'Experimental spectrum not normalized'
        for i, thrcnf in enumerate(thr_confs):
                assert np.isclose(sum(x[1] for x in thrcnf), 1), 'Theoretical spectrum %i not normalized' % i

        # Computing a common mass axis for all spectra
        exp_confs = [(m, i) for m, i in exp_confs]
        thr_confs = [[(m, i) for m, i in cfs] for cfs in thr_confs]
        global_mass_axis = set(x[0] for x in exp_confs)
        global_mass_axis.update(x[0] for s in thr_confs for x in s)
        global_mass_axis = sorted(global_mass_axis)
        if not quiet:
                print("Global mass axis computed")
        n = len(global_mass_axis)
        k = len(thr_confs)

        # Computing lengths of intervals between mz measurements (l_i variables)
        interval_lengths = [global_mass_axis[i+1] - global_mass_axis[i] for i in range(n-1)]
        if not quiet:
                print("Interval lengths computed")

        # linear program:
        program = lp.LpProblem('Dual L1 regression sparse', lp.LpMaximize)
        if not quiet:
                print("Linear program initialized")

        # variables:
        lpVars = []
        try:
                for i in range(n-2):
                        lpVars.append(lp.LpVariable('Z%i' % (i+1), None, None, lp.LpContinuous))
                lpVars.append(lp.LpVariable('Z%i' % (n-1), -interval_lengths[n-2], interval_lengths[n-2], lp.LpContinuous))
        except IndexError: #linear program makes no sense if n<=2 (n-number of points in mixture's spectrum)
                pass
        lpVars.append(lp.LpVariable('Z%i' % (n), None, None, lp.LpContinuous))
        lpVars.append(lp.LpVariable('Z%i' % (n+1), None, None, lp.LpContinuous))
        lpVars.append(lp.LpVariable('Z%i' % (n+2), 0, None, lp.LpContinuous))
        lpVars.append(lp.LpVariable('Z%i' % (n+3), 0, None, lp.LpContinuous))

        if not quiet:
                print("Variables created")

        # objective function:
        exp_vec = intensity_generator(exp_confs, global_mass_axis)  # generator of experimental intensity observations
        program += lp.lpSum(v*x for v, x in zip(exp_vec, lpVars[:n-1]+[0])).addInPlace(
                                lp.lpSum(v*x for v, x in zip([-1, 0, 0, -1], lpVars[n-1:]))), 'Dual_objective'

        # constraints:
        for j in range(k):
                thr_vec = intensity_generator(thr_confs[j], global_mass_axis)
                program += lp.lpSum(v*x for v, x in zip(thr_vec, lpVars[:n-1]+[0]) if v > 0.).addInPlace(
                                lp.lpSum(v*x for v, x in zip([-1, 0, 1, -1], lpVars[n-1:]))) <= 0, 'P_%i' % (j+1)

        exp_vec = intensity_generator(exp_confs, global_mass_axis)
        program += lp.lpSum(v*x for v, x in zip(exp_vec, lpVars[:n-1]+[0])).addInPlace(
                                lp.lpSum(v*x for v, x in zip([0, 1, -1, 0], lpVars[n-1:]))) <= 0, 'p0_prime'

        if not quiet:
                print('tsk tsk')

        for i in range(n-1):
                program +=  lpVars[i] - lpVars[n-1]  <=  penalty, 'g_%i' % (i+1)
                program +=  -lpVars[n] - lpVars[i] <= penalty_th, 'g_prime_%i' % (i+1)
        try:
                for i in range(n-2):
                        program += lpVars[i] - lpVars[i+1] <= interval_lengths[i], 'epsilon_plus_%i' % (i+1)
                        program += lpVars[i+1] - lpVars[i] <= interval_lengths[i], 'epsilon_minus_%i' % (i+1)
        except IndexError: #linear program makes no sense if n<=2 (n-number of points in mixture's spectrum)
                pass

        program += -lpVars[n-1] <= penalty, 'g_%i' % (n)
        program += -lpVars[n] <= penalty_th, 'g_prime_%i' % (n)

        if not quiet:
                print("Constraints written")

        if not quiet:
                print("Starting solver")

        #Solving
        LpSolverDefault.msg = not quiet
        program.solve(solver = solver)
        end = time()
        if not quiet:
                print("Solver finished.")
                print("Status:", lp.LpStatus[program.status])
                print("Optimal value:", lp.value(program.objective))
                print("Time:", end - start)
        constraints = program.constraints
        probs = [round(constraints['P_%i' % i].pi, 12) for i in range(1, k+1)]
        p0_prime = round(constraints['p0_prime'].pi, 12)
        exp_vec = list(intensity_generator(exp_confs, global_mass_axis))
        abyss = [round(constraints['g_%i' % i].pi, 12) for i in range(1, n+1) if exp_vec[i-1] > 0.]
        abyss_th = [round(constraints['g_prime_%i' % i].pi, 12) for i in range(1, n+1)]

        if not np.isclose(sum(probs)+sum(abyss), 1., atol=len(abyss)*1e-03):
                warn("""In dualdeconv3:
                Proportions of signal and noise sum to %f instead of 1.
                This may indicate improper results.
                Please check the deconvolution results and consider reporting this warning to the authors.
                                    """ % (sum(probs)+sum(abyss)))

        return {"probs": probs, "noise_in_theoretical": p0_prime, "trash": abyss, "theoretical_trash": abyss_th,
         "fun": lp.value(program.objective), 'status': program.status, 'global_mass_axis': global_mass_axis}


def dualdeconv4(exp_sp, thr_sps, penalty, penalty_th, quiet=True, solver=LpSolverDefault):

        """
        This function solves linear program describing optimal transport of signal between the experimental 
        spectrum and the list of theoretical (reference) spectra. 
        Two auxiliary points are introduced in order to remove noise from the experimental spectrum
        and from the combination of theoretical (reference) spectra, as described by Domżał et al., 2022. 
        Transport of signal between the two auxiliary points is allowed (with cost equal to penalty + penalty_th),
        however, it is not optimal so it never occurs. Mathematically, this formulation is equivalent to the one 
        implemented in dualdeconv3 and both give the same results up to roundoff errors.
        _____
        Parameters:
            exp_sp: Spectrum object
                Experimental spectrum.
            thr_sp: Spectrum object
                List of theoretical (reference) spectra.
            penalty: float
                Denoising penalty for the experimental spectrum.
            penalty_th: float
                Denoising penalty for the theoretical (reference) spectra.
            solver: 
                Which solver should be used. In case of problems with the default solver,
                pulp.GUROBI() is recommended (note that it requires obtaining a licence).
                To see all solvers available at your machine execute: pulp.listSolvers(onlyAvailable=True).
        
        _____
        Returns: dict
            Dictionary with the following entries:
            - probs: List containing proportions of consecutive theoretical (reference) spectra in the experimental
            spectrum. Note that they do not have to sum up to 1, because some part of the signal can be noise.
            - noise_in_theoretical: Proportion of noise present in the combination of theoretical
            (reference) spectra.
            - trash: Amount of noise in the consecutive m/z points of the experimental spectrum.
            - theoretical trash: Amount of noise present in the combination of theoretical (reference)
            spectra in consecutive m/z points from global mass axis.
            - fun: Optimal value of the objective function.
            - status: Status of the linear program.
            - global mass axis: All the m/z values from the experimental spectrum and from the theoretical 
            (reference) spectra in a sorted list. 
        """

        start = time()
        exp_confs = exp_sp.confs.copy()
        thr_confs = [thr_sp.confs.copy() for thr_sp in thr_sps]

        # Normalization check:
        assert np.isclose(sum(x[1] for x in exp_confs) , 1), 'Experimental spectrum not normalized'
        for i, thrcnf in enumerate(thr_confs):
                assert np.isclose(sum(x[1] for x in thrcnf), 1), 'Theoretical spectrum %i not normalized' % i

        # Computing a common mass axis for all spectra
        exp_confs = [(m, i) for m, i in exp_confs]
        thr_confs = [[(m, i) for m, i in cfs] for cfs in thr_confs]
        global_mass_axis = set(x[0] for x in exp_confs)
        global_mass_axis.update(x[0] for s in thr_confs for x in s)
        global_mass_axis = sorted(global_mass_axis)
        if not quiet:
                print("Global mass axis computed")
        n = len(global_mass_axis)
        k = len(thr_confs)

        # Computing lengths of intervals between mz measurements (l_i variables)
        interval_lengths = [global_mass_axis[i+1] - global_mass_axis[i] for i in range(n-1)]
        if not quiet:
                print("Interval lengths computed")

        # linear program:
        program = lp.LpProblem('Dual L1 regression sparse', lp.LpMaximize)
        if not quiet:
                print("Linear program initialized")

        # variables:
        lpVars = []
        try:
                for i in range(n-2):
                        lpVars.append(lp.LpVariable('Z%i' % (i+1), None, None, lp.LpContinuous))
                lpVars.append(lp.LpVariable('Z%i' % (n-1), -interval_lengths[n-2], interval_lengths[n-2], lp.LpContinuous))
        except IndexError: #linear program makes no sense if n<=2 (n-number of points in mixture's spectrum)
                pass
        lpVars.append(lp.LpVariable('Z%i' % n, None, None, lp.LpContinuous))
        lpVars.append(lp.LpVariable('Z%i' % (n+1), None, None, lp.LpContinuous))
        lpVars.append(lp.LpVariable('Z%i' % (n+2), 0, None, lp.LpContinuous))
        lpVars.append(lp.LpVariable('Z%i' % (n+3), 0, None, lp.LpContinuous))
        if not quiet:
                print("Variables created")

        # objective function:
        exp_vec = intensity_generator(exp_confs, global_mass_axis)  # generator of experimental intensity observations
        program += lp.lpSum(v*x for v, x in zip(exp_vec, lpVars[:n-1]+[0])).addInPlace(
                            lp.lpSum(v*x for v, x in zip([-1, 0, 0, -1], lpVars[n-1:]))), 'Dual_objective'

        # constraints:
        for j in range(k):
                thr_vec = intensity_generator(thr_confs[j], global_mass_axis)
                program += lp.lpSum(v*x for v, x in zip(thr_vec, lpVars[:n-1]+[0]) if v > 0.).addInPlace(
                            lp.lpSum(v*x for v, x in zip([-1, 0, 1, -1], lpVars[n-1:]))) <= -penalty, 'P_%i' % (j+1)

        exp_vec = intensity_generator(exp_confs, global_mass_axis)
        program += lp.lpSum(v*x for v, x in zip(exp_vec, lpVars[:n-1]+[0])).addInPlace(
                            lp.lpSum(v*x for v, x in zip([0, 1, -1, 0], lpVars[n-1:]))) <= penalty_th, 'p0_prime'

        if not quiet:
                print('tsk tsk')
        
        for i in range(n-1):
                program +=  lpVars[i] - lpVars[n-1]  <=  0, 'g_%i' % (i+1)
                program +=  -lpVars[i] - lpVars[n]  <= 0, 'g_prime_%i' % (i+1)
        try:
                for i in range(n-2):
                        program += lpVars[i] - lpVars[i+1] <= interval_lengths[i], 'epsilon_plus_%i' % (i+1)
                        program += lpVars[i+1] - lpVars[i] <= interval_lengths[i], 'epsilon_minus_%i' % (i+1)
        except IndexError: #linear program makes no sense if n<=2 (n-number of points in mixture's spectrum)
                pass

        program += -lpVars[n-1] <= 0, 'g_%i' % (n)
        program += -lpVars[n] <= 0, 'g_prime_%i' % (n)

        if not quiet:
                print("Constraints written")

        if not quiet:
                print("Starting solver")

        #Solving
        LpSolverDefault.msg = not quiet
        program.solve(solver = solver)
        end = time()
        if not quiet:
                print("Solver finished.")
                print("Status:", lp.LpStatus[program.status])
                print("Optimal value:", lp.value(program.objective))
                print("Time:", end - start)
        constraints = program.constraints
        probs = [round(constraints['P_%i' % i].pi, 12) for i in range(1, k+1)]
        p0_prime = round(constraints['p0_prime'].pi, 12)
        exp_vec = list(intensity_generator(exp_confs, global_mass_axis))
        abyss = [round(constraints['g_%i' % i].pi, 12) for i in range(1, n+1) if exp_vec[i-1] > 0.]
        abyss_th = [round(constraints['g_prime_%i' % i].pi, 12) for i in range(1, n+1)]
        if not np.isclose(sum(probs)+sum(abyss), 1., atol=len(abyss)*1e-03):
                warn("""In dualdeconv4:
                Proportions of signal and noise sum to %f instead of 1.
                This may indicate improper results.
                Please check the deconvolution results and consider reporting this warning to the authors.
                                    """ % (sum(probs)+sum(abyss)))

        return {"probs": probs, "noise_in_theoretical": p0_prime, "trash": abyss, "theoretical_trash": abyss_th, 
        "fun": lp.value(program.objective)+penalty, 'status': program.status, 'global_mass_axis': global_mass_axis}


def estimate_proportions(spectrum, query, MTD=0.25, MDC=1e-8,
                        MMD=-1, max_reruns=3, verbose=False, 
                        progress=True, MTD_th=0.1, solver=LpSolverDefault,
                        what_to_compare='concentration'):

    """
    Returns estimated proportions of molecules from query in spectrum.
    Performs initial filtering of formulas and experimental spectrum to speed up the computations.
    _____
    Parameters:
    spectrum: Spectrum object
        The experimental (subject) spectrum.
    query: list of Spectrum objects
        A list of theoretical (reference) spectra.
    MTD: Maximum Transport Distance, float
        Ion current from experimental spectrum will be transported up to this distance when estimating
        molecule proportions. Default is 1.
    MDC: Minimum Detectable Current, float
        If the isotopic envelope of an ion encompasses less than
        this amount of the total ion current, it is assumed that this ion
        is absent in the spectrum. Default is 1e-8.
    MMD: Maximum Mode Distance, float
        If there is no experimental peak within this distance from the
        highest peak of an isotopic envelope of a molecule,
        it is assumed that this molecule is absent in the spectrum.
        Setting this value to -1 disables filtering. Default is -1.
    max_reruns: int
        Due to numerical errors, some partial results may be inaccurate.
        If this is detected, then those results are recomputed for a maximal number of times
        given by this parameter. Default is 3.
    verbose: bool
        Print diagnostic messages? Default is False.
    progress: bool
        Whether to display progress bars during work. Default is True.
    MTD_th: Maximum Transport Distance for theoretical spectra, float
        If presence of noise in theoretical (reference, query) spectra is not expected, 
        then this parameter should be set to None. Otherwise, set its value to some positive real number.
        Ion current from theoretical spectra will be transported up to this distance 
        when estimating molecule proportions. Default is None.
    solver: 
        Which solver should be used. In case of problems with the default solver,
        pulp.GUROBI() is recommended (note that it requires obtaining a licence).
        To see all solvers available at your machine execute: pulp.listSolvers(onlyAvailable=True).
    what_to_compare:
        Should the resulting proportions correspond to concentrations or area under the curve? Default is
        'concentration'. Alternatively can be set to 'area'. This argument is used only for NMR spectra.
    _____
    Returns: dict
        A dictionary with the following entries:
        - proportions: List of proportions of query (i.e. theoretical, reference) spectra.
        - noise: List of intensities that could not be explained by the supplied formulas. 
        The intensities correspond to the m/z values of the experimental spectrum.
        If MTD_th parameter is not equal to None, then the dictionary contains also 
        the following entries:
        - noise_in_theoretical: List of intensities from query (i.e. theoretical, reference) spectra
        that do not correspond to any intensities in the experimental spectrum and therefore were 
        identified as noise. The intensities correspond to the m/z values from global mass axis.
        - proportion_of_noise_in_theoretical: Proportion of noise present in the combination of query
        (i.e. theoretical, reference) spectra.
        - global_mass_axis: All the m/z values from the experimental spectrum and from the query 
        (i.e. theoretical, reference) spectra in a sorted list. 
    """

    def progr_bar(x, **kwargs):
        if progress:
            return tqdm(x, **kwargs)
        else:
            return x
    try:
        exp_confs = spectrum.confs
    except:
        print("Could not retrieve the confs list. Is the supplied spectrum an object of class Spectrum?")
        raise
    assert abs(sum(x[1] for x in exp_confs) - 1.) < 1e-08, "The mixture's spectrum is not normalized."
    
    assert what_to_compare=='concentration' or what_to_compare=='area', 'Comparison of %s is not supported' %what_to_compare
    is_NMR_spectrum = [isinstance(sp, NMRSpectrum) for sp in [spectrum] + query]
    assert all(is_NMR_spectrum) or not any(is_NMR_spectrum), 'Spectra provided are of mixed types. \
            Please assert that either all or none of the spectra are NMR spectra.'
    nmr = all(is_NMR_spectrum)

    if not nmr:
        assert all(x[0] >= 0. for x in exp_confs), 'Found peaks with negative masses!'
    if any(x[1] < 0 for x in exp_confs):
        raise ValueError("""
        The mixture's spectrum cannot contain negative intensities. 
        Please remove them using e.g. the Spectrum.trim_negative_intensities() method.
        """)
                           
    vortex = [0.]*len(exp_confs)  # unexplained signal
    k = len(query)
    proportions = [0.]*k

    if MTD_th is None:
        MTD_max = MTD
    else:
        MTD_max = max(MTD, MTD_th)

    for i, q in enumerate(query):
        assert abs(sum(x[1] for x in q.confs) - 1.) < 1e-08, 'Reference spectrum %i is not normalized' %i
        if not nmr:
            assert all(x[0] >= 0 for x in q.confs), 'Reference spectrum %i has negative masses!' %i
        
    # Initial filtering of formulas
    envelope_bounds = []
    filtered = []
    for i in progr_bar(range(k), desc = "Initial filtering of formulas"):
        s = query[i]
        mode = s.get_modal_peak()[0]
        mn = s.confs[0][0]
        mx = s.confs[-1][0]
        matching_current = MDC==0. or sum(x[1] for x in misc.extract_range(
                                                                        exp_confs, mn - MTD_max, mx + MTD_max)) >= MDC
        matching_mode = MMD==-1 or abs(misc.closest(exp_confs, mode)[0] - mode) <= MMD

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
    for i in progr_bar(range(first_present, k), desc = "Computing chunks"):
        mn, mx, sp_id = envelope_bounds[i]
        if mn - prev_mx > 2*MTD_max:
            current_chunk += 1
            chunk_bounds.append( (prev_mn-MTD_max, prev_mx+MTD_max) )
            prev_mn = mn  # get lower bound of new chunk
        prev_mx = mx  # update the lower bound of current chunk
        chunkIDs[sp_id] = current_chunk
    chunk_bounds.append( (prev_mn-MTD_max, prev_mx+MTD_max) )
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
    for conf_id, cur_conf in progr_bar(enumerate(exp_confs), desc = "Splitting the experimental spectrum into chunks"):
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
        print("Ion currents in chunks:", chunk_TICs)

    # Deconvolving chunks:
    p0_prime = 0
    vortex_th = []
    global_mass_axis = []
    objective_function = 0
    for current_chunk_ID, conf_IDs in progr_bar(enumerate(exp_conf_chunks), desc="Deconvolving chunks",
                                                                            total=len(exp_conf_chunks)):
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

            rerun = 0
            success = False
            while not success:
                    rerun += 1
                    if rerun > max_reruns:
                            raise RuntimeError('Failed to deconvolve a fragment of the experimental spectrum \
                                                with mass (%f, %f)' % chunk_bounds[current_chunk_ID])
                    if MTD_th is None:
                        dec = dualdeconv2(chunkSp, thrSp, MTD, quiet=True, solver=solver)
                    else:
                        dec = dualdeconv4(chunkSp, thrSp, MTD, MTD_th, quiet=True, solver=solver)
                    if dec['status'] == 1:
                            success=True
                    else:
                            warn('Rerunning computations for chunk %i due to status %s' % (current_chunk_ID, 
                                                                                        lp.LpStatus[dec['status']]))
            if verbose:
                    print('Chunk %i deconvolution status:', lp.LpStatus[dec['status']])
                    print('Signal proportion in experimental spectrum:', sum(dec['probs']))
                    print('Noise proportion in experimental spectrum:', sum(dec['trash']))
                    print('Total explanation:', sum(dec['probs'])+sum(dec['trash']))
                    if MTD_th is not None:
                        print('Noise proportion in combination of theoretical spectra:', dec["noise_in_theoretical"])
            for i, p in enumerate(dec['probs']):
                original_thr_spectrum_ID = theoretical_spectra_IDs[i]
                proportions[original_thr_spectrum_ID] = p*chunk_TICs[current_chunk_ID]
            for i, p in enumerate(dec['trash']):
                original_conf_id = conf_IDs[i]
                vortex[original_conf_id] = p*chunk_TICs[current_chunk_ID]
            if MTD_th is not None:
                p0_prime = p0_prime + dec["noise_in_theoretical"]*chunk_TICs[current_chunk_ID]
                rescaled_vortex_th = [element*chunk_TICs[current_chunk_ID] for element in dec['theoretical_trash']]
                vortex_th = vortex_th + rescaled_vortex_th
                global_mass_axis = global_mass_axis + dec['global_mass_axis']
                
        objective_function = objective_function + dec['fun']

    if not np.isclose(sum(proportions)+sum(vortex), 1., atol=len(vortex)*1e-03):
        warn("""In estimate_proportions:
Proportions of signal and noise sum to %f instead of 1.
This may indicate improper results.
Please check the deconvolution results and consider reporting this warning to the authors.
                        """ % (sum(proportions)+sum(vortex)))
        
    compare_area = ((not nmr) or (nmr and what_to_compare=='area'))

    if compare_area:
        if MTD_th is not None:
            return {'proportions': proportions, 'noise': vortex, 'noise_in_theoretical': vortex_th, 
                'proportion_of_noise_in_theoretical': p0_prime, 'global_mass_axis': global_mass_axis, 
                   'Wasserstein distance': objective_function}
        else:
            return {'proportions': proportions, 'noise': vortex,
                    'Wasserstein distance': objective_function}
    else:
        queries_protons = [query_spec.protons for query_spec in query]
        rescaled_proportions = [prop/prot for prop, prot in zip(proportions, queries_protons)]
        rescaled_proportions = [prop/sum(rescaled_proportions) for prop in rescaled_proportions]
        return {'proportions': rescaled_proportions, 'Wasserstein distance': objective_function}