# -*- coding: utf-8 -*-
"""
Created on Sun Apr 22 20:29:34 2018

@author: Szymon Majewski
"""


import numpy as np
import math
import time

TEST_PROCEDURES = False
TEST_SOLVE_M = False
TEST_SOL_NORM_EQ = False


def FastIPMDeconv(emp_spectrum, thr_spectrums, eps=1e-6, max_iter=100):
    
    start_time = time.time()    # Starting time
    
    m = len(thr_spectrums)      # Number of theoretical spectra
    
    # Create a list with all masses, remembering which mass belongs to which spectrum
    masses = []
    for v in emp_spectrum.confs:
        masses += [(v[0], v[1], -1)]
    for i in range(m):
        spec = thr_spectrums[i]
        for v in spec.confs:
            masses += [(v[0], v[1], i)]
            
    if len(masses) == 0:
        print("Empty")
        return;
        
    # Sort
    masses.sort(key = lambda v: v[0])
    
    # Preprocessing
    # Compute the number of unique masses
    # Compute the lists of indices for theoretical spectrums
    # Compute the lists of masses on a given index
    # Compute the pdf of empirical distribution and the lenghts of segments
    
    n = 0  #Number of unique masses
    thr_indeces = [ [] for i in range(m)] # On what indexes does each spec has mass?
    masses_on_index = [[]] # Which specs hav what mass on the given index?
    b_list = []
    c_list = []
    cmass = masses[0][0]
    emp_dist = 0.0
    for v in masses:
        if v[0] != cmass:
            n+= 1
            masses_on_index += [[]]
            b_list += [v[0] - cmass]
            cmass = v[0]
            c_list += [emp_dist]
        if v[2] > -1:
            thr_indeces[v[2]] += [(n , v[1])]
            masses_on_index[n] += [(v[2], v[1])]
        else:
            emp_dist += v[1]
    
    n += 1
    c_list += [1.0]
    c = np.zeros(2*n + m)
    c[0:n] = c_list
    c[n:(2*n)] = - c[0:n]
    
    b = -np.array([0.0 for i in range(m)] + b_list)
    
    b_norm = np.linalg.norm(b)
    c_norm = np.linalg.norm(c)
    
    # Test positive definitines of A^T A
    test_A_mat = np.zeros((m,m))
    test_A_h = np.ones(n)
    A_trans_diag_A(thr_indeces, test_A_h, m, n, test_A_mat)
    
    if np.linalg.matrix_rank(test_A_mat, tol = 1e-08) < m:
        print("A^T A is not full rank. Problem is not well defined.")
        return
    
    # Set constants
    theta = 0.9
    gamma = 0.0
    
    # Set starting point
    x = np.ones(2*n+m)
    for i in range(n-1):
        x[i] = -b[m+i]/2
        x[n+i] = x[i];
    x[n-1] = 1.5
    x[2*n -1 ] = 0.5
    y = np.ones(m + n - 1)
    y[0:m] /= m
    y_init_help = np.zeros(n)
    A_right_mult(masses_on_index, y, m,n, y_init_help)
    y_init_help -= c[0:n]
    y[m:(m+n-1)] = abs(y_init_help[0:(n-1)]) + 1/n
    
    s = np.ones(2*n+m)
    s[0:(n-1)] = y[m:(m+n-1)] - y_init_help[0:(n-1)]
    s[n:(2*n-1)] = y[m:(m+n-1)] + y_init_help[0:(n-1)]
    s[(2*n):(2*n + m)] /= m
    
    x_help = np.zeros(2*n + m)
    y_help = np.zeros(n + m -1)
    
    NUM_ALG_ERR = False
    
    for step in range(max_iter):
        
        #print("Niedodatnie s: " , sum(s <= 0))
        A_tilde_left_mult(thr_indeces, x, m, n, y_help)
        r_p = b - y_help
        #print(r_p)
        
        A_tilde_right_mult(masses_on_index, y, m, n, x_help)
        r_d = c - x_help - s
        #print(r_d)
        
        mu = x.dot(s)/(2*n+m)
        #print(mu)
        
        r_p_norm = np.linalg.norm(r_p)
        r_p_rel = r_p_norm/max(1, b_norm)
        r_d_norm = np.linalg.norm(r_d)
        r_d_rel = r_d_norm/max(1, c_norm)
        #noncentrality = np.linalg.norm(x * s - mu) 
        
        #print("After " + str(step) + " iterations:")
        #print("r_p_norm = " + str(r_p_rel))
        #print("r_d_norm = " + str(r_d_rel))
        #print("duality gap = " + str((2*n +m)*mu))
        #print("noncentrality = ", noncentrality  )
        #print(y[0:m])
        
        
        if (r_p_rel < eps ) and (r_d_rel < eps) and (mu * (2*n + m) < eps):
            print("Termination condition reached successfully.")
            print("Total iterations : " +str(step))
            #print(x)
            #print(s)
            #print(y)
            break
        
        # Compute scaling factor
        try:
            dx, dy, ds = SolveNewtonEq(masses_on_index, thr_indeces, r_p, r_d, mu, 
                                   0.0, m, n, x, s, b)
        except Exception as e:
            print(e)
            NUM_ALG_ERR = True;
            break;
            
        alpha_p = 1.0
        alpha_d = 1.0
        
        for i in range(2*n+m):
            if dx[i] < 0:
                if (-x[i]/dx[i]) < alpha_p:
                    alpha_p = (-x[i]/dx[i])
        for i in range(2*n+m):
            if ds[i] < 0:
                if (-s[i]/ds[i]) < alpha_d:
                    alpha_d = (-s[i]/ds[i])
                    
        sigma = (s + alpha_d * ds).dot(x + alpha_p * dx)/s.dot(x)
        #print("Sigma " + str(sigma))
        
        gamma = math.pow(sigma,3)
        
        # Find search direction with chosen scaling factor
        try:
            dx, dy, ds = SolveNewtonEq(masses_on_index, thr_indeces, r_p, r_d, mu, 
                                   gamma, m, n, x, s, b, TEST_SOL_M=False, masses=masses)
        except Exception as e:
            print(e)
            NUM_ALG_ERR = True;
            break;
        
        alpha_p = 1.0/theta
        alpha_d = 1.0/theta
        
        for i in range(2*n+m):
            if dx[i] < 0:
                if (-x[i]) > alpha_p * dx[i]:
                    alpha_p = (-x[i]/dx[i])
        for i in range(2*n+m):
            if ds[i] < 0:
                if (-s[i]) > alpha_d * ds[i]:
                    alpha_d = (-s[i]/ds[i])
        
        alpha_p *= theta
        alpha_d *= theta
        
        x += alpha_p * dx
        s += alpha_d * ds
        y += alpha_d * dy
        x = np.maximum(x, 1e-20 * np.ones(2*n + m))
        s = np.maximum(s, 1e-20 * np.ones(2*n + m))
    
    if step == (max_iter - 1):
        print("WARNING: Maximum amount of iterations reached before the termination condition was met.")
        #print(x)
        #print(s)
        #print(y)
    if NUM_ALG_ERR:
        print("WARNING: Numerical algebra error encountered")
        print("Error in step", step)
        print(r_p_norm, r_p_rel, r_d_norm, r_d_rel, mu * (2*n + m))
        return None


    fun_val = np.zeros(n)
    A_right_mult(masses_on_index, y[0:m], m, n, fun_val)
    a_vals = abs(fun_val - c[0:n]) 
    
    # end_time = time.time()
    # print("Time: ", end_time - start_time)

    return {'weights':y[0:m], 'fun': - sum(a_vals[0:(n-1)] * b[m:m+n-1]), 
            'NUM_ALG_ERR' : NUM_ALG_ERR, 'steps' : step+1}

# Computes Av in linear time
# masses_on_index - list of masses for every index
# v - a numpy vector of length m
# w - a numpy vector of length n
# writes Av to w
def A_right_mult(masses_on_index, v, m, n, w):
    w[0] = 0.0
    for i in range(n):
        if i == 0:
            w[i] = 0.0
        else:
            w[i] = w[i-1]
        for ind, prob in masses_on_index[i]:
            w[i] += prob * v[ind] 
            
# Version for tilde(A)
# v - a numpy vector of length m + n -1
# w - a numpy vector of length 2n + m
# writes tilde(A)v to w
def A_tilde_right_mult(masses_on_index, v, m, n, w):
    A_right_mult(masses_on_index, v[0:m], m, n , w)
    w[n:(2*n)] = -w[0:n]
    w[0:n-1] -= v[m:(m+n-1)]
    w[n:(2*n-1)] -= v[m:(m+n-1)]
    w[(2*n):(2*n+m)] = -v[0:m]

# Computes vA in linear time
# thr_indeces - list of indexes for every theoretical spectrum
# v - a numpy vector of length n
# w - a numpy vector of length m
# writes vA to w
def A_left_mult(thr_indeces, v, m, n, w):
    v_sufsum = np.array([0.0 for i in range(n)])
    v_sufsum[n-1] = v[n-1]
    for i in range(n-2, -1, -1):
        v_sufsum[i] = v_sufsum[i+1] + v[i]
    for i in range(m):
        w[i] = 0.0
        for ind, prob in thr_indeces[i]:
            w[i] += prob * v_sufsum[ind]

# Version for tilde(A) 
# v - numpy vector of length 2n + m
# w - a numpy vector of length m + n - 1
# writhes v\tilde(A) to w
def A_tilde_left_mult(thr_indeces, v, m, n, w):
    w_h1 = np.zeros(m)
    w_h2 = np.zeros(m)
    A_left_mult(thr_indeces, v[0:n], m, n, w_h1)
    A_left_mult(thr_indeces, v[n:(2*n)], m, n, w_h2)
    w[0:m] = w_h1 - w_h2 
    w[0:m] -= v[(2*n):(2*n+m)]
    w[m:(m+n-1)] = - v[0:(n-1)] - v[n:(2*n - 1)]         

# Computes A^T H A, where H is a given diagonal matrix
#
# h - vector of length n with the diagonal of H 
# mat - m by m matrix, where the results will be written
def A_trans_diag_A(thr_indeces, h, m, n, mat):
    h_sufsum = np.array([0.0 for i in range(n)])
    h_sufsum[n-1] = h[n-1]
    for i in range(n-2, -1, -1):
        h_sufsum[i] = h_sufsum[i+1] + h[i]
    for i in range(m):
        for j in range(i+1, m):
            mat[i,j] = 0.0
            l = thr_indeces[j] + [(n+1,0.0)]
            j_iter = 0
            pdf = 0.0
            for ind, prob in thr_indeces[i]:
                while l[j_iter][0] <= ind:
                    pdf += l[j_iter][1]
                    j_iter += 1
                mat[i,j] += h_sufsum[ind] * prob * pdf
            l = thr_indeces[i] + [(n+1, 0.0)]
            i_iter = 0
            pdf = 0.0
            for ind, prob in thr_indeces[j]:
                while l[i_iter][0] < ind:
                    pdf += l[i_iter][1]
                    i_iter += 1
                mat[i,j] += h_sufsum[ind] * prob * pdf
            mat[j,i] = mat[i,j]
    for i in range(m):
        mat[i,i] = 0.0
        pdf  = 0.0
        for ind, prob in thr_indeces[i]:
            pdf += prob
            mat[i,i] += h_sufsum[ind] * prob * (2 * pdf - prob)
                
# Solve the first equation 
# q - vector of length m + n - 1
# p - vector of length m + n - 1
# Solves Lq = p, writes the result in q
def M_left_solve(thr_indeces, q, m, n, p, kj_inv):
    q[m:(m+n-1)] = p[m:(m+n-1)]
    q_help = np.zeros(n)
    q_help[0:(n-1)] = p[m:(m+n-1)]
    q_help[0:(n-1)] *= kj_inv
    p_help = np.zeros(m)
    A_left_mult(thr_indeces, q_help, m, n, p_help)
    q[0:m] = p[0:m] - p_help

# Solves Cq = p, writes the result in q
def M_mid_solve(thr_indeces, q, m, n, p, g_plus, h, add_diag):
    q[m:(m+n-1)] = p[m:(m+n-1)] / g_plus[0:(n-1)]
    mat = np.zeros((m,m))
    A_trans_diag_A(thr_indeces, h, m, n, mat)
    mat += np.diag(add_diag)
    #print(np.linalg.cond(mat))
    q[0:m] = np.linalg.solve(mat, p[0:m])
    #q[0:m] = np.linalg.lstsq(mat, p[0:m])[0]
        
# Solves Rq = p, writes the result in q
def M_right_solve(masses_on_index, q, m, n, p, kj_inv):
    q[0:(m + n -1)] = p[0:(m + n - 1)]
    q_help = np.zeros(n)
    A_right_mult(masses_on_index, p[0:m] , m, n, q_help)
    q[m:(m+n-1)] -= kj_inv * q_help[0:(n-1)]
    
# Solve the Newton equation, by solving the normal equation
# r_p - vector of primal residuals, length n + m - 1
# r_d - vector of dual residuals, lenght 2n
# mu - centrality
# gamma - contraction
# x - vector of primal variables, length 2n
# s - vectro of slack variables, lenght 2n
# dx - vector to which dx will be written
# dy - vector to which dy will be written
# ds - vector to which ds will be written
def SolveNewtonEq(masses_on_index, thr_indeces, r_p, r_d, mu, gamma,
                  m, n, x, s, b, TEST_SOL=False, TEST_SOL_M = False, masses=[]):
    
    dy = np.zeros(m + n - 1)
    ds = np.zeros(2 * n + m)
    dx = np.zeros(2 * n + m)
    
    g_plus = np.zeros(n)
    g_minus = np.zeros(n)
    g = np.zeros(2 * n + m)
    kj_inv = np.zeros(n-1)
    r = np.zeros(n + m -1)
    
    r_help = (x * r_d - gamma * mu ) / s
    A_tilde_left_mult(thr_indeces, r_help, m, n, r)
    r += b
        
    g = x / s;
    g_plus = g[0:n] + g[n:(2*n)]
    g_minus = g[n:(2*n)] - g[0:n]
    kj_inv = g_minus[0:(n-1)] / g_plus[0:(n-1)]
        
    r2 = np.zeros(m + n - 1)
    M_left_solve(thr_indeces, r2, m, n, r, kj_inv)
        
    r3 = np.zeros(m + n - 1)
    g_max = np.maximum(g[0:n], g[n:(2*n)])
    g_min = np.minimum(g[0:n], g[n:(2*n)])
    h = (g_max / g_plus)
    h = 4* h * g_min
    h[n-1] = g_plus[n-1]
    #h = g_plus - (g_minus) * (g_minus)/g_plus
    #h[n-1] = g_plus[n-1]
    M_mid_solve(thr_indeces, r3, m, n, r2, g_plus, h, g[(2*n):(2*n + m)])
        
    M_right_solve(masses_on_index, dy, m, n, r3, kj_inv)
        
    A_tilde_right_mult(masses_on_index, dy, m, n, ds)
    ds = r_d[0:(2*n+m)] - ds[0:(2*n+m)]
        
    dx = (mu * gamma - x * ds)/s
    dx -= x
    
    if TEST_SOL:
        Adx = np.zeros(n+m-1)
        A_tilde_left_mult(thr_indeces, dx, m, n, Adx)
        print("TSNEQ")
        print((np.linalg.norm(Adx - r_p)))
            
        Ady = np.zeros(2 * n + m)
        r_test = np.zeros(n + m -1)
        A_tilde_right_mult(masses_on_index, dy, m, n, Ady)
        print((np.linalg.norm(Ady + ds - r_d)))
        Ady = (Ady *x )/ s
        A_tilde_left_mult(thr_indeces, Ady, m, n, r_test)
        print((np.linalg.norm(r_test - r)))
            
        #print(dx * s + ds * x + x*s)
        #print(gamma * mu)
    if TEST_SOL_M:
            A = np.zeros((n, m))
            cmass = masses[0][0]
            row = 0
            for mass, prob, ind in masses:
                if mass > cmass:
                    row += 1
                    cmass = mass
                if ind > -1:
                    A[row,ind] = prob
            for i in range(1,n):
                A[i] += A[i-1]
            A_tilde = np.zeros((2*n+m, m+n-1))
            A_tilde[0:n, 0:m] = A
            A_tilde[n:(2*n), 0:m] = -A
            A_tilde[(2*n):(2*n + m), 0:m] = -np.diag(np.ones(m))
            for i in range(n-1):
                A_tilde[i, m+i] = - 1.0
                A_tilde[n+i, m+i] = -1.0
            G = np.diag(g)
            M = np.transpose(A_tilde).dot(G).dot(A_tilde)
            dy_prim = np.linalg.solve(M, r)
            print((np.linalg.norm(dy_prim - dy)))
            print((sum(abs(dy_prim - dy) > 1e-06)))
            print((np.linalg.norm(x[(2*n):(2*n + m)])))
            print((np.linalg.norm(M.dot(dy_prim) - r)))
            print((np.linalg.norm(M.dot(dy) - r)))
            pass
    
    return (dx, dy, ds)
    
def TestProcedures(emp_spectrum, thr_spectrums):
    
    m = len(thr_spectrums)
    
    # Create a list with all masses, remembering which mass belongs to which spectrum
    masses = []
    for v in emp_spectrum.confs:
        masses += [(v[0], v[1], -1)]
    for i in range(m):
        spec = thr_spectrums[i]
        for v in spec.confs:
            masses += [(v[0], v[1], i)]
            
    if len(masses) == 0:
        print("Empty")
        return;
        
    # Sort
    masses.sort(key = lambda v: v[0])
    
    # Preprocessing
    # Compute the number of unique masses
    # Compute the lists of indices for theoretical spectrums
    # Compute the lists of masses on a given index
    # Compute the pdf of empirical distribution and the lenghts of segments
    
    n = 0  #Number of unique masses
    thr_indeces = [ [] for i in range(m)]
    masses_on_index = [[]]
    b_list = []
    c_list = []
    cmass = masses[0][0]
    emp_dist = 0.0
    for v in masses:
        if v[0] != cmass:
            n+= 1
            masses_on_index += [[]]
            b_list += [v[0] - cmass]
            cmass = v[0]
            c_list += [emp_dist]
        if v[2] > -1:
            thr_indeces[v[2]] += [(n , v[1])]
            masses_on_index[n] += [(v[2], v[1])]
        else:
            emp_dist += v[1]
    
    n += 1
    c_list += [1.0]
    c = np.zeros(2*n)
    c[0:n] = c_list
    c[n:(2*n)] = - c[0:n]
    
    b = -np.array([0.0 for i in range(m)] + b_list)
    
    
    # Test subroutines
    print((n, m))
    A = np.zeros((n, m))
    cmass = masses[0][0]
    row = 0
    for mass, prob, ind in masses:
        if mass > cmass:
            row += 1
            cmass = mass
        if ind > -1:
            A[row,ind] = prob
    for i in range(1,n):
        A[i] += A[i-1]
    v = np.random.rand(n)
    w = np.random.rand(m)
    A_left_mult(thr_indeces, v, m, n, w)
    print((np.transpose(v).dot(A) - np.transpose(w)))
    v = np.random.rand(m, 1)
    w = np.random.rand(n, 1)
    A_right_mult(masses_on_index, v, m, n , w)
    diff = (w - A.dot(v))
    print((np.sqrt(np.sum(diff * diff))))
    mat = np.zeros((m,m))
    h = np.random.rand(n)
    prod = np.transpose(A).dot(np.diag(h)).dot(A)
    A_trans_diag_A(thr_indeces, h, m, n, mat)
    print((prod - mat))
        
    A_tilde = np.zeros((2*n, m+n-1))
    A_tilde[0:n, 0:m] = A
    A_tilde[n:(2*n), 0:m] = -A
    for i in range(n-1):
        A_tilde[i, m+i] = - 1.0
        A_tilde[n+i, m+i] = -1.0
        
    x_rand = np.zeros(2*n)
    y_rand = np.random.rand(n + m - 1)
    A_tilde_right_mult(masses_on_index, y_rand, m, n, x_rand)
    print((A_tilde.dot(y_rand) - x_rand))
        
    x_rand = np.random.rand(2*n)
    A_tilde_left_mult(thr_indeces, x_rand, m, n, y_rand)
    print((np.transpose(A_tilde).dot(x_rand) - y_rand))