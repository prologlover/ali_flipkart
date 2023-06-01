import numpy as np
from numpy.random import rand
from FS.functionHO import Fun
import math
import random



def init_position(lb, ub, N, dim):
    X = np.zeros([N, dim], dtype='float')
    for i in range(N):
        for d in range(dim):
            X[i, d] = lb[0, d] + (ub[0, d] - lb[0, d]) * rand()

    return X


def init_velocity(lb, ub, N, dim):
    V = np.zeros([N, dim], dtype='float')
    Vmax = np.zeros([1, dim], dtype='float')
    Vmin = np.zeros([1, dim], dtype='float')
    # Maximum & minimum velocity
    for d in range(dim):
        Vmax[0, d] = (ub[0, d] - lb[0, d]) / 2
        Vmin[0, d] = -Vmax[0, d]

    for i in range(N):
        for d in range(dim):
            V[i, d] = Vmin[0, d] + (Vmax[0, d] - Vmin[0, d]) * rand()

    print('vmax====', Vmax, '    vmin=', Vmin)
    return V, Vmax, Vmin

def levy_distribution(beta, dim):
    # Sigma
    nume  = math.gamma(1 + beta) * np.sin(np.pi * beta / 2)
    deno  = math.gamma((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2)
    sigma = (nume / deno) ** (1 / beta)
    # Parameter u & v
    u     = np.random.randn(dim) * sigma
    v     = np.random.randn(dim)
    # Step
    step  = u / abs(v) ** (1 / beta)
    LF    = 0.01 * step

    return LF

def binary_conversion(X, thres, N, dim):
    Xbin = np.zeros([N, dim], dtype='int')
    for i in range(N):
        for d in range(dim):
            if X[i, d] > thres:
                Xbin[i, d] = 1
            else:
                Xbin[i, d] = 0

    return Xbin


def boundary(x, lb, ub):
    if x < lb:
        x = lb
    if x > ub:
        x = ub

    return x


def jfs(xtrain, ytrain, opts):
    # Parameters
    ub = 1
    lb = 0
    thres = 0.5
    w = 0.9  # inertia weight
    c1 = 2  # acceleration factor
    c2 = 2  # acceleration factor
    b = 1  # constant
    beta = 1.5  # levy component


    N = opts['N']
    max_iter = opts['T']
    if 'w' in opts:
        w = opts['w']
    if 'c1' in opts:
        c1 = opts['c1']
    if 'c2' in opts:
        c2 = opts['c2']
    if 'b' in opts:
        b = opts['b']
    if 'beta' in opts:
        beta = opts['beta']

        # Dimension
    dim = np.size(xtrain, 1)
    if np.size(lb) == 1:
        ub = ub * np.ones([1, dim], dtype='float')
        lb = lb * np.ones([1, dim], dtype='float')

    # Initialize position & velocity
    X = init_position(lb, ub, N, dim)
    V, Vmax, Vmin = init_velocity(lb, ub, N, dim)

    # Pre
    fit = np.zeros([N, 1], dtype='float')
    Xf = np.zeros([1, dim], dtype='float')
    fitG = float('inf')
    Xpb = np.zeros([N, dim], dtype='float')
    fitP = float('inf') * np.ones([N, 1], dtype='float')
    curve = np.zeros([1, max_iter], dtype='float')
    t = 0
    Xrb = np.zeros([1, dim], dtype='float')
    fitR = float('inf')



    # Initialize position
    X = init_position(lb, ub, N, dim)

    # Pre
    Xf = np.zeros([1, dim], dtype='float')
    fitF = float('inf')

    Xbest = np.zeros([1, dim], dtype='float')



    hho_count=0
    ssa_count=0
    while t < max_iter:





            hho_count=hho_count+1

            # Binary conversion
            Xbin = binary_conversion(X, thres, N, dim)

            # Fitness
            for i in range(N):
                fit[i, 0] = Fun(xtrain, ytrain, Xbin[i, :], opts)
                if fit[i, 0] < fitR:
                    Xrb[0, :] = X[i, :]
                    fitR = fit[i, 0]




            # Store result
            curve[0, t] = fitR.copy()
            print("Iteration:", t + 1)
            print("Best (HHO):", curve[0, t])


            # Mean position of hawk (2)
            X_mu = np.zeros([1, dim], dtype='float')
            X_mu[0, :] = np.mean(X, axis=0)

            for i in range(N):
                # Random number in [-1,1]
                E0 = -1 + 2 * rand()
                # Escaping energy of rabbit (3)
                E = 2 * E0 * (1 - (t / max_iter))
                # Exploration phase
                if abs(E) >= 1:
                    # Define q in [0,1]
                    q = rand()
                    if q >= 0.5:
                        # Random select a hawk k
                        k = np.random.randint(low=0, high=N)
                        r1 = rand()
                        r2 = rand()
                        for d in range(dim):
                            # Position update (1)
                            X[i, d] = X[k, d] - r1 * abs(X[k, d] - 2 * r2 * X[i, d])
                            # Boundary
                            X[i, d] = boundary(X[i, d], lb[0, d], ub[0, d])

                    elif q < 0.5:
                        r3 = rand()
                        r4 = rand()
                        for d in range(dim):
                            # Update Hawk (1)
                            X[i, d] = (Xrb[0, d] - X_mu[0, d]) - r3 * (lb[0, d] + r4 * (ub[0, d] - lb[0, d]))
                            # Boundary
                            X[i, d] = boundary(X[i, d], lb[0, d], ub[0, d])

                # Exploitation phase
                elif abs(E) < 1:
                    # Jump strength
                    J = 2 * (1 - rand())
                    r = rand()
                    # {1} Soft besiege
                    if r >= 0.5 and abs(E) >= 0.5:
                        for d in range(dim):
                            # Delta X (5)
                            DX = Xrb[0, d] - X[i, d]
                            # Position update (4)
                            X[i, d] = DX - E * abs(J * Xrb[0, d] - X[i, d])
                            # Boundary
                            X[i, d] = boundary(X[i, d], lb[0, d], ub[0, d])

                    # {2} hard besiege
                    elif r >= 0.5 and abs(E) < 0.5:
                        for d in range(dim):
                            # Delta X (5)
                            DX = Xrb[0, d] - X[i, d]
                            # Position update (6)
                            X[i, d] = Xrb[0, d] - E * abs(DX)
                            # Boundary
                            X[i, d] = boundary(X[i, d], lb[0, d], ub[0, d])

                    # {3} Soft besiege with progressive rapid dives
                    elif r < 0.5 and abs(E) >= 0.5:
                        # Levy distribution (9)
                        LF = levy_distribution(beta, dim)
                        Y = np.zeros([1, dim], dtype='float')
                        Z = np.zeros([1, dim], dtype='float')

                        for d in range(dim):
                            # Compute Y (7)
                            Y[0, d] = Xrb[0, d] - E * abs(J * Xrb[0, d] - X[i, d])
                            # Boundary
                            Y[0, d] = boundary(Y[0, d], lb[0, d], ub[0, d])

                        for d in range(dim):
                            # Compute Z (8)
                            Z[0, d] = Y[0, d] + rand() * LF[d]
                            # Boundary
                            Z[0, d] = boundary(Z[0, d], lb[0, d], ub[0, d])

                            # Binary conversion
                        Ybin = binary_conversion(Y, thres, 1, dim)
                        Zbin = binary_conversion(Z, thres, 1, dim)
                        # fitness
                        fitY = Fun(xtrain, ytrain, Ybin[0, :], opts)
                        fitZ = Fun(xtrain, ytrain, Zbin[0, :], opts)
                        # Greedy selection (10)
                        if fitY < fit[i, 0]:
                            fit[i, 0] = fitY
                            X[i, :] = Y[0, :]
                        if fitZ < fit[i, 0]:
                            fit[i, 0] = fitZ
                            X[i, :] = Z[0, :]

                            # {4} Hard besiege with progressive rapid dives
                    elif r < 0.5 and abs(E) < 0.5:
                        # Levy distribution (9)
                        LF = levy_distribution(beta, dim)
                        Y = np.zeros([1, dim], dtype='float')
                        Z = np.zeros([1, dim], dtype='float')

                        for d in range(dim):
                            # Compute Y (12)
                            Y[0, d] = Xrb[0, d] - E * abs(J * Xrb[0, d] - X_mu[0, d])
                            # Boundary
                            Y[0, d] = boundary(Y[0, d], lb[0, d], ub[0, d])

                        for d in range(dim):
                            # Compute Z (13)
                            Z[0, d] = Y[0, d] + rand() * LF[d]
                            # Boundary
                            Z[0, d] = boundary(Z[0, d], lb[0, d], ub[0, d])

                            # Binary conversion
                        Ybin = binary_conversion(Y, thres, 1, dim)
                        Zbin = binary_conversion(Z, thres, 1, dim)
                        # fitness
                        fitY = Fun(xtrain, ytrain, Ybin[0, :], opts)
                        fitZ = Fun(xtrain, ytrain, Zbin[0, :], opts)
                        # Greedy selection (10)
                        if fitY < fit[i, 0]:
                            fit[i, 0] = fitY
                            X[i, :] = Y[0, :]
                        if fitZ < fit[i, 0]:
                            fit[i, 0] = fitZ
                            X[i, :] = Z[0, :]




            ssa_count=ssa_count+1
            # Binary conversion
            Xbin = binary_conversion(X, thres, N, dim)

            # Fitness
            for i in range(N):
                fit[i, 0] = Fun(xtrain, ytrain, Xbin[i, :], opts)
                if fit[i, 0] < fitF:
                    Xf[0, :] = X[i, :]
                    fitF = fit[i, 0]



            # Store result
            curve[0, t] = fitF.copy()
            print("Iteration:", t + 1)
            print("Best (SSA):", curve[0, t])



            # Compute coefficient, c1 (3.2)
            c1 = 2 * np.exp(-(4 * t / max_iter) ** 2)


            for i in range(N):
                # First leader update
                if i == 0:
                    for d in range(dim):
                        # Coefficient c2 & c3 [0 ~ 1]
                        c2 = rand()
                        c3 = rand()
                        # Leader update (3.1)
                        if c3 >= 0.5:
                            X[i, d] = Xf[0, d] + c1 * ((ub[0, d] - lb[0, d]) * c2 + lb[0, d])
                        else:
                            X[i, d] = Xf[0, d] - c1 * ((ub[0, d] - lb[0, d]) * c2 + lb[0, d])

                        # Boundary
                        X[i, d] = boundary(X[i, d], lb[0, d], ub[0, d])

                        # Salp update
                elif i >= 1:
                    for d in range(dim):
                        # Salp update by following front salp (3.4)
                        X[i, d] = (X[i, d] + X[i - 1, d]) / 2
                        # Boundary
                        X[i, d] = boundary(X[i, d], lb[0, d], ub[0, d])


            if fitR<fitF:
                print("HHO",fitF,fitR)
                for d in range(dim):
                    Xbest[0, d] =Xrb[0,d]

            else:
                print("ssa", fitF,fitR)
                for d in range(dim):
                    Xbest[0, d] = Xf[0,d]

            t += 1





    # Best feature subset

    Gbin = binary_conversion(Xbest, thres, 1, dim)
    Gbin = Gbin.reshape(dim)
    pos = np.asarray(range(0, dim))
    sel_index = pos[Gbin == 1]
    num_feat = len(sel_index)
    # Create dictionary
    ssa_data = {'sf': sel_index, 'c': curve, 'nf': num_feat}
    print("hho_count=",hho_count)
    print("ssa_count=",ssa_count)
    return ssa_data
