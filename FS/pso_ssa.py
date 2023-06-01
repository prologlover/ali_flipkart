import numpy as np
from numpy.random import rand
from FS.functionHO import Fun


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

    return V, Vmax, Vmin


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

    N = opts['N']
    max_iter = opts['T']
    if 'w' in opts:
        w = opts['w']
    if 'c1' in opts:
        c1 = opts['c1']
    if 'c2' in opts:
        c2 = opts['c2']

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



    # Initialize position
    X = init_position(lb, ub, N, dim)

    # Pre
    Xf = np.zeros([1, dim], dtype='float')
    fitF = float('inf')




    flag_fit=0.5
    pso_count=0
    ssa_count=0
    while t < max_iter:
        print("temp_flg=",flag_fit)
        if flag_fit<0.5:
            pso_count=pso_count+1

            # Binary conversion
            Xbin = binary_conversion(X, thres, N, dim)

            # Fitness
            for i in range(N):
                fit[i, 0] = Fun(xtrain, ytrain, Xbin[i, :], opts)
                if fit[i, 0] < fitP[i, 0]:
                    Xpb[i, :] = X[i, :]
                    fitP[i, 0] = fit[i, 0]
                if fitP[i, 0] < fitG:
                    Xf[0, :] = Xpb[i, :]
                    fitG = fitP[i, 0]



            # Store result
            curve[0, t] = fitG.copy()
            print("Iteration:", t + 1)
            print("Best (PSO):", curve[0, t])
            t += 1

            for i in range(N):
                for d in range(dim):
                    # Update velocity
                    r1 = rand()
                    r2 = rand()
                    V[i, d] = w * V[i, d] + c1 * r1 * (Xpb[i, d] - X[i, d]) + c2 * r2 * (Xf[0, d] - X[i, d])
                    # Boundary
                    V[i, d] = boundary(V[i, d], Vmin[0, d], Vmax[0, d])
                    # Update position
                    X[i, d] = X[i, d] + V[i, d]
                    # Boundary
                    X[i, d] = boundary(X[i, d], lb[0, d], ub[0, d])
            flag_fit = fitG + r1+r2


        else:
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

            t += 1

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

            flag_fit = fitF+c2+c3


    # Best feature subset

    Gbin = binary_conversion(Xf, thres, 1, dim)
    Gbin = Gbin.reshape(dim)
    pos = np.asarray(range(0, dim))
    sel_index = pos[Gbin == 1]
    num_feat = len(sel_index)
    # Create dictionary
    ssa_data = {'sf': sel_index, 'c': curve, 'nf': num_feat}
    print("pso_count=",pso_count)
    print("ssa_count=",ssa_count)
    return ssa_data
