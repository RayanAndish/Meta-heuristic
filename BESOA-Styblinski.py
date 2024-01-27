import random
import numpy as np
import time
import csv

# Styblinski-Tang function
def styblinski_tang(x):
    term1 = np.sum(np.power(x, 4) - 16 * np.power(x, 2) + 5 * x) / 2
    term2 = len(x) * np.power(0.5, 2)
    return term1 + term2

def selection(X, A, fitness, E, func, Gbest, ub, lb, ps, pe):
    N, dim = X.shape
    M = np.zeros((N, dim))
    for i in range(N):
        if fitness[i] < Gbest:
            M[i, :] = A[i, :]
        else:
            r = np.random.rand()
            if r < ps:
                M[i, :] = A[i, :]
            elif r < ps + pe:
                M[i, :] = E[i, :]
            else:
                M[i, :] = X[i, :]

            # Bound handling (element-wise comparison)
            M[i, :] = np.where(M[i, :] > ub, ub, M[i, :])
            M[i, :] = np.where(M[i, :] < lb, lb, M[i, :])

    fitness = fun_calcobjfunc(func, M)
    return M, fitness

# BESOA function
def BESOA(func, lb, ub, dim, N, max_iter, pa, pb, pc, ps, pm, pe):
    # Initialization
    X = initialization(N, dim, ub, lb)
    fitness = fun_calcobjfunc(func, X)
    Gbest, Gbest_idx = np.min(fitness), np.argmin(fitness)
    Xbest = np.copy(X[Gbest_idx, :])
    CNVG = np.zeros(max_iter)

    # Main loop
    for t in range(max_iter):
        A = np.zeros((N, dim))
        E = np.zeros((N, dim))
        for i in range(N):
            r = np.random.rand()

            if r < pa:
                A[i, :] = attack(X[i, :], Xbest, ub, lb)
            elif r < pa + pb:
                A[i, :] = explore(X[i, :], Xbest, ub, lb)
            elif r < pa + pb + pc:
                A[i, :] = scout(X[i, :], ub, lb)
            else:
                A[i, :] = eagle(X[i, :], X, fitness, ub, lb)  # Include the eagle function here

            # Mutation
            if np.random.rand() < pm:
                E[i, :] = np.random.rand(dim) * (ub - lb) + lb

        fitness = fun_calcobjfunc(func, A)
        X, fitness = selection(X, A, fitness, E, func, Gbest, ub, lb, ps, pe)
        Gbest_temp = np.min(fitness)

        if Gbest_temp < Gbest:
            Gbest = Gbest_temp
            Gbest_idx = np.argmin(fitness)
            Xbest = np.copy(X[Gbest_idx, :])
        CNVG[t] = Gbest

        # Print values for each iteration
        #print(f"Iteration {t + 1}: Best Solutions: {Xbest}, Best Fitness: {Gbest}")

    return Xbest, Gbest, CNVG

def initialization(N, dim, ub, lb):
    X = np.zeros((N, dim))
    for i in range(N):
        X[i, :] = np.random.rand(dim) * (np.array(ub) - np.array(lb)) + np.array(lb)
    return X

def fun_calcobjfunc(func, X):
    N, dim = X.shape
    fitness = np.zeros(N)
    for i in range(N):
        fitness[i] = func(X[i, :])
    return fitness

def attack(X, Xbest, ub, lb):
    F = 0.5 + np.random.rand() * 0.5
    return X + F * (Xbest - X)

def explore(X, Xbest, ub, lb):
    F = 0.5 + np.random.rand() * 0.5
    if np.random.rand() < 0.5:
        return X + F * (np.random.rand(len(X)) * (ub - lb) + lb - X)
    else:
        return Xbest + F * (np.random.rand(len(X)) * (ub - lb) + lb - X)

def scout(X, ub, lb):
    return np.random.rand(len(X)) * (ub - lb) + lb

def eagle(X, Xpop, fitness, ub, lb):
    N, dim = Xpop.shape
    idx = np.argsort(fitness)[:int(N/5)]
    Xbest = np.mean(Xpop[idx, :], axis=0)
    Xworst = Xpop[np.argmax(fitness), :]
    F = np.random.rand() * 2 - 1

    # Add your eagle logic here

# Define the search space
dim = 30

# Define the BESOA parameters
N = 50
max_iter = 150
pa = 0.1
pb = 0.2
pc = 0.3
ps = 0.3
pm = 0.1
pe = 0.1

# Run the algorithm with functions and store the results in a CSV file
with open('BESOA-Styblinski-D30-25000.csv', mode='w') as results_file:
    results_writer = csv.writer(results_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    results_writer.writerow(['Iteration', 'Best_Solutions', 'Best_Fitness', 'Execution_Time'])

    for t in range(max_iter):
        # Update Xbest in each iteration
        start_time = time.time()
        lb_sty = (-5)
        ub_sty = (5)
        Xbest, Gbest, CNVG = BESOA(func=styblinski_tang, lb=lb_sty, ub=ub_sty, dim=dim, N=N, max_iter=t+1, pa=pa, pb=pb, pc=pc, ps=ps, pm=pm, pe=pe)
        end_time = time.time()
        results_writer.writerow([t + 1, Xbest.tolist(), CNVG[t], Gbest])
        print(f"Iteration {t + 1}: Best Solutions: {Xbest}, Best Fitness: {Gbest}")

        # Also, write the final row
    results_writer.writerow([max_iter, Xbest.tolist(), Gbest, end_time - start_time])