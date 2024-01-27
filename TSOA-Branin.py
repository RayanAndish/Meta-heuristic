import random
import numpy as np
import time
import csv

#ackley function
#branin function
def branin(x):
    a = 1
    b = 5.1 / (4 * np.pi**2)
    c = 5 / np.pi
    r = 6
    s = 10
    t = 1 / (8 * np.pi)
    return a * (x[1] - b * x[0]**2 + c * x[0] - r)**2 + s * (1 - t) * np.cos(x[0]) + s

def tso(func, dim, lb, ub, max_iter, num_tuna, w, c1, c2):
    # initialize tuna fitness and velocities
    fitness = np.random.uniform(low=lb, high=ub, size=(num_tuna, dim))
    velocities = np.zeros_like(fitness)

    # initialize personal and global best fitness and function values
    pbest_fitness = fitness.copy()
    pbest_tunas = np.array([func(p) for p in pbest_fitness])
    gbest_index = np.argmin(pbest_tunas)
    gbest_fitness = pbest_fitness[gbest_index]
    gbest_tuna = pbest_tunas[gbest_index]

    # list for save value of each iteration
    all_solutions = []

    # perform TSO iterations
    for i in range(max_iter):
        # update tuna velocities
        r1 = np.random.uniform(size=(num_tuna, dim))
        r2 = np.random.uniform(size=(num_tuna, dim))
        velocities = w * velocities + c1 * r1 * (pbest_fitness - fitness) + c2 * r2 * (gbest_fitness - fitness)

        # update tuna fitness
        fitness = fitness + velocities

        # apply bounds
        fitness = np.clip(fitness, lb, ub)

        # evaluate function values
        values = np.array([func(p) for p in fitness])

        # Save value of each iteration
        all_solutions.append(fitness.copy())

        # update personal best fitness and function values
        mask = values < pbest_tunas
        pbest_fitness[mask] = fitness[mask]
        pbest_tunas[mask] = values[mask]

        # update global best fitness and function value
        gbest_index = np.argmin(pbest_tunas)
        gbest_fitness = pbest_fitness[gbest_index]
        gbest_tuna = pbest_tunas[gbest_index]

    return gbest_tuna, gbest_fitness, all_solutions

# set TSO parameters
max_iter = 25000
num_tuna = 20
w = 0.9
c1 = 2
c2 = 2

# Run the algorithm with functions and store the results in a CSV file
with open('TSOA-Branin-D4-25000.csv', mode='w') as results_file:
    results_writer = csv.writer(results_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    results_writer.writerow(['Iteration', 'Best_Solutions', 'Best_Fitness', 'Execution_Time'])

    # Run the algorithm with ackley function
    Bdim = 4 # Set dimension to 2 for the 2-dimensional Branin function
    start_time = time.time()
    lb_branin = np.array([-5, 0] * (Bdim // 2))
    ub_branin = np.array([10, 15] * (Bdim // 2))
    best_fitness, best_tuna, all_solutions = tso(branin, dim=Bdim, lb=lb_branin, ub=ub_branin, max_iter=max_iter, num_tuna=num_tuna, w=w, c1=c1, c2=c2)
    end_time = time.time()

    # Print Value of Each Iteration
    for t, solution in enumerate(all_solutions):
        best_fitness_t = np.min([branin(p) for p in solution])
        best_solution_t = solution[np.argmin([branin(p) for p in solution]), :]
        print(f"Iteration {t + 1}: Best Fitness = {best_fitness_t}, Best Solution = {best_solution_t}")
        # Export CSV file
        results_writer.writerow([t + 1, best_solution_t.tolist(), best_fitness_t, end_time - start_time])
