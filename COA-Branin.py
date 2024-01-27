import random
import numpy as np
import time
import csv
import pandas as pd

# Branin function
def branin(x):
    a = 1
    b = 5.1 / (4 * np.pi**2)
    c = 5 / np.pi
    r = 6
    s = 10
    t = 1 / (8 * np.pi)
    return a * (x[1] - b * x[0]**2 + c * x[0] - r)**2 + s * (1 - t) * np.cos(x[0]) + s

def evaluate_fitness(population, obj_func):
    return np.array([obj_func(ind) for ind in population])

def chitah_optimizer(obj_func, dim, pop_size, max_iter, lb, ub):
    # Initialize the population within the specified bounds
    population = np.random.uniform(low=lb, high=ub, size=(pop_size, dim))
    fitness_values = np.zeros(pop_size)
    best_solution = np.zeros(dim)
    best_fitness = np.inf

    all_solutions = []

    for iteration in range(max_iter):
        for i in range(pop_size):
            fitness_values[i] = obj_func(population[i, :])

            if fitness_values[i] < best_fitness:
                best_fitness = fitness_values[i]
                best_solution = population[i, :]

        all_solutions.append(population.copy())

        # Chitah Algorithm Steps
        a = 0.7 - 0.5 * iteration / max_iter  # Acceleration coefficient
        r1 = np.random.rand(pop_size, dim)
        r2 = np.random.rand(pop_size, dim)

        population = population + a * (r1 * (best_solution - population) + r2 * np.random.uniform(low=lb, high=ub, size=(pop_size, dim)))

        # Apply bounds
        population = np.minimum(np.maximum(population, lb), ub)

    return best_fitness, best_solution, all_solutions

# Example Usage with Branin Function
Bdim = 4
population_size = 30
max_iterations = 25000

# Run the algorithm with functions and store the results in a CSV file
with open('COA-Branin-D4-25000.csv', mode='w') as results_file:
    results_writer = csv.writer(results_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    results_writer.writerow(['Iteration', 'Best_Solutions', 'Best_Fitness', 'Execution_Time'])

    # Run the algorithm with Branin function
    start_time = time.time()
    lb_branin = [-5, 0, -5, 0]
    ub_branin = [10, 15, 10, 15]
    best_fitness, best_solution, all_solutions = chitah_optimizer(branin, dim=Bdim, pop_size=population_size, max_iter=max_iterations, lb=lb_branin, ub=ub_branin)
    end_time = time.time()

    for t in range(max_iterations):
        best_fitness_t = evaluate_fitness(all_solutions[t], branin).min()
        best_solution_t = all_solutions[t][np.argmin(evaluate_fitness(all_solutions[t], branin)), :]
        print(f"Iteration {t + 1}: Best Fitness = {best_fitness_t}, Best Solution = {best_solution_t}")
        # Export CSV file
        results_writer.writerow([t + 1, best_solution_t.tolist(), best_fitness_t, end_time - start_time])