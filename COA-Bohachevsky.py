import random
import numpy as np
import time
import csv
import pandas as pd

# Bohachevsky function
def bohachevsky(x):
    term1 = np.power(x[0], 2)
    term2 = 2 * np.power(x[1], 2)
    term3 = -0.3 * np.cos(3 * np.pi * x[0])
    term4 = -0.4 * np.cos(4 * np.pi * x[1])
    return term1 + term2 + term3 + term4 + 0.7

def evaluate_fitness(population, obj_func):
    return np.array([obj_func(ind) for ind in population])

def chitah_optimizer(obj_func, dim, pop_size, max_iter, lb, ub):
    # Initialize the population within the specified bounds
    population = np.random.uniform(low=lb, high=ub, size=(pop_size, dim))

    # Clip the values to ensure they are within the specified bounds
    population = np.clip(population, lb, ub)

    best_solution = np.zeros(dim)
    best_fitness = np.inf

    all_solutions = []

    for iteration in range(max_iter):
        for i in range(pop_size):
            fitness = obj_func(population[i, :])

            if fitness < best_fitness:
                best_fitness = fitness
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

# Example Usage with Bohachevsky Function
Bohadim = 4
population_size = 30
max_iterations = 25000

# Run the algorithm with functions and store the results in a CSV file
with open('COA-Bohachevsky-D4-25000.csv', mode='w') as results_file:
    results_writer = csv.writer(results_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    results_writer.writerow(['Iteration', 'Best_Solutions', 'Best_Fitness', 'Execution_Time'])

    # Run the algorithm with Bohachevsky function
    start_time = time.time()
    lb_boha = [-100] * 4
    ub_boha = [100] * 4
    best_fitness, best_solution, all_solutions = chitah_optimizer(bohachevsky, dim=Bohadim, pop_size=population_size, max_iter=max_iterations, lb=lb_boha, ub=ub_boha)
    end_time = time.time()

    for t in range(max_iterations):
        best_fitness_t = evaluate_fitness(all_solutions[t], bohachevsky).min()
        best_solution_t = all_solutions[t][np.argmin(evaluate_fitness(all_solutions[t], bohachevsky)), :]
        print(f"Iteration {t + 1}: Best Fitness = {best_fitness_t}, Best Solution = {best_solution_t}")
        # Export CSV file
        results_writer.writerow([t + 1, best_solution_t.tolist(), best_fitness_t, end_time - start_time])