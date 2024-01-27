import random
import numpy as np
import time
import csv
import pandas as pd

# Sphere function
def sphere(x):
    return sum([xi**2 for xi in x])

def evaluate_fitness(population, obj_func):
    return np.array([obj_func(ind) for ind in population])

def chitah_optimizer(obj_func, dim, pop, iter, lb, ub):
    population = np.random.uniform(low=lb, high=ub, size=(pop, dim))
    fitness_values = np.zeros(pop)
    best_solution = np.zeros(dim)
    best_fitness = np.inf

    all_solutions = []

    for iteration in range(iter):
        for i in range(pop):
            fitness_values[i] = obj_func(population[i, :])

            if fitness_values[i] < best_fitness:
                best_fitness = fitness_values[i]
                best_solution = population[i, :]

        all_solutions.append(population.copy())

        # Chitah Algorithm Steps
        a = 0.7 - 0.5 * iteration / iter  # Acceleration coefficient
        r1 = np.random.rand(pop, dim)
        r2 = np.random.rand(pop, dim)

        population = population + a * (r1 * (best_solution - population) + r2 * np.random.uniform(low=lb, high=ub, size=(pop, dim)))

        # Apply bounds
        population = np.minimum(np.maximum(population, lb), ub)

    return best_fitness, best_solution, all_solutions

# Example Usage with Ackley Function
dim = 30
population_size = 30
max_iterations = 25000

# Run the algorithm with functions and store the results in a CSV file
with open('COA-Sphere-D30-25000.csv', mode='w') as results_file:
    results_writer = csv.writer(results_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    results_writer.writerow(['Iteration', 'Best_Solutions', 'Best_Fitness', 'Execution_Time'])

    # Run the algorithm with ackley function
    start_time = time.time()
    lb_sphere = [-5] * dim
    ub_sphere = [12] * dim
    best_fitness, best_solution, all_solutions = chitah_optimizer(sphere, dim=dim, pop=population_size, iter=max_iterations, lb=lb_sphere, ub=ub_sphere)
    end_time = time.time()

    for t in range(max_iterations):
        best_fitness_t = evaluate_fitness(all_solutions[t], sphere).min()
        best_solution_t = all_solutions[t][np.argmin(evaluate_fitness(all_solutions[t], sphere)), :]
        print(f"Iteration {t + 1}: Best Fitness = {best_fitness_t}, Best Solution = {best_solution_t}")
        # Export CSV file
        results_writer.writerow([t + 1, best_solution_t.tolist(), best_fitness_t, end_time - start_time])