import random
import numpy as np
import time
import csv

# Rastrigin function
def rastrigin(x):
    A = 10
    n = len(x)
    return A * n + sum([(xi**2 - A * np.cos(2 * np.pi * xi)) for xi in x])

def archer_fish_hunter(func, n, dim, max_iter, bounds):
    # Initialize the population within the specified bounds
    pop = np.random.uniform([b[0] for b in bounds], [b[1] for b in bounds], size=(n, dim))

    # Initialize the best solution and its fitness
    best_fish = pop[0]
    best_fitness = func(best_fish)

    # Initialize lists to store solutions and fitness values
    all_solutions = [pop.copy()]  # لیستی برای ذخیره تمام موقعیت‌ها
    all_fitness = [np.apply_along_axis(func, 1, pop).copy()]  # لیستی برای ذخیره تمام مقادیر تابع هدف

    # Initialize the execution time
    start_time = time.time()

    # Perform the main loop for the specified number of iterations
    for i in range(max_iter):
        # Update the position of each fish by adding a random step
        step = np.random.uniform(-1, 1, size=(n, dim))
        pop += step

        # Keep the fish within the bounds of the search space
        for d in range(dim):
            pop[:, d] = np.clip(pop[:, d], bounds[d][0], bounds[d][1])

        # Evaluate the fitness of each fish
        fitness = np.apply_along_axis(func, 1, pop)

        # Update the best solution if necessary
        idx = np.argmin(fitness)
        if fitness[idx] < best_fitness:
            best_fish = pop[idx]
            best_fitness = fitness[idx]

        # افزودن اطلاعات به لیست‌ها در هر دور
        all_solutions.append(pop.copy())
        all_fitness.append(fitness.copy())

    # Calculate the execution time
    end_time = time.time()
    execution_time = end_time - start_time

    # Return the best solution, its fitness, and the lists of solutions and fitness values
    return best_fish, best_fitness, all_solutions, all_fitness, execution_time

# Run the algorithm with functions and store the results in a CSV file
with open('AFHA-Rastrigin-D30-25000.csv', mode='w') as results_file:
    results_writer = csv.writer(results_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    results_writer.writerow(['Iteration', 'Best_Solutions', 'Best_Fitness', 'Execution_Time'])

    dim=30
    best_fish, best_fitness, all_solutions, all_fitness, execution_time = archer_fish_hunter(func=rastrigin, n=50, dim=30, max_iter=25000, bounds=[(-5, 12)] * dim)

    for t in range(len(all_solutions)):
        best_fitness_t = min(all_fitness[t])
        best_solution_t = all_solutions[t][np.argmin(all_fitness[t]), :]

        # چاپ اطلاعات به همراه ذخیره در فایل CSV
        print(f"Iteration {t + 1}: Best Fitness = {best_fitness_t}")
        print(f"Best Solutions: {best_solution_t}")

        results_writer.writerow([t + 1, best_solution_t.tolist(), best_fitness_t, execution_time])
