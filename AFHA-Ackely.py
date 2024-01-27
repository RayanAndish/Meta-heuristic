import random
import numpy as np
import time
import csv

def ackley(x):
    n = len(x)
    sum_sq = np.sum(x**2)
    sum_cos = np.sum(np.cos(2*np.pi*x))
    return -20.0*np.exp(-0.2*np.sqrt(sum_sq/n)) - np.exp(sum_cos/n) + 20.0 + np.exp(1)

def archer_fish_hunter(func, n, dim, max_iter, bounds):
    # Initialize the population within the specified bounds
    pop = np.random.uniform([bounds[0][0]] * dim, [bounds[0][1]] * dim, size=(n, dim))

    # Initialize the best solution and its fitness
    best_fish = pop[0]
    best_fitness = func(best_fish)

    # Initialize lists to store solutions and fitness values
    all_solutions = [pop.copy()]
    all_fitness = [np.apply_along_axis(func, 1, pop).copy()]

    # Perform the main loop for the specified number of iterations
    for i in range(max_iter):
        # Update the position of each fish by adding a random step
        step = np.random.uniform(-1, 1, size=(n, dim))
        pop += step

        # Keep the fish within the bounds of the search space
        for d in range(dim):
            pop[:, d] = np.clip(pop[:, d], bounds[0][0], bounds[0][1])

        # Evaluate the fitness of each fish
        fitness = np.apply_along_axis(func, 1, pop)

        # Update the best solution if necessary
        idx = np.argmin(fitness)
        if fitness[idx] < best_fitness:
            best_fish = pop[idx]
            best_fitness = fitness[idx]

        # Insert Data in each iteration
        all_solutions.append(pop.copy())
        all_fitness.append(fitness.copy())

    # Return the best solution, its fitness, and the lists of solutions and fitness values
    return best_fish, best_fitness, all_solutions, all_fitness

# Run the algorithm with functions and store the results in a CSV file
with open('AFHA-Ackley-D30-25000.csv', mode='w') as results_file:
    results_writer = csv.writer(results_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    results_writer.writerow(['Iteration', 'Best_Solutions', 'Best_Fitness', 'Execution_Time'])

    # Initialize start_time before calling the algorithm
    dim = 30
    start_time = time.time()
    best_fish, best_fitness, all_solutions, all_fitness = archer_fish_hunter(func=ackley, n=50, dim=30, max_iter=25000, bounds=[(-32, 32)] * dim)
    end_time = time.time()  # زمان پایان اجرا
    execution_time = end_time - start_time
    for t in range(len(all_solutions)):
        best_fitness_t = min(all_fitness[t])
        best_solution_t = all_solutions[t][np.argmin(all_fitness[t]), :]

        # چاپ اطلاعات به همراه ذخیره در فایل CSV
        print(f"Iteration {t + 1}: Best Fitness = {best_fitness_t}")
        print(f"Best Solutions: {best_solution_t}")

        results_writer.writerow([t + 1, list(best_solution_t), best_fitness_t, execution_time])

print(f"The execution time is: {execution_time} seconds.")