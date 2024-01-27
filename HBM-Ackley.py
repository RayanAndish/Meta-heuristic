import numpy as np
import time
import csv


# Ackley function
def ackley(x):
    n = len(x)
    sum_sq = np.sum(x ** 2)
    sum_cos = np.sum(np.cos(2 * np.pi * x))
    return -20.0 * np.exp(-0.2 * np.sqrt(sum_sq / n)) - np.exp(sum_cos / n) + 20.0 + np.exp(1)


# Define the Honey Bee Mating algorithm
def initialize_population(bees, dim, search_space):
    return np.random.uniform(search_space[0], search_space[1], size=(bees, dim))


def evaluate_fitness(population, func):
    return np.apply_along_axis(func, 1, population)


def select_best_n(population, fitness, n):
    indices = np.argsort(fitness)[:n]
    return population[indices], fitness[indices]


def employeed_bee_phase(employed_bees, search_space):
    perturbation = np.random.uniform(-1, 1, size=employed_bees.shape)
    new_solutions = employed_bees + perturbation
    new_solutions = np.clip(new_solutions, search_space[0], search_space[1])
    return new_solutions


def onlooker_bee_phase(employed_bees, employed_fitness):
    probabilities = employed_fitness / np.sum(employed_fitness)
    selected_indices = np.random.choice(len(employed_bees), len(employed_bees), p=probabilities)
    onlooker_bees = employed_bees[selected_indices]
    return onlooker_bees


def update_population(employed_bees, onlooker_bees, search_space, func):
    employed_fitness = evaluate_fitness(employed_bees, func)

    new_employed_bees = employeed_bee_phase(employed_bees, search_space)

    onlooker_bees_fitness = evaluate_fitness(onlooker_bees, func)
    onlooker_bees_probabilities = onlooker_bees_fitness / np.sum(onlooker_bees_fitness)
    selected_indices = np.random.choice(len(onlooker_bees), len(onlooker_bees), p=onlooker_bees_probabilities)
    selected_onlooker_bees = onlooker_bees[selected_indices]

    new_onlooker_bees = employeed_bee_phase(selected_onlooker_bees, search_space)

    return new_employed_bees, new_onlooker_bees


def hbm_algorithm(func, bees, dim, search_space, iterations):
    population = initialize_population(bees, dim, search_space)

    # Initialize lists to store solutions and fitness values
    all_solutions = [population.copy()]
    all_fitness = [evaluate_fitness(population, func).copy()]

    for iteration in range(iterations):
        fitness = evaluate_fitness(population, func)
        employed_bees, employed_fitness = select_best_n(population, fitness, bees // 2)

        onlooker_bees = onlooker_bee_phase(employed_bees, employed_fitness)

        new_employed_bees, new_onlooker_bees = update_population(employed_bees, onlooker_bees, search_space, func)

        # Update the population
        population = np.vstack([new_employed_bees, new_onlooker_bees])

        best_solution, best_fitness = select_best_n(population, evaluate_fitness(population, func), 1)

        # Insert Data in each iteration
        all_solutions.append(population.copy())
        all_fitness.append(evaluate_fitness(population, func).copy())

        print(f"Iteration {iteration + 1}: Best Fitness = {best_fitness[0]}, Best Solution ={best_solution[0]}")

    return best_solution[0], best_fitness[0], all_solutions, all_fitness


# Run the algorithm with functions and store the results in a CSV file
with open('HBM-Ackley-D30-25000.csv', mode='w') as results_file:
    results_writer = csv.writer(results_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    results_writer.writerow(['Iteration', 'Best_Solutions', 'Best_Fitness', 'Execution_Time'])

    # Initialize start_time before calling the algorithm
    start_time = time.time()
    result = hbm_algorithm(func=ackley, bees=50, dim=30, search_space=(-32, 32), iterations=25000)

    if result is not None:
        best_solution, best_fitness, all_solutions, all_fitness = result
        end_time = time.time()  # End Execution Time
        execution_time = end_time - start_time

        for t in range(len(all_solutions)):
            best_fitness_t = min(all_fitness[t])
            best_solution_t = all_solutions[t][np.argmin(all_fitness[t]), :]
            print(f"Iteration {t + 1}: Best Fitness = {best_fitness_t}, Best Solutions ={best_solution_t}")
            # Export CSV file
            results_writer.writerow([t + 1, best_solution_t.tolist(), best_fitness_t, execution_time])