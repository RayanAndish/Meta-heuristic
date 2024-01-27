import random
import numpy as np
import time
import csv


# rosenbrock function
def rosenbrock(x):
    return sum([100 * (x[i + 1] - x[i] ** 2) ** 2 + (x[i] - 1) ** 2 for i in range(len(x) - 1)])


def AVA(obj_func, lb, ub, n_vultures, n_iterations):
    dim = len(lb)
    vultures = np.zeros((n_vultures, dim))
    fitness = np.zeros(n_vultures)
    all_solutions = []  # لیستی برای ذخیره تمام موقعیت‌ها
    all_fitness = []  # لیستی برای ذخیره تمام مقادیر تابع هدف

    for i in range(n_vultures):
        vultures[i, :] = [random.uniform(lb[j], ub[j]) for j in range(dim)]
        fitness[i] = obj_func(vultures[i, :])

    best_fitness = np.min(fitness)
    best_vulture = vultures[np.argmin(fitness), :]

    # افزودن اطلاعات اولیه به لیست‌ها
    all_solutions.append(vultures.copy())
    all_fitness.append(fitness.copy())

    start_time = time.time()  # زمان شروع اجرا

    for t in range(n_iterations):
        for i in range(n_vultures):
            j = random.randint(0, n_vultures - 1)
            while j == i:
                j = random.randint(0, n_vultures - 1)

            new_pos = vultures[i, :] + (best_vulture - vultures[i, :]) \
                      + (vultures[j, :] - vultures[i, :]) * random.uniform(-1, 1)

            for d in range(dim):
                if new_pos[d] < lb[d]:
                    new_pos[d] = lb[d]
                elif new_pos[d] > ub[d]:
                    new_pos[d] = ub[d]

            new_fitness = obj_func(new_pos)

            if new_fitness < fitness[i]:
                vultures[i, :] = new_pos
                fitness[i] = new_fitness

        if np.min(fitness) < best_fitness:
            best_fitness = np.min(fitness)
            best_vulture = vultures[np.argmin(fitness), :]

        # افزودن اطلاعات به لیست‌ها در هر دور
        all_solutions.append(vultures.copy())
        all_fitness.append(fitness.copy())

    end_time = time.time()  # زمان پایان اجرا
    execution_time = end_time - start_time

    return all_solutions, all_fitness, execution_time


# Define the dimension for the Greiwank function
dim = 30

# Run the algorithm with functions and store the results in a CSV file
with open('AVOA-Rosenbrock-D30-25000.csv', mode='w') as results_file:
    results_writer = csv.writer(results_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    results_writer.writerow(['Iteration', 'Best_Solutions', 'Best_Fitness', 'Execution_Time'])

    lb_rosen = [-5] * dim
    ub_rosen = [10] * dim
    all_solutions, all_fitness, execution_time = AVA(rosenbrock, lb=lb_rosen, ub=ub_rosen, n_vultures=10,
                                                     n_iterations=25000)

    for t in range(len(all_solutions)):
        best_fitness_t = min(all_fitness[t])
        best_solution_t = all_solutions[t][np.argmin(all_fitness[t]), :]

        # چاپ اطلاعات به همراه ذخیره در فایل CSV
        print(f"Iteration {t + 1}: Best Fitness = {best_fitness_t}")
        print(f"Best Solutions: {best_solution_t}")

        results_writer.writerow([t + 1, best_solution_t.tolist(), best_fitness_t, execution_time])