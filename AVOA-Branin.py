import random
import numpy as np
import time
import csv

#branin function
def branin(x):
    a = 1
    b = 5.1 / (4 * np.pi**2)
    c = 5 / np.pi
    r = 6
    s = 10
    t = 1 / (8 * np.pi)
    return a * (x[1] - b * x[0]**2 + c * x[0] - r)**2 + s * (1 - t) * np.cos(x[0]) + s

def AVA(obj_func, lb, ub, n_vultures, n_iterations):
    dim = len(lb)
    vultures = np.zeros((n_vultures, dim))
    fitness = np.zeros(n_vultures)
    all_solutions = []  # لیستی برای ذخیره تمام موقعیت‌ها
    all_fitness = []    # لیستی برای ذخیره تمام مقادیر تابع هدف

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

# Run the algorithm with functions and store the results in a CSV file
with open('AVOA-Branin-D4-25000.csv', mode='w') as results_file:
    results_writer = csv.writer(results_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    results_writer.writerow(['Iteration', 'Best_Solutions', 'Best_Fitness', 'Execution_Time'])

    Bdim = 4  # Set dimension to 2 for the 2-dimensional Branin function
    lb_branin = [-5, 0] * Bdim
    ub_branin = [10, 15] * Bdim
    all_solutions, all_fitness, execution_time = AVA(branin, lb=lb_branin, ub=ub_branin, n_vultures=10, n_iterations=25000)

    for t in range(len(all_solutions)):
        best_fitness_t = min(all_fitness[t])
        best_solution_t = all_solutions[t][np.argmin(all_fitness[t]), :]

        # چاپ اطلاعات به همراه ذخیره در فایل CSV
        print(f"Iteration {t + 1}: Best Fitness = {best_fitness_t}")
        print(f"Best Solutions: {best_solution_t}")

        results_writer.writerow([t + 1, best_solution_t.tolist(), best_fitness_t, execution_time])