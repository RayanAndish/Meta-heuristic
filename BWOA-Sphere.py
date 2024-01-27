import random
import numpy as np
import time
import csv


# Sphere function
def sphere(x):
    return sum([xi ** 2 for xi in x])


class BWOA:
    def __init__(self, function, dim, num_agents, iterations, alpha, beta, bounds):
        self.function = function
        self.dim = dim
        self.num_agents = num_agents
        self.iterations = iterations
        self.alpha = alpha
        self.best_agent = None
        self.beta = beta
        self.bounds = bounds
        self.best_fitness = float('inf')

    def execute(self):
        all_fitness = []
        all_agents = []

        for i in range(self.iterations):
            # Use a different random initial position in each iteration
            self.agents = np.random.uniform(low=np.array([b[0] for b in self.bounds]),
                                            high=np.array([b[1] for b in self.bounds]),
                                            size=(self.num_agents, self.dim))

            # Evaluate fitness for each agent
            fitness = np.array([self.function(agent) for agent in self.agents])

            # Update best agent
            best_idx = np.argmin(fitness)
            if fitness[best_idx] < self.best_fitness:
                self.best_agent = self.agents[best_idx]
                self.best_fitness = fitness[best_idx]

            all_agents.append(self.agents.copy())
            all_fitness.append(fitness.tolist())

            # Update agent positions
            for j in range(self.num_agents):
                if j == best_idx:
                    continue
                r_jk = np.random.uniform()
                r_ik = np.random.uniform()
                self.agents[j] += self.alpha * (self.best_agent - self.agents[j]) + self.beta * (
                            self.agents[best_idx] - self.agents[j]) * r_jk + self.beta * (
                                              self.agents[best_idx] - self.agents[j]) * r_ik

        # Return best fitness, best agent, and all fitness values for each iteration
        best_spider_idx = np.argmin([self.function(agent) for agent in self.agents])
        best_spider = self.agents[best_spider_idx]
        return self.best_fitness, self.best_agent, all_fitness, all_agents


# Run the algorithm with functions and store the results in a CSV file
with open('BWOA-Sphere-D30-25000.csv', mode='w') as results_file:
    results_writer = csv.writer(results_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    results_writer.writerow(['Iteration', 'Best_Solutions', 'Best_Fitness', 'Execution_Time'])

    # Number of different initializations
    num_initializations = 1

    for initialization in range(num_initializations):
        # Run the algorithm with function
        start_time = time.time()
        bwoa = BWOA(sphere, dim=30, num_agents=20, iterations=25000, alpha=0.1, beta=0.1, bounds=[(-5, 12, 5, 12)])
        best_fitness, best_agent, all_fitness, all_agents = bwoa.execute()
        end_time = time.time()

        # Write the results to CSV
        for iteration, (fitness_values, agents) in enumerate(zip(all_fitness, all_agents)):
            best_solutions = agents[np.argmin(fitness_values)]
            best_fitness = min(fitness_values)
            results_writer.writerow([iteration + 1, best_solutions.tolist(), best_fitness, end_time - start_time])

            # Print fitness values for each iteration
            print(f"Iteration {iteration + 1} - Best Solutions: {best_solutions}")
            print(f"Iteration {iteration + 1} - Best fitness: {best_fitness}")

        print(f"Execution time: {end_time - start_time} seconds")
