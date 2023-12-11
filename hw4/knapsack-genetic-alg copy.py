import random

# Define the fitness function
def fitness(chromosome, values, weights, max_weight):
    total_value = sum(chromosome[i] * values[i] for i in range(len(chromosome)))
    total_weight = sum(chromosome[i] * weights[i] for i in range(len(chromosome)))
    if total_weight <= max_weight:
        return total_value
    else:
        return 0

# Define the selection function based on roulette wheel selection
def select_parent(roulette_wheel):
    pick = random.uniform(0, sum(roulette_wheel.values()))
    current = 0
    for key, value in roulette_wheel.items():
        current += value
        if current > pick:
            return key

# Define the crossover function
def crossover(chromosome1, chromosome2, mask):
    return [chromosome1[i] if mask[i] == '1' else chromosome2[i] for i in range(len(mask))]

# Define the mutation function
def mutate(chromosome, position):
    chromosome[position] = 1 if chromosome[position] == 0 else 0
    return chromosome

# Parameters
weights = [5, 8, 10]
values = [570, 710, 640]
max_weight = 15
population = [[0, 0, 0], [0, 0, 1], [0, 1, 0], [1, 0, 0]]

# Run the Genetic Algorithm
best_chromosome = None
best_fitness = 0
for generation in range(2):  # Two generations as per the problem statement
    # Calculate fitness for each chromosome
    fitness_results = {tuple(chromo): fitness(chromo, values, weights, max_weight) for chromo in population}
    
    # Update the best chromosome
    for chromo, fit in fitness_results.items():
        if fit > best_fitness:
            best_chromosome = chromo
            best_fitness = fit
            
    # Selection - roulette wheel
    total_fitness = sum(fitness_results.values())
    roulette_wheel = {chromo: fit / total_fitness for chromo, fit in fitness_results.items()}
    
    # Crossover
    if generation == 0:  # First generation
        crossover_mask = '110'
    else:  # Second generation
        crossover_mask = '100'
    
    new_population = []
    for _ in range(len(population) // 2):  # Assuming even number in the population
        parent1 = select_parent(roulette_wheel)
        parent2 = select_parent(roulette_wheel)
        child1 = crossover(parent1, parent2, crossover_mask)
        child2 = crossover(parent2, parent1, crossover_mask)
        new_population.extend([child1, child2])
    
    # Mutation
    if generation == 1:  # Mutation occurs in the second generation
        for chromo in new_population:
            if chromo == tuple([1, 0, 1]):
                mutate(chromo, 2)  # Mutate the 3rd bit
    
    # Prepare for the next generation
    population = new_population

# The best chromosome after two generations
print(f"Final answer: Best chromosome is {best_chromosome} with a fitness of {best_fitness}")
