from music21 import converter, stream, note, chord, meter, tempo, key, instrument, scale
import random
import numpy as np
import copy
from utils import *
from fitness import *
from mutations import *


def initialize_population(melody, population_size):
    # Start with the original melody
    population = [copy.deepcopy(melody)]
    # population = [melody.copy()]
    for _ in range(population_size - 1):
        variation = copy.deepcopy(melody)
        variation_bars = chop_into_bars(variation)
        all_bars = []
        for n, bars in enumerate(variation_bars):
            if n < len(variation_bars)-1:
                note_idx = random.randint(0, len(bars)-1)
                # Apply random modification (split note, swap notes, add note sequence)
                modification_type = random.choice(['split', 'swap', 'add'])
                if modification_type == 'split':
                    if bars[note_idx][3] >= 0.5:
                        # Split a random note into two notes
                        bars[note_idx: note_idx + 1] = split_note(bars[note_idx])
                elif modification_type == 'swap':
                    # Swap two random notes within the melody
                    bars = swap_notes(bars)
                elif modification_type == 'add':
                    if bars[note_idx][3] >= 1:
                        # Add note sequence by extending a random note
                        bars[note_idx:note_idx + 1] = add_note_sequence(bars[note_idx])
            all_bars.append(bars)
        variation_joined = join_into_melody(all_bars)
        population.append(variation_joined)
    return population


def crossover(parent1, parent2):
    # Single point crossover: Split parents at a random point and swap the segments
    # Chop melody into bars
    bars_p1 = chop_into_bars(parent1)
    bars_p2 = chop_into_bars(parent2)
    # Randomly choose an indices within a bar
    crossover_point = random.randint(1, min(len(bars_p1), len(bars_p2)) - 1)
    child1 = parent1[:crossover_point] + parent2[crossover_point:]
    child2 = parent2[:crossover_point] + parent1[crossover_point:]
    return child1, child2


# Define fitness function (example: harmonic, similarity, rhythmic diversity)
def calculate_fitness(individual, original_melody, hyperparameters):
    # Calculate fitness based on harmony, similarity, rhythmic diversity
    # similarity = melody_similarity(individual, original_melody)
    # complexity = tempo_complexity(individual) * 100
    # harmony_score = harmony(individual, hyperparameters)
    # print('similairty:', similarity, 'complexity:', complexity, 'harmony_score:', harmony_score)
    similarity = melody_similarity(individual, original_melody, hyperparameters['w_similarity'])
    complexity = tempo_complexity(individual, hyperparameters['w_tempo'])
    harmony_score = harmony(individual, hyperparameters)
    # Calculate the fitness score
    fitness = hyperparameters['w_harmony'] * harmony_score + hyperparameters['w_similarity'] * similarity + hyperparameters['w_tempo'] * complexity
    if hyperparameters['print_metrics']:
        # fitness = similarity + complexity + harmony_score * hyperparameters['w_harmony']
        print(hyperparameters['w_harmony'] * harmony_score, similarity * hyperparameters['w_similarity'], hyperparameters['w_tempo'] * complexity)
    return fitness


# def NormalizeData(data):
#     normal_data = (data - np.min(data)) / (np.max(data) - np.min(data))
#     normal_data = normal_data.tolist()
#     return normal_data


# def selectParents(population, original_melody, hyperparameters):
#     """ Selects two sequences from the population. Probability of being selected is weighted by the fitness score of each sequence """
#     parentA, parentB = random.choices(population, weights=[calculate_fitness(genome, original_melody, hyperparameters) for genome in population], k=2)
#     return parentA, parentB


# Main genetic algorithm function
def genetic_algorithm(original_melody, population_size, generations, crossover_rate, mutation_rate, hyperparameters):
    population = initialize_population(original_melody, population_size)
    for generation in range(generations):
        population = sorted(population, key=lambda individual: calculate_fitness(individual, original_melody, hyperparameters), reverse=True)
        # Evaluate fitness for each individual in the population
        fitness_scores = [calculate_fitness(individual, original_melody, hyperparameters) for individual in population]
        if hyperparameters['verbose']:
            print(generation, np.round(sorted(fitness_scores, reverse=True)[:10], 2))
        # code to replace all negative value with 0
        # fitness_scores = NormalizeData(fitness_scores)
        # Select parents based on fitness scores (for simplicity, using roulette wheel selection)
        # choice_indices = np.random.choice(len(population), size=2, p=np.array(fitness_scores) / np.sum(fitness_scores))
        # parents = [population[i] for i in choice_indices]
        parents = population[0:2]
        parent_a = copy.deepcopy(parents[0])
        parent_b = copy.deepcopy(parents[1])
        # Apply crossover
        if np.random.rand() < crossover_rate:
            child1, child2 = crossover(parent_a, parent_b)
            # Apply mutation
            child1 = mutate(child1, mutation_rate, hyperparameters['scale_type'])
            child2 = mutate(child2, mutation_rate, hyperparameters['scale_type'])
        else:
            # Apply mutation
            child1 = mutate(parent_a, mutation_rate, hyperparameters['scale_type'])
            child2 = mutate(parent_b, mutation_rate, hyperparameters['scale_type'])
        
        child1_fitness = calculate_fitness(child1, original_melody, hyperparameters)
        child2_fitness = calculate_fitness(child2, original_melody, hyperparameters)
        parent_a_fitness = calculate_fitness(parent_a, original_melody, hyperparameters)
        parent_b_fitness = calculate_fitness(parent_b, original_melody, hyperparameters)

        # if child1_fitness >= min(fitness_scores):
        if child1_fitness >= min(parent_a_fitness, parent_b_fitness):
            min_fitness_index = fitness_scores.index(sorted(fitness_scores)[0])
            population[min_fitness_index] = child1
        # if child2_fitness >= min(fitness_scores):
        if child2_fitness >= min(parent_a_fitness, parent_b_fitness):
            min_fitness_index = fitness_scores.index(sorted(fitness_scores)[0])
            population[min_fitness_index] = child2

        # # Replace individuals in the population with the new offspring
        # min_fitness_index = fitness_scores.index(sorted(fitness_scores)[0])
        # second_min_fitness_index = fitness_scores.index(sorted(fitness_scores)[1])
        # population[min_fitness_index] = child1
        # population[second_min_fitness_index] = child2
    # Return the best individual after all generations
    # best_individual = max(population, key=lambda x: calculate_fitness(x, original_melody, hyperparameters))
    best_individual = sorted(population, key=lambda x: calculate_fitness(x, original_melody, hyperparameters), reverse=True)[:10]
    return best_individual


# Main genetic algorithm function
# def genetic_algorithm(original_melody, population_size, generations, crossover_rate, mutation_rate, hyperparameters):
#     """ Runs genetic algorithm until a genome with the specified MAX_FITNESS score has been reached"""

#     population = initialize_population(original_melody, population_size)

#     nextGeneration = []
#     for i in range(generations):
#         print(i)

#         population = sorted(population, key=lambda individual: calculate_fitness(individual, original_melody, hyperparameters), reverse=True)

#         nextGeneration = population[0:2]

#         for j in range(int(len(population) / 2) - 1):
#             parentA, parentB = selectParents(population, original_melody, hyperparameters)
#             if np.random.rand() < crossover_rate:
#                 child1, child2 = crossover(parentA, parentB)
#                 child1 = mutate(child1, mutation_rate, hyperparameters['scale_type'])
#                 child2 = mutate(child2, mutation_rate, hyperparameters['scale_type'])
#             else:
#                 child1 = mutate(parentA, mutation_rate, hyperparameters['scale_type'])
#                 child2 = mutate(parentB, mutation_rate, hyperparameters['scale_type'])

#             nextGeneration += [child1, child2]

#         population = nextGeneration

#     best_individuals = sorted(population, key=lambda x: calculate_fitness(x, original_melody, hyperparameters), reverse=True)[0:10]
#     return best_individuals


# Example usage
if __name__ == "__main__":
    # Load original melody from MIDI file
    original_melody = load_midi("Themes/twinkle-twinkle-little-star.mid")
    
    # Set genetic algorithm parameters
    hyperparameters = {'w_harmony': 1, 'w_similarity': 5, 'w_tempo': 5, 'scale_type': 'major', 
                       'print_metrics': False, 'verbose': True}
    population_size = 100
    generations = 50 #600
    crossover_rate = 0.5
    mutation_rate = 0.05

    # Initialize population
    population = initialize_population(original_melody, population_size)
    # print("Initial Population:")
    # print(population[0:2])

    # Run the genetic algorithm
    best_variations = genetic_algorithm(original_melody, population_size, generations, 
                                       crossover_rate, mutation_rate, hyperparameters)
    # print(best_variation)
    for i, piece in enumerate(best_variations):
        create_midi_file(piece, f"Output/variation_{i}.mid", bpm=120)
    
# Things to do:
# 1. Change the pitch mutations by incorporating more sequence patterns
# 2. If it's a minim, do not add pitch or rhythm mutation
# 3. Change initialization by adding mutations to different bars

# Best: 'w_harmony': 1, 'w_similarity': 15, 'w_tempo': 0 with 10 generations