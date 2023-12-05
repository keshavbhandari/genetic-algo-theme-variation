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
    for _ in range(population_size - 1):
        variation = copy.deepcopy(melody)
        variation_bars = chop_into_bars(variation)
        all_bars = []
        for n, bars in enumerate(variation_bars):
            if n < len(variation_bars)-1:
                note_idx = random.randint(1, len(bars)-2)
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
                    if np.random.rand() < 0.5:
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


# Main genetic algorithm function
def genetic_algorithm(original_melody, population_size, generations, crossover_rate, mutation_rate, hyperparameters):
    population = initialize_population(original_melody, population_size)
    population = sorted(population, key=lambda individual: calculate_fitness(individual, original_melody, hyperparameters), reverse=True)[0:100]

    for generation in range(generations):
        population = sorted(population, key=lambda individual: calculate_fitness(individual, original_melody, hyperparameters), reverse=True)
        # Evaluate fitness for each individual in the population
        fitness_scores = [calculate_fitness(individual, original_melody, hyperparameters) for individual in population]
        
        if hyperparameters['verbose']:
            print(generation, np.round(np.mean(fitness_scores), 2), np.round(sorted(fitness_scores, reverse=True)[:10], 2))

        # Select parents based on fitness scores (for simplicity, using roulette wheel selection)
        choice_indices = np.random.choice(len(population), size=2, p=np.array(fitness_scores) / np.sum(fitness_scores))
        parent_a = copy.deepcopy(population[choice_indices[0]])
        parent_b = copy.deepcopy(population[choice_indices[1]])
        
        # Apply crossover
        if np.random.rand() < crossover_rate:
            child1, _ = crossover(parent_a, parent_b)
            # Apply mutation
            child1 = mutate(child1, mutation_rate, hyperparameters['scale_type'])
            # child2 = mutate(child2, mutation_rate, hyperparameters['scale_type'])
        else:
            # Apply mutation
            parent = random.sample([parent_a, parent_b], 1)[0]
            child1 = copy.deepcopy(parent)
            child1 = mutate(child1, mutation_rate, hyperparameters['scale_type'])
            # child2 = mutate(parent_b, mutation_rate, hyperparameters['scale_type'])
        
        child1_fitness = calculate_fitness(child1, original_melody, hyperparameters)
        # child2_fitness = calculate_fitness(child2, original_melody, hyperparameters)
        parent_a_fitness = fitness_scores[choice_indices[0]] #calculate_fitness(parent_a, original_melody, hyperparameters)
        parent_b_fitness = fitness_scores[choice_indices[1]] #calculate_fitness(parent_b, original_melody, hyperparameters)

        # if child1_fitness >= min(fitness_scores):
        if child1_fitness >= min(parent_a_fitness, parent_b_fitness) and child1 not in population:
            min_fitness_index = fitness_scores.index(sorted(fitness_scores)[0])
            population[min_fitness_index] = child1

    # Return the best individual after all generations
    # best_individual = max(population, key=lambda x: calculate_fitness(x, original_melody, hyperparameters))
    best_individuals = sorted(population, key=lambda x: calculate_fitness(x, original_melody, hyperparameters), reverse=True)[:10]
    best_individuals = post_processing(best_individuals, hyperparameters)
    return best_individuals


# # Main genetic algorithm function
# def genetic_algorithm(original_melody, population_size, generations, crossover_rate, mutation_rate, hyperparameters):
#     """ Runs genetic algorithm until a genome with the specified MAX_FITNESS score has been reached"""

#     population = initialize_population(original_melody, population_size)

#     nextGeneration = []
#     for i in range(generations):
#         # print(i)

#         population = sorted(population, key=lambda individual: calculate_fitness(individual, original_melody, hyperparameters), reverse=True)
#         fitness_scores = [calculate_fitness(individual, original_melody, hyperparameters) for individual in population]
#         if hyperparameters['verbose']:
#             print(i, np.round(np.mean(fitness_scores), 2))
#         nextGeneration = population[0:2]

#         for j in range(int(len(population)) - 1):
#             # parentA, parentB = selectParents(population, original_melody, hyperparameters)
#             # Select parents based on fitness scores (for simplicity, using roulette wheel selection)
#             choice_indices = np.random.choice(len(population), size=2, p=np.array(fitness_scores) / np.sum(fitness_scores))
#             parents = [population[i] for i in choice_indices]
#             parentA, parentB = parents[0], parents[1]
#             if np.random.rand() < crossover_rate:
#                 child1, child2 = crossover(parentA, parentB)
#                 child1 = mutate(child1, mutation_rate, hyperparameters['scale_type'])
#             else:
#                 child1 = random.sample([parentA, parentB], 1)[0]
#                 # child1 = copy.deepcopy(parentA)
#                 child1 = mutate(child1, mutation_rate, hyperparameters['scale_type'])

#             nextGeneration += [child1]

#         population = nextGeneration

#     best_individuals = sorted(population, key=lambda x: calculate_fitness(x, original_melody, hyperparameters), reverse=True)[0:10]
#     return best_individuals


# Example usage
if __name__ == "__main__":
    # Load original melody from MIDI file
    original_melody, key_signature, key_type = load_midi("Themes/twinkle-twinkle-little-star.mid")
    # original_melody, key_signature, key_type = load_midi_v2("Themes/twinkle-twinkle-little-star.mid")
    # original_melody, key_signature, key_type = load_midi_v2("Themes/Bach_Minuet_in_G.mid")
    # original_melody, key_signature, key_type = load_midi_v2("Themes/Sweet.mid")
    create_midi_file(original_melody, f"Output/original.mid", bpm=120)
    
    # Set genetic algorithm parameters
    hyperparameters = {'w_harmony': 14, 'w_similarity': 1, 'w_tempo': 5, 'scale_type': key_type, 'key_signature': key_signature,
                       'print_metrics': False, 'verbose': True}
    print(hyperparameters)
    
    population_size = 1000
    generations = 600 #200 #50
    crossover_rate = 0.5
    mutation_rate = 0.05

    # Initialize population
    population = initialize_population(original_melody, population_size)

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


# Hill climbing
# Plateau, random restart, keep the best