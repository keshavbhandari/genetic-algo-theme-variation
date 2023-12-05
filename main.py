import os
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
            if n < len(variation_bars) - 1:
                note_idx = random.randint(1, len(bars) - 2)
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


def crossover(parents, pCO):
    # randomly pair up mating parents
    random.shuffle(parents)
    half = int(len(parents) * 0.5)
    pairs = zip(parents[:half], parents[half:])
    children = []
    for parent_a, parent_b in pairs:
        if random.random() <= pCO:
            bars_a = chop_into_bars(parent_a)
            bars_b = chop_into_bars(parent_b)
            crossover_point = random.randint(1, min(len(bars_a), len(bars_b)) - 1)
            child_a = parent_a[:crossover_point] + parent_b[crossover_point:]
            child_b = parent_b[:crossover_point] + parent_a[crossover_point:]
            children.append(copy.deepcopy(child_a))
            children.append(copy.deepcopy(child_b))
        else:
            children.append(copy.deepcopy(parent_a))
            children.append(copy.deepcopy(parent_b))
    return children


# Define fitness function (example: harmonic, similarity, rhythmic diversity)
def calculate_fitness(individual, original_melody, hyperparameters):
    # Calculate fitness based on harmony, similarity, rhythmic diversity
    # print('similairty:', similarity, 'complexity:', complexity, 'harmony_score:', harmony_score)
    similarity = melody_similarity(individual, original_melody, hyperparameters['w_similarity'])
    complexity = tempo_complexity(individual, hyperparameters['w_tempo'])
    harmony_score = harmony(individual, hyperparameters)
    # Calculate the fitness score
    fitness = hyperparameters['w_harmony'] * harmony_score + hyperparameters['w_similarity'] * similarity + \
              hyperparameters['w_tempo'] * complexity
    if hyperparameters['print_metrics']:
        # fitness = similarity + complexity + harmony_score * hyperparameters['w_harmony']
        print(hyperparameters['w_harmony'] * harmony_score, similarity * hyperparameters['w_similarity'],
              hyperparameters['w_tempo'] * complexity)
    return fitness


# Main genetic algorithm function
def sort_by_fitness(pool, original_melody, hyperparameters):
    fitness_scores = [calculate_fitness(individual, original_melody, hyperparameters) for individual in pool]
    sorted_zip = sorted(zip(fitness_scores, pool), reverse=True)
    pool = [x for _, x in sorted_zip]
    fitness_scores = [x for x, _ in sorted_zip]
    return pool, fitness_scores


def genetic_algorithm(original_melody, population_size, generations, crossover_rate, mutation_rate, hyperparameters):
    population = initialize_population(original_melody, population_size)
    population, fitness_scores = sort_by_fitness(population, original_melody, hyperparameters)

    for generation in range(generations):
        parents = population[:population_size]
        children = crossover(parents, crossover_rate)
        children = mutate(children, mutation_rate, hyperparameters['scale_type'])
        population = parents + children
        population, fitness_scores = sort_by_fitness(population, original_melody, hyperparameters)
        if hyperparameters['verbose']:
            print(generation, np.round(np.mean(fitness_scores), 2), np.round(fitness_scores[:10], 2))

    # Return the best individual after all generations
    best_individuals = population[:10]
    best_individuals = post_processing(best_individuals, hyperparameters)
    return best_individuals


if __name__ == "__main__":

    # track_name = 'twinkle-twinkle-little-star'
    track_name = 'Bach_Minuet_in_G'
    # track_name = 'Sweet'

    theme_midi_file = f'Themes/{track_name}.mid'
    output_folder = f'ablation/{track_name}'
    original_midi_file = f'{output_folder}/original.mid'
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # original_melody, key_signature, key_type = load_midi(theme_midi_file)
    original_melody, key_signature, key_type = load_midi_v2(theme_midi_file)
    create_midi_file(original_melody, original_midi_file, bpm=120)

    # Set genetic algorithm parameters
    hyperparameters = {'w_harmony': 1, 'w_similarity': 1, 'w_tempo': 0,
                       'scale_type': key_type, 'key_signature': key_signature,
                       'print_metrics': False, 'verbose': True}
    print(hyperparameters)

    population_size = 500
    generations = 50
    crossover_rate = 0.9
    mutation_rate = [0.03, 0.01]

    # Initialize population
    population = initialize_population(original_melody, population_size)

    # Run the genetic algorithm
    best_variations = genetic_algorithm(original_melody, population_size, generations,
                                        crossover_rate, mutation_rate, hyperparameters)
    # print(best_variation)
    output_sub_folder = f'{output_folder}/H-{hyperparameters["w_harmony"]}' \
                        f'_S-{hyperparameters["w_similarity"]}' \
                        f'_T-{hyperparameters["w_tempo"]}'

    if not os.path.exists(output_sub_folder):
        os.makedirs(output_sub_folder)

    for i, piece in enumerate(best_variations):
        create_midi_file(piece, f"{output_sub_folder}/variation_{i}.mid", bpm=120)
