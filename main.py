import math
from music21 import converter, stream, note, chord, meter, tempo, key, instrument, scale
import random
import numpy as np

def load_midi(file_path):
    # file_path = "Themes/Twinkle-Little-Star (Long Version).mid"
    midi_stream = converter.parse(file_path)
    melody = []
    voice_number = 1
    for element in midi_stream.recurse():
        # if isinstance(element, stream.Measure):
        #         melody.append(element)
        if isinstance(element, stream.Voice):
            voice_number += 1
        if voice_number % 2 == 0:
            if isinstance(element, note.Note):
                melody.append([element.pitch.midi, element.beat, element.offset, element.duration.quarterLength,
                               element.volume.velocity])
            elif isinstance(element, chord.Chord):
                # Extract only the leading (higher pitch) note from the chord
                leading_note = max(element.pitches, key=lambda x: x.midi)
                melody.append([leading_note.midi, element.beat, element.offset, element.duration.quarterLength,
                               element.volume.velocity])
    return melody


def create_midi_file(melody, output_file_path, bpm=120):
    # Create a stream for the melody
    # melody_stream = stream.Score()
    melody_part = stream.Part()
    # melody_stream.append(melody_part)

    # Add notes to the melody part
    for note_value in melody:
        pitch, beat, offset, duration, velocity = note_value
        n = note.Note(pitch, quarterLength=duration)
        n.offset = offset
        n.volume.velocity = int(velocity * 127)  # Velocity range: 0 to 127
        melody_part.append(n)

    # Set tempo and time signature
    melody_part.append(tempo.MetronomeMark(number=bpm))
    melody_part.append(meter.TimeSignature('4/4'))
    melody_part.append(key.KeySignature(0))

    # Set instrument for the treble clef track (assuming piano for simplicity)
    melody_part.insert(0, instrument.Piano())

    # Write the MIDI file
    melody_part = stream.Score([melody_part])

    # Write the melody to the MIDI file
    melody_part.write('midi', fp=output_file_path)
    print(f"Melody saved to {output_file_path}")


def swap_notes(melody):
    # Chop melody into bars
    bars = chop_into_bars(melody)
    # Randomly choose an indices within a bar
    bar_idx = random.sample(range(len(bars)), 1)[0]
    idx1 = random.randint(1, len(bars[bar_idx])+1)
    if idx1==len(bars[bar_idx]):
        idx2 = idx1-1
    else:
        idx2 = idx1+1
    melody[bar_idx+idx1][0], melody[bar_idx+idx2][0] = melody[bar_idx+idx2][0], melody[bar_idx+idx1][0]
    return melody


def split_note(note_value, n_splits=2):
    # Split note into n notes with half duration
    splits = []
    new_duration = note_value[3] / n_splits
    for i in range(n_splits):
        note = [note_value[0], note_value[1] + i * new_duration, note_value[2] + i * new_duration, new_duration,
                note_value[4]]
        splits.append(note)
    return splits


def add_note_sequence(note_value):
    # Create a chord degree note sequence (for simplicity, using a major chord)
    sequence = [note_value[0], note_value[0] - 7, note_value[0] - 3, note_value[0]]
    note_sequence = split_note(note_value, len(sequence))
    for i in range(len(sequence)):
        note_sequence[i][0] = sequence[i]
    return note_sequence


def initialize_population(melody, population_size):
    population = [melody.copy()]  # Start with the original melody
    for _ in range(population_size - 1):
        variation = melody.copy()
        for i in range(3):
            # Apply random modification (split note, swap notes, add note sequence)
            modification_type = random.choice(['split', 'swap', 'add'])
            if modification_type == 'split' and len(variation) > 1:
                # Split a random note into two notes
                note_idx = random.randint(1, len(variation) - 1)
                if variation[note_idx][3] >= 0.5:
                    variation[note_idx: note_idx + 1] = split_note(variation[note_idx])
            elif modification_type == 'swap' and len(variation) > 1:
                # Swap two random notes within the melody
                variation = swap_notes(variation)
            elif modification_type == 'add':
                # Add note sequence by extending a random note
                note_idx = random.randint(1, len(variation) - 1)
                if variation[note_idx][3] >= 1:
                    variation[note_idx:note_idx + 1] = add_note_sequence(variation[note_idx])
        population.append(variation)
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


# Define duration mutation: A note is randomly selected and its duration is
# either doubled or reduced to half
def duration_mutation(individual):
    note_idx = random.randint(0, len(individual) - 1)
    if individual[note_idx][3] > 1:
        individual[note_idx][3] /= 2
    elif individual[note_idx][3] < 1 and not individual[note_idx][2] + individual[note_idx][3] >= 4:
        individual[note_idx][3] *= 2
    else:
        if np.random.rand() < 0.5:
            individual[note_idx][3] /= 2
        else:
            if not individual[note_idx][2] + individual[note_idx][3] >= 4:
                individual[note_idx][3] *= 2
    return individual


# def get_harmonic_notes(note_value, scale_type):
#     if scale_type == 'major':
#         note_harmonic_degree = [i.midi for i in scale.MajorScale(note.Note(note_value).nameWithOctave).pitches]
#     elif scale_type == 'minor':
#         note_harmonic_degree = [i.midi for i in scale.MinorScale(note.Note(note_value).nameWithOctave).pitches]
#     else:
#         note_harmonic_degree = [i.midi for i in scale.DiatonicScale(note.Note(note_value).nameWithOctave).pitches]
#     one_octave_lower = [i-12 for i in note_harmonic_degree]
#     harmonic_notes = one_octave_lower + note_harmonic_degree
#     return harmonic_notes


def get_harmonic_notes(note_value, scale_type):
    if scale_type == 'major':
        note_harmonic_degree = [note_value-5, note_value, note_value+4]
    elif scale_type == 'minor':
        note_harmonic_degree = [note_value-5, note_value, note_value+3]
    return note_harmonic_degree


# Pitch Mutation: A note out of harmony will be selected and changed to
# one of a harmony degree based on the previous note.
def pitch_mutation(individual, scale_type):
    note_idx = random.randint(1, len(individual) - 1)
    prev_note_idx = note_idx - 1
    previous_note = individual[prev_note_idx][0]
    harmonic_notes = get_harmonic_notes(previous_note, scale_type)
    individual[note_idx][0] = random.choice(harmonic_notes)
    return individual


# Define mutation operator (for example, random note change)
def mutate(individual, mutation_rate, scale_type):
    for i in range(len(individual)):
        if np.random.rand() < mutation_rate:
            # Apply custom mutation (random note change, etc.)
            # if np.random.rand() < 0.5:
                # Add note sequence by extending a random note
                # note_idx = random.randint(0, len(individual) - 1)
                # if individual[note_idx][3]>=1:
                #     individual[note_idx:note_idx + 1] = add_note_sequence(individual[note_idx])
            if np.random.rand() < 0.5:
                individual = duration_mutation(individual)
                # individual[i][0] = np.random.randint(0, 128)  # MIDI note range
            else:
                individual = pitch_mutation(individual, scale_type)
    return individual


def chop_into_bars(melody):
    chop_at = [idx for idx, note in enumerate(melody) if note[1] == 1] + [len(melody)]
    chop_into = [melody[chop_at[i]:chop_at[i + 1]] for i in range(len(chop_at) - 1)]
    return chop_into


def join_into_melody(bars):
    melody = []
    for b in bars:
        melody += b
    return melody


def melody_similarity(melody_a, melody_b, weight=1):
    melody_a = chop_into_bars(melody_a)
    melody_b = chop_into_bars(melody_b)
    n_bars = min(len(melody_a), len(melody_b))
    ce_differences = [central_effect(melody_a[i]) - central_effect(melody_b[i]) for i in range(n_bars)]
    ce_distances = [np.sum(np.square(diff)) for diff in ce_differences]
    score = 100 - np.average(ce_distances) * weight
    return score


def central_effect(notes):
    positions = np.array([note_coordinate(n) for n in notes])
    durations = [n[3] for n in notes]
    weights = np.array([d / sum(durations) for d in durations])
    weighted_sum = np.sum(positions * weights[:, np.newaxis], axis=0)
    return weighted_sum


def note_coordinate(note, h=math.sqrt(1 / 6)):
    pitch_class_to_k = [0, 7, 2, 9, 4, 11, 6, 1, 8, 3, 10, 5]
    pitch_class = note[0] % 12
    octave = int(note[0] / 12)
    k = pitch_class_to_k[pitch_class] + 12 * octave
    t = k * math.pi / 2
    x = math.sin(t)
    y = math.cos(t)
    z = k * h
    return [x, y, z]


def tempo_complexity(melody, weight=1):
    max_metricity = [5,
                     5 + 4,
                     5 + 4 + 3,
                     5 + 4 + 3 * 2,
                     5 + 4 + 3 * 2 + 2,
                     5 + 4 + 3 * 2 + 2 * 2,
                     5 + 4 + 3 * 2 + 2 * 3,
                     5 + 4 + 3 * 2 + 2 * 4,
                     5 + 4 + 3 * 2 + 2 * 4 + 1,
                     5 + 4 + 3 * 2 + 2 * 4 + 2,
                     5 + 4 + 3 * 2 + 2 * 4 + 3,
                     5 + 4 + 3 * 2 + 2 * 4 + 4,
                     5 + 4 + 3 * 2 + 2 * 4 + 5,
                     5 + 4 + 3 * 2 + 2 * 4 + 6,
                     5 + 4 + 3 * 2 + 2 * 4 + 7,
                     5 + 4 + 3 * 2 + 2 * 4 + 8,
                     ]
    melody = chop_into_bars(melody)
    complexity = [max_metricity[len(bar)-1] - metricity(bar) for bar in melody]
    score = sum(complexity) / len(complexity) * weight
    return score


def metricity(notes):
    weights = [5, 1, 2, 1, 3, 1, 2, 1, 4, 1, 2, 1, 3, 1, 2, 1]
    positions = [int(n[2] * 4) for n in notes]
    # print('N', notes)
    # print('P', positions)
    metricity = sum([weights[p] for p in positions])
    return metricity


def harmony(melody, hyperparameters):
    harmonyScore = 0
    harmonic_interval_rules = {0: 3, 2: 3, 4: 3, 5: 3, 7: 3}
    for j, note_value in enumerate(melody):
        if j != 0:
            prev_note = melody[j-1][0]
            current_note = melody[j][0]
            # Calculate how many semitones away this note is from the previous one
            noteDifference = abs(current_note - prev_note)
            if noteDifference in harmonic_interval_rules.keys():
                harmonyScore += harmonic_interval_rules[noteDifference]
            if noteDifference > 7:
                harmonyScore -= 8
            # Rules 7-10
            if hyperparameters['scale_type'] == "major":
                if note.Note(current_note).name == "C":
                    harmonyScore += 4
                if note.Note(current_note).name in ["E", "G"]:
                    harmonyScore += 3
            elif hyperparameters['scale_type'] == "minor":
                if note.Note(current_note).name == "A":
                    harmonyScore += 4
                if note.Note(current_note).name in ["C", "E"]:
                    harmonyScore += 3
    return harmonyScore


# Define fitness function (example: harmonic, similarity, rhythmic diversity)
def calculate_fitness(individual, original_melody, hyperparameters):
    # Calculate fitness based on harmony, similarity, rhythmic diversity
    # similarity = melody_similarity(individual, original_melody)
    # complexity = tempo_complexity(individual) * 100
    # harmony_score = harmony(individual, hyperparameters)
    # print('similairty:', similarity, 'complexity:', complexity, 'harmony_score:', harmony_score)
    similarity = melody_similarity(individual, original_melody, hyperparameters['w_similarity'])
    complexity = tempo_complexity(individual, hyperparameters['w_tempo'])
    # Calculate the fitness score
    # fitness = hyperparameters['w_harmony'] * harmony_score + hyperparameters['w_similarity'] * similarity + hyperparameters['w_tempo'] * complexity
    harmony_score = harmony(individual, hyperparameters)
    fitness = similarity + complexity + harmony_score * hyperparameters['w_harmony']
    # print(similarity, complexity, harmony)
    return fitness


def NormalizeData(data):
    normal_data = (data - np.min(data)) / (np.max(data) - np.min(data))
    normal_data = normal_data.tolist()
    return normal_data


def selectParents(population, original_melody, hyperparameters):
    """ Selects two sequences from the population. Probability of being selected is weighted by the fitness score of each sequence """
    parentA, parentB = random.choices(population, weights=[calculate_fitness(genome, original_melody, hyperparameters) for genome in population], k=2)
    return parentA, parentB


# Main genetic algorithm function
def genetic_algorithm(original_melody, population_size, generations, crossover_rate, mutation_rate, hyperparameters):
    population = initialize_population(original_melody, population_size)
    for generation in range(generations):
        # Evaluate fitness for each individual in the population
        fitness_scores = [calculate_fitness(individual, original_melody, hyperparameters) for individual in population]
        print(sorted(fitness_scores, reverse=True)[:10])
        # code to replace all negative value with 0
        # fitness_scores = NormalizeData(fitness_scores)
        # Select parents based on fitness scores (for simplicity, using roulette wheel selection)
        choice_indices = np.random.choice(len(population), size=2, p=np.array(fitness_scores) / np.sum(fitness_scores))
        parents = [population[i] for i in choice_indices]
        # Apply crossover
        if np.random.rand() < crossover_rate:
            child1, child2 = crossover(parents[0], parents[1])
            # Apply mutation
            child1 = mutate(child1, mutation_rate, hyperparameters['scale_type'])
            child2 = mutate(child2, mutation_rate, hyperparameters['scale_type'])
        else:
            # Apply mutation
            child1 = mutate(parents[0], mutation_rate, hyperparameters['scale_type'])
            child2 = mutate(parents[1], mutation_rate, hyperparameters['scale_type'])
        # Replace individuals in the population with the new offspring
        # Replace individuals in the population with the new offspring
        min_fitness_index = fitness_scores.index(sorted(fitness_scores)[0])
        second_min_fitness_index = fitness_scores.index(sorted(fitness_scores)[1])
        population[min_fitness_index] = child1
        population[second_min_fitness_index] = child2
    # Return the best individual after all generations
    # best_individual = max(population, key=lambda x: calculate_fitness(x, original_melody, hyperparameters))
    best_individuals = sorted(population, key=lambda x: calculate_fitness(x, original_melody, hyperparameters), reverse=True)[0:10]
    return best_individuals


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
    hyperparameters = {'w_harmony': 1, 'w_similarity': 10, 'w_tempo': 1, 'scale_type': 'major'}
    population_size = 100
    generations = 50
    crossover_rate = 0.5
    mutation_rate = 0.001

    # Initialize population
    population = initialize_population(original_melody, population_size)
    # print("Initial Population:")
    # print(population[0:2])

    # Run the genetic algorithm
    best_variations = genetic_algorithm(original_melody, population_size, generations, 
                                       crossover_rate, mutation_rate, hyperparameters)
    # print(best_variation)
    for i, piece in enumerate(best_variations):
        create_midi_file(piece, f"output/variation_{i}.mid", bpm=120)
    
# Things to do:
# 1. Change cross over so that it doesn't take the last bar into consideration as this is the cadence.
# 2. Change the pitch mutations by incorporating more sequence patterns
# 3. Change melodic similarity fn
# 4. If it's a minim, do not add pitch or rhythm mutation
# 5. ``````select `top 2 parents by f score