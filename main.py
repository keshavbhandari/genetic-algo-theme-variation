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


def split_note(note_value, n_splits=2):
    # Split note into n notes with half duration
    splits = []
    new_duration = note_value[3] / n_splits
    for i in range(n_splits):
        note = [note_value[0], note_value[1] + i * new_duration, note_value[2] + i * new_duration, new_duration,
                note_value[4]]
        splits.append(note)
    return splits


def swap_notes(melody):
    # Chop melody into bars
    bars = chop_into_bars(melody)
    # Randomly choose an indices within a bar
    bar_idx = random.sample(range(len(bars)), 1)[0]
    idx1, idx2 = random.sample(range(len(bars[bar_idx])), 2)
    melody[bar_idx+idx1][0], melody[bar_idx+idx2][0] = melody[bar_idx+idx2][0], melody[bar_idx+idx1][0]
    return melody


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
        # measure_indices = [i for i, x in enumerate(variation) if isinstance(x, stream.Measure)]
        # Apply random modification (split note, swap notes, add note sequence)
        modification_type = random.choice(['split', 'swap', 'add'])
        if modification_type == 'split' and len(variation) > 1:
            # Split a random note into two notes
            # note_idx = choice([i for i in range(0, len(variation)-1) if not i in measure_indices])
            note_idx = random.randint(0, len(variation) - 1)
            # variation.pop(note_idx)
            variation[note_idx: note_idx + 1] = split_note(variation[note_idx])
            # variation.extend(split_note(variation[note_idx]))
        elif modification_type == 'swap' and len(variation) > 1:
            # Swap two random notes within the melody
            variation = swap_notes(variation)
        elif modification_type == 'add':
            # Add note sequence by extending a random note
            note_idx = random.randint(0, len(variation) - 1)
            variation[note_idx:note_idx + 1] = add_note_sequence(variation[note_idx])
        population.append(variation)
    return population


def crossover(parent1, parent2, crossover_rate):
    if np.random.rand() < crossover_rate:
        # Single point crossover: Split parents at a random point and swap the segments
        crossover_point = random.randint(1, min(len(parent1), len(parent2)) - 1)
        child1 = parent1[:crossover_point] + parent2[crossover_point:]
        child2 = parent2[:crossover_point] + parent1[crossover_point:]
        return child1, child2
    else:
        child1 = parent1.copy()
        child2 = parent2.copy()
        return child1, child2


# Define duration mutation: A note is randomly selected and its duration is
# either doubled or reduced to half
def duration_mutation(individual):
    note_idx = random.randint(0, len(individual) - 1)
    if individual[note_idx][3] > 1:
        individual[note_idx][3] /= 2
    elif individual[note_idx][3] < 1:
        individual[note_idx][3] *= 2
    else:
        if np.random.rand() < 0.5:
            individual[note_idx][3] /= 2
        else:
            individual[note_idx][3] *= 2
    return individual

def get_harmonic_notes(note_value, scale_type):
    if scale_type == 'major':
        note_harmonic_degree = [i.midi for i in scale.MajorScale(note.Note(note_value).nameWithOctave).pitches]
    elif scale_type == 'minor':
        note_harmonic_degree = [i.midi for i in scale.MajorScale(note.Note(note_value).nameWithOctave).pitches]
    else:
        note_harmonic_degree = [i.midi for i in scale.DiatonicScale(note.Note(note_value).nameWithOctave).pitches]
    one_octave_lower = [i-12 for i in note_harmonic_degree]
    harmonic_notes = one_octave_lower + note_harmonic_degree
    return harmonic_notes

# Pitch Mutation: A note out of harmony will be selected and changed to
# one of a harmony degree based on the previous note.
def pitch_mutation(individual, scale_type):
    note_idx = random.randint(1, len(individual) - 2)
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
            if np.random.rand() < 0.5:
                # Add note sequence by extending a random note
                note_idx = random.randint(0, len(individual) - 1)
                if individual[note_idx][3]>=1:
                    individual[note_idx:note_idx + 1] = add_note_sequence(individual[note_idx])
            elif np.random.rand() < 0.9:
                individual = duration_mutation(individual)
                # individual[i][0] = np.random.randint(0, 128)  # MIDI note range
            # else:
            #     individual = pitch_mutation(individual, scale_type)
    return individual


def chop_into_bars(melody):
    chop_at = [idx for idx, note in enumerate(melody) if note[1] == 1] + [len(melody)]
    chop_into = [melody[chop_at[i]:chop_at[i + 1]] for i in range(len(chop_at) - 1)]
    return chop_into


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
    metricity = sum([weights[p] for p in positions])
    return metricity


def harmony(melody):
    harmonyScore = 0
    harmonic_interval_rules = {0: 3, 2: 3, 4: 3, 5: 3, 7: 3}
    for j, note_value in enumerate(melody):
        if j != 0:
            prev_note = melody[j-1][0]
            current_note = melody[j][0]
            # Calculate how many semitones away this note is from the previous one
            noteDifference = abs(current_note - prev_note)
            if noteDifference in harmonic_interval_rules.keys():
                harmonyScore += noteDifference
            if noteDifference > 7:
                harmonyScore -= 8
    return harmonyScore



# Define fitness function (example: harmonic, similarity, rhythmic diversity)
def calculate_fitness(individual, original_melody, hyperparameters):
    # Calculate fitness based on harmony, similarity, rhythmic diversity
    similarity = melody_similarity(individual, original_melody, hyperparameters['w_similarity'])
    complexity = tempo_complexity(individual, hyperparameters['w_tempo'])
    # print(similarity, complexity)
    # Calculate the fitness score
    # fitness = hyperparameters['w_similarity'] * similarity + hyperparameters['w_tempo'] * complexity
    harmony_score = harmony(individual)
    fitness = harmony_score
    return fitness

def NormalizeData(data):
    normal_data = (data - np.min(data)) / (np.max(data) - np.min(data))
    normal_data = normal_data.tolist()
    return normal_data

# Main genetic algorithm function
def genetic_algorithm(original_melody, population_size, generations, crossover_rate, mutation_rate, scale_type, hyperparameters):
    population = initialize_population(original_melody, population_size)
    for generation in range(generations):
        # Evaluate fitness for each individual in the population
        fitness_scores = [calculate_fitness(individual, original_melody, hyperparameters) for individual in population]
        # code to replace all negative value with 0
        fitness_scores = NormalizeData(fitness_scores)
        # Select parents based on fitness scores (for simplicity, using roulette wheel selection)
        choice_indices = np.random.choice(len(population), size=2, p=np.array(fitness_scores) / np.sum(fitness_scores))
        # top_2_scores = sorted(fitness_scores)[-2:]
        # print(top_2_scores)
        # choice_indices = [fitness_scores.index(s) for s in top_2_scores]
        parents = [population[i] for i in choice_indices]
        # Apply crossover
        child1, child2 = crossover(parents[0], parents[1], crossover_rate)
        # Apply mutation
        child1 = mutate(child1, mutation_rate, scale_type)
        child2 = mutate(child2, mutation_rate, scale_type)
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


# Example usage
if __name__ == "__main__":
    # Load original melody from MIDI file
    original_melody = load_midi("Themes/twinkle-twinkle-little-star.mid")
    
    # Set genetic algorithm parameters
    hyperparameters = {'w_similarity': 0.5, 'w_tempo': 0.5}
    population_size = 100
    generations = 50 #600
    crossover_rate = 0.5
    mutation_rate = 0.05
    scale_type = "major"

    # Initialize population
    population = initialize_population(original_melody, population_size)
    # print("Initial Population:")
    # print(population[0:2])

    # Run the genetic algorithm
    best_variations = genetic_algorithm(original_melody, population_size, generations, 
                                       crossover_rate, mutation_rate, scale_type, hyperparameters)
    # print(best_variation)
    for i in range(len(best_variations)):
        create_midi_file(best_variations[i], f"output/variation_{i}.mid", bpm=120)
    
