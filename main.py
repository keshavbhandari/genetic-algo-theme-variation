from music21 import converter, stream, note, chord, meter, tempo, key, instrument
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
            voice_number+=1
        if voice_number % 2 == 0:
            if isinstance(element, note.Note):
                melody.append([element.pitch.midi, element.beat, element.offset, element.duration.quarterLength, element.volume.velocity])
            elif isinstance(element, chord.Chord):
                # Extract only the leading (higher pitch) note from the chord
                leading_note = max(element.pitches, key=lambda x: x.midi)
                melody.append([leading_note.midi, element.beat, element.offset, element.duration.quarterLength, element.volume.velocity])
    return melody

def create_midi_file(melody, output_file_path, bpm=120):
    # Create a stream for the melody
    # melody_stream = stream.Score()
    melody_part = stream.Part()
    # melody_stream.append(melody_part)
    
    # Add notes to the melody part
    for note_value in melody:
        pitch, beat, offset, duration,  velocity = note_value
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
    new_duration = note_value[3]/n_splits
    for i in range(n_splits):
        note = [note_value[0], note_value[1] + i * new_duration, note_value[2] + i * new_duration, new_duration, note_value[4]]
        splits.append(note)
    return splits

def swap_notes(melody):
    # Randomly choose two indices and swap the notes
    idx1, idx2 = random.sample(range(len(melody)), 2)
    melody[idx1][0], melody[idx2][0] = melody[idx2][0], melody[idx1][0]
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
        modification_type = 'add'
        if modification_type == 'split' and len(variation) > 1:
            # Split a random note into two notes
            # note_idx = choice([i for i in range(0, len(variation)-1) if not i in measure_indices])
            note_idx = random.randint(0, len(variation) - 1)
            # variation.pop(note_idx)
            variation[note_idx: note_idx+1] = split_note(variation[note_idx])
            # variation.extend(split_note(variation[note_idx]))
        elif modification_type == 'swap' and len(variation) > 1:
            # Swap two random notes within the melody
            variation = swap_notes(variation)
        elif modification_type == 'add':
            # Add note sequence by extending a random note
            note_idx = random.randint(0, len(variation) - 1)
            variation[note_idx:note_idx+1] = add_note_sequence(variation[note_idx])
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

# Pitch Mutation: A note out of harmony will be selected and changed to 
# one of a harmony degree based on the previous note.
def pitch_mutation(individual):
    note_idx = random.randint(1, len(individual) - 2)
    prev_note_idx = note_idx - 1
    previous_note = individual[prev_note_idx][0]
    chord_notes_previous_bar = [previous_note, previous_note-7, 
                                previous_note + 7]
    individual[note_idx][0] = min(random.choice(chord_notes_previous_bar), 127)
    return individual

# Define mutation operator (for example, random note change)
def mutate(individual, mutation_rate):
    for i in range(len(individual)):
        if np.random.rand() < mutation_rate:
            # Apply custom mutation (random note change, etc.)
            if np.random.rand() < 0.5:
                individual = pitch_mutation(individual)
                # individual[i][0] = np.random.randint(0, 128)  # MIDI note range
            else:
                individual = duration_mutation(individual)
    return individual

# Define fitness function (example: harmonic, similarity, rhythmic diversity)
def calculate_fitness(individual, original_melody):
    # Calculate fitness based on harmony, similarity, rhythmic diversity
    # ...
    fitness = 1  # Calculate the fitness score
    return fitness

# Main genetic algorithm function
def genetic_algorithm(original_melody, population_size, generations, crossover_rate, mutation_rate):
    population = initialize_population(original_melody, population_size)
    for generation in range(generations):
        # Evaluate fitness for each individual in the population
        fitness_scores = [calculate_fitness(individual, original_melody) for individual in population]
        # Select parents based on fitness scores (for simplicity, using roulette wheel selection)
        choice_indices = np.random.choice(len(population), size=2, p=np.array(fitness_scores) / np.sum(fitness_scores))
        parents = [population[i] for i in choice_indices]
        # Apply crossover
        child1, child2 = crossover(parents[0], parents[1], crossover_rate)
        # Apply mutation
        child1 = mutate(child1, mutation_rate)
        child2 = mutate(child2, mutation_rate)
        # Replace individuals in the population with the new offspring
        population[fitness_scores.index(min(fitness_scores))] = child1
        population[fitness_scores.index(min(fitness_scores))] = child2
    # Return the best individual after all generations
    best_individual = min(population, key=lambda x: calculate_fitness(x, original_melody))
    return best_individual

# Example usage
if __name__ == "__main__":

    # Load original melody from MIDI file
    original_melody = load_midi("Themes/twinkle-twinkle-little-star.mid")
    
    # Set genetic algorithm parameters
    population_size = 100
    generations = 20 #600
    crossover_rate = 0.5
    mutation_rate = 0.05

    # Initialize population
    population = initialize_population(original_melody, population_size)
    print("Initial Population:")
    print(population[0:2])

    # Run the genetic algorithm
    best_variation = genetic_algorithm(original_melody, population_size, generations, crossover_rate, mutation_rate)
    print(best_variation)
    create_midi_file(best_variation, "output/variation.mid", bpm=120)
    
