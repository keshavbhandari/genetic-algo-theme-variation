import random
import numpy as np
from utils import *
from music21 import converter, stream, note, chord, meter, tempo, key, instrument, scale


def swap_notes(bars):
    idx1 = random.randint(0, len(bars)-1)
    if idx1==len(bars)-1:
        idx2 = idx1-1
    else:
        idx2 = idx1+1
    bars[idx1][0], bars[idx2][0] = bars[idx2][0], bars[idx1][0]
    return bars


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


# # Define duration mutation: A note is randomly selected and its duration is
# # either doubled or reduced to half
# def duration_mutation(individual):
#     note_idx = random.randint(0, len(individual) - 1)
#     if individual[note_idx][3] > 1:
#         individual[note_idx][3] /= 2
#     elif individual[note_idx][3] < 1 and not individual[note_idx][2] + individual[note_idx][3] >= 4:
#         individual[note_idx][3] *= 2
#     else:
#         if np.random.rand() < 0.5:
#             individual[note_idx][3] /= 2
#         else:
#             if not individual[note_idx][2] + individual[note_idx][3] >= 4:
#                 individual[note_idx][3] *= 2
#     return individual


# Define duration mutation: A note is randomly selected and its duration is
# either doubled or reduced to half
def duration_mutation(individual):
    note_idx = random.randint(0, len(individual) - 1)
    if individual[note_idx][3] > 1 and not individual[note_idx][1] + individual[note_idx][3] >= 4:
        individual[note_idx][3] /= 2
    elif individual[note_idx][3] < 1 and not individual[note_idx][1] + individual[note_idx][3] >= 4:
        individual[note_idx][3] *= 2
    elif individual[note_idx][3] == 1 and not individual[note_idx][1] + individual[note_idx][3] >= 4:
        if np.random.rand() < 0.5:
            individual[note_idx][3] /= 2
        else:
            if not individual[note_idx][2] + individual[note_idx][3] >= 4:
                individual[note_idx][3] *= 2
    return individual


def get_harmonic_notes(note_value, scale_type):
    if scale_type == 'major':
        note_harmonic_degree = [i.midi for i in scale.MajorScale(note.Note(note_value).nameWithOctave).pitches]
    elif scale_type == 'minor':
        note_harmonic_degree = [i.midi for i in scale.MinorScale(note.Note(note_value).nameWithOctave).pitches]
    else:
        note_harmonic_degree = [i.midi for i in scale.DiatonicScale(note.Note(note_value).nameWithOctave).pitches]
    one_octave_lower = [i-12 for i in note_harmonic_degree]
    harmonic_notes = one_octave_lower + note_harmonic_degree
    return harmonic_notes


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
        if i not in [0, len(individual)] and np.random.rand() < mutation_rate:
            # Apply custom mutation (random note change, etc.)
            if np.random.rand() < 0.4:
                individual = duration_mutation(individual)
            else:
                individual = pitch_mutation(individual, scale_type)
    return individual


def post_processing(individuals, hyperparameter):
    for individual in individuals:
        # Fix cadence
        if hyperparameter['scale_type'] == 'major':
            individual[-1][0] = "C"
        elif hyperparameter['scale_type'] == 'minor':
            individual[-1][0] = "A"
    return individuals