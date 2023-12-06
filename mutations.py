import random
import numpy as np
from music21 import note, scale
import copy


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


def get_harmonic_notes(scale_type, key_signature):
    if scale_type == 'major':
        harmonic_notes = [i.midi for i in scale.MajorScale(key_signature).pitches]
    elif scale_type == 'minor':
        harmonic_notes = [i.midi for i in scale.MinorScale(key_signature).pitches]
    return harmonic_notes


def post_processing(individuals, hyperparameter):
    for individual in individuals:
        # Fix cadence
        # individual[-1][0] = hyperparameter["key_signature"]
        individual[-1][0] = individual[0][0]
    return individuals


def pitch_mutation(individual, harmonic_notes, note_idx):
    previous_note = individual[note_idx-1][0] % 12
    choices = [x for x in harmonic_notes if x < previous_note][-7:] + \
              [x for x in harmonic_notes if x >= previous_note][:7]
    jump = random.choice(choices) - previous_note
    individual[note_idx][0] = individual[note_idx-1][0] + jump
    return individual


def duration_mutation(individual, note_idx):
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

def mutate(pool, pMuta, harmonic_notes):
    mutated_pool = []
    for individual in pool:
        for note_idx in range(len(individual)):
            if random.random() < pMuta[0] and note_idx != 0:
                individual = pitch_mutation(individual, harmonic_notes, note_idx)
            if random.random() < pMuta[1]:
                individual = duration_mutation(individual, note_idx)
        mutated_pool.append(copy.deepcopy(individual))
    return mutated_pool
