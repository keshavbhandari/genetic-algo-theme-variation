from music21 import note
import math
from utils import *
import numpy as np


def melody_similarity(melody_a, melody_b):
    melody_a = chop_into_bars(melody_a)
    melody_b = chop_into_bars(melody_b)
    n_bars = min(len(melody_a), len(melody_b))
    ce_differences = [central_effect(melody_a[i]) - central_effect(melody_b[i]) for i in range(n_bars)]
    ce_distances = [np.sum(np.square(diff)) for diff in ce_differences]
    score = 1 - np.average(ce_distances)
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


def tempo_complexity(melody):
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
    score = sum(complexity) / len(complexity)
    return score


def metricity(notes):
    weights = [5, 1, 2, 1, 3, 1, 2, 1, 4, 1, 2, 1, 3, 1, 2, 1]
    positions = [int(n[2] * 4) for n in notes]
    # print('N', notes)
    # print('P', positions)
    metricity = sum([weights[p] for p in positions])
    return metricity


def harmony(melody, hyperparameters):
    scale_type = hyperparameters['scale_type']
    key_signature = hyperparameters['key_signature']
    tonic_class = note.Note(key_signature).pitch.pitchClass
    major_score_map = [5, 0, 2, 0,
                       4, 2, 0, 4,
                       0, 2, 0, 2]
    minor_score_map = [5, 0, 2, 4,
                       0, 2, 0, 4,
                       2, 0, 2, 0]
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
            if noteDifference > 7 and noteDifference:
                harmonyScore -= 8
            # Rules 7-10
            semi_to_tonic = (current_note - tonic_class + 12) % 12
            if scale_type == 'major':
                harmonyScore += major_score_map[semi_to_tonic]
            else:
                harmonyScore += minor_score_map[semi_to_tonic]

    return harmonyScore / len(melody)