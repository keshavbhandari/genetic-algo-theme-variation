from music21 import converter, stream, note, chord, meter, tempo, key, instrument, scale

def load_midi(file_path):
    # file_path = "Themes/Twinkle-Little-Star (Long Version).mid"
    midi_stream = converter.parse(file_path)
    key_signature = midi_stream.analyze('key').tonicPitchNameWithCase
    key_type = midi_stream.analyze('key').type
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
    return melody, key_signature, key_type


def load_midi_v2(file_path):
    # file_path = "Themes/Twinkle-Little-Star (Long Version).mid"
    midi_stream = converter.parse(file_path)
    key_signature = midi_stream.analyze('key').tonicPitchNameWithCase
    key_type = midi_stream.analyze('key').type
    midi_stream = midi_stream.parts[0]
    melody = []
    for element in midi_stream.recurse():
        if isinstance(element, note.Note):
            melody.append([element.pitch.midi, element.beat, element.offset, element.duration.quarterLength,
                            element.volume.velocity])
        elif isinstance(element, chord.Chord):
            # Extract only the leading (higher pitch) note from the chord
            leading_note = max(element.pitches, key=lambda x: x.midi)
            melody.append([leading_note.midi, element.beat, element.offset, element.duration.quarterLength,
                            element.volume.velocity])
    
    trimmed_melody = []
    bar_counter = 0
    for element in melody:
        if element[1] == 1:
            bar_counter += 1
        if bar_counter <=12:
            trimmed_melody.append(element)

    return trimmed_melody, key_signature, key_type


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

def chop_into_bars(melody):
    chop_at = [idx for idx, note in enumerate(melody) if note[1] == 1] + [len(melody)]
    chop_into = [melody[chop_at[i]:chop_at[i + 1]] for i in range(len(chop_at) - 1)]
    return chop_into

def join_into_melody(bars):
    melody = []
    for b in bars:
        melody += b
    return melody