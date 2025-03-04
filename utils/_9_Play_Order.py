import os
import time
import fluidsynth

# FluidSynth Paths
fluidsynth_bin_path = r"C:\tools\fluidsynth\bin"
soundfont_path = r"C:\tools\fluidsynth\MuseScore_General.sf2"

# MIDI note-to-number mapping (Middle C = 60)
MIDI_NOTES = {
    "C": 60, "C#": 61, "D": 62, "D#": 63, "E": 64, "F": 65, "F#": 66,
    "G": 67, "G#": 68, "A": 69, "A#": 70, "B": 71
}

# Beat duration mapping (assuming 120 BPM)
BASE_DURATION = 0.5  # Quarter note duration in seconds
BEAT_DURATIONS = {
    "4": BASE_DURATION * 4,  # Quarter note
    "2": BASE_DURATION * 2,  # Half note
    "1": BASE_DURATION,  # Whole note
    "1/2": BASE_DURATION / 2,  # Eighth note
    "1/4": BASE_DURATION / 4   # Sixteenth note
}

def setup_fluidsynth():
    """
    Initializes FluidSynth with the given SoundFont.
    """
    fs = fluidsynth.Synth()
    fs.start(driver="dsound")  # Use DirectSound for Windows
    sfid = fs.sfload(soundfont_path)
    fs.program_select(0, sfid, 0, 0)  # Select bank 0, preset 0 (default piano)
    return fs

def group_notes_by_x(NoteHead_Dictionary, x_tolerance=20):
    """
    Groups notes with similar x-positions to be played together within a single staff.

    :param NoteHead_Dictionary: Dictionary containing detected noteheads.
    :param x_tolerance: The threshold distance to group notes as simultaneous.
    :return: List of grouped notes [(x_group, [(pitch, beat, multiplier)])].
    """
    sorted_notes = sorted(NoteHead_Dictionary.items(), key=lambda item: item[0][0])  # Sort by x-position
    grouped_notes = []
    
    current_group = []
    current_x = None

    for (x, y, w, h), note_info in sorted_notes:
        pitch = note_info.get("Pitch", "Unknown")
        beat = note_info.get("Beat", "4")
        multiplier = note_info.get("Multiplier", 0)  # Get octave shift

        if current_x is None or abs(x - current_x) <= x_tolerance:
            current_group.append((pitch, beat, multiplier))
        else:
            grouped_notes.append((current_x, current_group))
            current_group = [(pitch, beat, multiplier)]

        current_x = x  # Update reference x-position

    if current_group:
        grouped_notes.append((current_x, current_group))

    return grouped_notes

def Play_Music(ALL_NOTE_DICT, bpm=120):
    """
    Plays back music using FluidSynth, processing each dictionary in order and applying octave scaling.

    :param ALL_NOTE_DICT: List of NoteHead_Dictionary from all grand staves.
    :param bpm: Beats per minute (default: 120 BPM).
    """
    fs = setup_fluidsynth()  # Initialize FluidSynth
    base_duration = 60 / bpm  # Duration of a quarter note in seconds

    print("Playing music with FluidSynth...")

    for staff_index, NoteHead_Dictionary in enumerate(ALL_NOTE_DICT):
        print(f"Playing Grand Staff {staff_index + 1}...")

        grouped_notes = group_notes_by_x(NoteHead_Dictionary)  # Group simultaneous notes for each staff
        print(grouped_notes)

        for x_position, note_group in grouped_notes:
            midi_notes = []
            max_duration = 0  # Find the longest duration in this group

            for pitch, beat, multiplier in note_group:
                if pitch in MIDI_NOTES:
                    midi_note = MIDI_NOTES[pitch] + (multiplier * 12)  # **Apply octave scaling**
                    midi_notes.append(midi_note)
                else:
                    print(f"Skipping unrecognized pitch: {pitch}")

                duration = base_duration * BEAT_DURATIONS.get(beat, BASE_DURATION)
                max_duration = max(max_duration, duration)  # Ensure all notes play together for the longest note

            # Play MIDI notes (chord)
            for note in midi_notes:
                fs.noteon(0, note, 100)  # Channel 0, Note, Velocity 100

            time.sleep(max_duration)  # Wait for the note(s) to finish

            # Stop MIDI notes
            for note in midi_notes:
                fs.noteoff(0, note)

    fs.delete()  # Stop FluidSynth
    print("Playback finished.")
