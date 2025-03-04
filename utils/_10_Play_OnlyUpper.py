import fluidsynth
import time

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
    Initializes FluidSynth with the given SoundFont and a valid audio driver.
    """
    fs = fluidsynth.Synth()

    try:
        fs.start(driver="dsound")  # Change to your system driver if needed
    except Exception as e:
        print(f"Error initializing FluidSynth: {e}")
        print("Trying alternative driver...")
        fs.start(driver="alsa" if os.name != "nt" else "dsound")  # Fallback for Linux/Windows

    sfid = fs.sfload(soundfont_path)
    if sfid == -1:
        print("Error loading SoundFont! Check the path.")
        return None

    fs.program_select(0, sfid, 0, 0)  # Select bank 0, preset 0 (default piano)
    return fs

def Play_Upper_Staff_Notes(ALL_NOTE_DICT, image_height, bpm=120):
    """
    Plays only the notes from the lower staff (center_y > image_height / 2).

    :param ALL_NOTE_DICT: List of NoteHead_Dictionary from all grand staves.
    :param image_height: The height of the sheet music image.
    :param bpm: Beats per minute (default: 120 BPM).
    """
    fs = setup_fluidsynth()
    if not fs:
        print("Failed to initialize FluidSynth. Exiting playback.")
        return

    base_duration = 60 / bpm  # Duration of a quarter note in seconds
    mid_line = image_height / 2  # Define middle of the image

    print("Playing lower staff notes with FluidSynth...")

    for staff_index, NoteHead_Dictionary in enumerate(ALL_NOTE_DICT):
        print(f"Processing Grand Staff {staff_index + 1}...")

        for (x, y, w, h), note_info in sorted(NoteHead_Dictionary.items(), key=lambda item: item[0][0]):
            center_x, center_y = note_info.get("Center_Point", (None, None))
            pitch = note_info.get("Pitch", "Unknown")
            beat = note_info.get("Beat", "4")

            if center_y is None or center_y >= mid_line:  # Skip notes in the upper staff
                continue

            if pitch not in MIDI_NOTES:
                print(f"Skipping unrecognized pitch: {pitch}")
                continue

            midi_note = MIDI_NOTES[pitch]
            duration = base_duration * BEAT_DURATIONS.get(beat, BASE_DURATION)

            # Play MIDI note
            fs.noteon(0, midi_note, 100)  # Channel 0, Note, Velocity 100
            time.sleep(duration)  # Hold the note
            fs.noteoff(0, midi_note)  # Stop the note

    fs.delete()  # Stop FluidSynth
    print("Playback finished.")
