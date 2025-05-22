import numpy as np
import sounddevice as sd
import mido
import time as pytime

# IMPORTANT SETTINGS
active_notes = {}  # Dictionary to track active notes and their waveforms
max_polyphony = 6  # Maximum number of simultaneous notes
sampling_rate = 22050



# Function to convert MIDI note to frequency
def midi_to_frequency(midi_note):
    return 440.0 * (2 ** ((midi_note - 69) / 12))

# Function to generate a sawtooth wave
def generate_sawtooth(frequency, duration, amplitude=1.0, sampling_rate=44100):
    num_samples = int(sampling_rate * duration)
    time = np.linspace(0, duration, num_samples, endpoint=False)
    cycles = frequency * time
    return amplitude * (cycles - np.floor(cycles)) * 2 - 1  # Scale to range [-1, 1]

# Function to generate an LFO waveform
def generate_lfo(frequency, duration, amplitude=1.0, sampling_rate=44100, lfo_type='sine'):
    num_samples = int(sampling_rate * duration)
    time = np.linspace(0, duration, num_samples, endpoint=False)
    if lfo_type == 'sine':
        return amplitude * np.sin(2 * np.pi * frequency * time)
    elif lfo_type == 'triangle':
        return amplitude * (2 * np.abs(2 * (time * frequency - np.floor(time * frequency + 0.5))) - 1)
    elif lfo_type == 'square':
        return amplitude * np.sign(np.sin(2 * np.pi * frequency * time))
    else:
        raise ValueError("Unsupported LFO type. Use 'sine', 'triangle', or 'square'.")

# Function to generate a sawtooth wave with LFO modulation
def generate_sawtooth_with_lfo(frequency, duration, lfo_frequency=5.0, lfo_amplitude=0.1, amplitude=1.0, sampling_rate=44100):
    num_samples = int(sampling_rate * duration)
    time = np.linspace(0, duration, num_samples, endpoint=False)
    cycles = frequency * time

    # Generate the LFO
    lfo = generate_lfo(lfo_frequency, duration, amplitude=lfo_amplitude, sampling_rate=sampling_rate)

    # Modulate the frequency with the LFO
    modulated_frequency = frequency + lfo
    modulated_cycles = np.cumsum(modulated_frequency / sampling_rate)

    # Generate the sawtooth wave with modulated frequency
    return amplitude * (modulated_cycles - np.floor(modulated_cycles)) * 2 - 1  # Scale to range [-1, 1]

# Function to apply an ADSR envelope
def apply_adsr(wave, sampling_rate, attack=0.01, decay=0.1, sustain=0.7, release=0.2):
    num_samples = len(wave)
    attack_samples = int(attack * sampling_rate)
    decay_samples = int(decay * sampling_rate)
    release_samples = int(release * sampling_rate)

    # Ensure the total ADSR duration does not exceed the waveform length
    total_adsr_samples = attack_samples + decay_samples + release_samples
    if total_adsr_samples > num_samples:
        # Scale down the ADSR parameters proportionally
        scale_factor = num_samples / total_adsr_samples
        attack_samples = int(attack_samples * scale_factor)
        decay_samples = int(decay_samples * scale_factor)
        release_samples = int(release_samples * scale_factor)

    sustain_samples = num_samples - (attack_samples + decay_samples + release_samples)
    if sustain_samples < 0:
        sustain_samples = 0  # Ensure sustain_samples is not negative

    # Generate ADSR envelope
    envelope = np.concatenate([
        np.linspace(0, 1, attack_samples),  # Attack
        np.linspace(1, sustain, decay_samples),  # Decay
        np.full(sustain_samples, sustain),  # Sustain
        np.linspace(sustain, 0, release_samples)  # Release
    ])
    return wave[:len(envelope)] * envelope  # Apply envelope

# Current state for polyphonic synth (up to 2 notes)


# Helper function for the audio callback
def audio_callback(outdata, frames, time, status):
    if status:
        print(status)
    # Mix all active notes
    if active_notes:
        mixed_wave = np.zeros(frames)  # Initialize with silence
        for note, wave in list(active_notes.items()):
            # Add the waveform to the mix, up to the requested number of frames
            mixed_wave[:len(wave[:frames])] += wave[:frames]
            # Remove played frames from the waveform
            active_notes[note] = wave[frames:]
            if len(active_notes[note]) == 0:  # Remove note if fully played
                del active_notes[note]
        # Ensure the mixed wave fits the output shape
        outdata[:] = mixed_wave.reshape(-1, 1)
    else:
        outdata.fill(0)  # Output silence if no active notes

# Create a single audio stream for all notes
stream = sd.OutputStream(
    samplerate=sampling_rate,
    channels=1,
    callback=audio_callback,
    blocksize=1024  # Larger block size for smoother playback
)
stream.start()
# Function to apply a low-pass filter
def low_pass_filter(wave, cutoff_frequency, sampling_rate):
    rc = 1.0 / (2 * np.pi * cutoff_frequency)
    dt = 1.0 / sampling_rate
    alpha = dt / (rc + dt)
    filtered_wave = np.zeros_like(wave)
    filtered_wave[0] = wave[0]  # Initialize the first sample
    for i in range(1, len(wave)):
        filtered_wave[i] = alpha * wave[i] + (1 - alpha) * filtered_wave[i - 1]
    return filtered_wave
# Open MIDI input
midi_input_name = mido.get_input_names()[0]  # Use the first available MIDI input
with mido.open_input(midi_input_name) as midi_input:
    print(f"Listening for MIDI input on {midi_input_name}...")

    for msg in midi_input:
        if msg.type == 'note_on' and msg.velocity > 0:  # Note-on message
            new_note = msg.note
            new_frequency = midi_to_frequency(new_note)
            print(f"Playing note {new_note} (Frequency: {new_frequency} Hz)")

            if len(active_notes) < max_polyphony:  # Allow only up to 2 notes
                duration = 10.0  # Note duration
                sawtooth = generate_sawtooth_with_lfo(
                    new_frequency, duration, lfo_frequency=5.0, lfo_amplitude=0.5, sampling_rate=sampling_rate
                )
                sawtooth = apply_adsr(sawtooth, sampling_rate)

                # Apply the low-pass filter
                cutoff_frequency = 1500  # Set the cutoff frequency in Hz
                sawtooth = low_pass_filter(sawtooth, cutoff_frequency, sampling_rate)

                active_notes[new_note] = sawtooth

        elif msg.type == 'note_off' or (msg.type == 'note_on' and msg.velocity == 0):  # Note-off message
            print(f"Note {msg.note} off")
            if msg.note in active_notes:  # Remove the corresponding note
                del active_notes[msg.note]
