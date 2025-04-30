# ... (Keep imports the same) ...
import numpy as np
import sounddevice as sd
from scipy.signal import sawtooth, square, butter, lfilter, lfilter_zi
import time
import math
import mido
import mido.backends.rtmidi
import threading
import queue
import keyboard # Optional for external control
import sys
import traceback
import random # Needed for arp random pattern
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import ttk
from tkinter import simpledialog, messagebox

# --- Default Parameters (Added Arp Params) ---
DEFAULT_PARAMS = {
    # Synth Core
    'osc_type': 'saw', 'attack': 0.01, 'decay': 0.2, 'sustain': 0.8, 'release': 0.3,
    'filter_cutoff': 5000, 'filter_resonance': 0.1, 'filter_env_amount': 0,
    'lfo_rate': 0.0, 'lfo_depth': 0.0, 'lfo_target': 'pitch', 'lfo_shape': 'sine',
    'volume': 0.6, 'analog_drive': 1.0, 'analog_drift': 0.0000,
    # FM
    'fm_mod_freq_ratio': 1.0, 'fm_mod_depth': 100.0,
    # Chorus FX
    'chorus_depth': 0.005, 'chorus_rate': 0.5,
    # --- Arpeggiator ---
    'arp_on': False,        # Arp enabled/disabled
    'arp_bpm': 120.0,       # Beats Per Minute
    'arp_rate': 16,         # Note division (4=1/4, 8=1/8, 16=1/16, 32=1/32)
    'arp_pattern': 'Up',    # 'Up', 'Down', 'UpDown', 'Random'
    'arp_octaves': 1,       # Range (1-4)
    # 'arp_gate': 0.95,     # Note length (optional, more complex to implement)
}

# --- Presets (Using only Default for brevity, add yours back!) ---
PRESETS = {
    "Default": DEFAULT_PARAMS.copy() # Start with default values
    # Add your other 150 presets back here!
    # e.g., "Pad: MW Pad": { ... 'arp_on': False, 'arp_bpm': 120, ... },
}

# --- Helper Functions (remain the same) ---
def midi_to_freq(note):
    return 432.0 * (2.0**((note - 69) / 12.0))

# --- Voice Class (remain the same) ---
class Voice:
    # (Keep the entire Voice class exactly as it was in the previous version)
    def __init__(self, note, velocity, params, sample_rate):
        self.note = note
        self.velocity = velocity / 127.0
        self.params = params # Expects a copy from the synth
        self.sample_rate = sample_rate
        self.freq = midi_to_freq(note) # Base carrier frequency
        self.carrier_phase = 0.0
        self.modulator_phase = 0.0 # For FM
        self.envelope_stage = 'attack'
        self.envelope_level = 0.0
        self.time_in_stage = 0.0
        self.release_start_level = 0.0
        self.filter_b = None
        self.filter_a = None
        self.filter_zi = None
        self._update_filter_coeffs(self.params['filter_cutoff'])
        if self.filter_b is not None and self.filter_a is not None:
            try:
                self.filter_zi = lfilter_zi(self.filter_b, self.filter_a)
                self.filter_zi = self.filter_zi.astype(np.float64)
            except ValueError as e:
                print(f"Warning: Init filter state failed: {e}", file=sys.stderr)
                filter_order = max(len(self.filter_a), len(self.filter_b)) - 1
                if filter_order > 0: self.filter_zi = np.zeros(filter_order, dtype=np.float64)
                else: self.filter_zi = None
        self.lfo_phase = np.random.rand() * 2 * np.pi
        self.drift_lfo_phase = np.random.rand() * 2 * np.pi

    def _update_filter_coeffs(self, cutoff_hz):
        p = self.params; sr = self.sample_rate; nyquist = sr / 2.0
        cutoff_hz = np.clip(cutoff_hz, 20, nyquist * 0.99)
        cutoff_norm = cutoff_hz / nyquist
        try:
            self.filter_b, self.filter_a = butter(2, cutoff_norm, btype='low', analog=False)
        except ValueError as e:
            print(f"Warning: butter failed cutoff {cutoff_hz}: {e}", file=sys.stderr)
            self.filter_b, self.filter_a = butter(2, 0.98, btype='low', analog=False)

    def process(self, num_samples):
        p = self.params; sr = self.sample_rate
        lfo_block = np.zeros(num_samples)
        if p['lfo_depth'] != 0 and p['lfo_rate'] > 0:
            lfo_freq_rad = 2 * np.pi * p['lfo_rate'] / sr
            t_phases = np.arange(self.lfo_phase, self.lfo_phase + num_samples * lfo_freq_rad, lfo_freq_rad)[:num_samples]
            shape = p.get('lfo_shape', 'sine')
            if shape == 'sine': lfo_block = np.sin(t_phases)
            elif shape == 'triangle': lfo_block = 2.0 * (np.abs((t_phases / np.pi) % 2.0 - 1.0)) - 1.0
            elif shape == 'saw': lfo_block = ((t_phases / np.pi) % 2.0) - 1.0
            elif shape == 'square': lfo_block = np.sign(np.sin(t_phases))
            self.lfo_phase = (t_phases[-1] + lfo_freq_rad) % (2 * np.pi)
        drift_block = np.zeros(num_samples)
        if p['analog_drift'] > 0:
            drift_lfo_rate = 0.15; drift_freq_rad = 2 * np.pi * drift_lfo_rate / sr
            t_drift_phases = np.arange(self.drift_lfo_phase, self.drift_lfo_phase + num_samples * drift_freq_rad, drift_freq_rad)[:num_samples]
            drift_block = np.sin(t_drift_phases)
            self.drift_lfo_phase = (t_drift_phases[-1] + drift_freq_rad) % (2 * np.pi)
        current_drift_block = drift_block * p['analog_drift'] * self.freq
        lfo_val = np.mean(lfo_block) if p['lfo_rate'] > 0 else 0.0
        lfo_target = p.get('lfo_target', 'pitch')
        lfo_val_pitch = lfo_val if lfo_target == 'pitch' else 0.0
        lfo_val_filter = lfo_val if lfo_target == 'filter' else 0.0
        lfo_val_amp = lfo_val if lfo_target == 'amp' else 0.0
        lfo_val_fm_depth = lfo_val if lfo_target == 'fm_depth' else 0.0
        env_level = self.envelope_level; time_in_st = self.time_in_stage
        carr_phase = self.carrier_phase; mod_phase = self.modulator_phase
        attack_samples = max(1, int(p['attack'] * sr)); decay_samples = max(1, int(p['decay'] * sr))
        release_samples = max(1, int(p['release'] * sr)); sustain_level = np.clip(p.get('sustain', 0.8), 0.0, 1.0)
        osc_phase_step_base = 2 * np.pi / sr; osc_type = p.get('osc_type', 'saw'); drive_gain = p.get('analog_drive', 1.0)
        osc_output_block = np.zeros(num_samples); env_block = np.zeros(num_samples)
        for i in range(num_samples):
            if self.envelope_stage == 'attack':
                env_level = min(1.0, time_in_st / attack_samples)
                if time_in_st >= attack_samples: self.envelope_stage = 'decay'; env_level = 1.0; time_in_st = 0
            elif self.envelope_stage == 'decay':
                env_level = sustain_level + (1.0 - sustain_level) * max(0.0, 1.0 - time_in_st / decay_samples)
                if time_in_st >= decay_samples: self.envelope_stage = 'sustain'; env_level = sustain_level; time_in_st = 0
            elif self.envelope_stage == 'sustain': env_level = sustain_level
            elif self.envelope_stage == 'release':
                env_level = self.release_start_level * max(0.0, 1.0 - time_in_st / release_samples)
                if time_in_st >= release_samples: self.envelope_stage = 'off'; env_level = 0.0
            elif self.envelope_stage == 'off':
                env_level = 0.0
                if i == 0: osc_output_block.fill(0.0); env_block.fill(0.0)
                else: osc_output_block[i:] = 0.0; env_block[i:] = 0.0
                break
            env_level = np.clip(env_level, 0.0, 1.0); env_block[i] = env_level
            if self.envelope_stage != 'sustain' and self.envelope_stage != 'off': time_in_st += 1
            current_drift = current_drift_block[i]; pitch_lfo_mod = lfo_val_pitch * p['lfo_depth']
            base_freq = max(1.0, self.freq + current_drift + pitch_lfo_mod)
            osc_val = 0.0
            if osc_type == 'fm':
                modulator_freq = base_freq * p['fm_mod_freq_ratio']; modulator_phase_increment = modulator_freq * osc_phase_step_base
                mod_phase = (mod_phase + modulator_phase_increment) % (2 * np.pi); modulator_output = np.sin(mod_phase)
                current_mod_depth = p['fm_mod_depth'] * env_level
                if lfo_target == 'fm_depth':
                     fm_lfo_mod_factor = 1.0 + p['lfo_depth'] * lfo_val_fm_depth * 0.5
                     current_mod_depth *= max(0.0, fm_lfo_mod_factor)
                freq_deviation = modulator_output * current_mod_depth; carrier_freq = max(1.0, base_freq + freq_deviation)
                carrier_phase_increment = carrier_freq * osc_phase_step_base; carr_phase = (carr_phase + carrier_phase_increment) % (2 * np.pi)
                osc_val = np.sin(carr_phase)
            else:
                carrier_phase_increment = base_freq * osc_phase_step_base; carr_phase = (carr_phase + carrier_phase_increment) % (2 * np.pi)
                if osc_type == 'saw': osc_val = (carr_phase / np.pi) - 1.0
                elif osc_type == 'square': osc_val = 1.0 if carr_phase < np.pi else -1.0
                elif osc_type == 'sine': osc_val = np.sin(carr_phase)
                elif osc_type == 'triangle': osc_val = 2.0 * (abs(carr_phase / np.pi - 1.0)) - 1.0
                else: osc_val = 0.0
            if drive_gain > 1.0: osc_val = np.tanh(osc_val * drive_gain)
            osc_output_block[i] = osc_val
        self.carrier_phase = carr_phase; self.modulator_phase = mod_phase
        self.envelope_level = env_level; self.time_in_stage = time_in_st
        avg_env_level = np.mean(env_block); filter_mod = avg_env_level * p['filter_env_amount']
        if lfo_target == 'filter': filter_mod += lfo_val_filter * p['lfo_depth']
        dynamic_cutoff = p['filter_cutoff'] + filter_mod
        self._update_filter_coeffs(dynamic_cutoff)
        filtered_block = osc_output_block
        if self.filter_b is not None and self.filter_a is not None and osc_type != 'fm':
            expected_zi_len = max(len(self.filter_a), len(self.filter_b)) - 1
            if expected_zi_len <= 0: self.filter_zi = None
            elif self.filter_zi is None or len(self.filter_zi) != expected_zi_len:
                self.filter_zi = np.zeros(expected_zi_len, dtype=np.float64)
            if self.filter_zi is not None:
                if self.filter_zi.dtype != np.float64: self.filter_zi = self.filter_zi.astype(np.float64)
                osc_output_block_64 = osc_output_block.astype(np.float64)
                try:
                    filtered_block_64, self.filter_zi = lfilter(self.filter_b, self.filter_a, osc_output_block_64, zi=self.filter_zi)
                    filtered_block = filtered_block_64.astype(np.float32)
                except ValueError as e:
                     print(f"ERROR lfilter: {e}", file=sys.stderr); traceback.print_exc(file=sys.stderr)
                     filtered_block = osc_output_block
                     self.filter_zi = np.zeros(expected_zi_len, dtype=np.float64)
        amp_mod = 1.0
        if lfo_target == 'amp' and p['lfo_depth'] > 0:
            amp_lfo_norm = (lfo_val_amp + 1.0) / 2.0
            amp_mod = (1.0 - p['lfo_depth']) + (p['lfo_depth'] * amp_lfo_norm)
            amp_mod = np.clip(amp_mod, 0.0, 1.0)
        output = filtered_block * env_block * self.velocity * amp_mod * p['volume']
        if self.envelope_stage == 'off' and len(output) > 0:
            off_indices = np.where(env_block <= 1e-6)[0]
            if len(off_indices) > 0: output[off_indices[0]:] = 0.0
        return output

    def note_off(self):
        if self.envelope_stage != 'off' and self.envelope_stage != 'release':
            self.release_start_level = self.envelope_level
            self.envelope_stage = 'release'; self.time_in_stage = 0

    def is_active(self): return self.envelope_stage != 'off'


# --- Synthesizer Class (Modified for Arp) ---
class Neptune1:
    def __init__(self, sample_rate=44100, block_size=512, max_voices=12):
        self.sample_rate = sample_rate
        self.block_size = block_size
        self.max_voices = max_voices
        self.params = DEFAULT_PARAMS.copy()
        self.voices = []
        self.lock = threading.Lock()
        self.preset_names = list(PRESETS.keys())
        self.current_preset_index = 0
        self.current_preset_name = "Custom"
        self.current_preset_name = self._find_preset_name(self.params)
        self.ui_update_callback = None
        self.chorus_lfo_phase = 0.0
        max_chorus_delay_sec = 0.05
        self.chorus_delay_line = np.zeros(int(sample_rate * max_chorus_delay_sec * 1.2), dtype=np.float32)
        self.chorus_delay_ptr = 0
        self.stream = None

        # --- Arpeggiator State ---
        self.held_notes = [] # List of currently physically held MIDI note numbers
        self.held_note_velocities = {} # Store velocity per note
        self._arp_thread = None
        self._arp_thread_running = False
        self._arp_step = 0
        self._arp_last_played_note = None # The actual MIDI note number played by arp
        self._arp_direction_updown = 1 # For UpDown pattern (1=up, -1=down)

        try:
            self.stream = sd.OutputStream(
                samplerate=sample_rate, blocksize=block_size, channels=1,
                callback=self._audio_callback, dtype='float32'
            )
            self.stream.start()
            print("Audio stream started.")
        except Exception as e:
            print(f"FATAL ERROR initializing audio stream: {e}", file=sys.stderr)
            raise

    # --- Methods unchanged: set_ui_update_callback, _trigger_ui_update, _find_preset_name ---
    def set_ui_update_callback(self, callback): self.ui_update_callback = callback
    def _trigger_ui_update(self):
        if self.ui_update_callback:
            try: self.ui_update_callback()
            except Exception as e: print(f"Error UI callback: {e}", file=sys.stderr); traceback.print_exc(file=sys.stderr)
    def _find_preset_name(self, params_to_find):
        for i, name in enumerate(self.preset_names):
            preset_params = PRESETS[name]; match = True
            for key in preset_params:
                 if key not in params_to_find or params_to_find[key] != preset_params[key]: match = False; break
            if not match: continue
            full_preset_params = DEFAULT_PARAMS.copy(); full_preset_params.update(preset_params)
            for key in DEFAULT_PARAMS:
                 # Include check for arp defaults if needed
                 if key not in params_to_find or params_to_find[key] != full_preset_params[key]: match = False; break
            if match: self.current_preset_index = i; return name
        if self.current_preset_name != "Custom":
            try: self.current_preset_index = self.preset_names.index(self.current_preset_name)
            except ValueError: self.current_preset_index = 0
        else: self.current_preset_index = 0
        return "Custom"


    # --- Modified set_params to handle arp on/off ---
    def set_params(self, params_dict, source="unknown"):
        arp_state_changed = False
        old_arp_on = self.params.get('arp_on', False)

        with self.lock:
            changed = False
            for key, value in params_dict.items():
                if key in self.params:
                    try:
                        current_value = self.params[key]
                        new_value = value # Assign first

                        # Attempt type conversion based on default type
                        default_type = type(DEFAULT_PARAMS.get(key, ''))
                        current_type = type(current_value)
                        if default_type == bool and not isinstance(new_value, bool):
                            new_value = str(value).lower() in ('true', '1', 't', 'yes', 'on')
                        elif default_type == float and not isinstance(new_value, float):
                            new_value = float(value)
                        elif default_type == int and not isinstance(new_value, int):
                             # Handle arp_rate/octaves carefully
                             if key in ['arp_rate', 'arp_octaves']:
                                 new_value = int(float(value)) # Allow float input from scale but store as int
                             else:
                                 new_value = int(value)
                        elif default_type == str and not isinstance(new_value, str):
                             new_value = str(value)

                        # Check if value actually changed
                        if new_value != current_value:
                            self.params[key] = new_value
                            changed = True
                            if key == 'arp_on':
                                arp_state_changed = True
                    except Exception as e:
                        print(f"Warn: Set param '{key}'='{value}' fail: {e}", file=sys.stderr)

            if changed:
                current_params_copy = self.params.copy()
                for voice in self.voices:
                    voice.params = current_params_copy # Update voices
                self.current_preset_name = self._find_preset_name(self.params)
                self._trigger_ui_update() # Update UI

        # --- Start/Stop Arp Thread outside lock if state changed ---
        if arp_state_changed:
            new_arp_on = self.params.get('arp_on', False)
            if new_arp_on and not old_arp_on:
                self._start_arp_thread()
            elif not new_arp_on and old_arp_on:
                self._stop_arp_thread()

    # --- Methods unchanged: load_preset, _apply_master_chorus, _audio_callback ---
    def load_preset(self, preset_name_or_index):
        preset_to_load = None; preset_name = "Unknown"; target_index = -1
        if isinstance(preset_name_or_index, int):
            if 0 <= preset_name_or_index < len(self.preset_names):
                target_index = preset_name_or_index; preset_name = self.preset_names[target_index]; preset_to_load = PRESETS[preset_name]
            else: print(f"Warn: Preset index {preset_name_or_index} OOR.", file=sys.stderr); return False
        elif isinstance(preset_name_or_index, str):
            if preset_name_or_index in PRESETS:
                 preset_name = preset_name_or_index; preset_to_load = PRESETS[preset_name]
                 try: target_index = self.preset_names.index(preset_name)
                 except ValueError: target_index = -1
            else: print(f"Warn: Preset '{preset_name_or_index}' not found.", file=sys.stderr); return False
        else: print("Warn: Invalid preset identifier.", file=sys.stderr); return False
        if preset_to_load:
            print(f"\nLoading preset {target_index}: {preset_name}")
            new_params = DEFAULT_PARAMS.copy(); new_params.update(preset_to_load)
            self.set_params(new_params, source="load_preset") # This handles UI update and arp thread now
            self.current_preset_index = target_index; self.current_preset_name = preset_name
            self._trigger_ui_update(); return True # Ensure final name/index is set
        return False

    def _apply_master_chorus(self, signal):
        p = self.params; sr = self.sample_rate; chorus_depth = p.get('chorus_depth', 0.0); chorus_rate = p.get('chorus_rate', 0.0)
        if chorus_depth <= 0 or chorus_rate <= 0: return signal
        lfo_freq_rad = 2 * np.pi * chorus_rate / sr; num_samples = len(signal); dl_len = len(self.chorus_delay_line)
        t = np.arange(self.chorus_lfo_phase, self.chorus_lfo_phase + num_samples * lfo_freq_rad, lfo_freq_rad)[:num_samples]
        chorus_lfo_signal = np.sin(t); self.chorus_lfo_phase = (t[-1] + lfo_freq_rad) % (2*np.pi)
        base_delay_sec = 0.015; delay_sec = base_delay_sec + chorus_lfo_signal * chorus_depth / 2.0
        min_delay_sec = 0.001; max_delay_sec = (dl_len / sr) * 0.95; delay_sec = np.clip(delay_sec, min_delay_sec, max_delay_sec)
        delay_samples = delay_sec * sr; delayed_output = np.zeros_like(signal); write_ptr = self.chorus_delay_ptr
        for i in range(num_samples):
            self.chorus_delay_line[write_ptr] = signal[i]
            read_ptr_float = (write_ptr - delay_samples[i] + dl_len) % dl_len; read_ptr_int = int(np.floor(read_ptr_float))
            read_ptr_frac = read_ptr_float - read_ptr_int; read_ptr_next = (read_ptr_int + 1) % dl_len
            delayed_sample = (1.0 - read_ptr_frac) * self.chorus_delay_line[read_ptr_int] + read_ptr_frac * self.chorus_delay_line[read_ptr_next]
            delayed_output[i] = delayed_sample; write_ptr = (write_ptr + 1) % dl_len
        self.chorus_delay_ptr = write_ptr; return 0.6 * signal + 0.4 * delayed_output

    def _audio_callback(self, outdata, frames, time_info, status):
        if status: print("Audio CB status:", status, file=sys.stderr)
        buffer = np.zeros((frames, 1), dtype=np.float32)
        with self.lock:
            active_voices_next = []; voices_to_process = self.voices[:]
            if not voices_to_process: outdata[:] = buffer; return
            for voice in voices_to_process:
                if voice.is_active():
                    try:
                        voice_output = voice.process(frames); buffer[:, 0] += voice_output
                        if voice.is_active(): active_voices_next.append(voice)
                    except Exception as e: print(f"ERR voice {voice.note}: {e}", file=sys.stderr); traceback.print_exc(file=sys.stderr)
            self.voices = active_voices_next
        try: buffer[:, 0] = self._apply_master_chorus(buffer[:, 0])
        except Exception as e: print(f"ERR chorus: {e}", file=sys.stderr); traceback.print_exc(file=sys.stderr)
        max_amp = np.max(np.abs(buffer));
        if max_amp > 1.0: np.clip(buffer, -1.0, 1.0, out=buffer)
        outdata[:] = buffer

    # --- Note On/Off (Modified for Arp) ---
    def note_on(self, note, velocity):
        """Handles MIDI Note On messages, considering the arpeggiator state."""
        with self.lock:
            arp_on = self.params.get('arp_on', False)
            if arp_on:
                # Arp is on: Add note to held list if not already present
                if note not in self.held_notes:
                    self.held_notes.append(note)
                    self.held_notes.sort() # Keep sorted for patterns
                    self.held_note_velocities[note] = velocity
                    # print(f"Arp Held Notes: {self.held_notes}") # Debug
            else:
                # Arp is off: Trigger voice directly
                self._trigger_voice_on(note, velocity)

    def note_off(self, note):
        """Handles MIDI Note Off messages, considering the arpeggiator state."""
        with self.lock:
            arp_on = self.params.get('arp_on', False)
            if arp_on:
                # Arp is on: Remove note from held list
                if note in self.held_notes:
                    self.held_notes.remove(note)
                    # No need to sort again after removal
                    if note in self.held_note_velocities:
                        del self.held_note_velocities[note]
                    # print(f"Arp Held Notes: {self.held_notes}") # Debug
                # If no notes are held anymore, stop the last arp note
                if not self.held_notes and self._arp_last_played_note is not None:
                    self._arp_stop_note(self._arp_last_played_note)
                    self._arp_last_played_note = None
                    self._arp_step = 0 # Reset step
                    self._arp_direction_updown = 1 # Reset direction
            else:
                # Arp is off: Trigger voice off directly
                self._trigger_voice_off(note)

    def _trigger_voice_on(self, note, velocity):
        """Internal method to actually start a synth voice (used by note_on and arp)."""
        # Assumes lock is already held
        if len(self.voices) >= self.max_voices:
            self.voices.pop(0)
        try:
            self.voices.append(Voice(note, velocity, self.params.copy(), self.sample_rate))
        except Exception as e:
            print(f"ERR voice note {note}: {e}", file=sys.stderr)
            traceback.print_exc(file=sys.stderr)

    def _trigger_voice_off(self, note):
        """Internal method to actually stop a synth voice (used by note_off and arp)."""
        # Assumes lock is already held
        for voice in self.voices:
            if voice.note == note and voice.envelope_stage not in ['release', 'off']:
                voice.note_off()
                break # Stop first matching voice

    # --- Methods unchanged: control_change, close ---
    def control_change(self, control, value):
        new_params = {}
        if control == 71: new_params['filter_resonance'] = np.clip((value / 127.0) * 0.95, 0.0, 0.95)
        elif control == 1:
             target = self.params.get('lfo_target', 'pitch'); max_depth = 0.0
             if target == 'pitch': max_depth = 12.0
             elif target == 'filter': max_depth = 6000.0
             elif target == 'amp': max_depth = 1.0
             elif target == 'fm_depth': max_depth = 2.0
             new_params['lfo_depth'] = (value / 127.0) * max_depth
        elif control == 7: new_params['volume'] = (value / 127.0) * 0.8
        elif control == 75: min_r, max_r = 0.1, 16.0; new_params['fm_mod_freq_ratio'] = min_r * (max_r / min_r)**(value / 127.0)
        elif control == 76: new_params['fm_mod_depth'] = (value / 127.0) * 2500.0
        elif control == 73: min_a, max_a = 0.001, 4.0; new_params['attack'] = min_a * (max_a / min_a)**(value / 127.0)
        elif control == 72: min_rl, max_rl = 0.01, 6.0; new_params['release'] = min_rl * (max_rl / min_rl)**(value / 127.0)
        # Add CCs for Arp params? e.g., CC 80 for Arp Rate, CC 81 for Arp Pattern
        # elif control == 80: new_params['arp_rate'] = [4, 8, 16, 32][value // 32] # Map 0-127 to 4 rates
        # elif control == 81: new_params['arp_pattern'] = ['Up', 'Down', 'UpDown', 'Random'][value // 32]
        if new_params: self.set_params(new_params, source="midi_cc")

    def close(self):
        print("Stopping synth...")
        self._stop_arp_thread() # Ensure arp thread is stopped first
        print("Stopping audio stream...")
        if self.stream:
            try: self.stream.stop(); self.stream.close()
            except Exception as e: print(f"Error closing audio stream: {e}", file=sys.stderr)
        self.stream = None; print("Audio stream stopped.")

# --- Arpeggiator Logic ---
    def _start_arp_thread(self):
        # (Keep this method the same)
        if self._arp_thread is not None and self._arp_thread.is_alive():
            print("Arp thread already running.")
            return
        self._arp_thread_running = True
        self._arp_step = 0 # Reset step counter
        self._arp_direction_updown = 1 # Reset direction for UpDown
        self._arp_last_played_note = None
        self._arp_thread = threading.Thread(target=self._arp_sequencer_loop, daemon=True)
        self._arp_thread.start()
        print("Arpeggiator thread started.")

    def _stop_arp_thread(self):
        # (Keep this method the same)
        if self._arp_thread is None or not self._arp_thread.is_alive():
            return
        self._arp_thread_running = False
        print("Waiting for arpeggiator thread to finish...")
        self._arp_thread.join(timeout=1.0)
        if self._arp_thread.is_alive():
            print("Warning: Arp thread did not exit cleanly.", file=sys.stderr)
        else:
            print("Arpeggiator thread stopped.")
        self._arp_thread = None
        with self.lock:
            if self._arp_last_played_note is not None:
                self._arp_stop_note(self._arp_last_played_note)
                self._arp_last_played_note = None

    def _arp_sequencer_loop(self):
        """The main loop for the arpeggiator thread."""
        while self._arp_thread_running:
            try:
                # Get current arp settings (copy to avoid race conditions)
                with self.lock:
                    arp_on = self.params.get('arp_on', False)
                    bpm = self.params.get('arp_bpm', 120.0)
                    rate = self.params.get('arp_rate', 16)
                    pattern = self.params.get('arp_pattern', 'Up')
                    octaves = self.params.get('arp_octaves', 1)
                    # Get sorted copy of physically held notes
                    local_held_notes = sorted(self.held_notes)

                if not arp_on or not local_held_notes:
                    # If arp turned off or no notes held, sleep briefly and check again
                    # Also, ensure any last playing arp note is stopped if notes released
                    if not local_held_notes and self._arp_last_played_note is not None:
                         with self.lock:
                             self._arp_stop_note(self._arp_last_played_note)
                    time.sleep(0.05) # Shorter sleep when inactive
                    continue

                # Calculate step duration
                if bpm <= 0 or rate <= 0:
                    time.sleep(0.1) # Avoid division by zero or invalid rate
                    continue
                steps_per_beat = rate / 4.0 # e.g., rate 16 -> 4 steps per beat
                beats_per_second = bpm / 60.0
                steps_per_second = steps_per_beat * beats_per_second
                step_duration_seconds = 1.0 / steps_per_second

                # --- Determine the sequence of notes to play ---
                note_sequence = []
                num_held = len(local_held_notes)

                # *** MODIFICATION START: Handle single note case ***
                if num_held == 1:
                    base_note = local_held_notes[0]
                    for o in range(octaves):
                        note_sequence.append(base_note + o * 12)
                    # Pattern is irrelevant for single note, just cycle through octaves
                    pattern = 'Up' # Force 'Up' logic for cycling
                # *** MODIFICATION END ***
                else: # Original logic for multiple notes
                    for o in range(octaves):
                        for note in local_held_notes:
                            note_sequence.append(note + o * 12)

                if not note_sequence:
                     time.sleep(0.05); continue # Should not happen if local_held_notes has items

                # --- Apply Pattern to select note ---
                current_note_to_play = None
                seq_len = len(note_sequence)

                # Ensure step is valid even if note sequence length changed drastically
                current_step = self._arp_step

                if pattern == 'Up':
                    step_index = current_step % seq_len
                    current_note_to_play = note_sequence[step_index]
                elif pattern == 'Down':
                    step_index = current_step % seq_len
                    current_note_to_play = note_sequence[seq_len - 1 - step_index]
                elif pattern == 'UpDown':
                     # Total steps in one full cycle (e.g., 0,1,2,1 for seq_len=3 -> 4 steps)
                    effective_len = max(1, seq_len * 2 - 2) if seq_len > 1 else 1
                    step_index = current_step % effective_len
                    if step_index < seq_len: # Going up
                        current_note_to_play = note_sequence[step_index]
                    else: # Going down (excluding endpoints repeated)
                         # Index calculation: effective_len - step_index gives index from the end *backwards*
                         # e.g. len=3 (notes 0,1,2), eff_len=4. steps 0,1,2,3
                         # step 3: 4 - 3 = 1 -> index 1 (note 1)
                         current_note_to_play = note_sequence[effective_len - step_index]

                elif pattern == 'Random':
                    current_note_to_play = random.choice(note_sequence)
                else: # Default to 'Up' if pattern is unknown
                    step_index = current_step % seq_len
                    current_note_to_play = note_sequence[step_index]

                # --- Play the note ---
                if current_note_to_play is not None:
                    # Find original velocity if possible
                    original_base_note = local_held_notes[0] # Use first held note's octave base
                    note_in_base_octave = current_note_to_play % 12 + (original_base_note // 12) * 12
                    # Try to find the velocity for the specific note in its original octave among held notes
                    velocity = self.held_note_velocities.get(note_in_base_octave, 100) # Default velocity 100

                    with self.lock: # Lock needed for triggering voices
                         self._arp_play_note(current_note_to_play, velocity)
                    self._arp_step += 1 # Increment step only after successfully playing

                # --- Sleep for the step duration ---
                sleep_time = step_duration_seconds
                time.sleep(sleep_time)

            except Exception as e:
                print(f"Error in arp sequencer loop: {e}", file=sys.stderr)
                traceback.print_exc(file=sys.stderr)
                time.sleep(0.5) # Avoid busy-looping on error

    def _arp_play_note(self, note, velocity):
        """Internal: Stops previous arp note and plays the new one.
           Corrected to always stop/start for retriggering.
        """
        # Assumes lock is held

        # --- FIX: Always stop the previously played arp note ---
        # Stop whatever note the arp played last, regardless of the new note number.
        if self._arp_last_played_note is not None:
            self._arp_stop_note(self._arp_last_played_note)

        # --- Always Trigger the new note ON ---
        # Since we stopped the previous one, we always want to start the new one.
        # The _trigger_voice_on handles voice limits etc.
        self._trigger_voice_on(note, velocity)


        # --- Update the last played note ---
        self._arp_last_played_note = note # Remember this note was the last one played by arp


    def _arp_stop_note(self, note):
        # (Keep this method the same)
        # Assumes lock is held
        self._trigger_voice_off(note)
        # Only clear last played note if it's the one we just stopped
        if self._arp_last_played_note == note:
             self._arp_last_played_note = None

# (The rest of the code, including Neptune1's other methods,
#  PARAM_CONFIG, helper functions, SynthUI class, midi_listener,
#  and the main execution block, remains exactly the same as before)
# --- Parameter Definitions for UI (Added Arp Params) ---
PARAM_CONFIG = {
    # Synth Core
    'Preset':       {'type': 'preset'},
    'osc_type':     {'type': 'options', 'options': ['saw', 'square', 'sine', 'triangle', 'fm']},
    'attack':       {'type': 'log', 'min': 0.001, 'max': 5.0},
    'decay':        {'type': 'log', 'min': 0.01, 'max': 5.0},
    'sustain':      {'type': 'lin', 'min': 0.0, 'max': 1.0},
    'release':      {'type': 'log', 'min': 0.01, 'max': 8.0},
    'filter_cutoff':{'type': 'log', 'min': 20.0, 'max': 18000.0},
    'filter_resonance': {'type': 'lin', 'min': 0.0, 'max': 0.95},
    'filter_env_amount': {'type': 'lin', 'min': -8000.0, 'max': 8000.0},
    'lfo_rate':     {'type': 'log', 'min': 0.01, 'max': 30.0},
    'lfo_depth':    {'type': 'lin', 'min': 0.0, 'max': 1.0},
    'lfo_target':   {'type': 'options', 'options': ['pitch', 'filter', 'amp', 'fm_depth']},
    'lfo_shape':    {'type': 'options', 'options': ['sine', 'triangle', 'saw', 'square']},
    'volume':       {'type': 'lin', 'min': 0.0, 'max': 1.0},
    'analog_drive': {'type': 'lin', 'min': 1.0, 'max': 5.0},
    'analog_drift': {'type': 'lin', 'min': 0.0, 'max': 0.005},
    # FM
    'fm_mod_freq_ratio': {'type': 'log', 'min': 0.1, 'max': 16.0},
    'fm_mod_depth': {'type': 'log', 'min': 1.0, 'max': 5000.0},
    # Chorus FX
    'chorus_depth': {'type': 'lin', 'min': 0.0, 'max': 0.02},
    'chorus_rate':  {'type': 'lin', 'min': 0.0, 'max': 5.0},
    # --- Arpeggiator ---
    'arp_on':       {'type': 'check'}, # Checkbutton
    'arp_bpm':      {'type': 'lin', 'min': 30.0, 'max': 240.0},
    'arp_rate':     {'type': 'options', 'options': [4, 8, 16, 32]}, # Representing 1/4, 1/8, 1/16, 1/32
    'arp_pattern':  {'type': 'options', 'options': ['Up', 'Down', 'UpDown', 'Random']},
    'arp_octaves':  {'type': 'options', 'options': [1, 2, 3, 4]}, # Could be lin slider 1-4 if preferred
}

# --- Helper Scaling Functions (remain the same) ---
def log_scale(value, min_val, max_val, scale_min=0, scale_max=100):
    if value <= scale_min: return min_val
    min_val_log = max(min_val, 1e-9); log_min = np.log(min_val_log); log_max = np.log(max_val)
    scale = (log_max - log_min) / (scale_max - scale_min)
    if scale == 0: return min_val # Avoid issues if min=max
    return np.exp(log_min + scale * (value - scale_min))
def inverse_log_scale(param_value, min_val, max_val, scale_min=0, scale_max=100):
    param_value = np.clip(param_value, min_val, max_val); min_val_log = max(min_val, 1e-9)
    log_min = np.log(min_val_log); log_max = np.log(max_val)
    scale = (log_max - log_min) / (scale_max - scale_min)
    if scale == 0: return scale_min
    param_value_log = max(param_value, 1e-9); val = scale_min + (np.log(param_value_log) - log_min) / scale
    return np.clip(val, scale_min, scale_max)


# --- Tkinter UI Class (Modified for Arp) ---
class SynthUI:
    def __init__(self, root, synth_instance):
        self.root = root; self.synth = synth_instance; self.controls = {}
        self.string_vars = {}; self.scale_vars = {}; self.check_vars = {} # Added check_vars
        self._update_lock = False
        self.root.title("Neptune 1"); self.root.protocol("WM_DELETE_WINDOW", self._on_closing)
        self.synth.set_ui_update_callback(self.schedule_ui_update)
        style = ttk.Style(); style.theme_use('clam') # Or 'alt', 'default', 'classic'
        # Styles (optional, adjust as needed)
        style.configure('TFrame', background='#EEE')
        style.configure('TLabel', background='#EEE')
        style.configure('TLabelFrame', background='#EEE')
        style.configure('TLabelFrame.Label', background='#EEE')
        style.configure('TScale', background='#EEE')
        style.configure('TCheckbutton', background='#EEE')

        main_frame = ttk.Frame(root, padding="10")
        main_frame.grid(row=0, column=0, sticky="nsew")
        root.grid_rowconfigure(0, weight=1); root.grid_columnconfigure(0, weight=1)

        # Frames (Add Arp Frame)
        top_frame = ttk.Frame(main_frame)
        osc_frame = ttk.LabelFrame(main_frame, text="Oscillator", padding="5")
        fm_frame = ttk.LabelFrame(main_frame, text="FM", padding="5")
        filter_frame = ttk.LabelFrame(main_frame, text="Filter", padding="5")
        env_frame = ttk.LabelFrame(main_frame, text="Envelope", padding="5")
        lfo_frame = ttk.LabelFrame(main_frame, text="LFO", padding="5")
        master_frame = ttk.LabelFrame(main_frame, text="Master/FX", padding="5")
        arp_frame = ttk.LabelFrame(main_frame, text="Arpeggiator", padding="5") # New Arp Frame

        # Layout frames
        top_frame.grid(row=0, column=0, columnspan=3, sticky="ew", pady=5)
        osc_frame.grid(row=1, column=0, padx=5, pady=5, sticky="ns")
        fm_frame.grid(row=2, column=0, padx=5, pady=5, sticky="nsew")
        filter_frame.grid(row=1, column=1, rowspan=2, padx=5, pady=5, sticky="ns")
        env_frame.grid(row=1, column=2, rowspan=2, padx=5, pady=5, sticky="ns")
        # Adjust LFO and Master/FX positions
        lfo_frame.grid(row=3, column=0, padx=5, pady=5, sticky="nsew") # LFO below OSC/FM
        master_frame.grid(row=3, column=1, padx=5, pady=5, sticky="nsew") # Master below Filter
        arp_frame.grid(row=3, column=2, padx=5, pady=5, sticky="nsew") # Arp below Env

        # Populate Top Frame (Preset)
        ttk.Label(top_frame, text="Preset:").grid(row=0, column=0, padx=5, sticky="w")
        preset_var = tk.StringVar(); self.string_vars['Preset'] = preset_var
        preset_combo = ttk.Combobox(top_frame, textvariable=preset_var, values=self.synth.preset_names, state='readonly', width=25)
        preset_combo.grid(row=0, column=1, padx=5, sticky="ew"); preset_combo.bind('<<ComboboxSelected>>', self._on_preset_change)
        self.controls['Preset'] = preset_combo; top_frame.grid_columnconfigure(1, weight=1)

        # Populate Sections
        self._create_controls(osc_frame, ['osc_type', 'analog_drive', 'analog_drift'])
        self._create_controls(fm_frame, ['fm_mod_freq_ratio', 'fm_mod_depth'])
        self._create_controls(filter_frame, ['filter_cutoff', 'filter_resonance', 'filter_env_amount'])
        self._create_controls(env_frame, ['attack', 'decay', 'sustain', 'release'])
        self._create_controls(lfo_frame, ['lfo_rate', 'lfo_depth', 'lfo_target', 'lfo_shape'])
        self._create_controls(master_frame, ['volume', 'chorus_depth', 'chorus_rate'])
        # Populate Arp section
        self._create_controls(arp_frame, ['arp_on', 'arp_bpm', 'arp_rate', 'arp_pattern', 'arp_octaves'])

        self.update_controls_from_synth() # Initial update

    # Modify _create_controls to handle 'check' type
    def _create_controls(self, parent_frame, param_names):
        for i, name in enumerate(param_names):
            config = PARAM_CONFIG.get(name);
            if not config: continue
            label = ttk.Label(parent_frame, text=f"{name.replace('_', ' ').title()}:")
            label.grid(row=i, column=0, padx=5, pady=2, sticky="w")
            control_type = config['type']

            if control_type == 'options':
                var = tk.StringVar(); combo = ttk.Combobox(parent_frame, textvariable=var, values=config['options'], state='readonly', width=12)
                combo.grid(row=i, column=1, padx=5, pady=2, sticky="ew"); combo.bind('<<ComboboxSelected>>', lambda e, p=name: self._on_value_change(p))
                self.controls[name] = combo; self.string_vars[name] = var
            elif control_type in ['lin', 'log']:
                var = tk.DoubleVar(); scale = ttk.Scale(parent_frame, from_=0, to=100, orient='horizontal', variable=var, length=150, command=lambda val, p=name: self._on_value_change(p))
                scale.grid(row=i, column=1, padx=5, pady=2, sticky="ew"); self.controls[name] = scale; self.scale_vars[name] = var
            elif control_type == 'check': # Handle Checkbutton
                var = tk.BooleanVar()
                # Place checkbox directly, no separate label needed if text is descriptive
                # Or use grid row=i, column=1 if keeping separate label
                check = ttk.Checkbutton(parent_frame, variable=var, command=lambda p=name: self._on_value_change(p))
                check.grid(row=i, column=1, padx=5, pady=2, sticky="w") # Align left
                self.controls[name] = check
                self.check_vars[name] = var # Store boolean var

            parent_frame.grid_columnconfigure(1, weight=1) # Allow control to expand

    # Modify _on_value_change to handle checkbuttons
    def _on_value_change(self, param_name):
        if self._update_lock or param_name not in self.controls: return
        widget = self.controls[param_name]; config = PARAM_CONFIG.get(param_name); new_value = None
        try:
            if isinstance(widget, ttk.Combobox):
                val_str = self.string_vars[param_name].get()
                # Try converting options like rate/octaves back to numbers
                try: new_value = int(val_str)
                except ValueError:
                     try: new_value = float(val_str)
                     except ValueError: new_value = val_str # Keep as string if not number
            elif isinstance(widget, ttk.Scale):
                scaled_val = self.scale_vars[param_name].get()
                if config['type'] == 'log': new_value = log_scale(scaled_val, config['min'], config['max'])
                else: new_value = config['min'] + (config['max'] - config['min']) * (scaled_val / 100.0)
            elif isinstance(widget, ttk.Checkbutton): # Handle Checkbutton
                 new_value = self.check_vars[param_name].get() # Get boolean value

            if new_value is not None: self.synth.set_params({param_name: new_value}, source="ui")
        except Exception as e: print(f"ERR UI change {param_name}: {e}", file=sys.stderr)

    # Modify update_controls_from_synth for checkbuttons
    def update_controls_from_synth(self):
        if self._update_lock: return
        self._update_lock = True
        try:
            current_params = self.synth.params; preset_name = self.synth.current_preset_name
            if self.string_vars['Preset'].get() != preset_name: self.string_vars['Preset'].set(preset_name)

            for name, widget in self.controls.items():
                if name == 'Preset' or name not in current_params: continue
                config = PARAM_CONFIG.get(name); current_val = current_params[name]
                try:
                    if isinstance(widget, ttk.Combobox):
                        # For options that are numbers (rate, octaves), ensure string conversion
                        if self.string_vars[name].get() != str(current_val):
                            self.string_vars[name].set(str(current_val))
                    elif isinstance(widget, ttk.Scale):
                        scaled_val = 0.0
                        if config['type'] == 'log': scaled_val = inverse_log_scale(current_val, config['min'], config['max'])
                        else: range_ = config['max'] - config['min']; scaled_val = ((current_val - config['min']) / range_) * 100.0 if range_ != 0 else 0.0
                        if abs(self.scale_vars[name].get() - scaled_val) > 0.1: self.scale_vars[name].set(scaled_val)
                    elif isinstance(widget, ttk.Checkbutton): # Handle checkbutton update
                        if self.check_vars[name].get() != bool(current_val):
                            self.check_vars[name].set(bool(current_val))

                except Exception as e: print(f"ERR UI update {name}: {e}", file=sys.stderr)
        finally: self._update_lock = False

    # Unchanged methods: _on_preset_change, schedule_ui_update, _on_closing
    def _on_preset_change(self, event=None):
        if self._update_lock: return
        selected_preset = self.string_vars['Preset'].get()
        self.synth.load_preset(selected_preset)
    def schedule_ui_update(self): self.root.after(10, self.update_controls_from_synth)
    def _on_closing(self):
        print("UI closing...")
        global midi_thread_running, midi_port, arp_thread_running # Make sure arp flag is global if used here
        midi_thread_running = False # Signal MIDI thread
        if self.synth: self.synth._stop_arp_thread() # Stop arp thread via synth method
        if midi_port and not midi_port.closed:
            print("Closing MIDI port...")
            try: midi_port.close()
            except Exception as e: print(f"Error closing MIDI port: {e}", file=sys.stderr)
        if self.synth: self.synth.close() # Close audio stream
        self.root.destroy()
        print("Cleanup complete.")


# --- MIDI Input Thread (remain the same) ---
# --- MIDI Input Thread (Corrected Syntax Errors) ---
midi_thread_running = True
midi_port = None

def midi_listener(synth_instance, port_name):
    """Listens for MIDI messages and calls synth methods (uses blocking iterator)."""
    global midi_thread_running, midi_port
    print(f"MIDI thread started for port '{port_name}'.")

    while midi_thread_running:
        try:
            if midi_port is None or midi_port.closed:
                 if not midi_thread_running: break # Exit if flag already false
                 print(f"Attempting to open MIDI port '{port_name}'...")
                 midi_port = mido.open_input(port_name) # Assign to global var
                 print(f"Listening on '{port_name}'...")

            # This loop blocks until a message arrives or the port is closed/errors
            for msg in midi_port:
                if not midi_thread_running: break # Check flag before processing

                # Process the message
                if msg.type == 'note_on' and msg.velocity > 0:
                    synth_instance.note_on(msg.note, msg.velocity)
                elif msg.type == 'note_off' or (msg.type == 'note_on' and msg.velocity == 0):
                    synth_instance.note_off(msg.note)
                elif msg.type == 'control_change':
                    synth_instance.control_change(msg.control, msg.value)
                elif msg.type == 'program_change':
                    synth_instance.load_preset(msg.program)
                # Add other message types if needed

            # If the inner loop finished, check the flag again before deciding to reopen
            if not midi_thread_running:
                print("MIDI thread: Stop flag detected.")
                break # Exit the outer while loop

        except (OSError, IOError, mido.MidiError) as e:
            print(f"MIDI Port Error on '{port_name}': {e}. Retrying...", file=sys.stderr)
            if midi_port and not midi_port.closed:
                try: midi_port.close()
                except Exception: pass # Ignore errors during close in except block
            midi_port = None # Mark as closed
            # Wait before retrying, checking the flag periodically
            # --- FIX 1: Corrected loop syntax ---
            for _ in range(20): # Check every 100ms for 2 seconds
                 time.sleep(0.1)
                 if not midi_thread_running:
                     break # Exit wait loop if stopped
            if not midi_thread_running:
                break # Exit outer loop if stopped during wait

        except Exception as e: # Catch any other unexpected errors
            print(f"\nUnexpected error in MIDI listener thread: {e}", file=sys.stderr)
            traceback.print_exc(file=sys.stderr)
            if midi_port and not midi_port.closed:
                try: midi_port.close()
                except Exception: pass
            midi_port = None
            # Wait longer after an unknown error
            # --- FIX 2: Corrected loop syntax ---
            for _ in range(50): # Check every 100ms for 5 seconds
                 time.sleep(0.1)
                 if not midi_thread_running:
                     break # Exit wait loop if stopped
            if not midi_thread_running:
                break # Exit outer loop if stopped during wait

        # If we reach here, the port might have closed normally or errored.
        # The loop will continue if midi_thread_running is True, attempting to reopen.
        if midi_port and midi_port.closed:
            print(f"MIDI port '{port_name}' appears closed.")
            midi_port = None

    # Final cleanup when thread exits
    if midi_port and not midi_port.closed:
        print("Closing MIDI port from thread exit...")
        try: midi_port.close()
        except Exception: pass
    print("MIDI thread finished.")


# --- Main Execution (remain mostly the same, ensure presets are loaded) ---
if __name__ == "__main__":
    SAMPLE_RATE = 44100
    BLOCK_SIZE = 1024 # Keep increased block size
    MAX_VOICES = 16   # Max voices for synth engine (arp uses these)

    # --- Select MIDI Port ---
    root_check = tk.Tk(); root_check.withdraw()
    selected_port_name = None
    while selected_port_name is None:
        try:
            print("Scanning MIDI ports..."); available_ports = mido.get_input_names()
            if not available_ports: messagebox.showerror("MIDI Error", "No MIDI input devices found."); exit()
            port_list_str = "\n".join([f"{i}: {name}" for i, name in enumerate(available_ports)])
            result = available_ports[0]
            if result is None: print("Cancelled."); exit()
            try:
                port_index = int(result)
                if 0 <= port_index < len(available_ports): selected_port_name = available_ports[port_index]; print(f"Selected: '{selected_port_name}'")
                else: messagebox.showwarning("Invalid", f"Index {port_index} OOR.", parent=root_check)
            except ValueError: messagebox.showwarning("Invalid", f"'{result}' not number.", parent=root_check)
        except Exception as e: messagebox.showerror("MIDI Error", f"Scan error: {e}"); exit()
    root_check.destroy()

    # --- Initialize Synth ---
    synth = None
    try:
        print("Initializing synth..."); synth = Neptune1(sample_rate=SAMPLE_RATE, block_size=BLOCK_SIZE, max_voices=MAX_VOICES)
        if synth.stream: print(f"Actual audio settings: SR={synth.stream.samplerate:.0f}, Block={synth.stream.blocksize}, Latency={synth.stream.latency*1000:.1f}ms")
        else: print("Warning: Synth stream init failed.", file=sys.stderr)
    except Exception as e: messagebox.showerror("Synth Error", f"Synth init failed: {e}"); exit(1)

    # --- Initialize UI ---
    root = tk.Tk(); app = SynthUI(root, synth)

    # --- Load Initial Preset (ensure it exists) ---
    initial_preset_name = "Default" # Or choose another from your list
    if initial_preset_name not in PRESETS: initial_preset_name = list(PRESETS.keys())[0] # Fallback
    synth.load_preset(initial_preset_name)

    # --- Start MIDI Listener Thread ---
    midi_thread_running = True
    midi_thread = threading.Thread(target=midi_listener, args=(synth, selected_port_name), daemon=True)
    midi_thread.start()

    # --- Start Tkinter Main Loop ---
    print("Starting UI...")
    try: root.mainloop()
    except KeyboardInterrupt: print("Ctrl+C..."); app._on_closing()

    print("Application finished.")
