import numpy as np
import sounddevice as sd
from scipy.signal import butter, lfilter, lfilter_zi
import time
import math
import mido
import mido.backends.rtmidi
import threading
import sys
import traceback
import random
try:
    import RPi.GPIO as GPIO
    RPI_GPIO_AVAILABLE = True
except ImportError:
    RPI_GPIO_AVAILABLE = False
    print("RPi.GPIO library not found. Encoder functionality will be disabled.", file=sys.stderr)
except RuntimeError:
    RPI_GPIO_AVAILABLE = False
    print("Error importing RPi.GPIO (may need sudo or not on RPi). Encoder functionality disabled.", file=sys.stderr)


# --- Default Parameters ---
DEFAULT_PARAMS = {
    'osc_type': 'saw', 'attack': 0.01, 'decay': 0.2, 'sustain': 0.8, 'release': 0.3,
    'filter_cutoff': 5000, 'filter_resonance': 0.1, 'filter_env_amount': 0,
    'volume': 0.6, 'analog_drive': 1.0, 'analog_drift': 0.0000,
    'chorus_depth': 0.005, 'chorus_rate': 0.5,
    'arp_on': False, 'arp_bpm': 120.0, 'arp_rate': 16,
    'arp_pattern': 'Up', 'arp_octaves': 1,
}

# --- Presets (REPLACE WITH YOUR 150 PRESETS) ---
PRESETS = {
    "Default": DEFAULT_PARAMS.copy(),
    "Pad: MW Pad": { 'osc_type': 'saw', 'attack': 0.8, 'decay': 1.5, 'sustain': 0.6, 'release': 1.8, 'filter_cutoff': 3800, 'filter_resonance': 0.1, 'filter_env_amount': 800, 'volume': 0.5, 'analog_drive': 1.5, 'analog_drift':0.0005, 'chorus_depth': 0.008, 'chorus_rate': 0.6, 'arp_on': False, 'arp_bpm': 120.0, 'arp_rate': 16, 'arp_pattern': 'Up', 'arp_octaves': 1},
    # --- ADD YOUR OTHER PRESETS HERE ---
}

# --- Helper Functions ---
def midi_to_freq(note): return 432.0 * (2.0**((note - 69) / 12.0))

# --- Voice Class ---
class Voice:
    def __init__(self, note, velocity, params, sample_rate):
        self.note = note; self.velocity = velocity / 127.0; self.params = params.copy()
        self.sample_rate = sample_rate; self.freq = midi_to_freq(note)
        self.carrier_phase = 0.0; self.envelope_stage = 'attack'
        self.envelope_level = 0.0; self.time_in_stage = 0.0; self.release_start_level = 0.0
        self.filter_b = None; self.filter_a = None; self.filter_zi = None
        self._update_filter_coeffs(self.params.get('filter_cutoff', 5000))
        if self.filter_b is not None and self.filter_a is not None:
            try: self.filter_zi = lfilter_zi(self.filter_b, self.filter_a).astype(np.float64)
            except ValueError as e:
                filter_order = max(len(self.filter_a), len(self.filter_b)) - 1
                self.filter_zi = np.zeros(filter_order, dtype=np.float64) if filter_order > 0 else None
            except Exception as e: self.filter_zi = None; print(f"Voice init filter zi error: {e}", file=sys.stderr)
        self.drift_lfo_phase = np.random.rand() * 2 * np.pi

    def _update_filter_coeffs(self, cutoff_hz):
        sr = self.sample_rate; nyquist = sr / 2.0
        cutoff_hz = np.clip(cutoff_hz, 20, nyquist * 0.99); cutoff_norm = cutoff_hz / nyquist
        try: self.filter_b, self.filter_a = butter(2, cutoff_norm, btype='low', analog=False)
        except ValueError: self.filter_b, self.filter_a = butter(2, 0.98, btype='low', analog=False)
        except Exception: self.filter_b, self.filter_a = None, None

    def process(self, num_samples):
        p = self.params; sr = self.sample_rate
        drift_block = np.zeros(num_samples, dtype=np.float32)
        if p.get('analog_drift',0.0) > 0:
            drift_lfo_rate = 0.15; drift_freq_rad = 2 * np.pi * drift_lfo_rate / sr
            t_drift_phases = np.arange(self.drift_lfo_phase, self.drift_lfo_phase + num_samples * drift_freq_rad, drift_freq_rad,dtype=np.float32)[:num_samples]
            if len(t_drift_phases) == num_samples :
                drift_block = np.sin(t_drift_phases); self.drift_lfo_phase = (t_drift_phases[-1] + drift_freq_rad) % (2 * np.pi)
        current_drift_block = drift_block * p.get('analog_drift',0.0) * self.freq

        env_level = self.envelope_level; time_in_st = self.time_in_stage; carr_phase = self.carrier_phase
        attack_time = max(0.001, p['attack']); decay_time = max(0.001,p['decay']); release_time = max(0.001,p['release'])
        attack_samples = int(attack_time * sr); decay_samples = int(decay_time * sr)
        release_samples = int(release_time * sr); sustain_level = np.clip(p.get('sustain', 0.8), 0.0, 1.0)
        osc_phase_step_base = 2 * np.pi / sr; osc_type = p.get('osc_type', 'saw'); drive_gain = p.get('analog_drive', 1.0)
        osc_output_block = np.zeros(num_samples, dtype=np.float32); env_block = np.zeros(num_samples, dtype=np.float32)

        for i in range(num_samples):
            if self.envelope_stage == 'attack':
                env_level = min(1.0, time_in_st / attack_samples if attack_samples > 0 else 1.0)
                if time_in_st >= attack_samples: self.envelope_stage = 'decay'; env_level = 1.0; time_in_st = 0
            elif self.envelope_stage == 'decay':
                env_level = sustain_level + (1.0 - sustain_level) * max(0.0, 1.0 - (time_in_st / decay_samples if decay_samples > 0 else 1.0))
                if time_in_st >= decay_samples: self.envelope_stage = 'sustain'; env_level = sustain_level; time_in_st = 0
            elif self.envelope_stage == 'sustain': env_level = sustain_level
            elif self.envelope_stage == 'release':
                env_level = self.release_start_level * max(0.0, 1.0 - (time_in_st / release_samples if release_samples > 0 else 1.0))
                if time_in_st >= release_samples: self.envelope_stage = 'off'; env_level = 0.0
            elif self.envelope_stage == 'off': env_level = 0.0
            if env_level == 0 and self.envelope_stage == 'off':
                if i == 0: osc_output_block.fill(0.0); env_block.fill(0.0); break
                else: osc_output_block[i:] = 0.0; env_block[i:] = 0.0; break
            env_level = np.clip(env_level, 0.0, 1.0); env_block[i] = env_level
            if self.envelope_stage != 'sustain' and self.envelope_stage != 'off': time_in_st += 1

            current_drift = current_drift_block[i]; base_freq = max(1.0, self.freq + current_drift)
            osc_val = 0.0; carrier_phase_increment = base_freq * osc_phase_step_base; carr_phase = (carr_phase + carrier_phase_increment) % (2 * np.pi)
            if osc_type == 'saw': osc_val = (carr_phase / np.pi) - 1.0
            elif osc_type == 'square': osc_val = 1.0 if carr_phase < np.pi else -1.0
            elif osc_type == 'sine': osc_val = np.sin(carr_phase)
            elif osc_type == 'triangle': osc_val = 2.0 * (abs(carr_phase / np.pi - 1.0)) - 1.0
            if drive_gain > 1.0: osc_val = np.tanh(osc_val * drive_gain)
            osc_output_block[i] = osc_val

        self.carrier_phase = carr_phase; self.envelope_level = env_level; self.time_in_stage = time_in_st
        avg_env_level = np.mean(env_block); filter_mod = avg_env_level * p['filter_env_amount']
        dynamic_cutoff = p['filter_cutoff'] + filter_mod; self._update_filter_coeffs(dynamic_cutoff)
        filtered_block = osc_output_block
        if self.filter_b is not None and self.filter_a is not None:
            expected_zi_len = max(len(self.filter_a), len(self.filter_b)) - 1
            if expected_zi_len <= 0: self.filter_zi = None
            elif self.filter_zi is None or len(self.filter_zi) != expected_zi_len:
                 try: self.filter_zi = np.zeros(expected_zi_len, dtype=np.float64)
                 except Exception: self.filter_zi = None
            if self.filter_zi is not None:
                if self.filter_zi.dtype != np.float64: self.filter_zi = self.filter_zi.astype(np.float64)
                osc_output_block_64 = osc_output_block.astype(np.float64)
                try: filtered_block_64, self.filter_zi = lfilter(self.filter_b, self.filter_a, osc_output_block_64, zi=self.filter_zi); filtered_block = filtered_block_64.astype(np.float32)
                except ValueError: filtered_block = osc_output_block; self.filter_zi = np.zeros(expected_zi_len, dtype=np.float64) if expected_zi_len > 0 else None
                except Exception: filtered_block = osc_output_block; self.filter_zi = np.zeros(expected_zi_len, dtype=np.float64) if expected_zi_len > 0 else None
        output = filtered_block * env_block * self.velocity * p['volume']
        if self.envelope_stage == 'off' and len(output) > 0:
            off_indices = np.where(env_block <= 1e-6)[0]
            if len(off_indices) > 0: output[off_indices[0]:] = 0.0
        return output
    def note_off(self):
        if self.envelope_stage != 'off' and self.envelope_stage != 'release': self.release_start_level = self.envelope_level; self.envelope_stage = 'release'; self.time_in_stage = 0
    def is_active(self): return self.envelope_stage != 'off'

# --- Synthesizer Class ---
class PySynthJunoMIDI:
    def __init__(self, sample_rate=44100, block_size=1024, max_voices=8):
        self.sample_rate = sample_rate; self.block_size = block_size; self.max_voices = max_voices
        self.params = DEFAULT_PARAMS.copy(); self.voices = []
        self.lock = threading.Lock(); self.preset_names = list(PRESETS.keys())
        self.current_preset_index = 0; self.current_preset_name = self._find_preset_name(self.params)
        self.chorus_lfo_phase = 0.0; max_chorus_delay_sec = 0.05
        self.chorus_delay_line = np.zeros(int(sample_rate * max_chorus_delay_sec * 1.2), dtype=np.float32)
        self.chorus_delay_ptr = 0; self._chorus_on_depth = self.params.get('chorus_depth', 0.005)
        self.stream = None; self.held_notes = []; self.held_note_velocities = {}
        self._arp_thread = None; self._arp_thread_running = False; self._arp_step = 0
        self._arp_last_played_note = None; self._arp_direction_updown = 1
        try:
            print("Available output devices:"); print(sd.query_devices())
            usb_device_index = None; devices = sd.query_devices()
            if devices is not None:
                for i, device in enumerate(devices):
                    if isinstance(device, dict) and 'USB Audio Device' in device.get('name', '') and device.get('max_output_channels', 0) > 0:
                        usb_device_index = i; print(f"Found USB Audio: {device.get('name')} (Idx: {i})"); break
            if usb_device_index is not None:
                try: sd.default.device = usb_device_index; print(f"Attempting to use USB Audio Idx {usb_device_index}.")
                except Exception as e_dev: print(f"Could not set default device: {e_dev}", file=sys.stderr)
            self.stream = sd.OutputStream(samplerate=sample_rate, blocksize=block_size, channels=1, callback=self._audio_callback, dtype='float32')
            self.stream.start(); print(f"Audio stream started on {self.stream.device} (Block: {self.stream.blocksize})")
        except sd.PortAudioError as e: print(f"FATAL PortAudioError: {e}", file=sys.stderr); raise
        except Exception as e: print(f"FATAL audio init error: {e}", file=sys.stderr); traceback.print_exc(file=sys.stderr); raise

    def _find_preset_name(self, params_to_find):
        for i, name in enumerate(self.preset_names):
            preset_params = PRESETS[name]; match = True
            for key in DEFAULT_PARAMS:
                if params_to_find.get(key, DEFAULT_PARAMS[key]) != preset_params.get(key, DEFAULT_PARAMS[key]): match=False; break
            if match: self.current_preset_index=i; return name
        self.current_preset_index=0; return "Custom"

    def set_params(self, params_dict, source="unknown"):
        arp_state_changed = False; old_arp_on = self.params.get('arp_on', False)
        with self.lock:
            changed = False
            for key, value in params_dict.items():
                if key in self.params:
                    try:
                        current_value = self.params[key]; new_value = value
                        default_type = type(DEFAULT_PARAMS.get(key, type(None)))
                        if default_type == bool and not isinstance(new_value, bool): new_value = str(value).lower() in ('true', '1', 't', 'yes', 'on')
                        elif default_type == float and not isinstance(new_value, (float, int)): new_value = float(value)
                        elif default_type == int and not isinstance(new_value, int): new_value = int(float(value))
                        elif default_type == str and not isinstance(new_value, str): new_value = str(value)
                        if new_value != current_value:
                            if key == 'chorus_depth' and new_value > 0: self._chorus_on_depth = new_value
                            self.params[key] = new_value; changed = True
                            if key == 'arp_on': arp_state_changed = True
                    except (ValueError, TypeError) as e: print(f"Warn: Param conversion {key}={value}: {e}", file=sys.stderr)
                    except Exception as e: print(f"Warn: Set param {key}={value}: {e}", file=sys.stderr)
            if changed:
                try:
                    current_params_copy = self.params.copy()
                    for voice in self.voices: voice.params = current_params_copy.copy()
                except Exception as e: print(f"Error updating voice params: {e}", file=sys.stderr); traceback.print_exc(file=sys.stderr)
                self.current_preset_name = self._find_preset_name(self.params)
        if arp_state_changed:
            new_arp_on = self.params.get('arp_on', False)
            if new_arp_on and not old_arp_on: self._start_arp_thread()
            elif not new_arp_on and old_arp_on: self._stop_arp_thread()

    def toggle_volume_mute(self):
        with self.lock:
            if not hasattr(self, 'is_volume_muted'): self.is_volume_muted = False
            if not hasattr(self, '_stored_volume'): self._stored_volume = self.params['volume']
            if not self.is_volume_muted:
                self._stored_volume = self.params['volume']; self.params['volume'] = 0.0; self.is_volume_muted = True; print("Volume Muted")
            else:
                self.params['volume'] = self._stored_volume; self.is_volume_muted = False; print(f"Volume Unmuted: {self.params['volume']:.2f}")
            params_copy = self.params.copy(); [v.params = params_copy.copy() for v in self.voices]
            self.current_preset_name = self._find_preset_name(self.params)

    def toggle_chorus(self): # Called by encoder if switch is defined for chorus_amount
        with self.lock: changed = False; current_depth = self.params.get('chorus_depth', 0.0)
        if current_depth > 0: self.params['chorus_depth'] = 0.0; print("Chorus OFF by Switch"); changed = True
        else: restore_depth = self._chorus_on_depth if self._chorus_on_depth > 0 else 0.005; self.params['chorus_depth'] = restore_depth; print(f"Chorus ON by Switch (Depth: {restore_depth:.3f})"); changed = True
        if changed:
             try:
                 with self.lock: current_params_copy = self.params.copy()
                 for voice in self.voices: voice.params = current_params_copy.copy()
             except Exception as e: print(f"Error updating voice params (chorus toggle): {e}", file=sys.stderr); traceback.print_exc(file=sys.stderr)
             self.current_preset_name = self._find_preset_name(self.params)

    def load_preset(self, preset_name_or_index):
        preset_to_load = None; preset_name = "Unknown"; target_index = -1
        try:
            if isinstance(preset_name_or_index, int):
                if 0 <= preset_name_or_index < len(self.preset_names): target_index = preset_name_or_index; preset_name = self.preset_names[target_index]; preset_to_load = PRESETS[preset_name]
                else: print(f"Warn: Idx {preset_name_or_index} OOR.", file=sys.stderr); return False
            elif isinstance(preset_name_or_index, str):
                if preset_name_or_index in PRESETS: preset_name = preset_name_or_index; preset_to_load = PRESETS[preset_name]; target_index = self.preset_names.index(preset_name)
                else: print(f"Warn: Preset '{preset_name_or_index}' not found.", file=sys.stderr); return False
            else: print("Warn: Invalid preset ID.", file=sys.stderr); return False
        except Exception as e: print(f"Error finding preset {preset_name_or_index}: {e}", file=sys.stderr); traceback.print_exc(file=sys.stderr); return False
        if preset_to_load:
            print(f"\nLoading preset {target_index if target_index !=-1 else ''}: {preset_name}")
            new_params = DEFAULT_PARAMS.copy();
            for k, v in preset_to_load.items():
                if k in new_params: new_params[k] = v
            self._chorus_on_depth = new_params.get('chorus_depth', 0.005)
            self.set_params(new_params, source="load_preset")
            self.current_preset_index = target_index; self.current_preset_name = preset_name; return True
        return False

    def _apply_master_chorus(self, signal):
        try:
            p = self.params; sr = self.sample_rate; chorus_depth = p.get('chorus_depth', 0.0); chorus_rate = p.get('chorus_rate', 0.0)
            if chorus_depth <= 0 or chorus_rate <= 0: return signal
            lfo_freq_rad = 2 * np.pi * chorus_rate / sr; num_samples = len(signal); dl_len = len(self.chorus_delay_line)
            if dl_len == 0: return signal
            t = np.arange(self.chorus_lfo_phase, self.chorus_lfo_phase + num_samples * lfo_freq_rad, lfo_freq_rad, dtype=np.float32)[:num_samples]
            if len(t) < num_samples: t = np.pad(t, (0, num_samples - len(t)), 'edge')
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
        except Exception as e: print(f"Error in chorus processing: {e}", file=sys.stderr); traceback.print_exc(file=sys.stderr); return signal

    def _audio_callback(self, outdata, frames, time_info, status):
        if status: print("Audio CB status:", status, file=sys.stderr)
        buffer = np.zeros((frames, 1), dtype=np.float32)
        try:
            with self.lock:
                active_voices_next = []; voices_to_process = self.voices[:]
                if not voices_to_process: outdata[:] = buffer; return
                for voice in voices_to_process:
                    if voice.is_active():
                        try: voice_output = voice.process(frames); buffer[:, 0] += voice_output;
                        if voice.is_active(): active_voices_next.append(voice)
                        except Exception as e: print(f"ERR voice {voice.note}: {e}", file=sys.stderr); traceback.print_exc(file=sys.stderr)
                self.voices = active_voices_next
            try: buffer[:, 0] = self._apply_master_chorus(buffer[:, 0])
            except Exception as e: print(f"ERR chorus in CB: {e}", file=sys.stderr); traceback.print_exc(file=sys.stderr)
            max_amp = np.max(np.abs(buffer));
            if max_amp > 1.0: np.clip(buffer, -1.0, 1.0, out=buffer)
            outdata[:] = buffer
        except Exception as e: print(f"FATAL Error in audio callback: {e}", file=sys.stderr); traceback.print_exc(file=sys.stderr); outdata.fill(0)

    def note_on(self, note, velocity):
        with self.lock:
            arp_on = self.params.get('arp_on', False)
            if arp_on:
                if note not in self.held_notes: self.held_notes.append(note); self.held_notes.sort(); self.held_note_velocities[note] = velocity
            else:
                try: self._trigger_voice_on(note, velocity)
                except Exception as e: print(f"Error triggering voice on {note}: {e}", file=sys.stderr); traceback.print_exc(file=sys.stderr)
    def note_off(self, note):
         with self.lock:
            arp_on = self.params.get('arp_on', False)
            if arp_on:
                if note in self.held_notes: self.held_notes.remove(note);
                if note in self.held_note_velocities: del self.held_note_velocities[note]
            else:
                 try: self._trigger_voice_off(note)
                 except Exception as e: print(f"Error triggering voice off {note}: {e}", file=sys.stderr); traceback.print_exc(file=sys.stderr)
    def _trigger_voice_on(self, note, velocity):
        if len(self.voices) >= self.max_voices: self.voices.pop(0)
        try: self.voices.append(Voice(note, velocity, self.params.copy(), self.sample_rate))
        except Exception as e: print(f"ERR creating voice for note {note}: {e}", file=sys.stderr); traceback.print_exc(file=sys.stderr)
    def _trigger_voice_off(self, note):
        try:
            for voice in self.voices:
                if voice.note == note and voice.envelope_stage not in ['release', 'off']: voice.note_off(); break
        except Exception as e: print(f"Error finding/stopping voice for note {note}: {e}", file=sys.stderr); traceback.print_exc(file=sys.stderr)
    def control_change(self, control, value): pass
    def close(self):
        print("Stopping synth...");
        try: self._stop_arp_thread()
        except Exception as e: print(f"Error stopping arp thread: {e}", file=sys.stderr); traceback.print_exc(file=sys.stderr)
        print("Stopping audio stream...")
        if self.stream:
            try: self.stream.stop(); self.stream.close()
            except sd.PortAudioError as e: print(f"PortAudioError closing audio stream: {e}", file=sys.stderr)
            except Exception as e: print(f"Error closing audio stream: {e}", file=sys.stderr); traceback.print_exc(file=sys.stderr)
        self.stream = None; print("Audio stream stopped.")
    def _start_arp_thread(self):
        if self._arp_thread is not None and self._arp_thread.is_alive(): return
        try:
            self._arp_thread_running = True; self._arp_step = 0; self._arp_direction_updown = 1; self._arp_last_played_note = None
            self._arp_thread = threading.Thread(target=self._arp_sequencer_loop, daemon=True); self._arp_thread.start()
            print("Arp thread started.")
        except Exception as e: print(f"Failed to start arp thread: {e}", file=sys.stderr); traceback.print_exc(file=sys.stderr); self._arp_thread_running = False
    def _stop_arp_thread(self):
        if self._arp_thread is None or not self._arp_thread.is_alive(): return
        try:
            self._arp_thread_running = False; print("Waiting for arp thread...")
            self._arp_thread.join(timeout=0.5)
            if self._arp_thread.is_alive(): print("Warn: Arp thread join timed out.", file=sys.stderr)
            else: print("Arp thread stopped.")
            self._arp_thread = None
            with self.lock:
                if self._arp_last_played_note is not None: self._arp_stop_note(self._arp_last_played_note); self._arp_last_played_note = None
        except Exception as e: print(f"Error stopping arp thread cleanly: {e}", file=sys.stderr); traceback.print_exc(file=sys.stderr); self._arp_thread = None
    def _arp_sequencer_loop(self):
        while self._arp_thread_running:
            try:
                with self.lock:
                    arp_on = self.params.get('arp_on', False); bpm = self.params.get('arp_bpm', 120.0)
                    rate = self.params.get('arp_rate', 16); pattern = self.params.get('arp_pattern', 'Up')
                    octaves = self.params.get('arp_octaves', 1); local_held_notes = sorted(self.held_notes)
                if not arp_on or not local_held_notes:
                    if not local_held_notes and self._arp_last_played_note is not None:
                         with self.lock: self._arp_stop_note(self._arp_last_played_note)
                    time.sleep(0.05); continue
                if bpm <= 0 or rate <= 0: time.sleep(0.1); continue
                steps_per_beat = rate / 4.0; beats_per_second = bpm / 60.0
                steps_per_second = steps_per_beat * beats_per_second; step_duration_seconds = 1.0 / steps_per_second
                note_sequence = []; num_held = len(local_held_notes)
                effective_pattern = pattern
                if num_held == 1:
                    base_note = local_held_notes[0]
                    for o in range(octaves): note_sequence.append(base_note + o * 12)
                    effective_pattern = 'Up'
                else:
                    for o in range(octaves):
                        for note in local_held_notes: note_sequence.append(note + o * 12)
                if not note_sequence: time.sleep(0.05); continue
                current_note_to_play = None; seq_len = len(note_sequence); current_step = self._arp_step
                if effective_pattern == 'Up': step_index = current_step % seq_len; current_note_to_play = note_sequence[step_index]
                elif effective_pattern == 'Down': step_index = current_step % seq_len; current_note_to_play = note_sequence[seq_len - 1 - step_index]
                elif effective_pattern == 'UpDown':
                     effective_len = max(1, seq_len * 2 - 2) if seq_len > 1 else 1; step_index = current_step % effective_len
                     if step_index < seq_len: current_note_to_play = note_sequence[step_index]
                     else: current_note_to_play = note_sequence[effective_len - step_index]
                elif effective_pattern == 'Random': current_note_to_play = random.choice(note_sequence)
                else: step_index = current_step % seq_len; current_note_to_play = note_sequence[step_index]
                if current_note_to_play is not None:
                    original_base_note_candidate = local_held_notes[current_step % num_held if num_held > 0 else 0]
                    velocity = self.held_note_velocities.get(original_base_note_candidate, 100)
                    with self.lock: self._arp_play_note(current_note_to_play, velocity)
                    self._arp_step += 1
                sleep_time = step_duration_seconds; time.sleep(sleep_time)
            except Exception as e: print(f"ERR arp loop: {e}", file=sys.stderr); traceback.print_exc(file=sys.stderr); time.sleep(0.5)
    def _arp_play_note(self, note, velocity):
        try:
            if self._arp_last_played_note is not None: self._arp_stop_note(self._arp_last_played_note)
            self._trigger_voice_on(note, velocity); self._arp_last_played_note = note
        except Exception as e: print(f"Error in _arp_play_note for note {note}: {e}", file=sys.stderr); traceback.print_exc(file=sys.stderr)
    def _arp_stop_note(self, note):
        try:
            self._trigger_voice_off(note)
            if self._arp_last_played_note == note: self._arp_last_played_note = None
        except Exception as e: print(f"Error in _arp_stop_note for note {note}: {e}", file=sys.stderr); traceback.print_exc(file=sys.stderr)

# --- Encoder Configuration (Using PHYSICAL BOARD Pin Numbers from Diagram) ---
ENCODER_PINS = {
    'osc_type':      (29, 31, None, 'osc_type', 'options', ['saw', 'square', 'sine', 'triangle']), # GPIO5, GPIO6
    'chorus_amount': (33, 35, None, 'chorus_depth', 'continuous', (0.0, 0.02, 0.0005)),           # GPIO13, GPIO19 (Controls depth, Rate fixed or another encoder)
    'volume':        (37, 36, 38, 'volume', 'continuous_toggle_func', (0.0, 1.0, 0.01), 'toggle_volume_mute'), # GPIO26, GPIO16, GPIO20
    'arp_rate_enc':  (22, 18, 16, 'arp_rate', 'options_toggle_param', [4, 8, 16, 32], 'arp_on'), # GPIO25, GPIO24, GPIO23
    'cutoff':        (15, 13, None, 'filter_cutoff', 'log_continuous', (20.0, 18000.0, 1.05)),     # GPIO22, GPIO27
    'resonance':     (11, 7, None, 'filter_resonance', 'continuous', (0.0, 0.95, 0.01)),         # GPIO17, GPIO4
    'env_amount':    (5, 3, None, 'filter_env_amount', 'continuous', (-8000.0, 8000.0, 100.0)),   # GPIO3, GPIO2
    'attack':        (40, 21, None, 'attack', 'log_continuous', (0.001, 5.0, 1.02)),             # GPIO21, GPIO9
    'decay':         (19, 23, None,  'decay', 'log_continuous', (0.01, 5.0, 1.02)),             # GPIO10, GPIO11
    'sustain':       (8, 10, None, 'sustain', 'continuous', (0.0, 1.0, 0.01)),                 # GPIO14(TXD), GPIO15(RXD) - If Serial disabled
    'release':       (12, 24, None, 'release', 'log_continuous', (0.01, 8.0, 1.02)),               # GPIO18(PCM_CLK), GPIO8(SPI_CE0) - If PCM/SPI0 disabled
}

# --- Rotary Encoder Class ---
class RotaryEncoder:
    def __init__(self, clk_pin, dt_pin, sw_pin, param_name, param_type, options_or_range, toggle_target=None, synth=None):
        self.clk_pin = clk_pin; self.dt_pin = dt_pin; self.sw_pin = sw_pin
        self.param_name = param_name; self.param_type = param_type
        self.options_or_range = options_or_range; self.toggle_target = toggle_target
        self.synth = synth; self.last_clk_state = GPIO.input(self.clk_pin); self.current_option_index = 0
        if self.param_type in ['options', 'options_toggle_param']:
            current_val = self.synth.params.get(self.param_name)
            if current_val in self.options_or_range:
                try: self.current_option_index = self.options_or_range.index(current_val)
                except ValueError: self.current_option_index = 0
        GPIO.add_event_detect(self.clk_pin, GPIO.FALLING, callback=self._clk_event, bouncetime=5)
        if self.sw_pin is not None:
            self.last_sw_state = GPIO.input(self.sw_pin)
            GPIO.add_event_detect(self.sw_pin, GPIO.FALLING, callback=self._sw_event, bouncetime=300)
        else: self.last_sw_state = None
    def _clk_event(self, channel):
        dt_state = GPIO.input(self.dt_pin); direction = 1 if dt_state == GPIO.HIGH else -1
        self._update_param(direction)
    def _sw_event(self, channel):
        if self.sw_pin is None: return
        time.sleep(0.05)
        if GPIO.input(self.sw_pin) == GPIO.LOW:
            print(f"Encoder '{self.param_name}' SW Press (Pin {self.sw_pin})")
            if self.param_type == 'toggle_param' or self.param_type == 'options_toggle_param':
                if self.toggle_target:
                    current_state = self.synth.params.get(self.toggle_target, False)
                    self.synth.set_params({self.toggle_target: not current_state})
                    print(f"Toggled {self.toggle_target}: {not current_state}")
            elif self.param_type == 'continuous_toggle_func':
                if self.toggle_target and hasattr(self.synth, self.toggle_target):
                    try: getattr(self.synth, self.toggle_target)()
                    except Exception as e: print(f"Error toggle func {self.toggle_target}: {e}", file=sys.stderr)
    def _update_param(self, direction):
        try:
            current_val = self.synth.params.get(self.param_name); new_val = current_val
            if self.param_type == 'continuous' or self.param_type == 'log_continuous':
                min_val, max_val, step = self.options_or_range
                if self.param_type == 'log_continuous': safe_current_val = max(current_val, 1e-9); factor = step if direction > 0 else 1.0 / step; new_val = safe_current_val * factor
                else: new_val = current_val + (direction * step)
                new_val = np.clip(new_val, min_val, max_val)
            elif self.param_type in ['options', 'options_toggle_param']:
                self.current_option_index = (self.current_option_index + direction) % len(self.options_or_range)
                new_val = self.options_or_range[self.current_option_index]
            if abs(new_val - current_val) > 1e-9:
                self.synth.set_params({self.param_name: new_val})
                display_val = f"{new_val:.3f}" if isinstance(new_val, float) else new_val
                print(f"Encoder '{self.param_name}' (CLK:{self.clk_pin}) -> {self.param_name}: {display_val}")
        except Exception as e: print(f"Error update param {self.param_name}: {e}", file=sys.stderr); traceback.print_exc(file=sys.stderr)
    def cleanup(self):
        try: GPIO.remove_event_detect(self.clk_pin)
        if self.sw_pin is not None: GPIO.remove_event_detect(self.sw_pin)
        except Exception: pass

# --- MIDI Input Thread ---
midi_thread_running = True; midi_port = None
def midi_listener(synth_instance, port_name):
    global midi_thread_running, midi_port
    print(f"MIDI thread for '{port_name}'.")
    while midi_thread_running:
        try:
            if midi_port is None or midi_port.closed:
                 if not midi_thread_running: break
                 print(f"Opening MIDI '{port_name}'..."); midi_port = mido.open_input(port_name); print(f"Listening '{port_name}'...")
            for msg in midi_port:
                if not midi_thread_running: break
                try:
                    if msg.type == 'note_on' and msg.velocity > 0: synth_instance.note_on(msg.note, msg.velocity)
                    elif msg.type == 'note_off' or (msg.type == 'note_on' and msg.velocity == 0): synth_instance.note_off(msg.note)
                except Exception as e: print(f"Error MIDI msg {msg}: {e}", file=sys.stderr); traceback.print_exc(file=sys.stderr)
            if not midi_thread_running: break
        except (OSError, IOError, mido.MidiError) as e:
            print(f"MIDI Port Error: {e}. Retrying...", file=sys.stderr)
            if midi_port and not midi_port.closed: try: midi_port.close()
            except Exception: pass; midi_port = None
            for _ in range(20): time.sleep(0.1);  if not midi_thread_running: break
            if not midi_thread_running: break
        except Exception as e:
            print(f"Unexpected MIDI listener error: {e}", file=sys.stderr); traceback.print_exc(file=sys.stderr)
            if midi_port and not midi_port.closed: try: midi_port.close()
            except Exception: pass; midi_port = None
            for _ in range(50): time.sleep(0.1); if not midi_thread_running: break
            if not midi_thread_running: break
        if midi_port and midi_port.closed: print(f"MIDI port '{port_name}' closed."); midi_port = None
    if midi_port and not midi_port.closed: print("Closing MIDI from thread exit..."); try: midi_port.close()
    except Exception: pass
    print("MIDI thread finished.")

# --- Main Execution ---
if __name__ == "__main__":
    SAMPLE_RATE = 44100; BLOCK_SIZE = 1024; MAX_VOICES = 6
    SYNTH_INSTANCE = None; ENCODERS_INITIALIZED = []; MIDI_THREAD_INSTANCE = None
    GPIO_MODE_SET = False
    if RPI_GPIO_AVAILABLE:
        try: GPIO.setmode(GPIO.BOARD); GPIO.setwarnings(False); GPIO_MODE_SET = True; print("GPIO mode set to BOARD.")
        except RuntimeError as e: print(f"ERR RPi.GPIO setup: {e}. Encoders disabled.", file=sys.stderr)
        except Exception as e: print(f"Unexpected ERR RPi.GPIO setup: {e}. Encoders disabled.", file=sys.stderr)
    else: print("RPi.GPIO not available. Encoders disabled.", file=sys.stderr)

    selected_port_name = None
    try:
        print("Scanning MIDI..."); available_ports = mido.get_input_names()
        if not available_ports: print("No MIDI inputs found.", file=sys.stderr)
        elif len(available_ports) == 1: selected_port_name = available_ports[0]; print(f"Auto MIDI: '{selected_port_name}'")
        else:
            print("Multi MIDI:"); [print(f"  {i}: {n}") for i,n in enumerate(available_ports)]
            for name in available_ports:
                if "midi" in name.lower() and "through" not in name.lower() and "virtual" not in name.lower(): selected_port_name = name; break
            if not selected_port_name: selected_port_name = available_ports[0]
            print(f"Auto MIDI: '{selected_port_name}' (from multiple)")
    except Exception as e: print(f"Error MIDI auto-detect: {e}", file=sys.stderr)

    try:
        print("Init synth..."); SYNTH_INSTANCE = PySynthJunoMIDI(sample_rate=SAMPLE_RATE, block_size=BLOCK_SIZE, max_voices=MAX_VOICES)
        if SYNTH_INSTANCE.stream: print(f"Audio: {SYNTH_INSTANCE.stream.device}, SR={SYNTH_INSTANCE.stream.samplerate:.0f}, Block={SYNTH_INSTANCE.stream.blocksize}, Latency={SYNTH_INSTANCE.stream.latency*1000:.1f}ms")
        else: print("CRITICAL: Synth stream failed.", file=sys.stderr);
        if GPIO_MODE_SET: GPIO.cleanup();
        sys.exit(1)
    except Exception as e: print(f"FATAL: Synth init: {e}", file=sys.stderr); traceback.print_exc(file=sys.stderr);
    if GPIO_MODE_SET: GPIO.cleanup();
    sys.exit(1)

    # Define toggle_volume_mute attached to the synth instance
    SYNTH_INSTANCE.is_volume_muted = False
    SYNTH_INSTANCE._stored_volume = SYNTH_INSTANCE.params['volume']
    def _synth_toggle_volume_mute_method(self_synth):
        with self_synth.lock:
            if not self_synth.is_volume_muted: self_synth._stored_volume = self_synth.params['volume']; self_synth.params['volume'] = 0.0; self_synth.is_volume_muted = True; print("Volume Muted")
            else: self_synth.params['volume'] = self_synth._stored_volume; self_synth.is_volume_muted = False; print(f"Volume Unmuted: {self_synth.params['volume']:.2f}")
            params_copy = self_synth.params.copy(); [v.params = params_copy.copy() for v in self_synth.voices]
            self_synth.current_preset_name = self_synth._find_preset_name(self_synth.params)
    SYNTH_INSTANCE.toggle_volume_mute = lambda: _synth_toggle_volume_mute_method(SYNTH_INSTANCE)


    print("\n--- Init Encoders (Check PHYSICAL BOARD pins!) ---")
    if RPI_GPIO_AVAILABLE and GPIO_MODE_SET:
        for name, config_tuple in ENCODER_PINS.items():
            if len(config_tuple) < 6: print(f"  Skipping misconfigured encoder '{name}'.", file=sys.stderr); continue
            clk, dt, sw, param, p_type, opts_range = config_tuple[:6]
            toggle = config_tuple[6] if len(config_tuple) > 6 else None
            try:
                GPIO.setup(clk, GPIO.IN, pull_up_down=GPIO.PUD_UP); GPIO.setup(dt, GPIO.IN, pull_up_down=GPIO.PUD_UP)
                if sw is not None: GPIO.setup(sw, GPIO.IN, pull_up_down=GPIO.PUD_UP)
                ENCODERS_INITIALIZED.append(RotaryEncoder(clk, dt, sw, param, p_type, opts_range, toggle, SYNTH_INSTANCE))
                sw_pin_str = f"SW:{sw}" if sw is not None else "SW:None"
                print(f"  Encoder '{name}' ({param}) on CLK:{clk}, DT:{dt}, {sw_pin_str}")
            except Exception as e: print(f"  ERR Init encoder '{name}': {e}. Check pins.", file=sys.stderr); traceback.print_exc(file=sys.stderr)
    else: print("  RPi.GPIO not active or mode not set. Encoders disabled.", file=sys.stderr)

    if selected_port_name:
        midi_thread_running = True
        MIDI_THREAD_INSTANCE = threading.Thread(target=midi_listener, args=(SYNTH_INSTANCE, selected_port_name), daemon=True)
        try: MIDI_THREAD_INSTANCE.start()
        except Exception as e: print(f"FATAL starting MIDI thread: {e}", file=sys.stderr); traceback.print_exc(file=sys.stderr);
        if SYNTH_INSTANCE: SYNTH_INSTANCE.close();
        if GPIO_MODE_SET: GPIO.cleanup();
        sys.exit(1)
    else: print("No MIDI port. MIDI input ignored.")

    initial_preset = "Default";
    if initial_preset not in PRESETS: initial_preset = list(PRESETS.keys())[0] if PRESETS else None
    if initial_preset and SYNTH_INSTANCE:
        try: SYNTH_INSTANCE.load_preset(initial_preset)
        except Exception as e: print(f"ERR loading initial preset '{initial_preset}': {e}", file=sys.stderr)
    elif not PRESETS: print("No presets defined. Using default parameters.", file=sys.stderr)


    print("\nNeptune 1 Synthesizer running. Ctrl+C to exit."); print("Control with rotary encoders.")
    try:
        while True: time.sleep(1)
    except KeyboardInterrupt: print("\nCtrl+C. Shutting down...")
    finally:
        print("Cleaning up..."); midi_thread_running = False
        if MIDI_THREAD_INSTANCE and MIDI_THREAD_INSTANCE.is_alive():
            print("Stopping MIDI thread...");
            if midi_port and not midi_port.closed: try: midi_port.close()
            except Exception: pass
            MIDI_THREAD_INSTANCE.join(timeout=0.5)
            if MIDI_THREAD_INSTANCE.is_alive(): print("Warn: MIDI thread join timed out.", file=sys.stderr)
        if SYNTH_INSTANCE: SYNTH_INSTANCE.close()
        if RPI_GPIO_AVAILABLE and GPIO_MODE_SET:
            for encoder in ENCODERS_INITIALIZED:
                try: encoder.cleanup()
                except Exception as e: print(f"Err cleaning encoder: {e}", file=sys.stderr)
            GPIO.cleanup(); print("GPIO cleaned up.")
        print("Synthesizer stopped.")
