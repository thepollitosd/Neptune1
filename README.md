# Neptune 1 - Headless Raspberry Pi Synthesizer

A polyphonic synthesizer designed to run headlessly on a Raspberry Pi, controlled by MIDI input and KY-040 rotary encoders.

## Features

*   Polyphonic sound engine (configurable number of voices)
*   Multiple oscillator types (Saw, Square, Sine, Triangle)
*   Low-pass filter with cutoff, resonance, and envelope amount control
*   ADSR envelope for amplitude
*   Chorus effect
*   Arpeggiator with various patterns, rates, and octave ranges
*   Parameter control via 11 KY-040 rotary encoders (configurable GPIO pins)
*   MIDI input for notes
*   Auto-detection for MIDI input port and USB audio output device
*   Console output for status and parameter changes

## Prerequisites

*   **Raspberry Pi:** Tested on Raspberry Pi 3B/3B+/4B. Other models with sufficient GPIOs should work.
*   **Raspberry Pi OS:** A recent version (e.g., Bullseye or Bookworm) with Python 3.7+ installed.
*   **USB Audio Interface:** Highly recommended for decent audio quality and lower latency. The built-in audio jack on older Pis is not ideal.
*   **MIDI Controller:** A USB MIDI keyboard or controller.
*   **KY-040 Rotary Encoders:** 11 encoders are expected for full parameter control as defined in the script.
*   **Breadboard and Jumper Wires:** For connecting the rotary encoders.
*   **Power Supply:** A stable power supply for your Raspberry Pi, especially when connecting multiple peripherals.

## Installation Steps

Follow these steps on your Raspberry Pi's terminal:

**1. Update System Packages:**
   It's good practice to start with an updated system.
   ```bash
   sudo apt update
   sudo apt upgrade -y
   ```

**2. Install System Dependencies for Audio and GPIO:**
   These libraries provide the underlying support needed by some Python packages.
   ```bash
   sudo apt install -y python3-dev libasound2-dev libportaudiocpp0 portaudio19-dev libjack-jackd2-dev python3-rpi.gpio python3-numpy python3-scipy
   ```
   *   `python3-dev`: Development headers for Python 3.
   *   `libasound2-dev`, `libportaudiocpp0`, `portaudio19-dev`, `libjack-jackd2-dev`: For `sounddevice` and audio backends.
   *   `python3-rpi.gpio`: For controlling GPIO pins.
   *   `python3-numpy`, `python3-scipy`: Often better to install system-wide versions on RPi if available and compatible.

**3. Install Python Packages using `pip3`:**
   Install the remaining Python libraries. It's recommended to do this within a virtual environment if you prefer, but for a dedicated synth project, global installation is also common on the Pi.
   ```bash
   pip3 install sounddevice mido python-rtmidi
   ```
   *   `sounddevice`: For audio playback.
   *   `mido`: For MIDI message handling.
   *   `python-rtmidi`: A reliable MIDI backend for `mido`.

   *(Note: `numpy` and `scipy` were installed via `apt` in the previous step, which is often more optimized for the RPi. If you encounter issues, you can try `pip3 install numpy scipy` as well, but the `apt` versions are usually preferred if they meet the requirements.)*

**4. Get the Synthesizer Code:**
   *   **Option A: Clone from Git (if you have a repository)**
      ```bash
      git clone https://github.com/thepollitosd/Neptune1/
      cd Neptune1
      ```
   *   **Option B: Manually Create the File**
      Copy the full Python script content and save it on your Raspberry Pi as `neptune1.py` (or your preferred name).

**5. Configure GPIO Pins for Encoders:**
   *   Open the `neptune1.py` script in a text editor (e.g., `nano neptune1.py`).
   *   Locate the `ENCODER_PINS` dictionary near the top of the script.
   *   **Crucially, update the physical board pin numbers in this dictionary to match how you have wired your 11 KY-040 rotary encoders to your Raspberry Pi's GPIO header.** Refer to a pinout diagram for your Raspberry Pi model (e.g., run `pinout` in the terminal or visit [pinout.xyz](https://pinout.xyz/)).
      *   The format is: `(CLK_PIN, DT_PIN, SW_PIN_or_None, parameter_name, ...)`
      *   Use `None` for `SW_PIN` if that encoder's switch is not used.
      *   Ensure all CLK, DT, and used SW pins are unique and valid GPIO pins.

**6. (Optional) Configure Audio Output:**
   The script attempts to auto-detect a USB audio device. If you want to force a specific audio output device:
   *   Run `aplay -l` to list playback hardware devices. Note the card number and device number.
   *   Run `python3 -m sounddevice` to list devices as `sounddevice` sees them. Note the index or name.
   *   You might need to modify the `sd.default.device` line in the `PySynthJunoMIDI.__init__` method or set the `AUDIODEV` environment variable (e.g., `export AUDIODEV=hw:1,0` before running the script, replacing `1,0` with your device). For `sounddevice`, using the device index or a unique part of its name is often easier.

**7. (Optional) Disable Serial Console (if using UART GPIOs):**
   If you've assigned encoders to the default UART pins (Physical pins 8 (GPIO14/TXD) and 10 (GPIO15/RXD)), you might need to disable the serial console:
   *   Run `sudo raspi-config`.
   *   Navigate to `Interface Options` -> `Serial Port`.
   *   When asked "Would you like a login shell to be accessible over serial?", select **No**.
   *   When asked "Would you like the serial port hardware to be enabled?", select **Yes** (this keeps the hardware usable by GPIO).
   *   Reboot the Raspberry Pi.

## Wiring the KY-040 Encoders

*   **VCC (+):** Connect to a **3.3V** pin on the Raspberry Pi.
*   **GND:** Connect to a **GND** (Ground) pin on the Raspberry Pi.
*   **CLK:** Connect to the designated CLK GPIO pin (as defined in `ENCODER_PINS`).
*   **DT:** Connect to the designated DT GPIO pin.
*   **SW (Switch):** Connect to the designated SW GPIO pin (if used). The script configures an internal pull-up resistor for SW pins, so an external one is usually not needed unless you experience issues.

**Refer to your Raspberry Pi's pinout diagram (`pinout` command or pinout.xyz) for correct physical pin numbering.**

## Running the Synthesizer

1.  Navigate to the directory where you saved `neptune1.py`.
2.  Connect your MIDI controller and USB audio interface.
3.  Run the script:
    ```bash
    python3 neptune1.py
    ```
    You might need `sudo python3 neptune1.py` if `RPi.GPIO` requires root access, though usually installing via `apt` handles permissions correctly for the `gpio` group.

4.  The script will attempt to auto-detect MIDI and audio devices and then print status messages to the console.
5.  Play notes on your MIDI controller and use the rotary encoders to adjust parameters.
6.  Press `Ctrl+C` in the terminal to stop the synthesizer.

## Troubleshooting

*   **"No sound":**
    *   Check audio output selection in `alsamixer` (run `alsamixer` in terminal, press F6 to select your USB sound card, ensure master and PCM volumes are up and not muted).
    *   Verify the `sounddevice` is using the correct output (check script console output).
    *   Ensure synth `volume` parameter isn't accidentally set to 0 by an encoder.
*   **"Encoders not working / erratic":**
    *   **Triple-check your wiring** against the `ENCODER_PINS` configuration and the Raspberry Pi's physical pin numbers.
    *   Ensure you are using `GPIO.setmode(GPIO.BOARD)`.
    *   Check for loose connections on the breadboard.
    *   The `bouncetime` in the `RotaryEncoder` class might need adjustment for your specific encoders.
    *   The CLK/DT logic in `RotaryEncoder._clk_event` might need to be flipped (`direction = 1 if dt_state == GPIO.LOW else -1`) if encoders turn parameters the wrong way.
*   **"Output underflow" or audio glitches:**
    *   The Raspberry Pi 3B might be struggling. Try reducing `MAX_VOICES` in `neptune1.py`.
    *   Ensure no other CPU-intensive processes are running.
    *   A larger `BLOCK_SIZE` (e.g., 1024 or even 2048) in `PySynthJunoMIDI.__init__` can help, at the cost of latency.
*   **"RPi.GPIO import error" or "RuntimeError: No access to /dev/gpiomem":**
    *   Make sure `python3-rpi.gpio` is installed correctly.
    *   You might need to run the script with `sudo`.
    *   Ensure your user is part of the `gpio` group (`sudo usermod -a -G gpio your_username`, then log out and back in).

Enjoy your Raspberry Pi synthesizer!
