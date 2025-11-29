# Sovra – Live X Spaces AI Co-Host 🎙🤖

Sovra is a **hands-free, always-listening AI co-host** for X (Twitter) Spaces.

It:

* Listens to your **real voice** through your MacBook mic
* Transcribes your speech locally using **OpenAI Whisper**
* Sends questions to your **local LLM** via the **oobabooga (text-generation-webui) OpenAI-compatible API**
* Speaks replies back using **macOS system TTS** (via `pyttsx3`) into **Ai + Speakers → BlackHole → X Space**

You control it entirely with **voice commands**:

* `SOVRA TURN ON` → Enable responses
* `SOVRA PAUSE` → Stop responding, but keep listening
* `SOVRA STOP` → Hard-interrupt Sovra mid-sentence
* `SOVRA ...` → Ask a question (e.g., `SOVRA what is Bitcoin?`)

---

## 1. High-Level Architecture

**Audio flow:**

1. You talk into your **MacBook Pro Microphone**.

2. `sounddevice` records short chunks (e.g., 6 seconds at 16 kHz).

3. `openai-whisper` (running locally via Python) transcribes the chunk to text.

4. The script looks for voice commands:

   * `SOVRA TURN ON`
   * `SOVRA PAUSE`
   * `SOVRA STOP`
   * or a question starting with `SOVRA ...`

5. For real questions, it sends the text to your **oobabooga OpenAI-compatible API** at:

   ```text
   https://YOUR_OOBABOOGA_API/v1/chat/completions
   ```

6. Your oobabooga LLM returns a reply.

7. `pyttsx3` speaks the reply using a macOS voice (e.g. Samantha/Alex/etc.).

8. Output goes to **Ai + Speakers**, which is routed through **BlackHole** into the X Space.

Everything stays **local + self-hosted** except the network hop to your own API endpoint.

---

## 2. Components & Dependencies

### 2.1. Python Version

* Recommended: **Python 3.11** (you’re already using this in `ai-space-env`)

### 2.2. Core Python Packages

Install these inside your virtualenv:

```bash
pip install \
  numpy \
  sounddevice \
  pyttsx3 \
  requests \
  openai-whisper
```

> Note:
> `openai-whisper` will automatically pull in:
>
> * `torch`
> * `tiktoken`
> * `numba`
> * `tqdm`
>   and their dependencies.

If you see **NumPy 1.x vs 2.x** compatibility warnings for other native modules, the simplest path is to keep **NumPy 2.x** (which Whisper + Torch is fine with) and slowly upgrade other problem packages as needed.

### 2.3. System Requirements (macOS)

* macOS with:

  * **BlackHole** (or similar virtual audio device) installed and configured.
  * A custom output device like **“Ai + Speakers”** that combines:

    * Your real speakers
    * BlackHole (so Spaces hears Sovra’s voice)
* Accessible **input device**:

  * `MacBook Pro Microphone` (or an external mic if you prefer)

### 2.4. LLM Backend (oobabooga / text-generation-webui)

You are running **oobabooga** with the built-in OpenAI-compatible API like:

```bash
python3 server.py \
  --listen \
  --listen-port 7860 \
  --share \
  --api \
  --api-port 8080 \
  --api-key YOUR_KEY \
  --trust-remote-code \
  --gradio-auth "12345" \
  --loader llama.cpp \
  --gpu-layers 40 \
  --extensions long_replies send_pictures
```

Then you have it reverse-proxied / exposed as:

```text
https://YOUR_OOBABOOGA_API
```

So the **OpenAI-style chat endpoint** the Python app hits is:

```text
https://YOUR_OOBABOOGA_API/v1/chat/completions
```

This is a standard OpenAI-compatible endpoint provided by oobabooga’s API server.

---

## 3. The Python App: `ai_x*.py` (Sovra)

The core script you’re running (e.g. `ai_x4.py`) does:

* Loads **Whisper** (`small` model by default) for local STT
* Talks to your **oobabooga API** for LLM replies
* Uses **pyttsx3** for TTS with selectable macOS voices
* Listens in a loop and reacts to **voice commands**

Key configuration at the top:

```python
INDEX_SERVER_BASE = "https://YOUR_OOBABOOGA_API"
CHAT_URL = f"{INDEX_SERVER_BASE}/v1/chat/completions"

API_KEY = "YOUR_KEY"

HEADERS = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {API_KEY}",
}
```

If you ever change the domain, port, or API key, you only need to update this section.

---

## 4. Audio Device Configuration

In the script:

```python
OUTPUT_DEVICE_INDEX = 4   # "Ai + Speakers"
INPUT_DEVICE_INDEX = 1    # "MacBook Pro Microphone"

sd.default.device = (OUTPUT_DEVICE_INDEX, INPUT_DEVICE_INDEX)
```

To discover your device indexes, run this once in a Python shell:

```python
import sounddevice as sd
print(sd.query_devices())
```

Look for lines like:

```text
  1 MacBook Pro Microphone, Core Audio (1 in, 0 out)
  4 Ai + Speakers, Core Audio (0 in, 2 out)
```

Then set `OUTPUT_DEVICE_INDEX` and `INPUT_DEVICE_INDEX` to those numbers.

---

## 5. Whisper Configuration

In the script:

```python
WHISPER_MODEL_NAME = "small"   # tiny, base, small, medium, large
print(f"Loading Whisper model: {WHISPER_MODEL_NAME} (this may take a bit)...")
whisper_model = whisper.load_model(WHISPER_MODEL_NAME)
```

You can switch model sizes depending on:

* **Speed vs accuracy trade-off**:

  * `tiny` / `base`: fastest, less accurate
  * `small` / `medium`: solid accuracy, moderate speed
  * `large`: best accuracy, slowest

The script records **6 second chunks** at 16 kHz:

```python
SAMPLE_RATE = 16000
CHUNK_SECONDS = 6.0
```

Each chunk is fed into Whisper for transcription.

---

## 6. Voice Selection & Speaking Speed

On startup, Sovra walks you through:

### 6.1. Voice Presets

The script scans macOS voices and offers English-filtered presets like:

* `Pro Female US (Samantha)`
* `Pro Male US (Alex)`
* `British Narrator (Daniel)`
* `Aussie Host (Karen)`
* `Irish Storyteller (Moira)`
* Fun robot voices like `Zarvox`, `Whisper`, `Trinoids`
* Kid and classic computer voices (`Junior`, `Fred`)

You’ll see a menu like:

```text
=== BuRtRoN.AI Voice Presets (English) ===
 1) [HIGH HUMAN] Pro Female US (Samantha) (Samantha)
 2) [HIGH HUMAN] Pro Male US (Alex) (Alex)
 3) [HIGH HUMAN] British Narrator (Daniel) (Daniel)
 ...
 L)  List ALL English voices (advanced/manual)
ENTER) Smart default (Samantha → Alex → Daniel → system default)
```

You can either:

* Choose a preset number (`1`, `2`, etc.)
* Press `L` to see all English-ish voices by raw index
* Just hit **Enter** for smart default selection

### 6.2. Speaking Speed

Then you’ll get:

```text
=== BuRtRoN.AI Speaking Speed ===
1) Slow        (~150)
2) Normal      (~185)  [default]
3) Fast        (~220)
4) Very Fast   (~260)
5) Custom value (enter a number like 200, 230, etc.)
ENTER) Keep default (185)
```

This sets the `pyttsx3` rate internally:

```python
DEFAULT_RATE = 185
engine.setProperty("rate", rate)
```

---

## 7. Voice Commands & Behavior

Once Sovra is online, she listens in a loop and uses Whisper to transcribe each audio chunk. The recognized text is passed into:

```python
handle_transcript(text, ai_enabled)
```

The command set:

### 7.1. Core Commands

* **Turn ON**
  Say: `SOVRA TURN ON`
  → Enables responses, and she replies:

  > “Sovra is now on and listening.”

* **Pause**
  Say: `SOVRA PAUSE`
  → Disables answering, but still listens for commands

  > “Sovra is now paused.”

* **Ask a Question**
  Say: `SOVRA what is Bitcoin?`
  → The script strips off `SOVRA`, sends the remainder (`what is Bitcoin?`) to the oobabooga API
  → Sovra speaks back the answer via TTS

### 7.2. Hard Stop / Interrupt

* **Stop Mid-Sentence**
  Say: `SOVRA STOP`
  → If Sovra is currently speaking, the script calls:

  ```python
  engine.stop()
  ```

  and sets `is_speaking = False`.
  This instantly cuts off the current utterance.

### 7.3. Cooldown Between Answers

After Sovra answers a question, the script waits briefly before listening again:

```python
COOLDOWN_AFTER_REPLY = 4.0  # seconds

if just_replied:
    time.sleep(COOLDOWN_AFTER_REPLY)
else:
    time.sleep(0.2)
```

This avoids her immediately responding to background chatter right after finishing a thought.

You can:

* Lower it if you want faster back-to-back interaction
* Increase it if you want more breathing room between answers

---

## 8. Running the App

Assuming you have your `ai-space-env` virtualenv and `ai_x4.py` (or similar) in `~/Desktop`:

```bash
cd ~/Desktop

# activate your venv
source ~/ai-space-env/bin/activate

# run Sovra
python ai_x4.py
```

You’ll see:

* Whisper model loading
* Voice preset selection
* Speed selection
* Then:

```text
Voice Commands:
  - "SOVRA TURN ON"  → Turn SOVRA ON (allow responses)
  - "SOVRA PAUSE"    → Pause SOVRA (no responses)
  - "SOVRA STOP"     → Immediately stop SOVRA mid-sentence
  - "SOVRA ..."      → Ask a question, e.g. 'SOVRA what is Bitcoin?'

Notes:
  - SOVRA listens continuously from your MacBook Pro Microphone.
  - Output voice goes to Ai + Speakers → BlackHole → X Spaces.
  - Press Ctrl+C in this terminal to stop the program.
```

Then just join a Space, route your audio correctly (mic: your mic, “speaker” output: Ai + Speakers), and talk.

---

## 9. Environment Summary

* **LLM / Chat Backend**:

  * [x] **oobabooga / text-generation-webui**
  * [x] Exposed via `https://YOUR_OOBABOOGA_API/v1/chat/completions`
  * [x] Auth: `Authorization: Bearer YOUR_KEY`

* **STT (Speech-to-Text)**:

  * [x] **openai-whisper** local model (`small` by default)

* **TTS (Text-to-Speech)**:

  * [x] macOS system TTS via `pyttsx3`
  * [x] Configurable English voices (Samantha, Alex, Daniel, etc.)
  * [x] Adjustable rate (speed)

* **Audio Routing**:

  * Input: `MacBook Pro Microphone`
  * Output: `Ai + Speakers` → BlackHole → X Spaces

---

## 10. Possible Future Enhancements

Some ideas you can bolt on later:

* Persistent **chat history** per Space session (feed prior turns into oobabooga for more context).
* Adaptive chunk length:

  * Shorter chunks for commands (e.g. 2–3 seconds)
  * Longer chunks when intentionally asking longer questions
* On-screen status display:

  * “Listening…”
  * “Transcribing…”
  * “Thinking…”
  * “Speaking…”
* Hotkey fallback:

  * Keyboard override to mute/unmute Sovra in case STT misfires in a noisy room.

