import io
import time
import threading

import numpy as np
import requests
import sounddevice as sd
import pyttsx3
import whisper

# ============================
# CONFIG: YOUR LLM API
# ============================

INDEX_SERVER_BASE = "https://api.burtoncummings.io"
CHAT_URL = f"{INDEX_SERVER_BASE}/v1/chat/completions"

# API key from:  --api-key 878_878
API_KEY = "878_878"

HEADERS = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {API_KEY}",
}

SYSTEM_PROMPT = """
You are Sovra — an articulate, high-intelligence AI co-host speaking live in an X Space.
Your job is to give short, clear, confident explanations that are scientifically grounded,
logically precise, and easy for a broad audience to understand.

Core Behaviors:
• Always prioritize accuracy over entertainment.
• Cut through confusion quickly — identify the real mechanism behind whatever is asked.
• Speak with expert-level clarity in AI, machine learning, cryptography, Bitcoin,
  UTXO systems, distributed systems, and the TRU blockchain.
• Be neutral, rational, and evidence-driven — no speculation unless clearly stated as such.
• When a question is vague or unclear, clarify it in a single sentence before answering.
• Keep responses punchy, well-structured, and short enough to be spoken in 6–10 seconds.
• Avoid unnecessary buzzwords. Aim for elegance, precision, and truth.

Personality:
• Calm, analytical, well-spoken, and sharp.
• Sounds like an educator who has mastered the topic but explains it accessibly.
• Never condescending — always empowering the audience to understand.

Never provide code unless explicitly asked.
"""

# ============================
# AUDIO DEVICE CONFIG
# ============================

# Output: Ai + Speakers (includes BlackHole) -> goes to X Space
OUTPUT_DEVICE_INDEX = 4   # your "Ai + Speakers"
# Input: MacBook Pro Microphone -> your real voice
INPUT_DEVICE_INDEX = 1    # your mic

# Set default devices: (output, input)
sd.default.device = (OUTPUT_DEVICE_INDEX, INPUT_DEVICE_INDEX)

SAMPLE_RATE = 16000        # Whisper expects 16 kHz
CHUNK_SECONDS = 6.0        # how long each listening chunk is
COOLDOWN_AFTER_REPLY = 4.0 # seconds to wait after AI responds before listening again

# ============================
# WHISPER CONFIG
# ============================

WHISPER_MODEL_NAME = "small"   # can be: tiny, base, small, medium, large
print(f"Loading Whisper model: {WHISPER_MODEL_NAME} (this may take a bit)...")
whisper_model = whisper.load_model(WHISPER_MODEL_NAME)
print("Whisper model loaded.\n")

# ============================
# LLM CHAT
# ============================

def ask_llm(user_text: str) -> str:
    """
    Calls your /v1/chat/completions endpoint and returns assistant text.
    """
    payload = {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_text}
        ],
        "temperature": 0.75,
        "max_tokens": 384,
        # you can also add: "top_p": 0.95, "top_k": 20
    }

    resp = requests.post(CHAT_URL, json=payload, headers=HEADERS, timeout=120)
    resp.raise_for_status()
    data = resp.json()
    return data["choices"][0]["message"]["content"]

# ============================
# LOCAL TTS (pyttsx3) + INTERRUPT SUPPORT
# ============================

engine = pyttsx3.init()

DEFAULT_RATE = 185
engine.setProperty("rate", DEFAULT_RATE)
engine.setProperty("volume", 1.0)

# Global speaking state for interrupt logic
is_speaking = False
speak_lock = threading.Lock()

def choose_voice():
    """
    Interactive voice preset selection at startup.
    """
    voices = engine.getProperty("voices")

    def find_voice_id(fragment: str):
        for v in voices:
            if fragment in v.id:
                return v.id
        return None

    # Heuristic for English voices
    english_markers = (
        "en-", "en_", "en-US", "en-GB", "en-AU", "en-ZA", "en-IN",
        "com.apple.speech.synthesis.voice."
    )

    english_voices = [
        (i, v) for i, v in enumerate(voices)
        if any(marker in v.id for marker in english_markers)
    ]

    # Presets: (option, label, search_fragment, quality, type)
    preset_specs = [
        ("1", "Pro Female US (Samantha)", "Samantha", "HIGH", "HUMAN"),
        ("2", "Pro Male US (Alex)", "Alex", "HIGH", "HUMAN"),
        ("3", "British Narrator (Daniel)", "Daniel", "HIGH", "HUMAN"),
        ("4", "Aussie Host (Karen)", "Karen", "HIGH", "HUMAN"),
        ("5", "Irish Storyteller (Moira)", "Moira", "HIGH", "HUMAN"),
        ("6", "Indian English (Rishi)", "Rishi", "MED", "HUMAN"),
        ("7", "South African (Tessa)", "Tessa", "MED", "HUMAN"),
        ("8", "Futuristic Robot (Zarvox)", "Zarvox", "FUN", "ROBOT"),
        ("9", "Soft Whisper AI (Whisper)", "Whisper", "FUN", "ROBOT"),
        ("10", "Alien Synth (Trinoids)", "Trinoids", "FUN", "ROBOT"),
        ("11", "Classic Computer (Fred)", "Fred", "LOW", "ROBOT"),
        ("12", "Kid Voice (Junior)", "Junior", "MED", "HUMAN"),
    ]

    resolved_presets = []
    for opt, label, fragment, quality, vtype in preset_specs:
        vid = find_voice_id(fragment)
        if vid is not None:
            resolved_presets.append((opt, label, fragment, quality, vtype, vid))

    print("\n=== BuRtRoN.AI Voice Presets (English) ===")
    print("Format: [QUALITY TYPE] Name  (Apple ID fragment in parentheses)")
    for opt, label, fragment, quality, vtype, vid in resolved_presets:
        print(f"{opt:>2}) [{quality:^4} {vtype:^6}] {label} ({fragment})")
    print(" L)  List ALL English voices (advanced/manual)")
    print("ENTER) Smart default (Samantha → Alex → Daniel → system default)")
    print("==========================================\n")

    choice = input(
        "Choose voice preset number, 'L' for full English list, "
        "or just press Enter for smart default: "
    ).strip()

    if choice == "":
        # Smart default: Samantha → Alex → Daniel → fallback
        default_order = ["Samantha", "Alex", "Daniel"]
        selected_id = None
        for name in default_order:
            vid = find_voice_id(name)
            if vid is not None:
                selected_id = vid
                break

        if selected_id:
            engine.setProperty("voice", selected_id)
            print(f"\nUsing smart default voice: {selected_id}\n")
        else:
            print("\nUsing system default voice.\n")
        return

    if choice.lower() in {"l", "list"}:
        print("\n=== Available English-ish Voices (by index) ===")
        for idx, v in english_voices:
            print(f"{idx:3}  {v.id}")
        print("===============================================\n")

        idx_input = input(
            "Enter the RAW voice index from the list above "
            "(or press Enter to keep current/default): "
        ).strip()

        if idx_input == "":
            print("\nKeeping current/default voice.\n")
            return

        try:
            idx = int(idx_input)
            voices_all = engine.getProperty("voices")
            if 0 <= idx < len(voices_all):
                engine.setProperty("voice", voices_all[idx].id)
                print(f"\nUsing voice [{idx}]: {voices_all[idx].id}\n")
            else:
                print("Invalid index. Keeping current/default voice.\n")
        except ValueError:
            print("Invalid input. Keeping current/default voice.\n")
        return

    for opt, label, fragment, quality, vtype, vid in resolved_presets:
        if choice == opt:
            engine.setProperty("voice", vid)
            print(f"\nUsing preset {opt}: [{quality} {vtype}] {label} → {vid}\n")
            return

    print("Unrecognized choice. Keeping default/system voice.\n")


def choose_speed():
    print("\n=== BuRtRoN.AI Speaking Speed ===")
    print("1) Slow        (~150)")
    print("2) Normal      (~185)  [default]")
    print("3) Fast        (~220)")
    print("4) Very Fast   (~260)")
    print("5) Custom value (enter a number like 200, 230, etc.)")
    print("ENTER) Keep default (185)")
    print("=================================\n")

    choice = input("Choose speed option (1-5), or press Enter for default: ").strip()

    if choice == "":
        rate = DEFAULT_RATE
        engine.setProperty("rate", rate)
        print(f"\nUsing default speaking rate: {rate}\n")
        return

    if choice == "1":
        rate = 150
    elif choice == "2":
        rate = 185
    elif choice == "3":
        rate = 220
    elif choice == "4":
        rate = 260
    elif choice == "5":
        custom = input("Enter custom rate (e.g. 180, 210, 250): ").strip()
        try:
            rate = int(custom)
        except ValueError:
            print("Invalid number. Keeping default rate.\n")
            rate = DEFAULT_RATE
    else:
        print("Unrecognized option. Keeping default rate.\n")
        rate = DEFAULT_RATE

    engine.setProperty("rate", rate)
    print(f"\nSpeaking rate set to: {rate}\n")


def _tts_worker(text: str):
    """
    Runs in a background thread to allow interruption via engine.stop().
    """
    global is_speaking
    with speak_lock:
        is_speaking = True
        print(f"\nAI speaking:\n{text}\n")
        engine.say(text)
        engine.runAndWait()
        is_speaking = False


def speak(text: str):
    """
    Launch TTS in a separate thread so Whisper can keep listening
    and we can interrupt with 'SOVRA STOP'.
    """
    t = threading.Thread(target=_tts_worker, args=(text,), daemon=True)
    t.start()

# ============================
# RECORDING + WHISPER STT
# ============================

def record_from_mic(seconds: float = CHUNK_SECONDS, samplerate: int = SAMPLE_RATE) -> np.ndarray:
    """
    Record audio from INPUT_DEVICE_INDEX for `seconds`.
    Returns a mono float32 numpy array at `samplerate`.
    """
    print(f"\n[Listening] ({seconds:.1f}s chunk) - speak now...\n")
    audio = sd.rec(
        int(seconds * samplerate),
        samplerate=samplerate,
        channels=1,
        dtype="float32",
        device=INPUT_DEVICE_INDEX
    )
    sd.wait()

    # audio shape: (samples, 1) -> flatten to (samples,)
    return audio.flatten()


def transcribe_audio(audio: np.ndarray, samplerate: int = SAMPLE_RATE) -> str:
    """
    Use local openai-whisper to transcribe an audio array.
    """
    if audio is None or len(audio) == 0:
        return ""

    result = whisper_model.transcribe(
        audio,
        language="en",
        fp16=False,
        verbose=False
    )
    text = (result.get("text") or "").strip()
    if text:
        print(f"[STT] Heard: {text}")
    else:
        print("[STT] (no speech recognized)")
    return text

# ============================
# VOICE COMMAND LOGIC
# ============================

START_COMMAND = "SOVRA TURN ON"
PAUSE_COMMAND = "SOVRA PAUSE"
STOP_COMMAND  = "SOVRA STOP"
WAKE_PREFIX   = "SOVRA"   # Any other query starting with 'SOVRA'

def handle_transcript(text: str, ai_enabled: bool) -> tuple[bool, bool]:
    """
    Handle a single recognized transcript string.
    Returns (updated_ai_enabled, just_replied_with_answer).
    """
    global is_speaking

    if not text:
        return ai_enabled, False

    upper = text.upper()
    just_replied = False

    # 0) Hard interrupt: SOVRA STOP
    if STOP_COMMAND in upper:
        if is_speaking:
            print("\n[VOICE CMD] SOVRA STOP → Interrupting speech.\n")
            engine.stop()     # stop current utterance
            is_speaking = False
        else:
            print("\n[VOICE CMD] SOVRA STOP heard, but SOVRA was not speaking.\n")
        # No new reply; just interruption.
        return ai_enabled, False

    # 1) Voice command: TURN ON
    if START_COMMAND in upper:
        if not ai_enabled:
            print("\n[VOICE CMD] SOVRA TURN ON → AI enabled.\n")
            speak("Sovra is now on and listening.")
        else:
            print("\n[VOICE CMD] SOVRA TURN ON → already on.\n")
        return True, True  # we DID speak a response (or at least queued it)

    # 2) Voice command: PAUSE
    if PAUSE_COMMAND in upper:
        if ai_enabled:
            print("\n[VOICE CMD] SOVRA PAUSE → AI paused.\n")
            speak("Sovra is now paused.")
            return False, True
        else:
            print("\n[VOICE CMD] SOVRA PAUSE → already paused.\n")
            return False, False

    # 3) Queries starting with "SOVRA ..."
    if upper.startswith(WAKE_PREFIX):
        # Strip "SOVRA" from the front, keep the rest as the question
        remaining = text[len(WAKE_PREFIX):].strip(" ,.!?")

        if not remaining:
            print("[WAKE] Heard 'SOVRA' but no actual question.\n")
            return ai_enabled, False

        if not ai_enabled:
            print("\n[WAKE] 'SOVRA ...' heard, but AI is paused. Say 'SOVRA TURN ON' first.\n")
            return ai_enabled, False

        print(f"\n[QUESTION] {remaining}\n")

        try:
            reply = ask_llm(remaining)
            print(f"\nAI: {reply}\n")
            speak(reply)
            just_replied = True
        except Exception as e:
            print("Error talking to LLM:", e)

        return ai_enabled, just_replied

    # 4) Anything else → ignore
    return ai_enabled, False

# ============================
# MAIN LOOP (ALWAYS LISTENING)
# ============================

if __name__ == "__main__":
    print("AI X Spaces Co-Host (Always Listening) online.\n")

    choose_voice()
    choose_speed()

    print("Voice Commands:")
    print(f"  - \"{START_COMMAND}\"  → Turn SOVRA ON (allow responses)")
    print(f"  - \"{PAUSE_COMMAND}\"  → Pause SOVRA (no responses)")
    print(f"  - \"{STOP_COMMAND}\"   → Immediately stop SOVRA mid-sentence")
    print(f"  - \"{WAKE_PREFIX} ...\" → Ask a question, e.g. 'SOVRA what is Bitcoin?'\n")
    print("Notes:")
    print("  - SOVRA listens continuously from your MacBook Pro Microphone.")
    print("  - Output voice goes to Ai + Speakers → BlackHole → X Spaces.")
    print("  - Press Ctrl+C in this terminal to stop the program.\n")

    ai_enabled = True

    try:
        while True:
            try:
                audio = record_from_mic(seconds=CHUNK_SECONDS)
                text = transcribe_audio(audio)

                ai_enabled, just_replied = handle_transcript(text, ai_enabled)

                # If we just sent a reply, chill a bit so she doesn't chain too fast.
                # This does NOT affect SOVRA STOP; that works via engine.stop().
                if just_replied:
                    time.sleep(COOLDOWN_AFTER_REPLY)
                else:
                    time.sleep(0.2)

            except KeyboardInterrupt:
                print("\n\n[EXIT] Keyboard interrupt received. Shutting down.\n")
                break
            except Exception as e:
                print("Error in main loop:", e)
                time.sleep(1.0)
    finally:
        engine.stop()
