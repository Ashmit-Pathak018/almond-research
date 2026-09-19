"""
silver_voice.py — Silver Voice Assistant (Phase 2)

Flow:
  1. Wake word listener ("Silver") via Google STT
  2. Match command against silver_actions.py playbook (pre-built, safe code)
  3a. If matched → speak what Silver will do → VOICE approval → run action
  3b. If NOT matched → send to Open Interpreter (LM Studio)
        → OI generates ONE code block
        → Silver SPEAKS "Should I proceed?" → listens for yes/no
        → monkeypatched input() handles the OI approval prompt
"""

import builtins
import io
import logging
import os
import re
import sys
import time
import threading
import asyncio
import numpy as np

# ── Fix CUDA DLL loading on Windows ──────────────────────────────────────────
import site
import os

_dll_paths = []
for site_pkg in site.getsitepackages():
    cublas_path = os.path.join(site_pkg, "nvidia", "cublas", "bin")
    cudnn_path  = os.path.join(site_pkg, "nvidia", "cudnn", "bin")
    nvrtc_path  = os.path.join(site_pkg, "nvidia", "cuda_nvrtc", "bin")
    
    for p in [cublas_path, cudnn_path, nvrtc_path]:
        if os.path.exists(p):
            _dll_paths.append(p)
            os.add_dll_directory(p)

# Append to PATH for C++ libraries that don't respect add_dll_directory
if _dll_paths:
    os.environ["PATH"] = os.pathsep.join(_dll_paths) + os.pathsep + os.environ.get("PATH", "")

from faster_whisper import WhisperModel

# ── Set LM Studio env vars BEFORE importing open-interpreter ──────────────────
os.environ["OPENAI_API_KEY"]  = "lm-studio"
os.environ["OPENAI_API_BASE"] = "http://localhost:1234/v1"

import pyaudio
import speech_recognition as sr
import interpreter as oi

from silver_actions import match_action

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("silver")

# ── Configuration ─────────────────────────────────────────────────────────────
WAKE_WORDS    = {"silver", "hey silver", "hi silver", "hello"}
TTS_VOICE     = "en-US-AvaNeural"  # Caring, Expressive, Pleasant female voice
TTS_RATE      = "+5%"
ENERGY_PAUSE  = 0.8
_SAMPLE_RATE  = 22050

YES_WORDS = {"yes", "yeah", "yep", "sure", "do it", "proceed", "go ahead",
             "go", "run it", "run", "confirm", "ok", "okay", "yup", "correct"}
NO_WORDS  = {"no", "nope", "cancel", "stop", "don't", "dont", "abort",
             "negative", "skip", "never mind", "nevermind"}

# ── Configure Open Interpreter ────────────────────────────────────────────────
oi.interpreter.llm.model          = "openai/local-model"
oi.interpreter.llm.api_base       = "http://localhost:1234/v1"
oi.interpreter.llm.api_key        = "lm-studio"
oi.interpreter.llm.max_tokens     = 2000
oi.interpreter.llm.context_window = 8192
oi.interpreter.auto_run           = False

oi.interpreter.system_message += (
    "\nYour name is Silver. You are a warm, caring, and highly capable female companion running locally on Windows."
    "\nYou act as a supportive friend who helps the user with whatever they need, while being technically adept."
    "\nYou have full computer control via Python and shell execution."
    "\n"
    "\nRULES (follow strictly):"
    "\n1. Write ONLY ONE code block per response. Consolidate everything into a single script."
    "\n2. Keep explanations to 1-2 sentences max. You are a voice assistant. Speak conversationally and warmly."
    "\n3. Never write code in multiple separate blocks — always merge into one."
    "\n4. For YouTube, ALWAYS use this pattern (never guess URLs):"
    "\n   import urllib.request, urllib.parse, re, webbrowser"
    "\n   q = urllib.parse.quote('SONG TITLE')"
    "\n   html = urllib.request.urlopen(f'https://www.youtube.com/results?search_query={q}', timeout=5).read().decode()"
    "\n   ids = re.findall(r'watch\\?v=(\\S{11})', html)"
    "\n   if ids: webbrowser.open(f'https://www.youtube.com/watch?v={ids[0]}')"
)

# ── Configure Faster Whisper (Local STT) ─────────────────────────────────────
log.info("Loading local Whisper model (base.en) onto GPU...")
_whisper_model = WhisperModel("base.en", device="cuda", compute_type="float16")
log.info("Whisper model loaded.")


# ── Global speech recognizer (shared) ────────────────────────────────────────
_recognizer = sr.Recognizer()
_recognizer.energy_threshold = 300
_recognizer.pause_threshold  = 1.5   # 1.5s pause won't cut you off
_recognizer.non_speaking_duration = 1.0 
_recognizer.dynamic_energy_threshold = True
_mic = sr.Microphone()

# ── Approval state (set by monkeypatched input, read by voice listener) ───────
_approval_event   = threading.Event()
_approval_result  = {"value": None}   # "y" or "n"
_waiting_approval = threading.Event() # signals the voice thread to listen


# ─────────────────────────────────────────────────────────────────────────────
# TTS
# ─────────────────────────────────────────────────────────────────────────────

def speak(text: str) -> None:
    """Speak text via Edge TTS synchronously."""
    import edge_tts
    from pydub import AudioSegment

    # Strip markdown / code blocks / asterisks
    text = re.sub(r"```.*?```", "", text, flags=re.DOTALL)
    text = re.sub(r"`[^`]*`", "", text)
    text = re.sub(r"\*{1,3}([^*]+)\*{1,3}", r"\1", text)
    text = re.sub(r"\s+", " ", text).strip()

    if not text:
        return

    log.info("🔊 Speaking: %s", text[:100])

    async def _synth() -> bytes:
        comm = edge_tts.Communicate(text, TTS_VOICE, rate=TTS_RATE)
        chunks = []
        async for item in comm.stream():
            if item["type"] == "audio":
                chunks.append(item["data"])
        return b"".join(chunks)

    loop = asyncio.new_event_loop()
    try:
        data = loop.run_until_complete(_synth())
    finally:
        loop.close()

    if not data:
        return

    from pydub import AudioSegment
    audio = AudioSegment.from_mp3(io.BytesIO(data))
    audio = audio.set_channels(1).set_sample_width(2).set_frame_rate(_SAMPLE_RATE)

    pa = pyaudio.PyAudio()
    s = pa.open(format=pyaudio.paInt16, channels=1, rate=_SAMPLE_RATE, output=True)
    s.write(audio.raw_data)
    s.stop_stream()
    s.close()
    pa.terminate()


# ─────────────────────────────────────────────────────────────────────────────
# STT helpers
# ─────────────────────────────────────────────────────────────────────────────

def _listen(timeout: float = 6.0, phrase_limit: float = 12.0) -> str | None:
    """Record one utterance and transcribe via local Faster Whisper."""
    try:
        with _mic as source:
            audio = _recognizer.listen(source, timeout=timeout, phrase_time_limit=phrase_limit)
        
        # Convert audio from 16-bit PCM to fp32 normalized numpy array for whisper
        pcm_data = audio.get_raw_data(convert_rate=16000, convert_width=2)
        audio_np = np.frombuffer(pcm_data, np.int16).astype(np.float32) / 32768.0
        
        # Transcribe with local GPU
        segments, _ = _whisper_model.transcribe(audio_np, beam_size=1)
        text = " ".join([s.text for s in segments]).strip().lower()
        if text:
            log.info("Whisper heard: '%s'", text)
        return text
    except sr.WaitTimeoutError:
        return None
    except sr.UnknownValueError:
        return None
    except Exception as e:
        log.warning("Whisper STT error: %s", e)
        return None


def _contains_wake_word(text: str | None) -> bool:
    return bool(text) and any(w in text for w in WAKE_WORDS)


def _strip_wake_word(text: str) -> str:
    for w in sorted(WAKE_WORDS, key=len, reverse=True):
        if text.startswith(w):
            return text[len(w):].lstrip(" ,.!?").strip()
    return text.strip()


def _ask_voice_approval() -> bool:
    """
    Ask for approval. VOICE and KEYBOARD run in parallel — whichever comes first wins.
      Keyboard: Y / Enter = yes    N / Esc = no
      Voice:    'yes/yeah/sure/...' = yes    'no/cancel/...' = no
    """
    import msvcrt
    import threading

    result = {"value": None}
    done   = threading.Event()

    # ── Voice listener thread ──────────────────────────────────────────────
    def _voice_thread():
        # Use a fresh, very sensitive recognizer (not the shared one)
        r = sr.Recognizer()
        r.energy_threshold        = 80    # very low — hears quiet speech
        r.dynamic_energy_threshold = False # don't let it ramp up
        r.pause_threshold          = 0.5

        time.sleep(0.5)  # let TTS audio fully settle before opening mic

        for attempt in range(3):
            if done.is_set():
                return
            try:
                with sr.Microphone() as source:
                    audio = r.listen(source, timeout=5, phrase_time_limit=3)
                
                # Convert PCM to fp32 numpy array for whisper
                pcm_data = audio.get_raw_data(convert_rate=16000, convert_width=2)
                audio_np = np.frombuffer(pcm_data, np.int16).astype(np.float32) / 32768.0
                
                segments, _ = _whisper_model.transcribe(audio_np, beam_size=1)
                text = " ".join([s.text for s in segments]).strip().lower()
                log.info("Voice approval heard (attempt %d): '%s'", attempt + 1, text)
                if done.is_set():
                    return
                if any(w in text for w in YES_WORDS):
                    result["value"] = True
                    done.set()
                    return
                elif any(w in text for w in NO_WORDS):
                    result["value"] = False
                    done.set()
                    return
                # Heard something but not yes/no — try again silently
            except (sr.WaitTimeoutError, sr.UnknownValueError):
                pass
            except sr.RequestError as e:
                log.warning("STT error in approval: %s", e)
                return

    # ── Keyboard listener thread ───────────────────────────────────────────
    def _keyboard_thread():
        while not done.is_set():
            if msvcrt.kbhit():
                key = msvcrt.getch()
                if key in (b'y', b'Y', b'\r', b'\n'):
                    result["value"] = True
                    done.set()
                    return
                elif key in (b'n', b'N', b'\x1b'):  # N or Esc
                    result["value"] = False
                    done.set()
                    return
            time.sleep(0.03)

    speak("Should I proceed?")
    print("\n  ↳  Say 'yes' or press  [ Y / Enter ]  to confirm")
    print("     Say 'no'  or press  [ N / Esc   ]  to cancel\n")

    t_voice = threading.Thread(target=_voice_thread,    daemon=True)
    t_key   = threading.Thread(target=_keyboard_thread, daemon=True)
    t_voice.start()
    t_key.start()

    # Wait up to 15 seconds for either to respond
    done.wait(timeout=15)

    if result["value"] is None:
        speak("No response — cancelling to be safe.")
        return False
    elif result["value"]:
        speak("Got it, running.")
        return True
    else:
        speak("OK, cancelled.")
        return False


# ─────────────────────────────────────────────────────────────────────────────
# Voice approval hook for Open Interpreter
# ─────────────────────────────────────────────────────────────────────────────

_original_input = builtins.input

def _voice_approval_input(prompt: str = "") -> str:
    """
    Replaces builtins.input so OI's 'run this code? [y/n]' prompt
    is handled by Silver's voice instead of keyboard.
    """
    # Only intercept if it looks like OI's code-approval prompt
    prompt_l = (prompt or "").lower()
    is_approval = any(x in prompt_l for x in ["run code", "[y/n]", "would you like", "execute", "(y/n)", "y/n"])

    if not is_approval:
        # Not a code approval — fall back to normal terminal input
        return _original_input(prompt)

    print(prompt)  # still show the prompt in terminal

    approved = _ask_voice_approval()
    answer = "y" if approved else "n"
    print(f"[Silver voice] → '{answer}'")
    return answer

# Install the hook
builtins.input = _voice_approval_input


# ─────────────────────────────────────────────────────────────────────────────
# Action playbook handler
# ─────────────────────────────────────────────────────────────────────────────

def _run_action_with_approval(command: str) -> bool:
    """
    Tries to match command against silver_actions.py.
    If matched: asks for voice approval, then executes.
    Returns True if an action was found and handled.
    """
    action = match_action(command)
    if action is None:
        return False

    log.info("Action matched: %s | %s", action.func.__name__, action.kwargs)
    speak(f"I'll {action.description}.")
    approved = _ask_voice_approval()
    if approved:
        try:
            result = action.func(**action.kwargs)
            speak(result or "Done.")
        except Exception as e:
            log.error("Action error: %s", e)
            speak(f"Something went wrong: {str(e)[:80]}")
    return True


# ─────────────────────────────────────────────────────────────────────────────
# Open Interpreter handler (for novel / complex tasks)
# ─────────────────────────────────────────────────────────────────────────────

def _send_to_oi(command: str) -> None:
    """
    Send the command to Open Interpreter.
    OI will write ONE code block. When it hits input() for approval,
    the monkeypatched version handles it via voice.
    The final text response is spoken back.
    """
    log.info("Sending to Open Interpreter: %s", command)
    print(f"\n🤖 Sending to Silver (OI): {command}\n")

    try:
        messages = oi.interpreter.chat(command, display=True)
        # Speak the last assistant text message
        for m in reversed(messages):
            if m.get("role") == "assistant" and m.get("type") == "message":
                txt = m.get("content", "").strip()
                if txt:
                    speak(txt)
                break
    except Exception as e:
        log.error("Open Interpreter error: %s", e)
        speak("Sorry, something went wrong. Check the terminal for details.")


# ─────────────────────────────────────────────────────────────────────────────
# Main dispatcher
# ─────────────────────────────────────────────────────────────────────────────

def handle_command(command: str) -> None:
    """Route command: playbook first, then OI."""
    print(f"\n🎤 You said: {command}\n")

    # 1. Try playbook
    if _run_action_with_approval(command):
        return

    # 2. Fall back to Open Interpreter
    log.info("No action matched — routing to Open Interpreter.")
    speak("I don't have a preset for that. Let me figure it out.")
    _send_to_oi(command)


# ─────────────────────────────────────────────────────────────────────────────
# Wake word loop
# ─────────────────────────────────────────────────────────────────────────────

def _wake_word_loop() -> None:
    log.info("Calibrating microphone for ambient noise (2s)...")
    with _mic as source:
        _recognizer.adjust_for_ambient_noise(source, duration=2)
    log.info("Calibration done. Energy threshold: %.0f", _recognizer.energy_threshold)

    speak("Silver is ready. Say 'Hey Silver' to wake me up.")
    print("\n" + "="*55)
    print("  👂  Silver is listening for: 'Silver' / 'Hey Silver'")
    print("="*55 + "\n")

    while True:
        try:
            # ── Phase 1: listen for wake word ────────────────────────
            text = _listen(timeout=60, phrase_limit=10)
            if not _contains_wake_word(text):
                if text:
                    log.debug("Ignored (no wake word): %s", text)
                continue

            remaining = _strip_wake_word(text or "")

            # ── Phase 2: get the actual command ───────────────────────
            if remaining and len(remaining.split()) >= 2:
                # Command was in the same utterance: "Silver open Chrome"
                handle_command(remaining)
            else:
                print("\n⚡ Wake word detected! Listening for your command...")
                speak("Yes?")
                cmd = _listen(timeout=8, phrase_limit=15)
                if cmd:
                    full = (remaining + " " + cmd).strip()
                    handle_command(full)
                else:
                    speak("I didn't catch that. Say 'Silver' again when ready.")

        except KeyboardInterrupt:
            log.info("Shutting down Silver.")
            speak("Goodbye!")
            sys.exit(0)
        except Exception as e:
            log.error("Wake loop error: %s", e, exc_info=True)
            time.sleep(1)


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    print("=" * 55)
    print("  🤖  A V A  —  Voice Companion  (Phase 2)")
    print("=" * 55)
    print(f"  LM Studio : {oi.interpreter.llm.api_base}")
    print(f"  Auto-run  : {'ON' if oi.interpreter.auto_run else 'OFF — voice approval required'}")
    print(f"  TTS Voice : {TTS_VOICE}")
    print(f"  Wake Words: {', '.join(sorted(WAKE_WORDS))}")
    print(f"  Playbook  : silver_actions.py  ({_count_actions()} actions loaded)")
    print("=" * 55 + "\n")
    _wake_word_loop()


def _count_actions() -> int:
    try:
        from silver_actions import REGISTRY
        return len(REGISTRY)
    except Exception:
        return 0


if __name__ == "__main__":
    main()
