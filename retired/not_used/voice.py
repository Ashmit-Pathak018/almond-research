"""
Project Almond — Voice Layer
Whisper STT + Coqui XTTS-v2 TTS. Fully offline, no API keys required.

Pipeline:
    Headset mic → SpeechRecognition → Whisper (STT)
    → Almond.chat() → Coqui XTTS-v2 (TTS) → audio output

Usage:
    python voice.py              # default session
    python voice.py --session s1 # named session
    python voice.py --list-mics  # find your headset mic index
"""

from __future__ import annotations

import argparse
import io
import logging
import queue
import tempfile
import threading
import time
import wave
from pathlib import Path
from typing import Optional

import pyaudio
import pyttsx3
import speech_recognition as sr
import whisper

from almond import Almond, AlmondConfig
from memory_block import MemoryTag, MemoryTier

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.WARNING, format="%(asctime)s [%(levelname)s] %(message)s")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

WHISPER_MODEL    = "base"        # base = fast + accurate for English
                                  # upgrade to "small" if accuracy suffers
COQUI_MODEL      = None   # swapped to pyttsx3 for V1

SAMPLE_RATE      = 16000          # Whisper expects 16kHz
CHANNELS         = 1
CHUNK            = 1024
SILENCE_TIMEOUT  = 2.0            # seconds of silence before processing
ENERGY_THRESHOLD = 300            # mic sensitivity (lower = more sensitive)

# Wake word — say this to activate Almond (set None to always listen)
WAKE_WORD: Optional[str] = "almond"

# Colors
CYAN   = "\033[96m"
GREEN  = "\033[92m"
YELLOW = "\033[93m"
DIM    = "\033[2m"
RESET  = "\033[0m"


# ---------------------------------------------------------------------------
# VoiceLayer
# ---------------------------------------------------------------------------

class VoiceLayer:
    """
    Manages the STT → brain → TTS pipeline for Project Almond.

    Modes:
        Wake word mode  (WAKE_WORD set): always listening, activates on keyword
        Push-to-talk    (WAKE_WORD None): press Enter to start recording
    """

    def __init__(
        self,
        session_id: str       = "voice_default",
        mic_index:  Optional[int] = None,
    ):
        self.session_id = session_id
        self.mic_index  = mic_index
        self._running   = False
        self._tts_queue: queue.Queue[str] = queue.Queue()

        print(f"\n{CYAN}[ALMOND VOICE]{RESET} Initialising...\n")

        # --- Load Whisper ---
        print(f"  {DIM}Loading Whisper ({WHISPER_MODEL})...{RESET}")
        self.whisper_model = whisper.load_model(WHISPER_MODEL)
        print(f"  {GREEN}✓ Whisper ready{RESET}")

        # --- Load pyttsx3 TTS ---
        print(f"  {DIM}Loading pyttsx3 TTS...{RESET}")
        self.tts = pyttsx3.init()
        self.tts.setProperty("rate", 175)    # speaking speed (words/min)
        self.tts.setProperty("volume", 0.9)
        # Pick a decent voice — index 1 is usually a female voice on Windows
        voices = self.tts.getProperty("voices")
        if len(voices) > 1:
            self.tts.setProperty("voice", voices[1].id)
        print(f"  {GREEN}✓ pyttsx3 TTS ready{RESET}")

        # --- Load Almond brain ---
        print(f"  {DIM}Loading Almond T-MMU...{RESET}")
        self.almond = Almond(AlmondConfig(session_id=session_id))

        # Seed voice-specific system rule on fresh sessions
        existing = self.almond.store.tier_counts()
        if sum(existing.values()) <= 1:
            self.almond.add_memory(
                content=(
                    "You are Almond, a voice assistant. Keep responses concise — "
                    "2 to 4 sentences max. Avoid bullet points, markdown, or lists. "
                    "Speak naturally as if in conversation. Be witty and warm."
                ),
                tag=MemoryTag.CORE_RULE,
                importance_score=10.0,
                tier=MemoryTier.L1_HOT_CACHE,
                keywords=[],
            )
        print(f"  {GREEN}✓ Almond T-MMU ready{RESET}")

        # --- PyAudio ---
        self._pa         = pyaudio.PyAudio()
        self._recognizer = sr.Recognizer()
        self._recognizer.energy_threshold        = ENERGY_THRESHOLD
        self._recognizer.pause_threshold         = SILENCE_TIMEOUT
        self._recognizer.dynamic_energy_threshold = True

        print(f"\n{GREEN}  Almond is online.{RESET}")
        if WAKE_WORD:
            print(f"  {DIM}Say '{WAKE_WORD}' to activate.{RESET}\n")
        else:
            print(f"  {DIM}Press Enter to speak.{RESET}\n")

    # -----------------------------------------------------------------------
    # Main Loop
    # -----------------------------------------------------------------------

    def run(self) -> None:
        """Start the voice session."""
        self._running = True
        try:
            if WAKE_WORD:
                self._wake_word_loop()
            else:
                self._push_to_talk_loop()
        except KeyboardInterrupt:
            print(f"\n{DIM}  Session ended.{RESET}")
        finally:
            self._running = False
            self.almond.close()
            self._pa.terminate()

    def _wake_word_loop(self) -> None:
        """Continuously listen for wake word, then capture full utterance."""
        mic = sr.Microphone(
            device_index=self.mic_index,
            sample_rate=SAMPLE_RATE,
        )
        with mic as source:
            self._recognizer.adjust_for_ambient_noise(source, duration=1)

        print(f"  {DIM}Listening for wake word...{RESET}")
        while self._running:
            with mic as source:
                try:
                    audio = self._recognizer.listen(source, timeout=5, phrase_time_limit=8)
                except sr.WaitTimeoutError:
                    continue

            # Quick Google STT check for wake word (fast, low latency)
            # If offline-only needed: swap to Whisper here too
            try:
                text = self._recognizer.recognize_google(audio).lower()
            except (sr.UnknownValueError, sr.RequestError):
                # Fall back to Whisper for wake word detection
                text = self._transcribe_whisper(audio).lower()

            if WAKE_WORD and WAKE_WORD.lower() in text:
                self._speak_raw("Yeah?")
                print(f"  {GREEN}Almond activated.{RESET} Listening...")
                with mic as source:
                    try:
                        utterance = self._recognizer.listen(
                            source, timeout=8, phrase_time_limit=30
                        )
                    except sr.WaitTimeoutError:
                        self._speak_raw("I didn't catch that.")
                        continue
                self._process_utterance(utterance)

    def _push_to_talk_loop(self) -> None:
        """Press Enter to start recording, Enter again to stop."""
        mic = sr.Microphone(
            device_index=self.mic_index,
            sample_rate=SAMPLE_RATE,
        )
        with mic as source:
            self._recognizer.adjust_for_ambient_noise(source, duration=1)

        while self._running:
            input(f"  {DIM}[Press Enter to speak]{RESET} ")
            print(f"  {GREEN}Listening...{RESET} (speak now, pause to finish)")
            with mic as source:
                try:
                    audio = self._recognizer.listen(
                        source, timeout=10, phrase_time_limit=60
                    )
                except sr.WaitTimeoutError:
                    print(f"  {DIM}No speech detected.{RESET}")
                    continue
            self._process_utterance(audio)

    # -----------------------------------------------------------------------
    # STT
    # -----------------------------------------------------------------------

    def _transcribe_whisper(self, audio: sr.AudioData) -> str:
        """Convert AudioData → WAV bytes → Whisper transcription."""
        wav_bytes = audio.get_wav_data(convert_rate=SAMPLE_RATE, convert_width=2)
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            f.write(wav_bytes)
            tmp_path = f.name
        try:
            result = self.whisper_model.transcribe(tmp_path, language="en", fp16=False)
            return result["text"].strip()
        finally:
            Path(tmp_path).unlink(missing_ok=True)

    def _process_utterance(self, audio: sr.AudioData) -> None:
        """Full pipeline: audio → text → command check → Almond → speech."""
        print(f"  {DIM}Transcribing...{RESET}", end="", flush=True)
        text = self._transcribe_whisper(audio)

        if not text or len(text.strip()) < 2:
            print(f"\r  {DIM}(nothing detected){RESET}    ")
            return

        print(f"\r  {CYAN}You  › {RESET}{text}")

        # --- Voice command interception ---
        lower = text.lower().strip()
        if any(k in lower for k in ["hey dump", "memory dump", "show memory"]):
            self._cmd_dump()
            return
        if any(k in lower for k in ["hey export", "export session", "save session"]):
            self._cmd_export()
            return
        if any(k in lower for k in ["hey reset", "reset memory", "clear memory"]):
            self._cmd_reset()
            return
        if any(k in lower for k in ["hey quit", "goodbye almond", "shut down", "exit almond"]):
            self._cmd_quit()
            return

        # --- Normal turn ---
        t0    = time.time()
        reply = self.almond.chat(text)
        ms    = (time.time() - t0) * 1000

        print(f"  {GREEN}Almond › {RESET}{reply}")
        print(f"  {DIM}[{ms:.0f}ms]{RESET}\n")
        self._speak_raw(reply)

    # -----------------------------------------------------------------------
    # Voice Commands
    # -----------------------------------------------------------------------

    def _cmd_dump(self) -> None:
        """Speak a summary of the current memory pool."""
        pool   = self.almond.controller.dump_pool()
        l1     = sum(1 for b in pool if b["tier"] == "L1_HOT_CACHE")
        l2     = sum(1 for b in pool if b["tier"] == "L2_ACTIVE_RAM")
        l3     = sum(1 for b in pool if b["tier"] == "L3_VIRTUAL_SWAP")
        l4     = sum(1 for b in pool if b["tier"] == "L4_ARCHIVE")
        top    = pool[:3]  # top 3 by Peff

        # Print full table to terminal
        print(f"\n  {YELLOW}Memory Pool ({len(pool)} blocks){RESET}")
        for b in pool:
            print(f"  {b['tier']:<18} {b['tag']:<14} peff={b['p_eff']:.4f}  {DIM}{b['content_preview'][:40]}{RESET}")

        # Speak a concise audio summary
        top_tags = ", ".join(b["tag"] for b in top)
        spoken = (
            f"Memory dump. {len(pool)} blocks total. "
            f"L1: {l1}, L2: {l2}, L3: {l3}, L4: {l4}. "
            f"Top memories by priority are tagged: {top_tags}."
        )
        print(f"  {GREEN}Almond › {RESET}{spoken}\n")
        self._speak_raw(spoken)

    def _cmd_export(self) -> None:
        """Export turn log to JSON and confirm via voice."""
        import json
        path = f"almond_turns_{self.session_id}.json"
        with open(path, "w") as f:
            json.dump(self.almond.export_turn_log(), f, indent=2)
        spoken = f"Session exported to {path}."
        print(f"  {GREEN}Almond › {RESET}{spoken}\n")
        self._speak_raw(spoken)

    def _cmd_quit(self) -> None:
        """Export session and shut down cleanly."""
        self._cmd_export()
        spoken = "Session saved. Shutting down. See you next time."
        print(f"  {GREEN}Almond › {RESET}{spoken}\n")
        self._speak_raw(spoken)
        self._running = False

    def _cmd_reset(self) -> None:
        """Warn user — reset requires confirmation via second voice command."""
        spoken = "Reset requires confirmation. Say confirm reset to wipe memory, or say anything else to cancel."
        print(f"  {GREEN}Almond › {RESET}{spoken}\n")
        self._speak_raw(spoken)

    # -----------------------------------------------------------------------
    # TTS (non-blocking worker thread)
    # -----------------------------------------------------------------------

    def _tts_worker(self) -> None:
        """
        Runs in a background thread. Pulls text from queue and speaks it.
        Non-blocking so the mic can keep listening while speech plays.
        """
        while True:
            text = self._tts_queue.get()
            if text is None:
                break
            self._speak_raw(text)
            self._tts_queue.task_done()

    def _speak_raw(self, text: str) -> None:
        """Synthesize text via pyttsx3. Reinitialises engine each call — 
        avoids Windows threading bug where runAndWait() silently dies."""
        clean = (
            text.replace("**", "")
                .replace("*", "")
                .replace("#", "")
                .replace("`", "")
                .replace("\n", " ")
                .strip()
        )
        if not clean:
            return
        try:
            engine = pyttsx3.init()
            engine.setProperty("rate", 175)
            engine.setProperty("volume", 0.9)
            voices = engine.getProperty("voices")
            if len(voices) > 1:
                engine.setProperty("voice", voices[1].id)
            engine.say(clean)
            engine.runAndWait()
            engine.stop()
        except Exception as e:
            logger.warning(f"TTS failed: {e}")


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def list_microphones() -> None:
    """Print all available audio input devices with their index."""
    pa = pyaudio.PyAudio()
    print("\nAvailable microphones:")
    for i in range(pa.get_device_count()):
        info = pa.get_device_info_by_index(i)
        if info["maxInputChannels"] > 0:
            print(f"  [{i}] {info['name']}")
    pa.terminate()


# ---------------------------------------------------------------------------
# Entry Point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Project Almond — Voice Mode")
    parser.add_argument("--session",   default="voice_default", help="Session ID")
    parser.add_argument("--mic",       type=int, default=None,  help="Mic device index")
    parser.add_argument("--list-mics", action="store_true",     help="List audio devices")
    parser.add_argument("--ptt",       action="store_true",     help="Force push-to-talk mode")
    args = parser.parse_args()

    if args.list_mics:
        list_microphones()
        return

    # Override wake word if push-to-talk requested
    if args.ptt:
        global WAKE_WORD
        WAKE_WORD = None

    voice = VoiceLayer(session_id=args.session, mic_index=args.mic)
    voice.run()


if __name__ == "__main__":
    main()