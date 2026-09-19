"""
silver_actions.py — Silver's Pre-Built Action Playbook

Silver checks this file FIRST before asking Open Interpreter to generate new code.
All actions here are tested, safe, and reliable on Windows.

HOW IT WORKS:
  1. match_action(text) tries to match the user's command to a known action.
  2. If matched, returns an ActionResult(description, func, kwargs).
  3. silver_voice.py gets voice approval, then calls func(**kwargs).
  4. If no match, OI generates code (with voice approval before execution).

TO ADD A NEW ACTION:
  1. Write a function below (see examples).
  2. Add keywords + function reference to the REGISTRY list at the bottom.
"""

import os
import re
import subprocess
import webbrowser
import urllib.parse
import urllib.request
import logging
from dataclasses import dataclass
from typing import Callable, Optional

log = logging.getLogger("silver.actions")


# ─────────────────────────────────────────────────────────────────────────────
# Return type
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ActionResult:
    description: str          # Silver speaks this before asking for approval
    func: Callable            # The actual function to call
    kwargs: dict              # Arguments to pass to func


# ─────────────────────────────────────────────────────────────────────────────
# Utility helpers
# ─────────────────────────────────────────────────────────────────────────────

def _open_app(path_or_name: str) -> str:
    """Open an application by its executable path or name."""
    try:
        os.startfile(path_or_name)
        return f"Opened {path_or_name}."
    except Exception:
        try:
            subprocess.Popen(path_or_name, shell=True)
            return f"Launched {path_or_name}."
        except Exception as e:
            return f"Could not open {path_or_name}: {e}"


def _youtube_play(query: str) -> str:
    """Search YouTube and open the first video result directly."""
    try:
        encoded = urllib.parse.quote(query)
        html = urllib.request.urlopen(
            f"https://www.youtube.com/results?search_query={encoded}", timeout=6
        ).read().decode()
        ids = re.findall(r"watch\?v=(\S{11})", html)
        if ids:
            url = f"https://www.youtube.com/watch?v={ids[0]}"
            webbrowser.open(url)
            return f"Playing '{query}' on YouTube."
        else:
            webbrowser.open(f"https://www.youtube.com/results?search_query={encoded}")
            return f"Opened YouTube search for '{query}'."
    except Exception as e:
        return f"YouTube error: {e}"


def _web_search(query: str, engine: str = "google") -> str:
    """Open a web search in the default browser."""
    engines = {
        "google": "https://www.google.com/search?q=",
        "bing":   "https://www.bing.com/search?q=",
        "youtube":"https://www.youtube.com/results?search_query=",
    }
    base = engines.get(engine, engines["google"])
    webbrowser.open(base + urllib.parse.quote(query))
    return f"Searching {engine} for '{query}'."


def _take_screenshot(filename: str = "") -> str:
    """Take a screenshot and save it to Desktop."""
    import pyautogui, datetime
    desktop = os.path.join(os.path.expanduser("~"), "Desktop")
    if not filename:
        filename = f"silver_screenshot_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
    path = os.path.join(desktop, filename)
    pyautogui.screenshot(path)
    return f"Screenshot saved to {path}."


def _type_text(text: str, delay: float = 0.05) -> str:
    """Type text at the current cursor position."""
    import pyautogui, time
    time.sleep(0.5)   # small delay so user can click into target window
    pyautogui.typewrite(text, interval=delay)
    return f"Typed: {text[:40]}{'...' if len(text) > 40 else ''}"


def _volume_up(steps: int = 5) -> str:
    """Increase system volume."""
    import pyautogui
    for _ in range(steps):
        pyautogui.press("volumeup")
    return f"Volume increased by {steps} steps."


def _volume_down(steps: int = 5) -> str:
    """Decrease system volume."""
    import pyautogui
    for _ in range(steps):
        pyautogui.press("volumedown")
    return f"Volume decreased by {steps} steps."


def _volume_mute() -> str:
    """Toggle mute."""
    import pyautogui
    pyautogui.press("volumemute")
    return "Toggled mute."


def _media_play_pause() -> str:
    """Toggle play/pause."""
    import pyautogui
    pyautogui.press("playpause")
    return "Toggled play/pause."


def _media_next_track() -> str:
    """Skip to next track."""
    import pyautogui
    pyautogui.press("nexttrack")
    return "Skipped to next track."


def _media_prev_track() -> str:
    """Go to previous track."""
    import pyautogui
    pyautogui.press("prevtrack")
    return "Went to previous track."


def _show_desktop() -> str:
    """Minimize all windows and show desktop."""
    import pyautogui
    pyautogui.hotkey("win", "d")
    return "Showing desktop."


def _lock_screen() -> str:
    """Lock the Windows session."""
    import pyautogui
    pyautogui.hotkey("win", "l")
    return "Screen locked."


def _open_file_explorer(path: str = "") -> str:
    """Open File Explorer at a given path (or default)."""
    target = path or os.path.expanduser("~")
    subprocess.Popen(["explorer.exe", target])
    return f"Opened File Explorer at {target}."


def _create_text_file(filename: str, content: str = "") -> str:
    """Create a text file on the Desktop."""
    desktop = os.path.join(os.path.expanduser("~"), "Desktop")
    path = os.path.join(desktop, filename if filename.endswith(".txt") else filename + ".txt")
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    os.startfile(path)
    return f"Created and opened {path}."


def _open_notepad(filepath: str = "") -> str:
    """Open Notepad, optionally with a file."""
    cmd = ["notepad.exe"]
    if filepath:
        cmd.append(filepath)
    subprocess.Popen(cmd)
    return "Opened Notepad."


def _open_calculator() -> str:
    subprocess.Popen("calc.exe")
    return "Opened Calculator."


def _open_task_manager() -> str:
    subprocess.Popen("taskmgr.exe")
    return "Opened Task Manager."


def _open_settings() -> str:
    subprocess.Popen(["start", "ms-settings:"], shell=True)
    return "Opened Windows Settings."


def _open_chrome(url: str = "") -> str:
    paths = [
        r"C:\Program Files\Google\Chrome\Application\chrome.exe",
        r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
    ]
    for p in paths:
        if os.path.exists(p):
            cmd = [p]
            if url:
                cmd.append(url)
            subprocess.Popen(cmd)
            return f"Opened Chrome{' at ' + url if url else ''}."
    # fallback via webbrowser
    webbrowser.open(url or "https://www.google.com")
    return "Opened browser."


def _open_spotify() -> str:
    paths = [
        os.path.join(os.environ.get("APPDATA", ""), "Spotify", "Spotify.exe"),
        r"C:\Program Files\WindowsApps\SpotifyAB.SpotifyMusic_*\Spotify.exe",
    ]
    for p in paths:
        if os.path.exists(p):
            subprocess.Popen([p])
            return "Opened Spotify."
    subprocess.Popen(["start", "spotify:"], shell=True)
    return "Launched Spotify."


def _open_vscode(path: str = "") -> str:
    cmd = ["code"]
    if path:
        cmd.append(path)
    try:
        subprocess.Popen(cmd)
        return "Opened VS Code."
    except FileNotFoundError:
        return "VS Code not found. Make sure it's in PATH."


def _open_discord() -> str:
    appdata = os.environ.get("LOCALAPPDATA", "")
    exe = os.path.join(appdata, "Discord", "Update.exe")
    if os.path.exists(exe):
        subprocess.Popen([exe, "--processStart", "Discord.exe"])
        return "Opened Discord."
    return "Discord not found at expected path."


def _open_steam() -> str:
    paths = [
        r"C:\Program Files (x86)\Steam\steam.exe",
        r"C:\Program Files\Steam\steam.exe",
    ]
    for p in paths:
        if os.path.exists(p):
            subprocess.Popen([p])
            return "Opened Steam."
    subprocess.Popen(["start", "steam:"], shell=True)
    return "Launched Steam."


def _paste_clipboard() -> str:
    """Paste clipboard contents at current cursor position."""
    import pyautogui
    pyautogui.hotkey("ctrl", "v")
    return "Pasted clipboard."


def _copy_selection() -> str:
    """Copy current selection to clipboard."""
    import pyautogui
    pyautogui.hotkey("ctrl", "c")
    return "Copied selection."


def _close_window() -> str:
    """Close the current active window."""
    import pyautogui
    pyautogui.hotkey("alt", "F4")
    return "Closed active window."


def _sleep_computer() -> str:
    """Put the computer to sleep."""
    subprocess.run(["rundll32.exe", "powrprof.dll,SetSuspendState", "0,1,0"])
    return "Going to sleep."


def _shutdown_computer() -> str:
    """Shutdown the computer (60s grace)."""
    subprocess.run(["shutdown", "/s", "/t", "60"])
    return "Shutting down in 60 seconds. Say 'Silver cancel shutdown' to abort."


def _cancel_shutdown() -> str:
    subprocess.run(["shutdown", "/a"])
    return "Shutdown cancelled."


def _restart_computer() -> str:
    subprocess.run(["shutdown", "/r", "/t", "60"])
    return "Restarting in 60 seconds."


def _open_task_view() -> str:
    import pyautogui
    pyautogui.hotkey("win", "tab")
    return "Opened Task View."


def _open_spotify_search(query: str) -> str:
    """Open Spotify and search for a song (via URI)."""
    uri = f"spotify:search:{urllib.parse.quote(query)}"
    subprocess.Popen(["start", uri], shell=True)
    return f"Searching Spotify for '{query}'."


# ─────────────────────────────────────────────────────────────────────────────
# REGISTRY
# Each entry: (keyword_list, description_template, func, arg_extractor)
# arg_extractor(text) → dict  (extracts arguments from the raw command)
# ─────────────────────────────────────────────────────────────────────────────

def _extract_after(text: str, *keywords) -> str:
    """Extract the portion of text after the first matching keyword."""
    text_l = text.lower()
    for kw in keywords:
        idx = text_l.find(kw)
        if idx != -1:
            return text[idx + len(kw):].strip(" ,.!?")
    return text


REGISTRY = [
    # ── YouTube ──────────────────────────────────────────────────────────────
    {
        "keywords": ["play", "youtube"],
        "description": lambda q: f"Play '{q}' on YouTube",
        "func": _youtube_play,
        "extract": lambda t: {"query": _extract_after(t, "play", "youtube", "put on", "open")},
    },
    # ── Spotify search ────────────────────────────────────────────────────────
    {
        "keywords": ["spotify"],
        "description": lambda q: f"Search Spotify for '{q}'",
        "func": _open_spotify_search,
        "extract": lambda t: {"query": _extract_after(t, "play", "search for", "search").replace("on spotify", "").replace("in spotify", "").strip(" ,.")},
    },
    # ── Web search ────────────────────────────────────────────────────────────
    {
        "keywords": ["search", "google", "look up", "look up", "find online"],
        "description": lambda q: f"Google search for '{q}'",
        "func": _web_search,
        "extract": lambda t: {"query": _extract_after(t, "search for", "google", "search", "look up", "find")},
    },
    # ── Chrome ───────────────────────────────────────────────────────────────
    {
        "keywords": ["open chrome", "launch chrome", "start chrome"],
        "description": lambda: "Open Google Chrome",
        "func": _open_chrome,
        "extract": lambda t: {"url": _extract_after(t, "and go to", "and open", "at", "go to")},
    },
    # ── Notepad ──────────────────────────────────────────────────────────────
    {
        "keywords": ["open notepad", "launch notepad", "notepad"],
        "description": lambda: "Open Notepad",
        "func": _open_notepad,
        "extract": lambda t: {},
    },
    # ── VS Code ──────────────────────────────────────────────────────────────
    {
        "keywords": ["open vs code", "open vscode", "launch vs code", "code editor"],
        "description": lambda: "Open VS Code",
        "func": _open_vscode,
        "extract": lambda t: {},
    },
    # ── Discord ──────────────────────────────────────────────────────────────
    {
        "keywords": ["open discord", "launch discord", "start discord"],
        "description": lambda: "Open Discord",
        "func": _open_discord,
        "extract": lambda t: {},
    },
    # ── Steam ────────────────────────────────────────────────────────────────
    {
        "keywords": ["open steam", "launch steam", "start steam"],
        "description": lambda: "Open Steam",
        "func": _open_steam,
        "extract": lambda t: {},
    },
    # ── Calculator ───────────────────────────────────────────────────────────
    {
        "keywords": ["open calculator", "open calc", "calculator"],
        "description": lambda: "Open Calculator",
        "func": _open_calculator,
        "extract": lambda t: {},
    },
    # ── Task Manager ─────────────────────────────────────────────────────────
    {
        "keywords": ["task manager", "open task manager"],
        "description": lambda: "Open Task Manager",
        "func": _open_task_manager,
        "extract": lambda t: {},
    },
    # ── File Explorer ─────────────────────────────────────────────────────────
    {
        "keywords": ["file explorer", "open explorer", "open files", "open folder"],
        "description": lambda: "Open File Explorer",
        "func": _open_file_explorer,
        "extract": lambda t: {},
    },
    # ── Screenshot ───────────────────────────────────────────────────────────
    {
        "keywords": ["screenshot", "take a screenshot", "capture screen", "screen capture"],
        "description": lambda: "Take a screenshot and save to Desktop",
        "func": _take_screenshot,
        "extract": lambda t: {},
    },
    # ── Type text ────────────────────────────────────────────────────────────
    {
        "keywords": ["type this", "type out", "type the words", "write this", "write that", "type that"],
        "description": lambda txt: f"Type: '{txt[:40]}'",
        "func": _type_text,
        "extract": lambda t: {"text": _extract_after(t, "type this", "type out", "type that", "write this", "write that")},
    },
    # ── Media controls ─────────────────────────────────────────────────────────
    {
        "keywords": ["pause music", "pause video", "pause playback", "play music", "resume music", "play video", "resume playback", "pause", "resume"],
        "description": lambda: "Toggle play/pause",
        "func": _media_play_pause,
        "extract": lambda t: {},
    },
    {
        "keywords": ["next track", "next song", "skip song", "skip track", "play next"],
        "description": lambda: "Skip to next track",
        "func": _media_next_track,
        "extract": lambda t: {},
    },
    {
        "keywords": ["previous track", "previous song", "go back", "last song"],
        "description": lambda: "Go to previous track",
        "func": _media_prev_track,
        "extract": lambda t: {},
    },
    # ── Volume up/down/mute ──────────────────────────────────────────────────
    {
        "keywords": ["volume up", "turn up", "louder", "increase volume"],
        "description": lambda: "Turn volume up",
        "func": _volume_up,
        "extract": lambda t: {},
    },
    {
        "keywords": ["volume down", "turn down", "quieter", "decrease volume", "lower volume"],
        "description": lambda: "Turn volume down",
        "func": _volume_down,
        "extract": lambda t: {},
    },
    {
        "keywords": ["mute", "unmute", "silence"],
        "description": lambda: "Toggle mute",
        "func": _volume_mute,
        "extract": lambda t: {},
    },
    # ── Desktop / window control ─────────────────────────────────────────────
    {
        "keywords": ["show desktop", "minimize all", "minimize everything"],
        "description": lambda: "Show desktop",
        "func": _show_desktop,
        "extract": lambda t: {},
    },
    {
        "keywords": ["close window", "close this", "close app"],
        "description": lambda: "Close the active window",
        "func": _close_window,
        "extract": lambda t: {},
    },
    {
        "keywords": ["task view", "switch window", "open task view"],
        "description": lambda: "Open Task View",
        "func": _open_task_view,
        "extract": lambda t: {},
    },
    # ── Lock / sleep / shutdown ───────────────────────────────────────────────
    {
        "keywords": ["lock screen", "lock my screen", "lock computer"],
        "description": lambda: "Lock the screen",
        "func": _lock_screen,
        "extract": lambda t: {},
    },
    {
        "keywords": ["sleep", "put computer to sleep", "hibernate"],
        "description": lambda: "Put the computer to sleep",
        "func": _sleep_computer,
        "extract": lambda t: {},
    },
    {
        "keywords": ["cancel shutdown", "abort shutdown", "stop shutdown"],
        "description": lambda: "Cancel the pending shutdown",
        "func": _cancel_shutdown,
        "extract": lambda t: {},
    },
    {
        "keywords": ["shutdown", "shut down", "turn off computer", "power off"],
        "description": lambda: "Shutdown the computer in 60 seconds",
        "func": _shutdown_computer,
        "extract": lambda t: {},
    },
    {
        "keywords": ["restart", "reboot"],
        "description": lambda: "Restart the computer in 60 seconds",
        "func": _restart_computer,
        "extract": lambda t: {},
    },
    # ── Windows Settings ─────────────────────────────────────────────────────
    {
        "keywords": ["open settings", "windows settings", "system settings"],
        "description": lambda: "Open Windows Settings",
        "func": _open_settings,
        "extract": lambda t: {},
    },
    # ── Clipboard ────────────────────────────────────────────────────────────
    {
        "keywords": ["paste", "paste that", "paste clipboard"],
        "description": lambda: "Paste clipboard contents",
        "func": _paste_clipboard,
        "extract": lambda t: {},
    },
    {
        "keywords": ["copy", "copy that", "copy selection"],
        "description": lambda: "Copy current selection",
        "func": _copy_selection,
        "extract": lambda t: {},
    },
]


# ─────────────────────────────────────────────────────────────────────────────
# Matcher
# ─────────────────────────────────────────────────────────────────────────────

def match_action(user_text: str) -> Optional[ActionResult]:
    """
    Try to match the user's command to a pre-built action.
    Returns an ActionResult if found, otherwise None.
    Longer/more-specific keyword matches are preferred.
    """
    text_lower = user_text.lower()
    best: Optional[tuple] = None  # (match_length, entry)

    for entry in REGISTRY:
        for kw in entry["keywords"]:
            if kw in text_lower:
                if best is None or len(kw) > best[0]:
                    best = (len(kw), entry)

    if best is None:
        return None

    entry = best[1]
    func = entry["func"]
    kwargs = entry["extract"](user_text)

    # Build the description (lambda may or may not take args)
    try:
        first_val = next((v for v in kwargs.values() if v), "")
        desc = entry["description"](first_val) if kwargs and first_val else entry["description"]()
    except TypeError:
        try:
            desc = entry["description"]()
        except Exception:
            desc = f"Run action: {func.__name__}"

    return ActionResult(description=desc, func=func, kwargs=kwargs)
