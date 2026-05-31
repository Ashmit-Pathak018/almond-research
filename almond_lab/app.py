"""
Project Almond Lab — Streamlit Control Center (v2.0)
====================================================
Mission Control Edition — Full observability for the T-MMU pipeline.

Run from the ROOT directory:
$ streamlit run almond_lab/app.py
"""

import json
import os
import re
import signal
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import pandas as pd
import psutil
import streamlit as st

root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(root_dir)

try:
    from memory_store import MemoryStore, MemoryTier
    STORE_AVAILABLE = True
except ImportError:
    STORE_AVAILABLE = False


# ============================================================================
# CONSTANTS
# ============================================================================

RESULTS_DIR  = Path(root_dir) / "longmem_eval_results"
LOG_FILE     = Path(root_dir) / ".almond_live.log"
HISTORY_FILE = Path(root_dir) / ".almond_run_history.json"
LM_STUDIO    = "http://localhost:1234"

DATASETS = [
    "data/longmemeval_oracle.json",
    "data/longmemeval_s_cleaned.json",
    "longmemeval_dataset.json",
]
ABLATIONS = ["none", "no_intent", "no_keyword", "no_recency", "no_peff"]

STATUS_COLOR = {
    "READY":     "#3CC97A",
    "RUNNING":   "#3ECFB2",
    "COMPLETED": "#C8922A",
    "FAILED":    "#E05C5C",
    "CANCELLED": "#7A7265",
    "WAITING":   "#4A9EE0",
}


# ============================================================================
# PAGE CONFIG
# ============================================================================

st.set_page_config(
    page_title="Almond Lab",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================================
# SESSION STATE
# ============================================================================

_defaults = {
    "is_running":      False,
    "process_pid":     None,
    "run_status":      "READY",
    "last_return_code": None,
    "last_run_config": {},
    "run_start_time":  None,
}
for _k, _v in _defaults.items():
    if _k not in st.session_state:
        st.session_state[_k] = _v


# ============================================================================
# DESIGN SYSTEM — NEURAL AMBER v2
# ============================================================================

st.markdown(
    """
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@300;400;500;600&family=Syne:wght@600;700;800&display=swap" rel="stylesheet">

<style>
/* ── TOKENS ─────────────────────────────────────────────────────── */
:root{
  --bg:#08080A; --bg-r:#0F0F13; --bg-s:#141418; --bg-o:#1A1A20;
  --border:rgba(200,146,42,.15); --border-hi:rgba(200,146,42,.4);
  --amber:#C8922A; --amber-d:#8A6118; --amber-g:rgba(200,146,42,.08);
  --mint:#3ECFB2; --red:#E05C5C; --green:#3CC97A; --blue:#4A9EE0;
  --text:#E8E2D5; --muted:#7A7265; --faint:#2A2520;
  --mono:'IBM Plex Mono',monospace; --display:'Syne',sans-serif;
  --r:4px; --rl:8px;
}

/* ── GLOBAL ─────────────────────────────────────────────────────── */
/* Removed overly aggressive font-family override on stApp to protect iframes/canvases */
.stApp{background:var(--bg)!important;}
header[data-testid="stHeader"]{background:transparent!important;}
.main .block-container{padding:1.6rem 2.25rem 4rem!important;max-width:1480px!important;}
h1{font-family:var(--display)!important;font-size:1.85rem!important;font-weight:800!important;
   color:var(--text)!important;letter-spacing:-.02em!important;margin-bottom:.1rem!important;line-height:1.1!important;}
h2,h3{font-family:var(--display)!important;font-weight:700!important;color:var(--text)!important;}
h3{font-size:.85rem!important;color:var(--amber)!important;text-transform:uppercase;letter-spacing:.08em!important;}

/* SAFELY target text without nuking Material Icons or DataFrames */
.stMarkdown, .stText, p, li, label[data-testid="stWidgetLabel"] {font-family:var(--mono);color:var(--text);}
.stMarkdown p{color:var(--muted)!important;font-size:.8rem!important;line-height:1.7!important;}
hr{border:none!important;border-top:1px solid var(--border)!important;margin:1.25rem 0!important;}
::-webkit-scrollbar{width:4px;height:4px;}
::-webkit-scrollbar-track{background:var(--bg);}
::-webkit-scrollbar-thumb{background:var(--amber-d);border-radius:99px;}
::-webkit-scrollbar-thumb:hover{background:var(--amber);}

/* ── SIDEBAR ────────────────────────────────────────────────────── */
[data-testid="stSidebar"]{background:var(--bg-r)!important;border-right:1px solid var(--border)!important;}
[data-testid="stSidebar"] .block-container{padding:1.2rem 1rem!important;}
[data-testid="stSidebar"] hr{border-color:var(--border)!important;margin:.85rem 0!important;}
[data-testid="stSidebar"] .stRadio>label{display:none!important;}
[data-testid="stSidebar"] .stRadio div[role="radiogroup"]{gap:.2rem!important;display:flex;flex-direction:column;}
[data-testid="stSidebar"] .stRadio div[role="radiogroup"] label{
  display:flex!important;align-items:center!important;padding:.48rem .7rem!important;
  border-radius:var(--r)!important;border:1px solid transparent!important;
  font-size:.79rem!important;color:var(--muted)!important;cursor:pointer!important;
  text-transform:none!important;letter-spacing:0!important;transition:all .12s!important;}
[data-testid="stSidebar"] .stRadio div[role="radiogroup"] label:hover{
  background:var(--amber-g)!important;border-color:var(--border)!important;color:var(--text)!important;}

/* ── METRICS ────────────────────────────────────────────────────── */
[data-testid="metric-container"]{
  background:var(--bg-s)!important;border:1px solid var(--border)!important;
  border-radius:var(--rl)!important;padding:.85rem 1rem!important;position:relative!important;overflow:hidden!important;}
[data-testid="metric-container"]::before{
  content:'';position:absolute;top:0;left:0;width:100%;height:2px;
  background:linear-gradient(90deg,var(--amber),transparent);}
[data-testid="metric-container"] [data-testid="stMetricLabel"]{
  font-size:.58rem!important;font-weight:500!important;color:var(--muted)!important;
  text-transform:uppercase!important;letter-spacing:.12em!important;}
[data-testid="metric-container"] [data-testid="stMetricValue"]{
  font-family:var(--display)!important;font-size:1.45rem!important;font-weight:800!important;color:var(--text)!important;}
[data-testid="metric-container"] [data-testid="stMetricDelta"]{font-size:.68rem!important;color:var(--mint)!important;}

/* ── BUTTONS ────────────────────────────────────────────────────── */
.stButton button{
  font-family:var(--mono)!important;font-size:.77rem!important;font-weight:600!important;
  border-radius:var(--r)!important;border:1px solid var(--border-hi)!important;
  background:var(--bg-o)!important;color:var(--text)!important;transition:all .12s!important;}
.stButton button:hover{background:var(--amber-g)!important;border-color:var(--amber)!important;color:var(--amber)!important;}
.stButton button[kind="primary"]{background:var(--amber)!important;border-color:var(--amber)!important;color:#080808!important;font-weight:700!important;}
.stButton button[kind="primary"]:hover{background:#DFA02E!important;box-shadow:0 0 20px rgba(200,146,42,.3)!important;color:#080808!important;}
.stButton button[disabled]{opacity:.35!important;}

/* ── FORM ELEMENTS ──────────────────────────────────────────────── */
.stSelectbox>div>div,.stTextInput>div>div>input{
  background:var(--bg-s)!important;border:1px solid var(--border)!important;
  border-radius:var(--r)!important;color:var(--text)!important;font-size:.8rem!important;}
.stTextInput>div>div>input{font-family:var(--mono)!important;}
label[data-testid="stWidgetLabel"] p{
  font-size:.62rem!important;color:var(--muted)!important;text-transform:uppercase!important;letter-spacing:.1em!important;font-weight:500!important;}
[data-baseweb="popover"]{background:var(--bg-o)!important;border:1px solid var(--border-hi)!important;border-radius:var(--r)!important;}
[data-baseweb="menu"] li{font-family:var(--mono)!important;font-size:.8rem!important;color:var(--text)!important;background:transparent!important;}
[data-baseweb="menu"] li:hover{background:var(--amber-g)!important;}
[data-testid="stSlider"] [data-baseweb="slider"] div[role="slider"]{
  background:var(--amber)!important;border:2px solid var(--amber)!important;box-shadow:0 0 8px rgba(200,146,42,.4)!important;}

/* ── PROGRESS BAR ───────────────────────────────────────────────── */
[data-testid="stProgressBar"]>div{background:var(--faint)!important;border-radius:99px!important;height:5px!important;}
[data-testid="stProgressBar"]>div>div{background:var(--amber)!important;border-radius:99px!important;}

/* ── TABS ───────────────────────────────────────────────────────── */
.stTabs [data-baseweb="tab-list"]{background:var(--bg-r)!important;border-bottom:1px solid var(--border)!important;gap:0!important;padding:0!important;}
.stTabs [data-baseweb="tab"]{font-family:var(--mono)!important;font-size:.7rem!important;font-weight:600!important;
  text-transform:uppercase!important;letter-spacing:.08em!important;color:var(--muted)!important;
  background:transparent!important;border-bottom:2px solid transparent!important;padding:.6rem 1rem!important;transition:all .12s!important;}
.stTabs [aria-selected="true"]{color:var(--amber)!important;border-bottom-color:var(--amber)!important;}
.stTabs [data-baseweb="tab-panel"]{background:var(--bg-s)!important;border:1px solid var(--border)!important;
  border-top:none!important;border-radius:0 0 var(--rl) var(--rl)!important;padding:1.1rem!important;}

/* ── EXPANDERS ──────────────────────────────────────────────────── */
/* Removed font-family override here to allow chevron icons to render! */
.streamlit-expanderHeader{background:var(--bg-s)!important;border:1px solid var(--border)!important;
  border-radius:var(--r)!important;font-size:.77rem!important;
  color:var(--text)!important;padding:.6rem .85rem!important;transition:all .12s!important;}
.streamlit-expanderHeader:hover{border-color:var(--border-hi)!important;color:var(--amber)!important;}
.streamlit-expanderContent{background:var(--bg-r)!important;border:1px solid var(--border)!important;
  border-top:none!important;border-radius:0 0 var(--r) var(--r)!important;padding:.9rem!important;}

/* ── ALERTS ─────────────────────────────────────────────────────── */
div[data-testid="stNotification"]{font-family:var(--mono)!important;font-size:.77rem!important;border-radius:var(--r)!important;}
div[data-testid="stNotification"][data-type="info"]{background:rgba(62,207,178,.06)!important;border-left:3px solid var(--mint)!important;}
div[data-testid="stNotification"][data-type="warning"]{background:rgba(200,146,42,.08)!important;border-left:3px solid var(--amber)!important;}
div[data-testid="stNotification"][data-type="error"]{background:rgba(224,92,92,.07)!important;border-left:3px solid var(--red)!important;}
div[data-testid="stNotification"][data-type="success"]{background:rgba(60,201,122,.07)!important;border-left:3px solid var(--green)!important;}

/* ── CODE + DATAFRAME ───────────────────────────────────────────── */
.stCodeBlock pre,.stCodeBlock code{background:var(--bg-r)!important;color:var(--mint)!important;font-family:var(--mono)!important;font-size:.75rem!important;}
[data-testid="stDataFrame"]{border:1px solid var(--border)!important;border-radius:var(--rl)!important;overflow:hidden!important;}

/* ── FORM SUBMIT BUTTON ─────────────────────────────────────────── */
.stFormSubmitButton button{width:100%!important;}

/* ── ANIMATIONS ─────────────────────────────────────────────────── */
@keyframes pulse{0%,100%{opacity:1}50%{opacity:.25}}
@keyframes blink{0%,100%{opacity:1}50%{opacity:0}}
</style>
"""
    ,
    unsafe_allow_html=True,
)

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def get_system_stats() -> dict:
    ram = psutil.virtual_memory()
    cpu = psutil.cpu_percent(interval=0.1)
    return {
        "ram_used":    round(ram.used / 1024**3, 2),
        "ram_total":   round(ram.total / 1024**3, 2),
        "ram_pct":     ram.percent,
        "cpu_pct":     cpu,
    }


@st.cache_data(ttl=4)
def check_lm_studio() -> dict:
    """Hit LM Studio /v1/models. Cached 4s so sidebar doesn't lag."""
    import urllib.request
    try:
        with urllib.request.urlopen(f"{LM_STUDIO}/v1/models", timeout=2) as r:
            data = json.loads(r.read())
            models = data.get("data", [])
            name = models[0]["id"].split("/")[-1] if models else "Unknown"
            return {"ok": True, "model": name, "count": len(models)}
    except Exception:
        return {"ok": False, "model": None, "count": 0}


def run_preflight(dataset: str) -> list[dict]:
    checks = []

    ds_path = Path(root_dir) / dataset
    checks.append({
        "label": "Dataset File", "ok": ds_path.exists(), "critical": True,
        "detail": str(ds_path.name) if ds_path.exists() else f"Not found: {dataset}",
    })

    script = Path(root_dir) / "eval_unified.py"
    checks.append({
        "label": "eval_unified.py", "ok": script.exists(), "critical": True,
        "detail": "Found in root" if script.exists() else "Missing from root directory",
    })

    RESULTS_DIR.mkdir(exist_ok=True)
    checks.append({
        "label": "Results Directory", "ok": True, "critical": False,
        "detail": str(RESULTS_DIR.name),
    })

    lm = check_lm_studio()
    checks.append({
        "label": "LM Studio", "ok": lm["ok"], "critical": True,
        "detail": f"{lm['model']}" if lm["ok"] else "Offline — start LM Studio",
    })

    major, minor = sys.version_info[:2]
    py_ok = (major, minor) >= (3, 9)
    checks.append({
        "label": f"Python {major}.{minor}", "ok": py_ok, "critical": False,
        "detail": "OK" if py_ok else "3.9+ recommended",
    })

    return checks


def get_log_lines() -> list[str]:
    if not LOG_FILE.exists():
        return []
    try:
        return LOG_FILE.read_text(encoding="utf-8", errors="replace").splitlines()
    except Exception:
        return []


_Q_RE    = re.compile(r'[Qq]uestion[\s#]*(\d+)\s*[/|]\s*(\d+)')
_TS_RE   = re.compile(r'\b(\d{2}:\d{2}:\d{2})\b')
_STAGES  = [
    ("Loading Dataset",      ["loading", "dataset loaded", "loaded dataset"]),
    ("Connecting to Model",  ["connecting", "lm studio", "openai"]),
    ("Building Index",       ["embed", "index", "vector"]),
    ("Retrieving Memories",  ["retriev", "recall", "memory"]),
    ("Generating Response",  ["generat", "running question", "processing"]),
    ("Saving Results",       ["saving", "writing", "report"]),
    ("Complete",             ["evaluation done", "complete", "finished"]),
]


def parse_progress(lines: list[str]) -> dict:
    out = {"stage": "Initializing...", "current_q": 0, "total_q": 0, "pct": 0.0, "error": False}

    for line in reversed(lines[-120:]):
        m = _Q_RE.search(line)
        if m:
            out["current_q"] = int(m.group(1))
            out["total_q"]   = int(m.group(2))
            if out["total_q"] > 0:
                out["pct"] = out["current_q"] / out["total_q"]
            break

    for line in reversed(lines[-40:]):
        ll = line.lower()
        if any(e in ll for e in ["error", "traceback", "exception"]):
            out["error"] = True
        for label, kws in _STAGES:
            if any(kw in ll for kw in kws):
                out["stage"] = label
                break
        else:
            continue
        break

    if out["current_q"] > 0:
        out["stage"] = f"Question {out['current_q']} / {out['total_q']}"

    return out


def process_alive() -> bool:
    pid = st.session_state.get("process_pid")
    if pid is None:
        return False
    try:
        p = psutil.Process(pid)
        return p.is_running() and p.status() != psutil.STATUS_ZOMBIE
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        return False


def start_evaluation(dataset: str, limit: int, ablation: str):
    script = Path(root_dir) / "eval_unified.py"
    LOG_FILE.write_text("")          # clear old log
    log_f = open(LOG_FILE, "w", encoding="utf-8")
    proc  = subprocess.Popen(
        [sys.executable, str(script),
         "--dataset", dataset, "--limit", str(limit), "--ablation", ablation],
        cwd=root_dir,
        stdout=log_f,
        stderr=subprocess.STDOUT,
        text=True,
    )
    log_f.close()                    # child inherits dup'd FD — safe to close parent handle

    st.session_state.process_pid    = proc.pid
    st.session_state.is_running     = True
    st.session_state.run_status     = "RUNNING"
    st.session_state.run_start_time = time.time()
    st.session_state.last_run_config = {
        "dataset": dataset, "limit": limit, "ablation": ablation,
    }


def cancel_evaluation():
    pid = st.session_state.get("process_pid")
    if pid:
        try:
            os.kill(pid, signal.SIGTERM)
        except (ProcessLookupError, PermissionError, AttributeError):
            pass
    st.session_state.is_running    = False
    st.session_state.run_status    = "CANCELLED"
    st.session_state.process_pid   = None


def get_all_reports() -> list[Path]:
    if not RESULTS_DIR.exists():
        return []
    reports = (
        list(RESULTS_DIR.glob("longmem_report*.json"))
        + list(RESULTS_DIR.glob("partial_report.json"))
    )
    return sorted(reports, key=os.path.getmtime, reverse=True)


def load_history() -> list[dict]:
    if not HISTORY_FILE.exists():
        return []
    try:
        return json.loads(HISTORY_FILE.read_text())
    except Exception:
        return []


def append_history(config: dict, summary: dict, report_name: str):
    h = load_history()
    h.insert(0, {
        "date":          datetime.now().strftime("%Y-%m-%d %H:%M"),
        "dataset":       config.get("dataset", "?"),
        "limit":         config.get("limit", "?"),
        "ablation":      config.get("ablation", "none"),
        "accuracy":      summary.get("accuracy", 0),
        "avg_latency_ms": summary.get("avg_latency_ms", 0),
        "avg_pollution": summary.get("avg_pollution", 0),
        "report_file":   report_name,
    })
    HISTORY_FILE.write_text(json.dumps(h[:50], indent=2))


# ============================================================================
# HTML COMPONENT HELPERS
# ============================================================================

def _badge(status: str) -> str:
    c   = STATUS_COLOR.get(status, "#7A7265")
    anim = "animation:pulse 1.5s ease-in-out infinite;" if status == "RUNNING" else ""
    return (
        f'<span style="display:inline-flex;align-items:center;gap:.35rem;padding:.22rem .6rem;'
        f'border-radius:3px;font-size:.62rem;font-weight:700;letter-spacing:.1em;'
        f'text-transform:uppercase;font-family:\'IBM Plex Mono\',monospace;'
        f'background:{c}18;border:1px solid {c}55;color:{c};">'
        f'<span style="width:6px;height:6px;border-radius:50%;background:{c};flex-shrink:0;{anim}"></span>'
        f'{status}</span>'
    )


def _slabel(text: str, mt: str = "1.1rem") -> None:
    st.markdown(
        f'<p style="font-size:.58rem;text-transform:uppercase;letter-spacing:.14em;'
        f'color:#7A7265;font-weight:500;margin-top:{mt};margin-bottom:.55rem;">{text}</p>',
        unsafe_allow_html=True,
    )


def _card(html_inner: str, border_color: str = "rgba(200,146,42,.15)",
          bg: str = "var(--bg-s,#141418)", radius: str = "8px") -> None:
    st.markdown(
        f'<div style="background:{bg};border:1px solid {border_color};'
        f'border-radius:{radius};padding:.9rem 1rem;">{html_inner}</div>',
        unsafe_allow_html=True,
    )


# ============================================================================
# SIDEBAR
# ============================================================================

with st.sidebar:
    st.markdown(
        '<div style="display:flex;align-items:center;gap:.55rem;margin-bottom:1.1rem;">'
        '<span style="font-size:1.25rem;line-height:1;">🧠</span>'
        '<div><div style="font-family:\'Syne\',sans-serif;font-size:1.05rem;font-weight:800;'
        'color:#C8922A;line-height:1;letter-spacing:-.01em;">ALMOND LAB</div>'
        '<div style="font-size:.52rem;color:#7A7265;text-transform:uppercase;letter-spacing:.16em;margin-top:2px;">'
        'T-MMU v2 · Mission Control</div></div></div>',
        unsafe_allow_html=True,
    )

    page = st.radio(
        "nav",
        ["🚀  Mission Control", "📊  Analysis", "🔍  Memory Inspector", "📈  Run History"],
        label_visibility="collapsed",
    )

    st.markdown("<hr>", unsafe_allow_html=True)

    # ── Run Status ──────────────────────────────────────────────────────────
    elapsed_str = ""
    if st.session_state.is_running and st.session_state.run_start_time:
        e = int(time.time() - st.session_state.run_start_time)
        elapsed_str = f' <span style="color:#3A352D;font-size:.6rem;">({e//60:02d}:{e%60:02d})</span>'

    st.markdown(
        f'<div style="margin-bottom:.7rem;">'
        f'<div style="font-size:.55rem;color:#7A7265;text-transform:uppercase;letter-spacing:.12em;margin-bottom:.35rem;">Run Status</div>'
        f'{_badge(st.session_state.run_status)}{elapsed_str}</div>',
        unsafe_allow_html=True,
    )

    # ── LM Studio Pill ──────────────────────────────────────────────────────
    lm_sb = check_lm_studio()
    if lm_sb["ok"]:
        model_disp = lm_sb["model"][:22] if lm_sb["model"] else "Unknown"
        st.markdown(
            f'<div style="background:rgba(60,201,122,.06);border:1px solid rgba(60,201,122,.25);'
            f'border-radius:4px;padding:.55rem .8rem;font-size:.72rem;">'
            f'<div style="color:#3CC97A;font-size:.56rem;text-transform:uppercase;letter-spacing:.12em;'
            f'font-weight:700;margin-bottom:.25rem;">● LM Studio · Online</div>'
            f'<div style="color:#E8E2D5;font-size:.76rem;">{model_disp}</div></div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f'<div style="background:rgba(224,92,92,.06);border:1px solid rgba(224,92,92,.25);'
            f'border-radius:4px;padding:.55rem .8rem;font-size:.72rem;">'
            f'<div style="color:#E05C5C;font-size:.56rem;text-transform:uppercase;letter-spacing:.12em;'
            f'font-weight:700;margin-bottom:.25rem;">○ LM Studio · Offline</div>'
            f'<div style="color:#7A7265;">Start LM Studio to enable eval</div></div>',
            unsafe_allow_html=True,
        )

    st.markdown("<hr>", unsafe_allow_html=True)

    # ── System Telemetry ────────────────────────────────────────────────────
    stats = get_system_stats()
    st.markdown(
        '<div style="font-size:.55rem;color:#7A7265;text-transform:uppercase;letter-spacing:.13em;margin-bottom:.5rem;">System</div>',
        unsafe_allow_html=True,
    )
    st.metric("RAM", f"{stats['ram_used']}/{stats['ram_total']} GB", f"{stats['ram_pct']}%")
    st.metric("CPU", f"{stats['cpu_pct']}%")

    if st.session_state.is_running and st.session_state.run_start_time:
        e = int(time.time() - st.session_state.run_start_time)
        st.metric("Elapsed", f"{e//60:02d}:{e%60:02d}")


# ============================================================================
# PAGE 1 — MISSION CONTROL
# ============================================================================

if "Mission" in page:

    # ── Handle process completion ────────────────────────────────────────────
    if st.session_state.is_running and not process_alive():
        lines     = get_log_lines()
        tail      = "\n".join(lines[-25:]).lower()
        completed = any(w in tail for w in ["evaluation done", "report saved", "completed", "success"])
        st.session_state.run_status  = "COMPLETED" if completed else "FAILED"
        st.session_state.is_running  = False
        st.session_state.process_pid = None

        if completed:
            reports = get_all_reports()
            if reports and st.session_state.last_run_config:
                try:
                    rdata = json.loads(reports[0].read_text())
                    append_history(
                        st.session_state.last_run_config,
                        rdata.get("summary", {}),
                        reports[0].name,
                    )
                except Exception:
                    pass

    # ── Header ───────────────────────────────────────────────────────────────
    h_col, status_col = st.columns([3, 1])
    with h_col:
        st.markdown("<h1>Mission Control</h1>", unsafe_allow_html=True)
        st.markdown(
            '<p style="color:#7A7265;font-size:.8rem;margin-bottom:1.4rem;">Configure, launch, and observe evaluation runs in real time.</p>',
            unsafe_allow_html=True,
        )
    with status_col:
        st.markdown(
            f'<div style="display:flex;justify-content:flex-end;align-items:flex-start;'
            f'padding-top:.35rem;gap:.5rem;">{_badge(st.session_state.run_status)}</div>',
            unsafe_allow_html=True,
        )

    # ── Top Row: Pre-flight + LM Studio ─────────────────────────────────────
    pf_col, lm_col = st.columns([1.15, 1], gap="large")

    check_dataset = st.session_state.last_run_config.get("dataset", DATASETS[0])
    checks   = run_preflight(check_dataset)
    all_crit = all(c["ok"] for c in checks if c["critical"])

    with pf_col:
        _slabel("Pre-flight Checks", "0")
        rows = ""
        for c in checks:
            if c["ok"]:
                icon, col = "✓", "#3CC97A"
            elif c["critical"]:
                icon, col = "✗", "#E05C5C"
            else:
                icon, col = "⚠", "#C8922A"
            rows += (
                f'<div style="display:flex;align-items:center;gap:.65rem;padding:.38rem .7rem;'
                f'background:rgba(255,255,255,.02);border:1px solid rgba(255,255,255,.04);border-radius:4px;">'
                f'<span style="color:{col};font-weight:700;font-size:.82rem;min-width:.9rem;">{icon}</span>'
                f'<div>'
                f'<div style="font-size:.78rem;color:#E8E2D5;">{c["label"]}</div>'
                f'<div style="font-size:.66rem;color:#7A7265;">{c["detail"]}</div>'
                f'</div></div>'
            )
        st.markdown(f'<div style="display:flex;flex-direction:column;gap:.28rem;">{rows}</div>', unsafe_allow_html=True)

        if not all_crit:
            st.markdown(
                '<div style="margin-top:.6rem;padding:.45rem .75rem;background:rgba(224,92,92,.07);'
                'border-left:3px solid #E05C5C;border-radius:0 4px 4px 0;font-size:.73rem;color:#E05C5C;">'
                'Critical checks failed — resolve before launching.</div>',
                unsafe_allow_html=True,
            )

    with lm_col:
        _slabel("LM Studio Monitor", "0")
        lm = check_lm_studio()
        if lm["ok"]:
            _card(
                f'<div style="font-size:.56rem;text-transform:uppercase;letter-spacing:.12em;'
                f'color:#3CC97A;font-weight:700;margin-bottom:.6rem;">● Connected</div>'
                f'<div style="font-size:.82rem;color:#E8E2D5;margin-bottom:.2rem;">{lm["model"]}</div>'
                f'<div style="font-size:.68rem;color:#7A7265;">localhost:1234 · {lm["count"]} model(s)</div>',
                border_color="rgba(60,201,122,.2)",
                bg="rgba(60,201,122,.04)",
            )
        else:
            _card(
                '<div style="font-size:.56rem;text-transform:uppercase;letter-spacing:.12em;'
                'color:#E05C5C;font-weight:700;margin-bottom:.6rem;">○ Offline</div>'
                '<div style="font-size:.78rem;color:#7A7265;line-height:1.6;">'
                'Open LM Studio and load a model<br>before running evaluation.</div>',
                border_color="rgba(224,92,92,.2)",
                bg="rgba(224,92,92,.04)",
            )

    st.markdown("<hr>", unsafe_allow_html=True)

    # ── Configuration ─────────────────────────────────────────────────────────
    _slabel("Evaluation Configuration")
    cfg1, cfg2, cfg3 = st.columns(3)
    with cfg1:
        dataset = st.selectbox("Dataset", DATASETS, disabled=st.session_state.is_running)
    with cfg2:
        limit = st.slider("Question Limit", 1, 500, 5, disabled=st.session_state.is_running)
    with cfg3:
        ablation = st.selectbox("Ablation Mode", ABLATIONS, disabled=st.session_state.is_running)

    # ── Run Controls ─────────────────────────────────────────────────────────
    run_col, cancel_col, status_col2 = st.columns([.45, .45, 3])
    can_run = all_crit and not st.session_state.is_running

    with run_col:
        if st.button("▶  RUN", type="primary", disabled=not can_run, use_container_width=True):
            start_evaluation(dataset, limit, ablation)
            st.rerun()

    with cancel_col:
        if st.button("■  CANCEL", disabled=not st.session_state.is_running, use_container_width=True):
            cancel_evaluation()
            st.rerun()

    with status_col2:
        cfg_str = ""
        if st.session_state.last_run_config:
            c = st.session_state.last_run_config
            ds_short = Path(c.get("dataset", "")).name
            cfg_str = (
                f' <span style="color:#3A352D;font-size:.65rem;">'
                f'· {ds_short} · {c.get("limit","?")}q · ablation:{c.get("ablation","?")}</span>'
            )
        st.markdown(
            f'<div style="display:flex;align-items:center;padding-top:.5rem;gap:.5rem;">'
            f'{_badge(st.session_state.run_status)}{cfg_str}</div>',
            unsafe_allow_html=True,
        )

    # ── Live Progress ─────────────────────────────────────────────────────────
    if st.session_state.run_status in ("RUNNING", "COMPLETED", "FAILED"):
        st.markdown("<hr>", unsafe_allow_html=True)
        _slabel("Progress")

        lines = get_log_lines()
        prog  = parse_progress(lines)

        pct = prog["pct"]
        if st.session_state.run_status == "COMPLETED":
            pct = 1.0

        pb_col, pct_col = st.columns([4, 1])
        with pb_col:
            st.progress(pct)
        with pct_col:
            label = f"{int(pct*100)}%"
            if prog["total_q"] > 0:
                label = f"Q {prog['current_q']} / {prog['total_q']}"
            st.markdown(
                f'<div style="font-size:.76rem;color:#7A7265;padding-top:.32rem;'
                f'text-align:right;">{label}</div>',
                unsafe_allow_html=True,
            )

        s_color = {
            "RUNNING":   "#3ECFB2",
            "COMPLETED": "#C8922A",
            "FAILED":    "#E05C5C",
        }.get(st.session_state.run_status, "#7A7265")

        s_icon = "●" if st.session_state.run_status == "RUNNING" else (
            "✓" if st.session_state.run_status == "COMPLETED" else "✗"
        )

        error_badge = (
            '<span style="color:#E05C5C;margin-left:.75rem;font-size:.72rem;">⚠ Errors detected in log</span>'
            if prog["error"]
            else ""
        )

        st.markdown(
            f'<div style="font-size:.8rem;margin-top:.2rem;">'
            f'<span style="color:{s_color};">{s_icon}</span>'
            f'<span style="color:#7A7265;margin:0 .4rem;">Stage:</span>'
            f'<span style="color:{s_color};">{prog["stage"]}</span>'
            f'{error_badge}'
            f'</div>',
            unsafe_allow_html=True,
        )

    # ── Live Log Console ──────────────────────────────────────────────────────
    if st.session_state.run_status in ("RUNNING", "COMPLETED", "FAILED", "CANCELLED"):
        st.markdown("<hr>", unsafe_allow_html=True)
        _slabel("Live Log Console")

        lines = get_log_lines()

        if lines:
            # Build colored log HTML
            log_rows = []
            for raw in lines[-400:]:
                line = raw.rstrip()
                if not line:
                    continue
                ts_m = _TS_RE.search(line)
                ts   = ts_m.group(1) if ts_m else "        "
                ll   = line.lower()

                if any(e in ll for e in ["error", "traceback", "exception", "failed"]):
                    txt_c = "#E05C5C"
                elif any(w in ll for w in ["passed", "success", "complete", "done", "saved"]):
                    txt_c = "#3CC97A"
                elif any(w in ll for w in ["warn", "warning"]):
                    txt_c = "#C8922A"
                elif any(w in ll for w in ["connect", "retriev", "generat", "loading", "embed"]):
                    txt_c = "#3ECFB2"
                else:
                    txt_c = "#7A7265"

                ts_span = f'<span style="color:#2A2520;user-select:none;margin-right:.75rem;">{ts}</span>'
                log_rows.append(
                    f'<div style="padding:.06rem 0;white-space:pre-wrap;word-break:break-all;">'
                    f'{ts_span}<span style="color:{txt_c};">{line}</span></div>'
                )

            st.markdown(
                '<div style="background:#04040A;border:1px solid rgba(200,146,42,.12);'
                'border-radius:6px;padding:.8rem 1rem;max-height:400px;overflow-y:auto;'
                'font-family:\'IBM Plex Mono\',monospace;font-size:.72rem;line-height:1.55;">'
                + "\n".join(log_rows)
                + '<div id="logend"></div>'
                '</div>'
                '<script>setTimeout(()=>{var e=document.getElementById("logend");'
                'if(e)e.scrollIntoView();},80);</script>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                '<div style="background:#04040A;border:1px solid rgba(200,146,42,.1);'
                'border-radius:6px;padding:1.75rem;text-align:center;font-size:.76rem;color:#2A2520;">'
                'Waiting for output...</div>',
                unsafe_allow_html=True,
            )

    # ── Auto-refresh ──────────────────────────────────────────────────────────
    if st.session_state.is_running:
        time.sleep(1.5)
        st.rerun()


# ============================================================================
# PAGE 2 — ANALYSIS
# ============================================================================

elif "Analysis" in page:
    st.markdown("<h1>Analysis</h1>", unsafe_allow_html=True)
    st.markdown(
        '<p style="color:#7A7265;font-size:.8rem;margin-bottom:1.4rem;">'
        'Benchmark results, failure explanations, and cognitive retrieval traces.</p>',
        unsafe_allow_html=True,
    )

    reports = get_all_reports()
    if not reports:
        st.info("No reports found. Run a benchmark from Mission Control first.")
    else:
        selected_name = st.selectbox("Benchmark Report", [r.name for r in reports])
        selected_path = RESULTS_DIR / selected_name

        try:
            data = json.loads(selected_path.read_text(encoding="utf-8"))
        except Exception as e:
            st.error(f"Could not read report: {e}")
            st.stop()

        summary = data.get("summary", {})

        # ── KPI Cards ─────────────────────────────────────────────────────────
        _slabel("Summary Metrics", "0")
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Accuracy",     f"{summary.get('accuracy', 0)}%")
        k2.metric("Questions",    f"{summary.get('questions', 0)}")
        k3.metric("Avg Latency",  f"{summary.get('avg_latency_ms', 0)} ms")
        k4.metric("Avg Pollution", f"{summary.get('avg_pollution', 0)}")

        st.markdown("<hr>", unsafe_allow_html=True)

        p1, p2, p3 = st.columns(3)
        p1.metric("Ablation Mode",  summary.get("ablation", "none"))
        p2.metric("Avg Retrieved",  summary.get("avg_retrieved_blocks", 0))
        p3.metric("Avg Rejected",   summary.get("avg_rejected_blocks", 0))

        # ── Tabs: Results | Failures | Traces ─────────────────────────────────
        tab_res, tab_fail, tab_trace = st.tabs([
            "All Results", "Failure Explainer", "Retrieval Traces"
        ])

        results = data.get("results", [])

        with tab_res:
            if results:
                df = pd.DataFrame(results)
                cols = ["index", "passed", "question_type", "question",
                        "expected_answer", "model_response", "latency_ms"]
                df = df[[c for c in cols if c in df.columns]]
                def color_bool(v):
                    return f"color:{'#3CC97A' if v else '#E05C5C'};font-weight:bold;"
                st.dataframe(
                    df.style.map(color_bool, subset=["passed"]),
                    use_container_width=True, height=440,
                )
            else:
                st.info("No per-question results in this report.")

        with tab_fail:
            _slabel("Failed Questions — Root Cause Analysis", "0")
            failed = [r for r in results if not r.get("passed", True)]
            if not failed:
                st.success("All questions passed in this run.")
            else:
                for r in failed:
                    traces = data.get("retrieval_traces", [])
                    trace  = next(
                        (t for t in traces
                         if t.get("query","").strip() == r.get("question","").strip()),
                        None,
                    )
                    retrieved_ct = len(trace.get("retrieved_ids", [])) if trace else "?"
                    rejections   = trace.get("rejection_reasons", []) if trace else []
                    pollution    = trace.get("pollution_score", "?") if trace else "?"

                    # Infer root cause
                    if retrieved_ct == 0 or retrieved_ct == "?":
                        root = "No relevant memory retrieved from the vault."
                    elif str(pollution) != "?" and float(str(pollution)) > 0.6:
                        root = f"High pollution score ({pollution}) — noisy context injected."
                    elif rejections:
                        root = f"{len(rejections)} memory block(s) rejected before retrieval."
                    else:
                        root = "Memory retrieved but model generated incorrect response."

                    with st.expander(
                        f"✗  Q{r.get('index','?')} — {r.get('question', '')[:72]}..."
                    ):
                        fc1, fc2 = st.columns(2)
                        with fc1:
                            st.markdown(
                                f'<div style="font-size:.72rem;color:#7A7265;text-transform:uppercase;'
                                f'letter-spacing:.1em;margin-bottom:.3rem;">Expected</div>'
                                f'<div style="font-size:.82rem;color:#3CC97A;background:rgba(60,201,122,.05);'
                                f'padding:.5rem .75rem;border-radius:4px;border:1px solid rgba(60,201,122,.2);">'
                                f'{r.get("expected_answer","—")}</div>',
                                unsafe_allow_html=True,
                            )
                        with fc2:
                            st.markdown(
                                f'<div style="font-size:.72rem;color:#7A7265;text-transform:uppercase;'
                                f'letter-spacing:.1em;margin-bottom:.3rem;">Model Response</div>'
                                f'<div style="font-size:.82rem;color:#E05C5C;background:rgba(224,92,92,.05);'
                                f'padding:.5rem .75rem;border-radius:4px;border:1px solid rgba(224,92,92,.2);">'
                                f'{r.get("model_response","—")}</div>',
                                unsafe_allow_html=True,
                            )

                        st.markdown(
                            f'<div style="margin-top:.75rem;padding:.5rem .75rem;'
                            f'background:rgba(200,146,42,.06);border-left:3px solid #C8922A;'
                            f'border-radius:0 4px 4px 0;font-size:.76rem;">'
                            f'<span style="color:#7A7265;text-transform:uppercase;font-size:.6rem;'
                            f'letter-spacing:.1em;">Root Cause</span><br>'
                            f'<span style="color:#C8922A;">{root}</span><br>'
                            f'<span style="color:#3A352D;font-size:.68rem;">'
                            f'Memories retrieved: {retrieved_ct} · '
                            f'Rejected blocks: {len(rejections)} · '
                            f'Pollution: {pollution}</span></div>',
                            unsafe_allow_html=True,
                        )

        with tab_trace:
            _slabel("Per-Query Retrieval Decisions", "0")
            traces = data.get("retrieval_traces", [])
            if not traces:
                st.info("No retrieval traces found in this report.")
            else:
                for tr in traces:
                    q         = tr.get("query", "Unknown")
                    accepted  = tr.get("retrieved_ids", [])
                    rejections= tr.get("rejection_reasons", [])
                    icon      = "🟢" if accepted else "🔴"
                    with st.expander(
                        f"{icon}  {q[:72]}{'...' if len(q)>72 else ''}  "
                        f"— accepted:{len(accepted)}  rejected:{len(rejections)}"
                    ):
                        st.markdown(
                            f'<div style="font-size:.76rem;color:#E8E2D5;margin-bottom:.7rem;">'
                            f'<span style="font-size:.6rem;color:#7A7265;text-transform:uppercase;'
                            f'letter-spacing:.1em;">Query</span><br>{q}</div>',
                            unsafe_allow_html=True,
                        )
                        pollution = tr.get("pollution_score", 0)
                        pc = "#E05C5C" if float(pollution or 0) > 0.5 else "#3CC97A"
                        st.markdown(
                            f'<div style="font-size:.73rem;margin-bottom:.8rem;">'
                            f'Pollution: <span style="color:{pc};font-weight:600;">{pollution}</span></div>',
                            unsafe_allow_html=True,
                        )
                        c1, c2 = st.columns(2)
                        with c1:
                            st.markdown(
                                '<div style="font-size:.6rem;text-transform:uppercase;letter-spacing:.1em;'
                                'color:#3CC97A;font-weight:700;margin-bottom:.4rem;">✓ Promoted to L2</div>',
                                unsafe_allow_html=True,
                            )
                            if accepted:
                                for a in accepted:
                                    st.code(a, language="text")
                            else:
                                st.markdown(
                                    '<span style="font-size:.73rem;color:#7A7265;">No memories passed.</span>',
                                    unsafe_allow_html=True,
                                )
                        with c2:
                            st.markdown(
                                '<div style="font-size:.6rem;text-transform:uppercase;letter-spacing:.1em;'
                                'color:#E05C5C;font-weight:700;margin-bottom:.4rem;">✗ Rejected</div>',
                                unsafe_allow_html=True,
                            )
                            if rejections:
                                for rej in rejections:
                                    if isinstance(rej, dict):
                                        reason = rej.get("reason", "Unknown")
                                        score  = rej.get("similarity", rej.get("hybrid_score", "?"))
                                        st.error(f"**{reason}** — score: {score}")
                                    else:
                                        st.error(str(rej))
                            else:
                                st.markdown(
                                    '<span style="font-size:.73rem;color:#7A7265;">No rejections.</span>',
                                    unsafe_allow_html=True,
                                )


# ============================================================================
# PAGE 3 — MEMORY INSPECTOR
# ============================================================================

elif "Memory" in page:
    h1c, h2c = st.columns([0.85, 0.15])
    with h1c:
        st.markdown("<h1>Memory Inspector</h1>", unsafe_allow_html=True)
        st.markdown(
            '<p style="color:#7A7265;font-size:.8rem;margin-bottom:1.4rem;">'
            'Browse the live SQLite T-MMU vault across all memory tiers.</p>',
            unsafe_allow_html=True,
        )
    with h2c:
        st.write("")
        st.write("")
        if st.button("↻ Refresh", use_container_width=True):
            st.rerun()

    if not STORE_AVAILABLE:
        st.error("Cannot import `MemoryStore`. Ensure `almond_lab/` is inside the Almond root.")
    else:
        db_path = Path(root_dir) / "longmem_almond.db"
        if not db_path.exists():
            st.warning("No database found (`longmem_almond.db`). Run a chat or eval to create one.")
        else:
            try:
                store  = MemoryStore(db_path=str(db_path))
                counts = store.tier_counts()

                _slabel("Vault Overview", "0")
                v1, v2, v3, v4 = st.columns(4)
                v1.metric("L1 · Hot Cache",    counts.get(MemoryTier.L1_HOT_CACHE.value, 0))
                v2.metric("L2 · Active RAM",   counts.get(MemoryTier.L2_ACTIVE_RAM.value, 0))
                v3.metric("L3 · Virtual Swap", counts.get(MemoryTier.L3_VIRTUAL_SWAP.value, 0))
                v4.metric("L4 · Archive",      counts.get(MemoryTier.L4_ARCHIVE.value, 0))

                st.markdown("<hr>", unsafe_allow_html=True)

                tabs = st.tabs(["L1  Rules", "L2  Active Context", "L3  Vector DB", "L4  Archive"])

                def render_tier(tier_enum, tab_obj):
                    with tab_obj:
                        blocks = store.get_all(tier_enum)
                        if not blocks:
                            st.info(f"No blocks in **{tier_enum.value}**")
                        else:
                            rows = [
                                {
                                    "ID":       b.id[:8],
                                    "Tag":      b.tag.value,
                                    "P_eff":    round(b.p_eff, 4),
                                    "Content":  b.content,
                                    "Accesses": b.access_count,
                                }
                                for b in sorted(blocks, key=lambda x: x.p_eff, reverse=True)
                            ]
                            search = st.text_input(
                                "Filter content", key=f"search_{tier_enum.value}",
                                placeholder="Type to filter..."
                            )
                            df = pd.DataFrame(rows)
                            if search:
                                mask = df.apply(
                                    lambda r: search.lower() in str(r).lower(), axis=1
                                )
                                df = df[mask]
                            st.dataframe(df, use_container_width=True, height=400)

                render_tier(MemoryTier.L1_HOT_CACHE,    tabs[0])
                render_tier(MemoryTier.L2_ACTIVE_RAM,   tabs[1])
                render_tier(MemoryTier.L3_VIRTUAL_SWAP, tabs[2])
                render_tier(MemoryTier.L4_ARCHIVE,      tabs[3])

            except Exception as e:
                st.error(f"Error reading database: {e}")
            finally:
                if "store" in locals() and hasattr(store, "_conn"):
                    store._conn.close()


# ============================================================================
# PAGE 4 — RUN HISTORY
# ============================================================================

elif "History" in page:
    st.markdown("<h1>Run History</h1>", unsafe_allow_html=True)
    st.markdown(
        '<p style="color:#7A7265;font-size:.8rem;margin-bottom:1.4rem;">'
        'Review and compare previous evaluation runs.</p>',
        unsafe_allow_html=True,
    )

    history = load_history()

    if not history:
        st.info("No run history yet. Completed evaluations are logged here automatically.")
    else:
        _slabel("Previous Runs", "0")

        for i, run in enumerate(history):
            acc = run.get("accuracy", 0)
            acc_col = "#3CC97A" if acc >= 80 else "#C8922A" if acc >= 60 else "#E05C5C"
            ablation_badge = (
                f'<span style="background:rgba(200,146,42,.1);border:1px solid rgba(200,146,42,.3);'
                f'border-radius:3px;padding:.1rem .4rem;font-size:.62rem;color:#C8922A;">'
                f'{run.get("ablation","none")}</span>'
            )
            ds_name = Path(run.get("dataset", "?")).name

            with st.expander(
                f"{'[Latest]  ' if i == 0 else ''}{run.get('date','?')}  ·  {ds_name}  "
                f"·  Accuracy: {acc}%"
            ):
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Accuracy",    f"{acc}%")
                m2.metric("Avg Latency", f"{run.get('avg_latency_ms', 0)} ms")
                m3.metric("Pollution",   f"{run.get('avg_pollution', 0)}")
                m4.metric("Questions",   f"{run.get('limit', '?')}")

                st.markdown(
                    f'<div style="margin-top:.7rem;font-size:.74rem;color:#7A7265;display:flex;gap:1.5rem;">'
                    f'<span>Dataset: <span style="color:#E8E2D5;">{ds_name}</span></span>'
                    f'<span>Ablation: {ablation_badge}</span>'
                    f'<span>Report: <span style="color:#E8E2D5;">{run.get("report_file","?")}</span></span>'
                    f'</div>',
                    unsafe_allow_html=True,
                )

        # Comparison table
        if len(history) > 1:
            st.markdown("<hr>", unsafe_allow_html=True)
            _slabel("Comparison Table")
            df_hist = pd.DataFrame([
                {
                    "Date":      r.get("date", "?"),
                    "Dataset":   Path(r.get("dataset","?")).name,
                    "Ablation":  r.get("ablation","?"),
                    "Questions": r.get("limit","?"),
                    "Accuracy":  f"{r.get('accuracy',0)}%",
                    "Latency ms": r.get("avg_latency_ms", 0),
                    "Pollution": r.get("avg_pollution", 0),
                }
                for r in history
            ])
            st.dataframe(df_hist, use_container_width=True, height=300)