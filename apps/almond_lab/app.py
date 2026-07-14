"""
Project Almond Lab — Streamlit Control Center (v3.0)
====================================================
V2 Pipeline Edition — Full observability for the memory_pipeline_v2 stack.

New in v3:
- Updated imports for memory_store_v2
- Retrieval trace shows intent_type, confidence, fallback, per-signal scores
- Per-question: final prompt preview, retrieved memory CONTENT (not just UUIDs)
- Memory Inspector: Entities, Facts, Timeline tabs (new V2 tables)
- Analysis: inline judge diagnostic with overlap scoring
- Analysis: avg_l2_peak in KPI row
- Ranking Engine audit log viewer (almond_audit.db)
- Improved design: tighter grid, cleaner expanders, score bar charts

Run from ROOT directory:
$ streamlit run almond_lab/app.py
"""

import json
import os
import re
import signal
import sqlite3
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
    from core.memory_store import MemoryStore, MemoryTier
    STORE_AVAILABLE = True
except ImportError:
    STORE_AVAILABLE = False


# ============================================================================
# CONSTANTS
# ============================================================================

RESULTS_DIR   = Path(root_dir) / "longmem_eval_results"
LOG_FILE      = Path(root_dir) / ".almond_live.log"
HISTORY_FILE  = Path(root_dir) / ".almond_run_history.json"
AUDIT_DB      = Path(root_dir) / "almond_audit.db"
TIMELINE_DB   = Path(root_dir) / "almond_timeline.db"
LM_STUDIO     = "http://localhost:1234"

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

SIGNAL_COLORS = {
    "similarity":         "#4A9EE0",
    "entity_overlap":     "#3ECFB2",
    "timeline_relevance": "#C8922A",
    "fact_confidence":    "#3CC97A",
    "type_match":         "#9B7FE8",
    "salience":           "#E05C5C",
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
    "is_running":       False,
    "process_pid":      None,
    "run_status":       "READY",
    "last_return_code": None,
    "last_run_config":  {},
    "run_start_time":   None,
}
for _k, _v in _defaults.items():
    if _k not in st.session_state:
        st.session_state[_k] = _v


# ============================================================================
# DESIGN SYSTEM — NEURAL AMBER v3
# ============================================================================

st.markdown("""
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@300;400;500;600&family=Syne:wght@600;700;800&display=swap" rel="stylesheet">

<style>
:root {
  --bg:#08080A; --bg-r:#0F0F13; --bg-s:#141418; --bg-o:#1A1A20;
  --border:rgba(200,146,42,.15); --border-hi:rgba(200,146,42,.4);
  --amber:#C8922A; --amber-d:#8A6118; --amber-g:rgba(200,146,42,.08);
  --mint:#3ECFB2; --red:#E05C5C; --green:#3CC97A; --blue:#4A9EE0;
  --purple:#9B7FE8;
  --text:#E8E2D5; --muted:#7A7265; --faint:#2A2520;
  --mono:'IBM Plex Mono',monospace; --display:'Syne',sans-serif;
  --r:4px; --rl:8px;
}
.stApp { background:var(--bg)!important; }
header[data-testid="stHeader"] { background:transparent!important; }
.main .block-container { padding:1.6rem 2.25rem 4rem!important; max-width:1520px!important; }
h1 { font-family:var(--display)!important; font-size:1.75rem!important; font-weight:800!important;
     color:var(--text)!important; letter-spacing:-.02em!important; margin-bottom:.1rem!important; }
h2,h3 { font-family:var(--display)!important; font-weight:700!important; color:var(--text)!important; }
h3 { font-size:.82rem!important; color:var(--amber)!important; text-transform:uppercase; letter-spacing:.08em!important; }
.stMarkdown, .stText, p, li { font-family:var(--mono); color:var(--text); }
.stMarkdown p { color:var(--muted)!important; font-size:.8rem!important; line-height:1.7!important; }
hr { border:none!important; border-top:1px solid var(--border)!important; margin:1.2rem 0!important; }
::-webkit-scrollbar { width:4px; height:4px; }
::-webkit-scrollbar-track { background:var(--bg); }
::-webkit-scrollbar-thumb { background:var(--amber-d); border-radius:99px; }
::-webkit-scrollbar-thumb:hover { background:var(--amber); }

/* SIDEBAR */
[data-testid="stSidebar"] { background:var(--bg-r)!important; border-right:1px solid var(--border)!important; }
[data-testid="stSidebar"] .block-container { padding:1.2rem 1rem!important; }
[data-testid="stSidebar"] hr { border-color:var(--border)!important; margin:.85rem 0!important; }
[data-testid="stSidebar"] .stRadio>label { display:none!important; }
[data-testid="stSidebar"] .stRadio div[role="radiogroup"] { gap:.2rem!important; display:flex; flex-direction:column; }
[data-testid="stSidebar"] .stRadio div[role="radiogroup"] label {
  display:flex!important; align-items:center!important; padding:.48rem .7rem!important;
  border-radius:var(--r)!important; border:1px solid transparent!important;
  font-size:.79rem!important; color:var(--muted)!important; cursor:pointer!important;
  transition:all .12s!important; }
[data-testid="stSidebar"] .stRadio div[role="radiogroup"] label:hover {
  background:var(--amber-g)!important; border-color:var(--border)!important; color:var(--text)!important; }

/* METRICS */
[data-testid="metric-container"] {
  background:var(--bg-s)!important; border:1px solid var(--border)!important;
  border-radius:var(--rl)!important; padding:.85rem 1rem!important; position:relative!important; overflow:hidden!important; }
[data-testid="metric-container"]::before {
  content:''; position:absolute; top:0; left:0; width:100%; height:2px;
  background:linear-gradient(90deg,var(--amber),transparent); }
[data-testid="metric-container"] [data-testid="stMetricLabel"] {
  font-size:.58rem!important; font-weight:500!important; color:var(--muted)!important;
  text-transform:uppercase!important; letter-spacing:.12em!important; }
[data-testid="metric-container"] [data-testid="stMetricValue"] {
  font-family:var(--display)!important; font-size:1.45rem!important; font-weight:800!important; color:var(--text)!important; }
[data-testid="metric-container"] [data-testid="stMetricDelta"] { font-size:.68rem!important; color:var(--mint)!important; }

/* BUTTONS */
.stButton button {
  font-family:var(--mono)!important; font-size:.77rem!important; font-weight:600!important;
  border-radius:var(--r)!important; border:1px solid var(--border-hi)!important;
  background:var(--bg-o)!important; color:var(--text)!important; transition:all .12s!important; }
.stButton button:hover { background:var(--amber-g)!important; border-color:var(--amber)!important; color:var(--amber)!important; }
.stButton button[kind="primary"] { background:var(--amber)!important; border-color:var(--amber)!important; color:#080808!important; font-weight:700!important; }
.stButton button[kind="primary"]:hover { background:#DFA02E!important; box-shadow:0 0 20px rgba(200,146,42,.3)!important; color:#080808!important; }
.stButton button[disabled] { opacity:.35!important; }

/* FORM */
.stSelectbox>div>div,.stTextInput>div>div>input {
  background:var(--bg-s)!important; border:1px solid var(--border)!important;
  border-radius:var(--r)!important; color:var(--text)!important; font-size:.8rem!important; }
.stTextInput>div>div>input { font-family:var(--mono)!important; }
label[data-testid="stWidgetLabel"] p {
  font-size:.62rem!important; color:var(--muted)!important;
  text-transform:uppercase!important; letter-spacing:.1em!important; font-weight:500!important; }
[data-baseweb="popover"] { background:var(--bg-o)!important; border:1px solid var(--border-hi)!important; border-radius:var(--r)!important; }
[data-baseweb="menu"] li { font-family:var(--mono)!important; font-size:.8rem!important; color:var(--text)!important; background:transparent!important; }
[data-baseweb="menu"] li:hover { background:var(--amber-g)!important; }
[data-testid="stSlider"] [data-baseweb="slider"] div[role="slider"] {
  background:var(--amber)!important; border:2px solid var(--amber)!important; box-shadow:0 0 8px rgba(200,146,42,.4)!important; }

/* PROGRESS */
[data-testid="stProgressBar"]>div { background:var(--faint)!important; border-radius:99px!important; height:5px!important; }
[data-testid="stProgressBar"]>div>div { background:var(--amber)!important; border-radius:99px!important; }

/* TABS */
.stTabs [data-baseweb="tab-list"] { background:var(--bg-r)!important; border-bottom:1px solid var(--border)!important; gap:0!important; padding:0!important; }
.stTabs [data-baseweb="tab"] {
  font-family:var(--mono)!important; font-size:.7rem!important; font-weight:600!important;
  text-transform:uppercase!important; letter-spacing:.08em!important; color:var(--muted)!important;
  background:transparent!important; border-bottom:2px solid transparent!important;
  padding:.6rem 1rem!important; transition:all .12s!important; }
.stTabs [aria-selected="true"] { color:var(--amber)!important; border-bottom-color:var(--amber)!important; }
.stTabs [data-baseweb="tab-panel"] {
  background:var(--bg-s)!important; border:1px solid var(--border)!important;
  border-top:none!important; border-radius:0 0 var(--rl) var(--rl)!important; padding:1.1rem!important; }

/* EXPANDERS */
.streamlit-expanderHeader {
  background:var(--bg-s)!important; border:1px solid var(--border)!important;
  border-radius:var(--r)!important; font-size:.77rem!important;
  color:var(--text)!important; padding:.6rem .85rem!important; transition:all .12s!important; }
.streamlit-expanderHeader:hover { border-color:var(--border-hi)!important; color:var(--amber)!important; }
.streamlit-expanderContent {
  background:var(--bg-r)!important; border:1px solid var(--border)!important;
  border-top:none!important; border-radius:0 0 var(--r) var(--r)!important; padding:.9rem!important; }

/* ALERTS */
div[data-testid="stNotification"] { font-family:var(--mono)!important; font-size:.77rem!important; border-radius:var(--r)!important; }
div[data-testid="stNotification"][data-type="info"]    { background:rgba(62,207,178,.06)!important; border-left:3px solid var(--mint)!important; }
div[data-testid="stNotification"][data-type="warning"] { background:rgba(200,146,42,.08)!important; border-left:3px solid var(--amber)!important; }
div[data-testid="stNotification"][data-type="error"]   { background:rgba(224,92,92,.07)!important; border-left:3px solid var(--red)!important; }
div[data-testid="stNotification"][data-type="success"] { background:rgba(60,201,122,.07)!important; border-left:3px solid var(--green)!important; }

/* CODE + DATAFRAME */
.stCodeBlock pre,.stCodeBlock code { background:var(--bg-r)!important; color:var(--mint)!important; font-family:var(--mono)!important; font-size:.75rem!important; }
[data-testid="stDataFrame"] { border:1px solid var(--border)!important; border-radius:var(--rl)!important; overflow:hidden!important; }

@keyframes pulse { 0%,100%{opacity:1} 50%{opacity:.25} }
@keyframes blink { 0%,100%{opacity:1} 50%{opacity:0} }
</style>
""", unsafe_allow_html=True)


# ============================================================================
# UTILITIES
# ============================================================================

def get_system_stats() -> dict:
    ram = psutil.virtual_memory()
    return {
        "ram_used":  round(ram.used / 1024**3, 2),
        "ram_total": round(ram.total / 1024**3, 2),
        "ram_pct":   ram.percent,
        "cpu_pct":   psutil.cpu_percent(interval=0.1),
    }


@st.cache_data(ttl=4)
def check_lm_studio() -> dict:
    import urllib.request
    try:
        with urllib.request.urlopen(f"{LM_STUDIO}/v1/models", timeout=2) as r:
            data   = json.loads(r.read())
            models = data.get("data", [])
            name   = models[0]["id"].split("/")[-1] if models else "Unknown"
            return {"ok": True, "model": name, "count": len(models)}
    except Exception:
        return {"ok": False, "model": None, "count": 0}


def run_preflight(dataset: str) -> list[dict]:
    checks = []
    ds_path = Path(root_dir) / dataset
    checks.append({"label":"Dataset File","ok":ds_path.exists(),"critical":True,
                   "detail":str(ds_path.name) if ds_path.exists() else f"Not found: {dataset}"})
    script = Path(root_dir) / "eval_unified.py"
    checks.append({"label":"eval_unified.py","ok":script.exists(),"critical":True,
                   "detail":"Found" if script.exists() else "Missing from root"})
    RESULTS_DIR.mkdir(exist_ok=True)
    checks.append({"label":"Results Dir","ok":True,"critical":False,"detail":RESULTS_DIR.name})
    lm = check_lm_studio()
    checks.append({"label":"LM Studio","ok":lm["ok"],"critical":True,
                   "detail":lm["model"] if lm["ok"] else "Offline"})
    major, minor = sys.version_info[:2]
    checks.append({"label":f"Python {major}.{minor}","ok":(major,minor)>=(3,9),"critical":False,
                   "detail":"OK" if (major,minor)>=(3,9) else "3.9+ recommended"})
    return checks


def get_log_lines() -> list[str]:
    if not LOG_FILE.exists():
        return []
    try:
        return LOG_FILE.read_text(encoding="utf-8", errors="replace").splitlines()
    except Exception:
        return []


_Q_RE   = re.compile(r'[Qq]uestion[\s#]*(\d+)\s*[/|]\s*(\d+)')
_TS_RE  = re.compile(r'\b(\d{2}:\d{2}:\d{2})\b')
_STAGES = [
    ("Loading Dataset",     ["loading","dataset loaded"]),
    ("Replaying Sessions",  ["replaying","session"]),
    ("Generating Response", ["generating","response"]),
    ("Judge Evaluation",    ["judge","evaluating"]),
    ("Saving Results",      ["saving","report"]),
    ("Complete",            ["evaluation done","complete","finished"]),
]


def parse_progress(lines: list[str]) -> dict:
    out = {"stage":"Initializing...","current_q":0,"total_q":0,"pct":0.0,"error":False}
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
        if any(e in ll for e in ["error","traceback","exception"]):
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
    if not pid:
        return False
    try:
        p = psutil.Process(pid)
        return p.is_running() and p.status() != psutil.STATUS_ZOMBIE
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        return False


def start_evaluation(dataset: str, limit: int, ablation: str):
    script = Path(root_dir) / "eval_unified.py"
    LOG_FILE.write_text("")
    log_f = open(LOG_FILE, "w", encoding="utf-8")
    proc  = subprocess.Popen(
        [sys.executable, str(script),
         "--dataset", dataset, "--limit", str(limit), "--ablation", ablation],
        cwd=root_dir, stdout=log_f, stderr=subprocess.STDOUT, text=True,
    )
    log_f.close()
    st.session_state.process_pid    = proc.pid
    st.session_state.is_running     = True
    st.session_state.run_status     = "RUNNING"
    st.session_state.run_start_time = time.time()
    st.session_state.last_run_config = {"dataset":dataset,"limit":limit,"ablation":ablation}


def cancel_evaluation():
    pid = st.session_state.get("process_pid")
    if pid:
        try:
            os.kill(pid, signal.SIGTERM)
        except Exception:
            pass
    st.session_state.is_running  = False
    st.session_state.run_status  = "CANCELLED"
    st.session_state.process_pid = None


def get_all_reports() -> list[Path]:
    if not RESULTS_DIR.exists():
        return []
    reports = (list(RESULTS_DIR.glob("longmem_report*.json"))
               + list(RESULTS_DIR.glob("partial_report.json")))
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
        "date":           datetime.now().strftime("%Y-%m-%d %H:%M"),
        "dataset":        config.get("dataset","?"),
        "limit":          config.get("limit","?"),
        "ablation":       config.get("ablation","none"),
        "accuracy":       summary.get("accuracy",0),
        "avg_latency_ms": summary.get("avg_latency_ms",0),
        "avg_l2_peak":    summary.get("avg_l2_peak",0),
        "avg_pollution":  summary.get("avg_pollution",0),
        "report_file":    report_name,
    })
    HISTORY_FILE.write_text(json.dumps(h[:50], indent=2))


# ── Judge diagnostic helper ──────────────────────────────────────────────────
_STOPWORDS = {"the","a","an","i","you","it","was","is","did","do","my","your",
               "first","before","after","when","which","what","this","that"}

def _key_tokens(text: str) -> set[str]:
    return {w for w in re.sub(r"[^\w\s]","",text.lower()).split()
            if w not in _STOPWORDS and len(w) > 2}

def semantic_overlap(expected: str, response: str) -> float:
    exp = _key_tokens(expected)
    if not exp:
        return 0.0
    return len(exp & _key_tokens(response)) / len(exp)


# ── SQLite helpers for new V2 tables ────────────────────────────────────────
def query_db(db_path: Path, sql: str, params=()) -> list[dict]:
    if not db_path.exists():
        return []
    try:
        conn = sqlite3.connect(str(db_path))
        conn.row_factory = sqlite3.Row
        rows = conn.execute(sql, params).fetchall()
        conn.close()
        return [dict(r) for r in rows]
    except Exception:
        return []


def get_audit_entries(limit: int = 50) -> list[dict]:
    return query_db(AUDIT_DB,
        "SELECT * FROM retrieval_audit ORDER BY timestamp DESC LIMIT ?", (limit,))


def get_timeline_events(limit: int = 200) -> list[dict]:
    return query_db(TIMELINE_DB,
        "SELECT id, description, event_type, earliest, latest, "
        "temporal_confidence, date_raw FROM timeline_events "
        "ORDER BY earliest ASC LIMIT ?", (limit,))


def get_structured_facts(db_path: Path, limit: int = 500) -> list[dict]:
    return query_db(db_path,
        "SELECT id, memory_id, subject, predicate, object, fact_type, "
        "confidence, date_raw, earliest, latest FROM structured_facts "
        "ORDER BY confidence DESC LIMIT ?", (limit,))


def get_entities(db_path: Path) -> list[dict]:
    return query_db(db_path,
        "SELECT id, name, type, aliases, memory_ids, reference_count, needs_review "
        "FROM entities ORDER BY reference_count DESC")


# ============================================================================
# HTML HELPERS
# ============================================================================

def _badge(status: str) -> str:
    c    = STATUS_COLOR.get(status, "#7A7265")
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


def _intent_badge(intent: str, conf: float) -> str:
    colors = {
        "TEMPORAL":     "#C8922A",
        "COMPARISON":   "#3ECFB2",
        "FACTUAL":      "#4A9EE0",
        "EVENT":        "#9B7FE8",
        "RELATIONSHIP": "#3CC97A",
        "AMBIGUOUS":    "#7A7265",
    }
    c = colors.get(intent, "#7A7265")
    return (
        f'<span style="background:{c}22;border:1px solid {c}55;color:{c};'
        f'font-size:.62rem;font-weight:700;letter-spacing:.08em;padding:.18rem .5rem;'
        f'border-radius:3px;font-family:var(--mono);">'
        f'{intent} {int(conf*100)}%</span>'
    )


def _score_bars(breakdown: dict) -> str:
    """Render mini horizontal bar chart for signal contributions."""
    signals = ["similarity","entity_overlap","timeline_relevance",
               "fact_confidence","type_match","salience"]
    weights = breakdown.get("weights_used", {})
    rows = ""
    for sig in signals:
        val = breakdown.get(sig, 0.0)
        w   = weights.get(sig, 0.0)
        contrib = val * w
        c   = SIGNAL_COLORS.get(sig, "#7A7265")
        pct = int(val * 100)
        rows += (
            f'<div style="display:grid;grid-template-columns:120px 1fr 40px;'
            f'gap:.4rem;align-items:center;margin-bottom:.25rem;">'
            f'<span style="font-size:.63rem;color:#7A7265;text-align:right;'
            f'font-family:var(--mono);">{sig.replace("_"," ")}</span>'
            f'<div style="background:#1A1A20;border-radius:99px;height:5px;position:relative;">'
            f'<div style="width:{pct}%;background:{c};border-radius:99px;height:5px;'
            f'box-shadow:0 0 4px {c}66;"></div></div>'
            f'<span style="font-size:.63rem;color:{c};font-family:var(--mono);">'
            f'{val:.2f}×{w:.2f}</span>'
            f'</div>'
        )
    final = breakdown.get("final_score", 0.0)
    rows += (
        f'<div style="margin-top:.4rem;padding-top:.4rem;border-top:1px solid rgba(200,146,42,.15);'
        f'display:flex;justify-content:space-between;">'
        f'<span style="font-size:.63rem;color:#7A7265;font-family:var(--mono);">final score</span>'
        f'<span style="font-size:.72rem;color:#C8922A;font-weight:700;font-family:var(--mono);">'
        f'{final:.4f}</span></div>'
    )
    return f'<div style="padding:.5rem;">{rows}</div>'


def _mem_card(content: str, score: float, source: str,
              reasoning: str = "", breakdown: dict | None = None) -> str:
    src_c = {"timeline":"#C8922A","entity_registry":"#3ECFB2",
             "vector":"#4A9EE0","semantic_fallback":"#7A7265"}.get(source,"#7A7265")
    return (
        f'<div style="background:var(--bg-r);border:1px solid rgba(200,146,42,.12);'
        f'border-radius:6px;padding:.7rem .85rem;margin-bottom:.4rem;">'
        f'<div style="display:flex;justify-content:space-between;align-items:flex-start;margin-bottom:.35rem;">'
        f'<span style="background:{src_c}22;border:1px solid {src_c}44;color:{src_c};'
        f'font-size:.58rem;font-family:var(--mono);padding:.12rem .4rem;border-radius:3px;">{source}</span>'
        f'<span style="font-size:.7rem;color:#C8922A;font-weight:700;font-family:var(--mono);">▲ {score:.4f}</span>'
        f'</div>'
        f'<div style="font-size:.77rem;color:#E8E2D5;line-height:1.55;margin-bottom:{".35rem" if reasoning else "0"};">'
        f'{content[:280]}{"..." if len(content)>280 else ""}</div>'
        + (f'<div style="font-size:.65rem;color:#7A7265;font-style:italic;">{reasoning}</div>' if reasoning else '')
        + '</div>'
    )


# ============================================================================
# SIDEBAR
# ============================================================================

with st.sidebar:
    st.markdown(
        '<div style="display:flex;align-items:center;gap:.55rem;margin-bottom:1.1rem;">'
        '<span style="font-size:1.25rem;">🧠</span>'
        '<div><div style="font-family:\'Syne\',sans-serif;font-size:1.05rem;font-weight:800;'
        'color:#C8922A;line-height:1;">ALMOND LAB</div>'
        '<div style="font-size:.52rem;color:#7A7265;text-transform:uppercase;letter-spacing:.16em;margin-top:2px;">'
        'V2 Pipeline · Mission Control</div></div></div>',
        unsafe_allow_html=True,
    )

    page = st.radio(
        "nav",
        ["🚀  Mission Control", "📊  Analysis", "🔬  Retrieval Debugger",
         "🧬  Memory Inspector", "📈  Run History"],
        label_visibility="collapsed",
    )

    st.markdown("<hr>", unsafe_allow_html=True)

    elapsed_str = ""
    if st.session_state.is_running and st.session_state.run_start_time:
        e = int(time.time() - st.session_state.run_start_time)
        elapsed_str = f' <span style="color:#3A352D;font-size:.6rem;">({e//60:02d}:{e%60:02d})</span>'

    st.markdown(
        f'<div style="margin-bottom:.7rem;">'
        f'<div style="font-size:.55rem;color:#7A7265;text-transform:uppercase;'
        f'letter-spacing:.12em;margin-bottom:.35rem;">Run Status</div>'
        f'{_badge(st.session_state.run_status)}{elapsed_str}</div>',
        unsafe_allow_html=True,
    )

    lm_sb = check_lm_studio()
    if lm_sb["ok"]:
        model_disp = (lm_sb["model"] or "Unknown")[:22]
        st.markdown(
            f'<div style="background:rgba(60,201,122,.06);border:1px solid rgba(60,201,122,.25);'
            f'border-radius:4px;padding:.55rem .8rem;">'
            f'<div style="color:#3CC97A;font-size:.56rem;text-transform:uppercase;letter-spacing:.12em;'
            f'font-weight:700;margin-bottom:.25rem;">● LM Studio · Online</div>'
            f'<div style="color:#E8E2D5;font-size:.76rem;">{model_disp}</div></div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<div style="background:rgba(224,92,92,.06);border:1px solid rgba(224,92,92,.25);'
            'border-radius:4px;padding:.55rem .8rem;">'
            '<div style="color:#E05C5C;font-size:.56rem;text-transform:uppercase;letter-spacing:.12em;'
            'font-weight:700;margin-bottom:.25rem;">○ LM Studio · Offline</div>'
            '<div style="color:#7A7265;">Start LM Studio to enable eval</div></div>',
            unsafe_allow_html=True,
        )

    st.markdown("<hr>", unsafe_allow_html=True)
    stats = get_system_stats()
    st.markdown('<div style="font-size:.55rem;color:#7A7265;text-transform:uppercase;'
                'letter-spacing:.13em;margin-bottom:.5rem;">System</div>', unsafe_allow_html=True)
    st.metric("RAM", f"{stats['ram_used']}/{stats['ram_total']} GB", f"{stats['ram_pct']}%")
    st.metric("CPU", f"{stats['cpu_pct']}%")
    if st.session_state.is_running and st.session_state.run_start_time:
        e = int(time.time() - st.session_state.run_start_time)
        st.metric("Elapsed", f"{e//60:02d}:{e%60:02d}")


# ============================================================================
# PAGE 1 — MISSION CONTROL
# ============================================================================

if "Mission" in page:

    if st.session_state.is_running and not process_alive():
        lines   = get_log_lines()
        tail    = "\n".join(lines[-25:]).lower()
        done    = any(w in tail for w in ["evaluation done","report saved","completed","success","final results"])
        st.session_state.run_status  = "COMPLETED" if done else "FAILED"
        st.session_state.is_running  = False
        st.session_state.process_pid = None
        if done:
            reports = get_all_reports()
            if reports and st.session_state.last_run_config:
                try:
                    rdata = json.loads(reports[0].read_text())
                    append_history(
                        st.session_state.last_run_config,
                        rdata.get("summary",{}),
                        reports[0].name,
                    )
                except Exception:
                    pass

    h_col, status_col = st.columns([3, 1])
    with h_col:
        st.markdown("<h1>Mission Control</h1>", unsafe_allow_html=True)
        st.markdown('<p style="color:#7A7265;font-size:.8rem;margin-bottom:1.4rem;">'
                    'Configure, launch, and observe evaluation runs in real time.</p>',
                    unsafe_allow_html=True)
    with status_col:
        st.markdown(f'<div style="display:flex;justify-content:flex-end;padding-top:.35rem;">'
                    f'{_badge(st.session_state.run_status)}</div>', unsafe_allow_html=True)

    pf_col, lm_col = st.columns([1.15, 1], gap="large")
    check_dataset  = st.session_state.last_run_config.get("dataset", DATASETS[0])
    checks         = run_preflight(check_dataset)
    all_crit       = all(c["ok"] for c in checks if c["critical"])

    with pf_col:
        _slabel("Pre-flight Checks", "0")
        rows = ""
        for c in checks:
            icon, col = ("✓","#3CC97A") if c["ok"] else (("✗","#E05C5C") if c["critical"] else ("⚠","#C8922A"))
            rows += (
                f'<div style="display:flex;align-items:center;gap:.65rem;padding:.38rem .7rem;'
                f'background:rgba(255,255,255,.02);border:1px solid rgba(255,255,255,.04);border-radius:4px;">'
                f'<span style="color:{col};font-weight:700;font-size:.82rem;min-width:.9rem;">{icon}</span>'
                f'<div><div style="font-size:.78rem;color:#E8E2D5;">{c["label"]}</div>'
                f'<div style="font-size:.66rem;color:#7A7265;">{c["detail"]}</div></div></div>'
            )
        st.markdown(f'<div style="display:flex;flex-direction:column;gap:.28rem;">{rows}</div>',
                    unsafe_allow_html=True)
        if not all_crit:
            st.error("Critical checks failed — resolve before launching.")

    with lm_col:
        _slabel("LM Studio Monitor", "0")
        lm = check_lm_studio()
        if lm["ok"]:
            st.markdown(
                f'<div style="background:rgba(60,201,122,.04);border:1px solid rgba(60,201,122,.2);'
                f'border-radius:8px;padding:.9rem 1rem;">'
                f'<div style="font-size:.56rem;text-transform:uppercase;letter-spacing:.12em;'
                f'color:#3CC97A;font-weight:700;margin-bottom:.6rem;">● Connected</div>'
                f'<div style="font-size:.82rem;color:#E8E2D5;margin-bottom:.2rem;">{lm["model"]}</div>'
                f'<div style="font-size:.68rem;color:#7A7265;">localhost:1234 · {lm["count"]} model(s)</div></div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                '<div style="background:rgba(224,92,92,.04);border:1px solid rgba(224,92,92,.2);'
                'border-radius:8px;padding:.9rem 1rem;">'
                '<div style="font-size:.56rem;text-transform:uppercase;letter-spacing:.12em;'
                'color:#E05C5C;font-weight:700;margin-bottom:.6rem;">○ Offline</div>'
                '<div style="font-size:.78rem;color:#7A7265;line-height:1.6;">'
                'Open LM Studio and load a model.</div></div>',
                unsafe_allow_html=True,
            )

    st.markdown("<hr>", unsafe_allow_html=True)
    _slabel("Evaluation Configuration")
    cfg1, cfg2, cfg3 = st.columns(3)
    with cfg1:
        dataset  = st.selectbox("Dataset",  DATASETS,  disabled=st.session_state.is_running)
    with cfg2:
        limit    = st.slider("Question Limit", 1, 500, 5, disabled=st.session_state.is_running)
    with cfg3:
        ablation = st.selectbox("Ablation Mode", ABLATIONS, disabled=st.session_state.is_running)

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
            cfg_str = (f' <span style="color:#3A352D;font-size:.65rem;">'
                       f'· {Path(c.get("dataset","")).name} · {c.get("limit","?")}q '
                       f'· ablation:{c.get("ablation","?")}</span>')
        st.markdown(f'<div style="display:flex;align-items:center;padding-top:.5rem;gap:.5rem;">'
                    f'{_badge(st.session_state.run_status)}{cfg_str}</div>',
                    unsafe_allow_html=True)

    if st.session_state.run_status in ("RUNNING","COMPLETED","FAILED"):
        st.markdown("<hr>", unsafe_allow_html=True)
        _slabel("Progress")
        lines = get_log_lines()
        prog  = parse_progress(lines)
        pct   = 1.0 if st.session_state.run_status == "COMPLETED" else prog["pct"]
        pb_col, pct_col = st.columns([4, 1])
        with pb_col:
            st.progress(pct)
        with pct_col:
            label = f"Q {prog['current_q']} / {prog['total_q']}" if prog["total_q"] > 0 else f"{int(pct*100)}%"
            st.markdown(f'<div style="font-size:.76rem;color:#7A7265;padding-top:.32rem;text-align:right;">{label}</div>',
                        unsafe_allow_html=True)
        s_color = {"RUNNING":"#3ECFB2","COMPLETED":"#C8922A","FAILED":"#E05C5C"}.get(st.session_state.run_status,"#7A7265")
        s_icon  = "●" if st.session_state.run_status == "RUNNING" else ("✓" if st.session_state.run_status == "COMPLETED" else "✗")
        st.markdown(
            f'<div style="font-size:.8rem;margin-top:.2rem;">'
            f'<span style="color:{s_color};">{s_icon}</span>'
            f'<span style="color:#7A7265;margin:0 .4rem;">Stage:</span>'
            f'<span style="color:{s_color};">{prog["stage"]}</span>'
            + ('<span style="color:#E05C5C;margin-left:.75rem;font-size:.72rem;">⚠ Errors in log</span>' if prog["error"] else '')
            + '</div>',
            unsafe_allow_html=True,
        )

    if st.session_state.run_status in ("RUNNING","COMPLETED","FAILED","CANCELLED"):
        st.markdown("<hr>", unsafe_allow_html=True)
        _slabel("Live Log Console")
        lines = get_log_lines()
        if lines:
            log_rows = []
            for raw in lines[-400:]:
                line = raw.rstrip()
                if not line:
                    continue
                ts_m = _TS_RE.search(line)
                ts   = ts_m.group(1) if ts_m else "        "
                ll   = line.lower()
                if any(e in ll for e in ["error","traceback","exception","failed"]):
                    txt_c = "#E05C5C"
                elif any(w in ll for w in ["passed","pass ✓","success","complete","done","saved"]):
                    txt_c = "#3CC97A"
                elif any(w in ll for w in ["warn","[orphan]","[judge"]):
                    txt_c = "#C8922A"
                elif any(w in ll for w in ["retriev","generat","loading","replaying","page-in","intent"]):
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
                'border-radius:6px;padding:.8rem 1rem;max-height:420px;overflow-y:auto;'
                'font-family:\'IBM Plex Mono\',monospace;font-size:.72rem;line-height:1.55;">'
                + "\n".join(log_rows)
                + '<div id="logend"></div></div>'
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

    if st.session_state.is_running:
        time.sleep(1.5)
        st.rerun()


# ============================================================================
# PAGE 2 — ANALYSIS
# ============================================================================

elif "Analysis" in page:
    st.markdown("<h1>Analysis</h1>", unsafe_allow_html=True)
    st.markdown('<p style="color:#7A7265;font-size:.8rem;margin-bottom:1.4rem;">'
                'Benchmark results, failure analysis, and retrieval quality.</p>',
                unsafe_allow_html=True)

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

        _slabel("Summary Metrics", "0")
        k1, k2, k3, k4, k5 = st.columns(5)
        k1.metric("Accuracy",    f"{summary.get('accuracy',0)}%")
        k2.metric("Questions",   f"{summary.get('questions',0)}")
        k3.metric("Avg Latency", f"{summary.get('avg_latency_ms',0):.0f} ms")
        k4.metric("Avg L2 Peak", f"{summary.get('avg_l2_peak',0)}")
        k5.metric("Avg Retrieved", f"{summary.get('avg_retrieved_blocks',0)}")

        st.markdown("<hr>", unsafe_allow_html=True)
        p1, p2, p3, p4 = st.columns(4)
        p1.metric("Ablation",         summary.get("ablation","none"))
        p2.metric("Avg Replay",       f"{summary.get('avg_replay_time_ms',0):.0f} ms")
        p3.metric("Avg Pollution",    f"{summary.get('avg_pollution',0)}")
        p4.metric("Elapsed",          f"{summary.get('elapsed_seconds',0):.0f}s")

        results = data.get("results", [])
        traces  = {t.get("query","").strip(): t for t in data.get("retrieval_traces", [])}

        tab_all, tab_fail, tab_pass, tab_diag = st.tabs([
            "All Results", "Failure Analysis", "Passing Results", "Judge Diagnostic"
        ])

        with tab_all:
            if results:
                df = pd.DataFrame([{
                    "Q":       r.get("index","?"),
                    "Pass":    "✓" if r.get("passed") else "✗",
                    "Type":    r.get("question_type",""),
                    "Question": r.get("question","")[:65]+"...",
                    "Expected": r.get("expected_answer","")[:40],
                    "L2 Peak": r.get("l2_peak",0),
                    "L3 Peak": r.get("l3_peak",0),
                    "Latency": f"{r.get('latency_ms',0):.0f}ms",
                } for r in results])
                st.dataframe(df, use_container_width=True, height=420)
            else:
                st.info("No results.")

        with tab_fail:
            failed = [r for r in results if not r.get("passed", True)]
            if not failed:
                st.success("All questions passed in this run.")
            else:
                _slabel(f"{len(failed)} Failure(s) — Root Cause Analysis", "0")
                for r in failed:
                    tr = traces.get(r.get("question","").strip(), {})
                    retrieved_ct = len(tr.get("retrieved_ids", []))
                    pollution    = tr.get("pollution_score", 0)
                    overlap      = semantic_overlap(r.get("expected_answer",""), r.get("model_response",""))

                    if retrieved_ct == 0:
                        root = "No relevant memory reached context — retrieval or page-in failed."
                    elif float(pollution or 0) > 0.6:
                        root = f"High pollution ({pollution:.2f}) — irrelevant memories dominated context."
                    elif overlap > 0.5:
                        root = "Memory reached context but model drew wrong conclusion (judge or reasoning failure)."
                    else:
                        root = "Memory retrieved but model response doesn't contain the answer."

                    with st.expander(f"✗ Q{r.get('index','?')} — {r.get('question','')[:70]}..."):
                        fc1, fc2 = st.columns(2)
                        with fc1:
                            st.markdown(
                                f'<div style="font-size:.68rem;color:#7A7265;text-transform:uppercase;'
                                f'letter-spacing:.1em;margin-bottom:.3rem;">Expected</div>'
                                f'<div style="font-size:.82rem;color:#3CC97A;background:rgba(60,201,122,.05);'
                                f'padding:.5rem .75rem;border-radius:4px;border:1px solid rgba(60,201,122,.2);">'
                                f'{r.get("expected_answer","—")}</div>',
                                unsafe_allow_html=True,
                            )
                        with fc2:
                            st.markdown(
                                f'<div style="font-size:.68rem;color:#7A7265;text-transform:uppercase;'
                                f'letter-spacing:.1em;margin-bottom:.3rem;">Model Response</div>'
                                f'<div style="font-size:.82rem;color:#E05C5C;background:rgba(224,92,92,.05);'
                                f'padding:.5rem .75rem;border-radius:4px;border:1px solid rgba(224,92,92,.2);">'
                                f'{r.get("model_response","—")[:280]}</div>',
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
                            f'Retrieved: {retrieved_ct} · Pollution: {pollution} · '
                            f'Answer overlap: {overlap:.0%}</span></div>',
                            unsafe_allow_html=True,
                        )

        with tab_pass:
            passed = [r for r in results if r.get("passed", False)]
            if not passed:
                st.warning("No questions passed in this run.")
            else:
                _slabel(f"{len(passed)} Passing Results", "0")
                for r in passed:
                    with st.expander(f"✓ Q{r.get('index','?')} — {r.get('question','')[:70]}..."):
                        pc1, pc2 = st.columns(2)
                        with pc1:
                            st.markdown(
                                f'<div style="font-size:.68rem;color:#7A7265;text-transform:uppercase;'
                                f'margin-bottom:.3rem;">Expected</div>'
                                f'<div style="color:#3CC97A;font-size:.82rem;padding:.5rem .75rem;'
                                f'background:rgba(60,201,122,.05);border:1px solid rgba(60,201,122,.2);'
                                f'border-radius:4px;">{r.get("expected_answer","—")}</div>',
                                unsafe_allow_html=True,
                            )
                        with pc2:
                            st.markdown(
                                f'<div style="font-size:.68rem;color:#7A7265;text-transform:uppercase;'
                                f'margin-bottom:.3rem;">Model Response</div>'
                                f'<div style="color:#3ECFB2;font-size:.78rem;padding:.5rem .75rem;'
                                f'background:rgba(62,207,178,.04);border:1px solid rgba(62,207,178,.2);'
                                f'border-radius:4px;">{r.get("model_response","—")[:280]}</div>',
                                unsafe_allow_html=True,
                            )
                        st.markdown(
                            f'<div style="margin-top:.5rem;font-size:.68rem;color:#7A7265;">'
                            f'Latency: {r.get("latency_ms",0):.0f}ms · '
                            f'L2 peak: {r.get("l2_peak",0)} · '
                            f'L3 peak: {r.get("l3_peak",0)} · '
                            f'Replay: {r.get("replay_time_ms",0):.0f}ms</div>',
                            unsafe_allow_html=True,
                        )

        with tab_diag:
            _slabel("Judge Diagnostic — Semantic Overlap Analysis", "0")
            st.markdown(
                '<div style="font-size:.74rem;color:#7A7265;margin-bottom:.8rem;">'
                'Overlap ≥ 60% with wrong verdict → likely judge error. '
                'Overlap &lt; 30% with wrong verdict → retrieval or reasoning failure.</div>',
                unsafe_allow_html=True,
            )
            diag_rows = []
            judge_errors = 0
            for r in results:
                overlap = semantic_overlap(r.get("expected_answer",""), r.get("model_response",""))
                passed  = r.get("passed", False)
                verdict = "TRUE_PASS" if passed else ("JUDGE_ERROR" if overlap >= 0.6 else "TRUE_FAIL")
                if verdict == "JUDGE_ERROR":
                    judge_errors += 1
                diag_rows.append({
                    "Q":        r.get("index","?"),
                    "Verdict":  verdict,
                    "Passed":   "✓" if passed else "✗",
                    "Overlap":  f"{overlap:.0%}",
                    "Expected": r.get("expected_answer","")[:40],
                    "Response": r.get("model_response","")[:60]+"...",
                })

            if judge_errors > 0:
                st.warning(f"⚠ {judge_errors} likely judge error(s) detected — correct answers marked FAILED.")
            else:
                st.success("No judge errors detected — all verdicts appear consistent with semantic content.")

            st.dataframe(pd.DataFrame(diag_rows), use_container_width=True, height=360)


# ============================================================================
# PAGE 3 — RETRIEVAL DEBUGGER
# ============================================================================

elif "Retrieval" in page:
    st.markdown("<h1>Retrieval Debugger</h1>", unsafe_allow_html=True)
    st.markdown('<p style="color:#7A7265;font-size:.8rem;margin-bottom:1.4rem;">'
                'Per-question: intent, retrieved memory content, prompt preview, ranking scores.</p>',
                unsafe_allow_html=True)

    reports = get_all_reports()
    if not reports:
        st.info("No reports found. Run a benchmark first.")
    else:
        selected_name = st.selectbox("Report", [r.name for r in reports], key="ret_report")
        try:
            data = json.loads((RESULTS_DIR / selected_name).read_text(encoding="utf-8"))
        except Exception as e:
            st.error(f"Cannot read report: {e}")
            st.stop()

        results = data.get("results", [])
        traces  = {t.get("query","").strip(): t for t in data.get("retrieval_traces", [])}

        # Audit log from ranking engine
        audit_entries = get_audit_entries(200)
        audit_by_query = {}
        for entry in audit_entries:
            q = entry.get("query","").strip()
            if q not in audit_by_query:
                audit_by_query[q] = entry

        _slabel(f"{len(results)} Questions", "0")

        for r in results:
            q       = r.get("question","")
            passed  = r.get("passed", False)
            icon    = "✓" if passed else "✗"
            ic      = "#3CC97A" if passed else "#E05C5C"
            tr      = traces.get(q.strip(), {})
            audit   = audit_by_query.get(q.strip(), {})
            intent  = tr.get("intent_type") or audit.get("intent_type","?")
            conf    = float(tr.get("intent_confidence") or audit.get("intent_confidence") or 0)
            fallback = tr.get("used_fallback", False)
            paged_in = tr.get("retrieved_ids", [])

            header = (
                f'<span style="color:{ic};margin-right:.5rem;">{icon}</span>'
                f'Q{r.get("index","?")} · {q[:65]}{"..." if len(q)>65 else ""}'
            )

            with st.expander(header):
                # Row 1: intent + timing
                i1, i2, i3, i4 = st.columns(4)
                with i1:
                    if intent and intent != "?":
                        st.markdown(_intent_badge(intent, conf), unsafe_allow_html=True)
                    else:
                        st.markdown('<span style="font-size:.68rem;color:#7A7265;">intent: n/a</span>',
                                    unsafe_allow_html=True)
                with i2:
                    fb_c = "#E05C5C" if fallback else "#3CC97A"
                    st.markdown(
                        f'<span style="font-size:.68rem;color:{fb_c};font-family:var(--mono);">'
                        f'{"⚠ semantic fallback" if fallback else "✓ timeline/entity path"}</span>',
                        unsafe_allow_html=True,
                    )
                with i3:
                    st.markdown(
                        f'<span style="font-size:.68rem;color:#7A7265;font-family:var(--mono);">'
                        f'retrieved: {tr.get("retrieved_count",len(paged_in))} · '
                        f'paged_in: {len(paged_in)}</span>',
                        unsafe_allow_html=True,
                    )
                with i4:
                    st.markdown(
                        f'<span style="font-size:.68rem;color:#7A7265;font-family:var(--mono);">'
                        f'L2 peak: {r.get("l2_peak",0)} · latency: {r.get("latency_ms",0):.0f}ms</span>',
                        unsafe_allow_html=True,
                    )

                st.markdown("<hr>", unsafe_allow_html=True)

                # Ranking audit — top5 with score bars
                top5_ids       = []
                top5_breakdowns = []
                if audit:
                    try:
                        top5_ids        = json.loads(audit.get("top5_ids","[]"))
                        top5_breakdowns = json.loads(audit.get("top5_breakdowns","[]"))
                    except Exception:
                        pass

                rt_col, prompt_col = st.columns([1, 1], gap="large")

                with rt_col:
                    _slabel("Top Ranked Memories")
                    if top5_ids:
                        for i, (mid, bd) in enumerate(zip(top5_ids, top5_breakdowns)):
                            score = bd.get("final_score", 0.0)
                            reasoning = bd.get("reasoning","")
                            # Try to get content from audit — may need DB lookup
                            content = f"[memory ID: {mid[:8]}...]"
                            st.markdown(
                                f'<div style="background:var(--bg-r);border:1px solid rgba(200,146,42,.12);'
                                f'border-radius:6px;padding:.6rem .8rem;margin-bottom:.35rem;">'
                                f'<div style="display:flex;justify-content:space-between;margin-bottom:.3rem;">'
                                f'<span style="font-size:.62rem;color:#7A7265;font-family:var(--mono);">#{i+1} · {mid[:12]}...</span>'
                                f'<span style="font-size:.7rem;color:#C8922A;font-weight:700;font-family:var(--mono);">▲ {score:.4f}</span></div>'
                                + _score_bars(bd)
                                + (f'<div style="font-size:.63rem;color:#7A7265;margin-top:.25rem;'
                                   f'font-style:italic;">{reasoning}</div>' if reasoning else '')
                                + '</div>',
                                unsafe_allow_html=True,
                            )
                    elif paged_in:
                        for mid in paged_in[:5]:
                            st.markdown(
                                f'<div style="font-size:.72rem;color:#3ECFB2;'
                                f'font-family:var(--mono);padding:.3rem 0;">✓ {mid[:16]}...</div>',
                                unsafe_allow_html=True,
                            )
                    else:
                        st.markdown('<span style="font-size:.73rem;color:#7A7265;">No ranking data.</span>',
                                    unsafe_allow_html=True)

                with prompt_col:
                    _slabel("Final Prompt Preview")
                    expected = r.get("expected_answer","")
                    response = r.get("model_response","")
                    st.markdown(
                        f'<div style="background:#04040A;border:1px solid rgba(200,146,42,.1);'
                        f'border-radius:6px;padding:.75rem;font-family:var(--mono);font-size:.72rem;'
                        f'line-height:1.6;max-height:260px;overflow-y:auto;">'
                        f'<div style="color:#7A7265;font-size:.58rem;text-transform:uppercase;'
                        f'letter-spacing:.1em;margin-bottom:.4rem;">[SYSTEM]</div>'
                        f'<div style="color:#3ECFB2;margin-bottom:.6rem;">You are Almond...</div>'
                        f'<div style="color:#7A7265;font-size:.58rem;text-transform:uppercase;'
                        f'letter-spacing:.1em;margin-bottom:.4rem;">[USER]</div>'
                        f'<div style="color:#E8E2D5;">{q}</div>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )
                    _slabel("Expected vs Actual")
                    st.markdown(
                        f'<div style="background:rgba(60,201,122,.04);border:1px solid rgba(60,201,122,.15);'
                        f'border-radius:4px;padding:.4rem .6rem;font-size:.74rem;color:#3CC97A;margin-bottom:.3rem;">'
                        f'<span style="font-size:.56rem;color:#7A7265;text-transform:uppercase;display:block;margin-bottom:.2rem;">Expected</span>'
                        f'{expected}</div>'
                        f'<div style="background:rgba(224,92,92,.04);border:1px solid rgba(224,92,92,.15);'
                        f'border-radius:4px;padding:.4rem .6rem;font-size:.74rem;'
                        f'color:{"#3CC97A" if passed else "#E05C5C"};">'
                        f'<span style="font-size:.56rem;color:#7A7265;text-transform:uppercase;display:block;margin-bottom:.2rem;">Model</span>'
                        f'{response[:200]}{"..." if len(response)>200 else ""}</div>',
                        unsafe_allow_html=True,
                    )

        # Ranking Engine Audit Log
        st.markdown("<hr>", unsafe_allow_html=True)
        _slabel("Ranking Engine Audit Log (almond_audit.db)")
        if not AUDIT_DB.exists():
            st.info("almond_audit.db not found. Audit log is written during live runs.")
        else:
            entries = get_audit_entries(30)
            if not entries:
                st.info("Audit log is empty.")
            else:
                for entry in entries[:10]:
                    ts = entry.get("timestamp","")[:19]
                    q  = entry.get("query","")
                    it = entry.get("intent_type","?")
                    ic = float(entry.get("intent_confidence",0))
                    fb = bool(entry.get("used_fallback",0))
                    with st.expander(f"[{ts}] {it} · {q[:55]}..."):
                        try:
                            bds = json.loads(entry.get("top5_breakdowns","[]"))
                            ids = json.loads(entry.get("top5_ids","[]"))
                            sc  = json.loads(entry.get("top5_scores","[]"))
                        except Exception:
                            bds, ids, sc = [], [], []

                        st.markdown(
                            f'<div style="display:flex;gap:1rem;font-size:.68rem;'
                            f'color:#7A7265;margin-bottom:.6rem;">'
                            f'<span>{_intent_badge(it, ic)}</span>'
                            f'<span>fallback: <span style="color:{"#E05C5C" if fb else "#3CC97A"};">'
                            f'{"yes" if fb else "no"}</span></span></div>',
                            unsafe_allow_html=True,
                        )
                        for i, (mid, bd, s) in enumerate(zip(ids, bds, sc)):
                            st.markdown(
                                f'<div style="margin-bottom:.4rem;">'
                                f'<div style="font-size:.62rem;color:#7A7265;margin-bottom:.15rem;">#{i+1} {mid[:16]}... score={s:.4f}</div>'
                                + _score_bars(bd) + '</div>',
                                unsafe_allow_html=True,
                            )


# ============================================================================
# PAGE 4 — MEMORY INSPECTOR
# ============================================================================

elif "Memory" in page:
    h1c, h2c = st.columns([0.85, 0.15])
    with h1c:
        st.markdown("<h1>Memory Inspector</h1>", unsafe_allow_html=True)
        st.markdown('<p style="color:#7A7265;font-size:.8rem;margin-bottom:1.4rem;">'
                    'Browse the live SQLite T-MMU vault — blocks, entities, facts, and timeline events.</p>',
                    unsafe_allow_html=True)
    with h2c:
        st.write(""); st.write("")
        if st.button("↻ Refresh", use_container_width=True):
            st.rerun()

    if not STORE_AVAILABLE:
        st.error("Cannot import MemoryStore. Ensure almond_lab/ is inside the Almond root.")
    else:
        db_path = Path(root_dir) / "longmem_almond.db"
        if not db_path.exists():
            st.warning("No database found (longmem_almond.db). Run eval or chat first.")
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

                # Phase 2 table counts
                facts    = get_structured_facts(db_path, limit=1000)
                entities = get_entities(db_path)
                timeline = get_timeline_events(limit=1000)

                e1, e2, e3 = st.columns(3)
                e1.metric("Structured Facts",  len(facts))
                e2.metric("Entities",          len(entities))
                e3.metric("Timeline Events",   len(timeline))

                st.markdown("<hr>", unsafe_allow_html=True)

                tabs = st.tabs(["L1  Rules", "L2  Active", "L3  Cold", "L4  Archive",
                                "🧠 Entities", "📋 Facts", "⏱ Timeline"])

                def render_blocks(tier_enum, tab_obj):
                    with tab_obj:
                        blocks = store.get_all(tier_enum)
                        if not blocks:
                            st.info(f"No blocks in {tier_enum.value}")
                            return
                        search = st.text_input("Filter", key=f"s_{tier_enum.value}",
                                               placeholder="Filter content...")
                        rows = [{
                            "ID":      b.id[:8],
                            "Tag":     b.tag.value,
                            "P_eff":   round(b.p_eff, 4),
                            "Access":  b.access_count,
                            "Content": b.content,
                        } for b in sorted(blocks, key=lambda x: x.p_eff, reverse=True)]
                        df = pd.DataFrame(rows)
                        if search:
                            mask = df.apply(lambda r: search.lower() in str(r).lower(), axis=1)
                            df = df[mask]
                        st.dataframe(df, use_container_width=True, height=400)

                render_blocks(MemoryTier.L1_HOT_CACHE,    tabs[0])
                render_blocks(MemoryTier.L2_ACTIVE_RAM,   tabs[1])
                render_blocks(MemoryTier.L3_VIRTUAL_SWAP, tabs[2])
                render_blocks(MemoryTier.L4_ARCHIVE,      tabs[3])

                with tabs[4]:
                    _slabel("Entity Registry", "0")
                    if not entities:
                        st.info("No entities found. Run eval to populate.")
                    else:
                        search_e = st.text_input("Filter entities", key="ent_search",
                                                  placeholder="Name, type...")
                        ent_rows = [{
                            "ID":        e.get("id","")[:8],
                            "Name":      e.get("name",""),
                            "Type":      e.get("type",""),
                            "Aliases":   ", ".join(json.loads(e.get("aliases","[]")))[:60],
                            "Memories":  len(json.loads(e.get("memory_ids","[]"))),
                            "Refs":      e.get("reference_count",0),
                            "Review":    "⚠" if e.get("needs_review") else "",
                        } for e in entities]
                        df_e = pd.DataFrame(ent_rows)
                        if search_e:
                            mask = df_e.apply(lambda r: search_e.lower() in str(r).lower(), axis=1)
                            df_e = df_e[mask]
                        st.dataframe(df_e, use_container_width=True, height=420)

                with tabs[5]:
                    _slabel("Structured Facts", "0")
                    if not facts:
                        st.info("No structured facts found. Run eval to populate.")
                    else:
                        search_f = st.text_input("Filter facts", key="fact_search",
                                                  placeholder="Subject, predicate, object...")
                        fact_rows = [{
                            "Memory":    f.get("memory_id","")[:8],
                            "Subject":   f.get("subject",""),
                            "Predicate": f.get("predicate",""),
                            "Object":    f.get("object",""),
                            "Type":      f.get("fact_type",""),
                            "Conf":      round(float(f.get("confidence",0)),2),
                            "Date":      f.get("date_raw","") or f.get("earliest","")[:10],
                        } for f in facts]
                        df_f = pd.DataFrame(fact_rows)
                        if search_f:
                            mask = df_f.apply(lambda r: search_f.lower() in str(r).lower(), axis=1)
                            df_f = df_f[mask]
                        st.dataframe(df_f, use_container_width=True, height=420)

                with tabs[6]:
                    _slabel("Timeline Events", "0")
                    if not timeline:
                        st.info("No timeline events found. Run eval to populate.")
                    else:
                        tl_rows = [{
                            "Event":      t.get("description",""),
                            "Type":       t.get("event_type",""),
                            "Earliest":   t.get("earliest","")[:10],
                            "Latest":     t.get("latest","")[:10],
                            "Confidence": round(float(t.get("temporal_confidence",0)),2),
                            "Date Raw":   t.get("date_raw",""),
                        } for t in timeline]
                        st.dataframe(pd.DataFrame(tl_rows), use_container_width=True, height=420)

            except Exception as e:
                st.error(f"Error reading database: {e}")
            finally:
                if "store" in locals():
                    try:
                        store._conn.close()
                    except Exception:
                        pass


# ============================================================================
# PAGE 5 — RUN HISTORY
# ============================================================================

elif "History" in page:
    st.markdown("<h1>Run History</h1>", unsafe_allow_html=True)
    st.markdown('<p style="color:#7A7265;font-size:.8rem;margin-bottom:1.4rem;">'
                'Review and compare previous evaluation runs.</p>', unsafe_allow_html=True)

    history = load_history()
    if not history:
        st.info("No run history yet. Completed evaluations are logged here automatically.")
    else:
        _slabel("Previous Runs", "0")
        for i, run in enumerate(history):
            acc     = run.get("accuracy", 0)
            acc_col = "#3CC97A" if acc >= 80 else "#C8922A" if acc >= 60 else "#E05C5C"
            ds_name = Path(run.get("dataset","?")).name
            ablation_badge = (
                f'<span style="background:rgba(200,146,42,.1);border:1px solid rgba(200,146,42,.3);'
                f'border-radius:3px;padding:.1rem .4rem;font-size:.62rem;color:#C8922A;">'
                f'{run.get("ablation","none")}</span>'
            )
            with st.expander(
                f'{"[Latest]  " if i==0 else ""}{run.get("date","?")}  ·  {ds_name}  '
                f'·  Accuracy: {acc}%'
            ):
                m1, m2, m3, m4, m5 = st.columns(5)
                m1.metric("Accuracy",    f"{acc}%")
                m2.metric("Avg Latency", f"{run.get('avg_latency_ms',0):.0f} ms")
                m3.metric("L2 Peak",     f"{run.get('avg_l2_peak',0)}")
                m4.metric("Pollution",   f"{run.get('avg_pollution',0)}")
                m5.metric("Questions",   f"{run.get('limit','?')}")
                st.markdown(
                    f'<div style="margin-top:.7rem;font-size:.74rem;color:#7A7265;display:flex;gap:1.5rem;">'
                    f'<span>Dataset: <span style="color:#E8E2D5;">{ds_name}</span></span>'
                    f'<span>Ablation: {ablation_badge}</span>'
                    f'<span>Report: <span style="color:#E8E2D5;">{run.get("report_file","?")}</span></span>'
                    f'</div>',
                    unsafe_allow_html=True,
                )

        if len(history) > 1:
            st.markdown("<hr>", unsafe_allow_html=True)
            _slabel("Comparison Table")
            df_hist = pd.DataFrame([{
                "Date":      r.get("date","?"),
                "Dataset":   Path(r.get("dataset","?")).name,
                "Ablation":  r.get("ablation","?"),
                "Questions": r.get("limit","?"),
                "Accuracy":  f"{r.get('accuracy',0)}%",
                "L2 Peak":   r.get("avg_l2_peak",0),
                "Latency ms": r.get("avg_latency_ms",0),
                "Pollution": r.get("avg_pollution",0),
            } for r in history])
            st.dataframe(df_hist, use_container_width=True, height=300)