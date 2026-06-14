"""
app.py – Streamlit UI for Live Vision Agent
Run: streamlit run app.py
"""

import asyncio
import queue
import threading
import time
from datetime import datetime

import cv2
import streamlit as st

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Live Vision Agent",
    page_icon="👁️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Styling ───────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600&family=JetBrains+Mono:wght@400;500&display=swap');

  /* ── Root tokens ── */
  :root {
    --bg:        #0d0f14;
    --surface:   #161922;
    --border:    #252933;
    --accent:    #5b8fff;
    --accent-dim:#1e2e55;
    --green:     #3ecf8e;
    --red:       #ff5b5b;
    --text:      #e4e8f0;
    --muted:     #6b7280;
    --mono:      'JetBrains Mono', monospace;
    --sans:      'Inter', sans-serif;
    --radius:    10px;
  }

  /* ── Global reset ── */
  html, body, [data-testid="stAppViewContainer"] {
    background: var(--bg) !important;
    color: var(--text) !important;
    font-family: var(--sans) !important;
  }
  [data-testid="stHeader"] { display: none !important; }
  [data-testid="stToolbar"] { display: none !important; }
  footer { display: none !important; }
  section[data-testid="stSidebar"] { display: none !important; }

  /* ── Remove default padding ── */
  .block-container {
    padding: 2rem 2.5rem !important;
    max-width: 1280px;
  }

  /* ── Masthead ── */
  .masthead {
    display: flex;
    align-items: center;
    gap: 14px;
    margin-bottom: 2rem;
    border-bottom: 1px solid var(--border);
    padding-bottom: 1.25rem;
  }
  .masthead-icon {
    width: 38px; height: 38px;
    background: var(--accent-dim);
    border: 1px solid var(--accent);
    border-radius: 8px;
    display: flex; align-items: center; justify-content: center;
    font-size: 18px;
  }
  .masthead-title {
    font-size: 1.15rem;
    font-weight: 600;
    letter-spacing: -0.02em;
    color: var(--text);
  }
  .masthead-sub {
    font-size: 0.78rem;
    color: var(--muted);
    margin-top: 1px;
  }
  .status-pill {
    margin-left: auto;
    display: flex;
    align-items: center;
    gap: 7px;
    font-size: 0.78rem;
    font-weight: 500;
    color: var(--muted);
    background: var(--surface);
    border: 1px solid var(--border);
    padding: 5px 12px;
    border-radius: 999px;
  }
  .dot {
    width: 7px; height: 7px;
    border-radius: 50%;
    background: var(--muted);
  }
  .dot.live { background: var(--green); box-shadow: 0 0 6px var(--green); }
  .dot.error { background: var(--red); }

  /* ── Panel card ── */
  .panel {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 0;
    overflow: hidden;
  }
  .panel-header {
    display: flex;
    align-items: center;
    gap: 8px;
    padding: 10px 16px;
    border-bottom: 1px solid var(--border);
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: var(--muted);
  }
  .panel-header span { color: var(--accent); font-size: 0.9rem; }
  .panel-body { padding: 14px 16px; }

  /* ── Webcam placeholder ── */
  .cam-placeholder {
    background: #0a0c10;
    border-radius: 6px;
    aspect-ratio: 16/9;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    gap: 10px;
    color: var(--muted);
    font-size: 0.82rem;
  }
  .cam-placeholder .cam-icon { font-size: 2.2rem; opacity: 0.35; }

  /* ── Transcript feed ── */
  .transcript-wrap {
    height: 340px;
    overflow-y: auto;
    display: flex;
    flex-direction: column;
    gap: 10px;
    padding-right: 4px;
  }
  .transcript-wrap::-webkit-scrollbar { width: 4px; }
  .transcript-wrap::-webkit-scrollbar-thumb {
    background: var(--border); border-radius: 4px;
  }
  .transcript-empty {
    flex: 1;
    display: flex;
    align-items: center;
    justify-content: center;
    flex-direction: column;
    gap: 8px;
    color: var(--muted);
    font-size: 0.82rem;
    opacity: 0.6;
  }
  .message {
    background: var(--accent-dim);
    border: 1px solid rgba(91,143,255,0.2);
    border-radius: 8px;
    padding: 10px 13px;
    font-size: 0.85rem;
    line-height: 1.55;
    color: var(--text);
  }
  .message-meta {
    font-family: var(--mono);
    font-size: 0.67rem;
    color: var(--muted);
    margin-bottom: 4px;
  }
  .message-text { color: var(--text); }

  /* ── Control bar ── */
  .ctrl-bar {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 14px 20px;
    display: flex;
    align-items: center;
    gap: 16px;
    margin-top: 1.25rem;
  }
  .ctrl-label {
    font-size: 0.78rem;
    color: var(--muted);
    font-family: var(--mono);
  }

  /* ── Error banner ── */
  .err-banner {
    background: rgba(255,91,91,0.08);
    border: 1px solid rgba(255,91,91,0.3);
    border-radius: var(--radius);
    padding: 10px 16px;
    font-size: 0.82rem;
    color: var(--red);
    margin-bottom: 1rem;
  }

  /* ── Override Streamlit buttons ── */
  .stButton > button {
    font-family: var(--sans) !important;
    font-size: 0.85rem !important;
    font-weight: 500 !important;
    border-radius: 8px !important;
    border: none !important;
    padding: 0.5rem 1.4rem !important;
    transition: opacity 0.15s, transform 0.1s !important;
  }
  .stButton > button:hover { opacity: 0.88 !important; transform: translateY(-1px); }
  .stButton > button:active { transform: translateY(0) !important; }

  /* Start button – accent */
  div[data-testid="column"]:nth-child(1) .stButton > button {
    background: var(--accent) !important;
    color: #fff !important;
  }
  /* Stop button – red tint */
  div[data-testid="column"]:nth-child(2) .stButton > button {
    background: rgba(255,91,91,0.15) !important;
    color: var(--red) !important;
    border: 1px solid rgba(255,91,91,0.3) !important;
  }
  /* Clear button – subtle */
  div[data-testid="column"]:nth-child(3) .stButton > button {
    background: var(--border) !important;
    color: var(--muted) !important;
  }

  /* ── Streamlit image caption removal ── */
  .stImage > div > div > p { display: none !important; }

  /* ── Info metrics row ── */
  .metrics-row {
    display: flex;
    gap: 12px;
    margin-top: 12px;
  }
  .metric-card {
    flex: 1;
    background: #0d0f14;
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 10px 14px;
    font-family: var(--mono);
  }
  .metric-label { font-size: 0.65rem; color: var(--muted); text-transform: uppercase; letter-spacing: 0.06em; }
  .metric-value { font-size: 1.05rem; font-weight: 500; color: var(--text); margin-top: 3px; }
  .metric-value.green { color: var(--green); }
  .metric-value.accent { color: var(--accent); }
</style>
""", unsafe_allow_html=True)


# ── Session state bootstrap ────────────────────────────────────────────────────
def _init_state():
    defaults = {
        "running": False,
        "transcripts": [],        # list of {"ts": str, "text": str}
        "error": None,
        "msg_count": 0,
        "session_start": None,
        # thread-safe queues for cross-thread comms
        "transcript_q": queue.Queue(),
        "stop_event": threading.Event(),
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

_init_state()


# ── Agent runner (background thread) ─────────────────────────────────────────
def _run_agent(transcript_q: queue.Queue, stop_event: threading.Event):
    """
    Imports and runs LiveVisionAgent in a background thread.
    Monkey-patches the receive loop to also push transcripts to our queue.
    """
    try:
        import sys, os
        sys.path.insert(0, os.path.dirname(__file__))
        from agents import LiveVisionAgent               # your existing class
        from google.genai.live import AsyncSession       # for type hint only

        agent = LiveVisionAgent()

        # ── Patch _receive_responses to forward transcripts ──────────────────
        original_receive = agent._receive_responses.__func__

        async def patched_receive(self):
            while not stop_event.is_set():
                async for message in self._session.receive():
                    if stop_event.is_set():
                        return
                    content = message.server_content
                    if content is None:
                        continue
                    if content.interrupted:
                        await self._drain_playback_queue()
                    if content.model_turn:
                        for part in content.model_turn.parts:
                            if part.inline_data and part.inline_data.data:
                                await self._playback_queue.put(part.inline_data.data)
                    if content.output_transcription and content.output_transcription.text:
                        txt = content.output_transcription.text.strip()
                        if txt:
                            ts = datetime.now().strftime("%H:%M:%S")
                            transcript_q.put({"ts": ts, "text": txt})
                            print("Model:", txt)

        import types as _types
        agent._receive_responses = _types.MethodType(patched_receive, agent)

        # ── Run until stop_event is set ───────────────────────────────────────
        async def _run():
            from google import genai
            from src.config import GEMINI_API_KEY, MODEL
            from agents import LIVE_CONFIG
            import asyncio

            try:
                async with agent._client.aio.live.connect(
                    model=MODEL, config=LIVE_CONFIG
                ) as session:
                    agent._session = session
                    tasks = [
                        asyncio.create_task(agent._stream_microphone()),
                        asyncio.create_task(agent._stream_camera()),
                        asyncio.create_task(agent._receive_responses()),
                        asyncio.create_task(agent._playback_audio()),
                    ]
                    # Poll stop_event and cancel when signalled
                    while not stop_event.is_set():
                        await asyncio.sleep(0.3)
                    for t in tasks:
                        t.cancel()
                    await asyncio.gather(*tasks, return_exceptions=True)
            finally:
                agent._audio.close()
                agent._camera.close()
                await agent._client.aio.aclose()

        asyncio.run(_run())

    except ImportError as e:
        transcript_q.put({"ts": datetime.now().strftime("%H:%M:%S"),
                          "text": f"[IMPORT ERROR] {e}"})
    except RuntimeError as e:
        # e.g. camera not found, mic not found
        transcript_q.put({"ts": datetime.now().strftime("%H:%M:%S"),
                          "text": f"[DEVICE ERROR] {e}"})
    except Exception as e:
        transcript_q.put({"ts": datetime.now().strftime("%H:%M:%S"),
                          "text": f"[ERROR] {e}"})


# ── Webcam preview (Streamlit-side, independent of agent) ────────────────────
def _grab_webcam_frame():
    """Returns a BGR frame or None."""
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        return None
    ok, frame = cap.read()
    cap.release()
    return frame if ok else None


# ── Drain transcript queue into session state ─────────────────────────────────
def _drain_transcripts():
    q: queue.Queue = st.session_state["transcript_q"]
    new_items = []
    while not q.empty():
        try:
            new_items.append(q.get_nowait())
        except queue.Empty:
            break
    if new_items:
        st.session_state["transcripts"].extend(new_items)
        st.session_state["msg_count"] += len(new_items)
    # Detect device errors pushed as messages
    for item in new_items:
        if "[ERROR]" in item["text"] or "[DEVICE ERROR]" in item["text"] or "[IMPORT ERROR]" in item["text"]:
            st.session_state["error"] = item["text"]
            st.session_state["running"] = False


# ── Start / Stop helpers ──────────────────────────────────────────────────────
def start_agent():
    stop_event: threading.Event = st.session_state["stop_event"]
    stop_event.clear()
    st.session_state["error"] = None
    st.session_state["session_start"] = time.time()
    st.session_state["running"] = True
    t = threading.Thread(
        target=_run_agent,
        args=(st.session_state["transcript_q"], stop_event),
        daemon=True,
    )
    t.start()


def stop_agent():
    st.session_state["stop_event"].set()
    st.session_state["running"] = False
    st.session_state["session_start"] = None


def clear_transcripts():
    st.session_state["transcripts"] = []
    st.session_state["msg_count"] = 0
    st.session_state["error"] = None


# ── Drain queue on every rerun ────────────────────────────────────────────────
_drain_transcripts()

# ── Compute live uptime ───────────────────────────────────────────────────────
def _uptime() -> str:
    if st.session_state["session_start"] is None:
        return "—"
    elapsed = int(time.time() - st.session_state["session_start"])
    m, s = divmod(elapsed, 60)
    return f"{m:02d}:{s:02d}"


# ══════════════════════════════════════════════════════════════════════════════
# ── RENDER ────────────────────────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════

running = st.session_state["running"]

# ── Masthead ──────────────────────────────────────────────────────────────────
dot_cls = "dot live" if running else ("dot error" if st.session_state["error"] else "dot")
status_label = "Live" if running else ("Error" if st.session_state["error"] else "Idle")

st.markdown(f"""
<div class="masthead">
  <div class="masthead-icon">👁️</div>
  <div>
    <div class="masthead-title">Live Vision Agent</div>
    <div class="masthead-sub">Gemini Multimodal · Real-time Vision &amp; Voice</div>
  </div>
  <div class="status-pill">
    <div class="{dot_cls}"></div>
    {status_label}
  </div>
</div>
""", unsafe_allow_html=True)

# ── Error banner ──────────────────────────────────────────────────────────────
if st.session_state["error"]:
    st.markdown(f"""
    <div class="err-banner">
      ⚠️ {st.session_state['error']}
    </div>
    """, unsafe_allow_html=True)

# ── Main columns ──────────────────────────────────────────────────────────────
left, right = st.columns([1.05, 1], gap="medium")

# ── LEFT: Webcam ──────────────────────────────────────────────────────────────
with left:
    st.markdown("""
    <div class="panel">
      <div class="panel-header"><span>◉</span> Camera Feed</div>
    </div>
    """, unsafe_allow_html=True)

    cam_placeholder = st.empty()

    if running:
        frame = _grab_webcam_frame()
        if frame is not None:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            cam_placeholder.image(frame_rgb, use_container_width=True)
        else:
            cam_placeholder.markdown("""
            <div class="cam-placeholder">
              <div class="cam-icon">📷</div>
              <div>Camera unavailable or in use by agent</div>
            </div>""", unsafe_allow_html=True)
    else:
        cam_placeholder.markdown("""
        <div class="cam-placeholder" style="min-height:260px;">
          <div class="cam-icon">📷</div>
          <div>Start the agent to see camera preview</div>
        </div>""", unsafe_allow_html=True)

    # ── Metrics row ───────────────────────────────────────────────────────────
    st.markdown(f"""
    <div class="metrics-row">
      <div class="metric-card">
        <div class="metric-label">Session Time</div>
        <div class="metric-value accent">{_uptime()}</div>
      </div>
      <div class="metric-card">
        <div class="metric-label">Responses</div>
        <div class="metric-value green">{st.session_state['msg_count']}</div>
      </div>
      <div class="metric-card">
        <div class="metric-label">Model</div>
        <div class="metric-value" style="font-size:0.78rem;">gemini-live</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

# ── RIGHT: Transcript ─────────────────────────────────────────────────────────
with right:
    st.markdown("""
    <div class="panel">
      <div class="panel-header"><span>≡</span> Gemini Transcript</div>
    </div>
    """, unsafe_allow_html=True)

    transcripts = st.session_state["transcripts"]

    if transcripts:
        msgs_html = ""
        for entry in reversed(transcripts[-40:]):   # newest first, cap at 40
            msgs_html += f"""
            <div class="message">
              <div class="message-meta">{entry['ts']}</div>
              <div class="message-text">{entry['text']}</div>
            </div>"""
        st.markdown(f'<div class="transcript-wrap">{msgs_html}</div>',
                    unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="transcript-wrap">
          <div class="transcript-empty">
            <div style="font-size:1.6rem">🎙️</div>
            <div>Gemini's spoken responses appear here</div>
            <div style="font-size:0.72rem;opacity:0.5;">Start the agent and speak to begin</div>
          </div>
        </div>""", unsafe_allow_html=True)

# ── Control bar ───────────────────────────────────────────────────────────────
st.markdown('<div style="height:1.25rem"></div>', unsafe_allow_html=True)
c1, c2, c3, c_space = st.columns([1, 1, 1, 3])

with c1:
    if st.button("▶  Start", disabled=running, use_container_width=True):
        start_agent()
        st.rerun()

with c2:
    if st.button("■  Stop", disabled=not running, use_container_width=True):
        stop_agent()
        st.rerun()

with c3:
    if st.button("↺  Clear", use_container_width=True):
        clear_transcripts()
        st.rerun()

# ── Auto-refresh while running ────────────────────────────────────────────────
if running:
    time.sleep(1.0)
    st.rerun()