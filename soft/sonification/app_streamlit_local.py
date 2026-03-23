# app_streamlit_local.py
"""
Local Streamlit sonification demo app.

Run with:
    streamlit run app_streamlit_local.py

Architecture:
- This file: Streamlit UI only
- sonification_engine.py: all audio/state/playback logic

No FastAPI, no uvicorn, no server/client split.
Audio plays locally via sounddevice.
"""

import time
import copy
import random
import streamlit as st

from sonification_engine import (
    SonificationEngine,
    DEFAULT_GAIN_PRESETS,
    compute_raw_state,
    parse_line,
)

# ─────────────────────────────────────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Sonification Demo",
    page_icon="🎵",
    layout="wide",
)

# ─────────────────────────────────────────────────────────────────────────────
# Session state bootstrap
# ─────────────────────────────────────────────────────────────────────────────
def _init_session():
    if "engine" not in st.session_state:
        st.session_state.engine = SonificationEngine()
    if "scenario_loaded" not in st.session_state:
        st.session_state.scenario_loaded = False
    if "scenario_mode" not in st.session_state:
        st.session_state.scenario_mode = "demo"
    if "gain_presets" not in st.session_state:
        st.session_state.gain_presets = copy.deepcopy(DEFAULT_GAIN_PRESETS)
    if "calm_threshold" not in st.session_state:
        st.session_state.calm_threshold = 28.0
    if "epi" not in st.session_state:
        st.session_state.epi = 1.0
    if "summary_line" not in st.session_state:
        st.session_state.summary_line = ""
    if "creator_manual_blocks" not in st.session_state:
        st.session_state.creator_manual_blocks = []
    if "creator_lines" not in st.session_state:
        st.session_state.creator_lines = []
    if "creator_blocks" not in st.session_state:
        st.session_state.creator_blocks = []

_init_session()
engine: SonificationEngine = st.session_state.engine


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────
LAYER_NAMES = ["messiaen", "breathing", "ocean", "birds"]
STATE_NAMES = ["NO_PRESENCE", "ECLIPSE", "CALM", "EXCITED"]

def _push_config():
    """Push current slider state to the engine."""
    engine.apply_config(
        calm_threshold=st.session_state.calm_threshold,
        epi=st.session_state.epi,
        gain_presets=st.session_state.gain_presets,
    )

def _recompute_summary():
    if st.session_state.scenario_mode == "demo" and st.session_state.scenario_loaded:
        _push_config()
        st.session_state.summary_line = engine.compute_internal_demo_summary()


def _generate_scenario(blocks, threshold: float):
    """
    Generate scenario lines from a list of state blocks.

    blocks: list of {"state": str, "duration": int}
    threshold: float — calm/excited boundary

    Returns a list of strings (one per second).
    """
    br_ranges = {
        "CALM":        (threshold * 0.50, threshold * 0.92),
        "EXCITED":     (threshold * 1.08, threshold * 1.70),
        "NO_PRESENCE": (18.0, 24.0),   # presence=0, value doesn't affect state
        "ECLIPSE":     (threshold * 0.60, threshold * 1.40),  # moon=0, any rate
    }

    lines = []
    t = 0
    current_br = threshold * 0.75  # start somewhere in the calm range

    for block in blocks:
        state    = block["state"]
        duration = block["duration"]

        br_min, br_max = br_ranges[state]
        target_br = random.uniform(br_min, br_max)

        presence = 0 if state == "NO_PRESENCE" else 1
        moon     = 0 if state == "ECLIPSE"     else 1

        for _ in range(duration):
            # smooth random walk toward target, with small noise
            current_br += (target_br - current_br) * 0.25 + random.gauss(0, 0.25)
            current_br  = max(8.0, min(60.0, current_br))

            # occasionally drift the target within the state range
            if random.random() < 0.08:
                target_br = random.uniform(br_min, br_max)

            hh = t // 3600
            mm = (t % 3600) // 60
            ss = t % 60
            dev  = round(random.uniform(0.10, 0.25), 3)
            line = f"{hh:02d}:{mm:02d}:{ss:02d},{current_br:.2f},{dev:.3f},{presence},{moon}"
            lines.append(line)
            t += 1

    return lines


# ─────────────────────────────────────────────────────────────────────────────
# ── SIDEBAR ──────────────────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("🎵 Sonification")
    st.divider()

    # ── Scenario chooser ──────────────────────────────────────────────────────
    st.subheader("Scenario")
    mode = st.radio(
        "Mode",
        options=["Internal demo", "Upload TXT"],
        index=0 if st.session_state.scenario_mode == "demo" else 1,
        key="mode_radio",
    )
    st.session_state.scenario_mode = "demo" if mode == "Internal demo" else "upload"

    if st.session_state.scenario_mode == "demo":
        if st.button("Load internal demo", use_container_width=True):
            try:
                n = engine.load_internal_demo()
                st.session_state.scenario_loaded = True
                _recompute_summary()
                st.success(f"Loaded {n} rows from internal demo.")
            except FileNotFoundError as e:
                st.error(str(e))
    else:
        uploaded = st.file_uploader("Upload .txt scenario", type=["txt"])
        if uploaded is not None:
            lines = uploaded.read().decode("utf-8").splitlines()
            n = engine.load_uploaded_scenario(lines)
            st.session_state.scenario_loaded = True
            st.session_state.summary_line = ""
            st.success(f"Loaded {n} valid rows.")

    st.divider()

    # ── Global controls ───────────────────────────────────────────────────────
    st.subheader("Global controls")

    calm_threshold = st.slider(
        "calm_threshold",
        min_value=18.0, max_value=33.0,
        value=float(st.session_state.calm_threshold),
        step=0.5,
        key="calm_threshold_slider",
    )
    st.session_state.calm_threshold = calm_threshold

    epi = st.slider(
        "epi",
        min_value=1.0, max_value=2.0,
        value=float(st.session_state.epi),
        step=0.05,
        key="epi_slider",
    )
    st.session_state.epi = epi

    st.divider()

    # ── Per-state gain sliders ────────────────────────────────────────────────
    st.subheader("Per-state gains")

    gp = st.session_state.gain_presets

    for state in STATE_NAMES:
        with st.expander(state):
            for layer in LAYER_NAMES:
                key = f"gain_{state}_{layer}"
                val = st.slider(
                    layer,
                    min_value=0.0, max_value=1.0,
                    value=float(gp[state][layer]),
                    step=0.01,
                    key=key,
                )
                gp[state][layer] = val

    st.session_state.gain_presets = gp

    # Push config whenever sidebar is rendered
    _push_config()
    if st.session_state.scenario_mode == "demo" and st.session_state.scenario_loaded:
        _recompute_summary()


# ─────────────────────────────────────────────────────────────────────────────
# ── MAIN AREA ────────────────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────
st.title("Sonification Demo")

tab_play, tab_create = st.tabs(["▶ Playback", "🛠 Create Scenario"])


# ═════════════════════════════════════════════════════════════════════════════
# TAB 1 — Playback
# ═════════════════════════════════════════════════════════════════════════════
with tab_play:
    if not st.session_state.scenario_loaded:
        st.info("Load a scenario from the sidebar to begin.")
    else:
        # ── Play / Stop ───────────────────────────────────────────────────────
        col_play, col_stop, col_status = st.columns([1, 1, 4])

        with col_play:
            if st.button("▶ Play", use_container_width=True,
                         disabled=engine.is_playing):
                _push_config()
                engine.start()
                st.rerun()

        with col_stop:
            if st.button("⏹ Stop", use_container_width=True,
                         disabled=not engine.is_playing):
                engine.stop(immediate=True)
                st.rerun()

        with col_status:
            if engine.is_playing:
                st.success("▶ Playing")
            else:
                snap = engine.get_snapshot()
                if snap.get("finished"):
                    st.info("Scenario finished.")
                else:
                    st.warning("Stopped.")

        st.divider()

        # ── Summary line (demo only) ──────────────────────────────────────────
        if st.session_state.scenario_mode == "demo":
            summary = st.session_state.summary_line
            if summary:
                st.subheader("Scenario summary")
                st.markdown(
                    "_=NO\\_PRESENCE  C=CALM  E=EXCITED  M=ECLIPSE  ~=TRANSITION"
                )
                st.code(summary, language=None)
                st.divider()

        # ── Live display ──────────────────────────────────────────────────────
        snap = engine.get_snapshot()

        if not engine.is_playing and snap.get("tick_index", 0) == 0:
            st.caption("Press ▶ Play to start.")
        else:
            st.subheader("Live state")

            left, right = st.columns(2)

            with left:
                st.markdown("**Sensor values**")
                presence_val = snap.get("presence", 0)
                moon_val     = snap.get("moon", 1)

                presence_str = "person in the room" if presence_val == 1 else "—"
                moon_str     = "eclipse" if moon_val == 0 else "—"

                st.markdown(f"presence: **{presence_str}**")
                st.markdown(f"moon: **{moon_str}**")

                st.divider()

                br    = snap.get("breathing_rate", 0.0)
                breff = snap.get("br_eff", 0.0)
                thr   = snap.get("calm_threshold", st.session_state.calm_threshold)

                st.markdown("**Breathing**")
                st.markdown(f"breathing_rate = {br:.2f} / calm_threshold = {thr:.1f}")
                st.markdown(f"**breathing rate × epi = {breff:.2f}**")

                st.divider()

                st.markdown("**State**")
                if snap.get("transitioning"):
                    st.markdown(
                        f"TRANSITIONING TO: **{snap.get('to_state', '?')}**  "
                        f"*(from {snap.get('from_state', '?')})*"
                    )
                else:
                    st.markdown(f"STATE: **{snap.get('confirmed_state', 'NO_PRESENCE')}**")

            with right:
                st.markdown("**Current gains**")
                gains = snap.get("gains", {})

                for layer in LAYER_NAMES:
                    g       = gains.get(layer, 0.0)
                    bar_pct = int(g * 100)
                    st.markdown(
                        f"`{layer:<12}` {g:.3f}  "
                        f"{'█' * (bar_pct // 5)}{'░' * (20 - bar_pct // 5)}"
                    )

                orch_g   = gains.get("orchestral", 0.0)
                orch_bar = int(orch_g * 100)
                st.markdown(
                    f"`{'orchestral':<12}` {orch_g:.3f}  "
                    f"{'█' * (orch_bar // 5)}{'░' * (20 - orch_bar // 5)}"
                )

        # ── Auto-refresh while playing ────────────────────────────────────────
        if engine.is_playing:
            time.sleep(0.5)
            st.rerun()


# ═════════════════════════════════════════════════════════════════════════════
# TAB 2 — Scenario Creator
# ═════════════════════════════════════════════════════════════════════════════
with tab_create:
    st.subheader("Scenario Creator")
    st.caption(
        "Generate a `.txt` scenario file to load into the Playback tab. "
        "One row per second — each row sets breathing rate, presence, and moon state."
    )

    st.divider()

    creator_mode = st.radio(
        "Builder mode",
        ["🎲 Random sequence", "✏️ Manual sequence"],
        horizontal=True,
        key="creator_mode",
    )

    st.divider()

    CREATOR_THRESHOLD = 18.0

    # ─────────────────────────────────────────────────────────────────────────
    # Random mode
    # ─────────────────────────────────────────────────────────────────────────
    if creator_mode == "🎲 Random sequence":
        c1, c2, c3 = st.columns(3)
        with c1:
            total_dur = st.number_input(
                "Total duration (seconds)",
                min_value=10, max_value=3600, value=120, step=10,
                key="creator_total",
            )
        with c2:
            min_block = st.number_input(
                "Min block length (s)",
                min_value=2, max_value=120, value=8, step=1,
                key="creator_min_block",
            )
        with c3:
            max_block = st.number_input(
                "Max block length (s)",
                min_value=2, max_value=120, value=20, step=1,
                key="creator_max_block",
            )

        if min_block > max_block:
            st.warning("Min block length must be ≤ max block length.")

        states_to_include = st.multiselect(
            "States to include",
            options=["NO_PRESENCE", "CALM", "EXCITED", "ECLIPSE"],
            default=["CALM", "EXCITED"],
            key="creator_states",
        )

        if st.button("Generate random scenario", use_container_width=True,
                     key="creator_gen_random"):
            if not states_to_include:
                st.error("Select at least one state.")
            elif min_block > max_block:
                st.error("Fix the block length range first.")
            else:
                blocks    = []
                remaining = int(total_dur)
                while remaining > 0:
                    dur   = min(random.randint(int(min_block), int(max_block)), remaining)
                    state = random.choice(states_to_include)
                    blocks.append({"state": state, "duration": dur})
                    remaining -= dur

                st.session_state.creator_lines  = _generate_scenario(blocks, CREATOR_THRESHOLD)
                st.session_state.creator_blocks = blocks

    # ─────────────────────────────────────────────────────────────────────────
    # Manual mode
    # ─────────────────────────────────────────────────────────────────────────
    else:
        st.markdown("**Add state blocks in order:**")

        ca, cb, cc = st.columns([2, 1, 1])
        with ca:
            new_state = st.selectbox(
                "State",
                ["CALM", "EXCITED", "NO_PRESENCE", "ECLIPSE"],
                key="creator_new_state",
            )
        with cb:
            new_dur = st.number_input(
                "Duration (s)", min_value=1, max_value=3600,
                value=15, key="creator_new_dur",
            )
        with cc:
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("＋ Add block", use_container_width=True, key="creator_add"):
                st.session_state.creator_manual_blocks.append(
                    {"state": new_state, "duration": int(new_dur)}
                )

        manual_blocks = st.session_state.creator_manual_blocks

        if manual_blocks:
            total_s = sum(b["duration"] for b in manual_blocks)
            st.markdown(f"**Sequence — {total_s}s total:**")

            for i, b in enumerate(manual_blocks):
                r1, r2, r3 = st.columns([4, 1, 1])
                sym = {"NO_PRESENCE": "_", "CALM": "C",
                       "EXCITED": "E", "ECLIPSE": "M"}.get(b["state"], "?")
                with r1:
                    st.markdown(
                        f"`{i+1}.` **{b['state']}** &nbsp;·&nbsp; {b['duration']}s &nbsp;"
                        f"<span style='color:grey'>({sym * min(b['duration'], 30)}"
                        f"{'…' if b['duration'] > 30 else ''})</span>",
                        unsafe_allow_html=True,
                    )
                with r3:
                    if st.button("✕", key=f"creator_del_{i}"):
                        st.session_state.creator_manual_blocks.pop(i)
                        st.rerun()

            st.markdown("")
            col_clear, col_gen = st.columns([1, 2])
            with col_clear:
                if st.button("Clear all", use_container_width=True, key="creator_clear"):
                    st.session_state.creator_manual_blocks = []
                    st.rerun()
            with col_gen:
                if st.button("Generate from sequence", use_container_width=True,
                             key="creator_gen_manual"):
                    st.session_state.creator_lines  = _generate_scenario(
                        manual_blocks, CREATOR_THRESHOLD
                    )
                    st.session_state.creator_blocks = list(manual_blocks)
        else:
            st.caption("No blocks added yet.")

    # ─────────────────────────────────────────────────────────────────────────
    # Preview + Download
    # ─────────────────────────────────────────────────────────────────────────
    if st.session_state.creator_lines:
        lines  = st.session_state.creator_lines
        blocks = st.session_state.creator_blocks

        st.divider()
        st.markdown(f"**Generated scenario — {len(lines)} seconds**")

        # visual state bar
        STATE_SYM = {"NO_PRESENCE": "_", "CALM": "C", "EXCITED": "E", "ECLIPSE": "M"}
        summary_str = "".join(
            STATE_SYM.get(b["state"], "?") * b["duration"] for b in blocks
        )
        st.code(summary_str, language=None)
        st.caption("_=NO_PRESENCE  C=CALM  E=EXCITED  M=ECLIPSE")

        with st.expander("Preview first 15 lines"):
            st.code("\n".join(lines[:15]), language=None)

        st.download_button(
            label="⬇ Download scenario.txt",
            data="\n".join(lines) + "\n",
            file_name="scenario.txt",
            mime="text/plain",
            use_container_width=True,
        )
