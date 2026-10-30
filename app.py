from __future__ import annotations
import math
import json
import uuid
from pathlib import Path
import sys
import numpy as np
import streamlit as st

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app_components.typing_capture import render_typing_capture
from src.config import MAX_HOLD_SECONDS
from src.features import macd_with_count
from src.inference import (
    InferenceError,
    analyse_keystrokes,
    load_macd_model,
)

APP_TITLE = "Parkinson's Keystroke Analysis"
MIN_VALID_KEYSTROKES = 200
MAX_VALID_KEYSTROKES = 1000
PASSAGE = (
    "A calm and steady routine can help people complete everyday activities. "
    "Typing uses coordinated movements of the fingers, hands and eyes. "
    "Each key press has a beginning and an end, and the time between those "
    "events can be measured. Please type this passage at your natural pace, "
    "without trying to type faster than usual. If you make a mistake, continue "
    "naturally and keep your usual rhythm. The purpose of this short research "
    "activity is to study timing patterns, not typing speed or spelling."
)


def validate_capture_payload(payload):
    """Validate the browser-to-Python capture payload."""
    if not isinstance(payload, dict):
        raise InferenceError("Invalid keystroke capture payload.")

    hold_times = payload.get("hold_times_s")
    if not isinstance(hold_times, list):
        raise InferenceError(
            "Keystroke timing observations must be a list."
        )

    received_count = len(hold_times)
    if received_count > MAX_VALID_KEYSTROKES:
        raise InferenceError(
            f"Received {received_count} timing values. "
            f"Maximum allowed is {MAX_VALID_KEYSTROKES}."
        )

    browser_count = payload.get("valid_keystrokes")
    if (
        isinstance(browser_count, bool)
        or browser_count != received_count
    ):
        raise InferenceError(
            "Browser-reported keystroke count does not match "
            "the received timing observations."
        )

    elapsed = payload.get("elapsed_seconds")
    try:
        elapsed = float(elapsed)
    except (TypeError, ValueError, OverflowError):
        elapsed = None

    if elapsed is not None and (
        not math.isfinite(elapsed) or elapsed <= 0
    ):
        elapsed = None

    return {
        "hold_times": hold_times,
        "received_count": received_count,
        "elapsed_seconds": elapsed,
    }


def verify_analysis_result(hold_times, result):
    """Verify inference features and output against canonical calculations."""
    try:
        values = np.asarray(list(hold_times), dtype=float).reshape(-1)
    except (TypeError, ValueError, OverflowError) as exc:
        raise InferenceError(
            "Keystroke timings could not be verified as numeric values."
        ) from exc

    valid = values[
        np.isfinite(values)
        & (values >= 0.0)
        & (values < MAX_HOLD_SECONDS)
    ]
    if valid.size < 2:
        raise InferenceError(
            "Insufficient valid observations for feature verification."
        )

    ql = float(result["percentile_low"])
    qh = float(result["percentile_high"])
    expected_macd, expected_pairs = macd_with_count(
        valid,
        ql=ql,
        qh=qh,
    )
    expected_statistics = {
        "valid_keystrokes": int(valid.size),
        "macd_pairs": expected_pairs,
        "macd_seconds": expected_macd,
        "mean_hold_seconds": float(np.mean(valid)),
        "median_hold_seconds": float(np.median(valid)),
        "std_hold_seconds": float(np.std(valid, ddof=0)),
    }

    for name, expected in expected_statistics.items():
        actual = result.get(name)
        if isinstance(expected, (int, np.integer)):
            matches = (
                not isinstance(actual, bool)
                and actual == expected
            )
        else:
            try:
                matches = bool(
                    np.isclose(
                        float(actual),
                        expected,
                        rtol=1e-12,
                        atol=1e-12,
                    )
                )
            except (TypeError, ValueError, OverflowError):
                matches = False

        if not matches:
            raise InferenceError(
                f"Feature verification failed for {name}."
            )

    model = load_macd_model()
    expected_logit = (
        model["coef"] * expected_macd + model["intercept"]
    )
    if expected_logit >= 0:
        exp_value = math.exp(-expected_logit)
        expected_output = 1.0 / (1.0 + exp_value)
    else:
        exp_value = math.exp(expected_logit)
        expected_output = exp_value / (1.0 + exp_value)
    try:
        output_matches = bool(
            np.isclose(
                float(result["model_output_score"]),
                expected_output,
                rtol=1e-12,
                atol=1e-12,
            )
        )
    except (KeyError, TypeError, ValueError, OverflowError):
        output_matches = False

    if not output_matches:
        raise InferenceError(
            "Model verification failed for logistic-regression output."
        )

    return {
        "feature_consistency_passed": True,
        "verified_observation_count": int(valid.size),
        "model_consistency_passed": True,
    }


def load_training_distribution():
    """Load and validate MACD percentiles from the exported model."""
    try:
        model = json.loads(
            (ROOT / "models" / "macd_model.json").read_text(
                encoding="utf-8"
            )
        )
    except (OSError, json.JSONDecodeError) as exc:
        raise InferenceError(
            "The exported model training distribution could not be loaded."
        ) from exc

    if not isinstance(model, dict) or model.get("feature") != "macd":
        raise InferenceError(
            "The exported model has no valid MACD training distribution."
        )
    statistics = model.get("training_stats")
    percentile_names = ("min", "p1", "p5", "median", "p95", "p99", "max")
    if not isinstance(statistics, dict):
        raise InferenceError(
            "The exported model has no valid training distribution."
        )

    try:
        distribution = {
            name: float(statistics[name])
            for name in percentile_names
        }
    except (KeyError, TypeError, ValueError, OverflowError) as exc:
        raise InferenceError(
            "The exported model training percentiles are incomplete."
        ) from exc

    values = tuple(distribution.values())
    if not all(math.isfinite(value) for value in values) or any(
        lower > upper for lower, upper in zip(values, values[1:])
    ):
        raise InferenceError(
            "The exported model training percentiles are invalid."
        )
    return distribution


def compare_with_training_distribution(macd_value):
    """Describe the MACD position within the exported training percentiles."""
    try:
        value = float(macd_value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise InferenceError(
            "MACD cannot be compared with the training distribution."
        ) from exc
    if not math.isfinite(value):
        raise InferenceError(
            "MACD cannot be compared with the training distribution."
        )

    distribution = load_training_distribution()
    boundaries = (
        ("minimum", distribution["min"]),
        ("1st percentile", distribution["p1"]),
        ("5th percentile", distribution["p5"]),
        ("median", distribution["median"]),
        ("95th percentile", distribution["p95"]),
        ("99th percentile", distribution["p99"]),
        ("maximum", distribution["max"]),
    )

    if value < boundaries[0][1]:
        position = "Below the training minimum"
    elif value > boundaries[-1][1]:
        position = "Above the training maximum"
    else:
        position = "Within the observed training range"
        for (lower_label, lower_value), (upper_label, upper_value) in zip(
            boundaries,
            boundaries[1:],
        ):
            if lower_value <= value <= upper_value:
                position = (
                    f"Between the training {lower_label} "
                    f"and {upper_label}"
                )
                break

    return {
        "training_distribution_position": position,
        "training_distribution_min": distribution["min"],
        "training_distribution_p1": distribution["p1"],
        "training_distribution_p5": distribution["p5"],
        "training_distribution_median": distribution["median"],
        "training_distribution_p95": distribution["p95"],
        "training_distribution_p99": distribution["p99"],
        "training_distribution_max": distribution["max"],
    }


st.set_page_config(
    page_title=APP_TITLE,
    layout="wide",
)

st.markdown(
    """<style>
.block-container {
    max-width: 1400px;
    padding-top: 2rem;
    padding-bottom: 2rem;
}
.section-label {
    font-size: 1.15rem !important;
    font-weight: 800 !important;
    color: #172033 !important;
}
div[data-testid="stCaptionContainer"] {
    font-size: 0.95rem !important;
    color: #475569 !important;
}
/* Metric cards */
[data-testid="stMetric"] {
    background: #ffffff !important;
    border: 1px solid #dce5f0 !important;
    border-radius: 14px !important;
    padding: 20px !important;
    box-shadow: 0 2px 8px rgba(15, 23, 42, 0.04);
}

/* Metric labels */
[data-testid="stMetricLabel"],
[data-testid="stMetricLabel"] p,
[data-testid="stMetricLabel"] span {
    color: #475569 !important;
    font-size: 14px !important;
    font-weight: 650 !important;
}

/* Metric values */
[data-testid="stMetricValue"],
[data-testid="stMetricValue"] div,
[data-testid="stMetricValue"] span {
    color: #1e40af !important;
    font-size: 28px !important;
    font-weight: 750 !important;
}

/* Results section headings */
[data-testid="stMarkdownContainer"] h1,
[data-testid="stMarkdownContainer"] h2,
[data-testid="stMarkdownContainer"] h3,
[data-testid="stMarkdownContainer"] h4 {
    color: #172033 !important;
    font-weight: 750 !important;
}

/* Regular results text */
[data-testid="stMarkdownContainer"] p,
[data-testid="stMarkdownContainer"] li {
    color: #334155 !important;
}

/* Captions */
[data-testid="stCaptionContainer"],
[data-testid="stCaptionContainer"] p {
    color: #64748b !important;
}

/* Primary action buttons */
.stButton > button[kind="primary"] {
    background: #2563eb !important;
    border-color: #2563eb !important;
    color: #ffffff !important;
    font-weight: 700 !important;
    border-radius: 10px !important;
}

.stButton > button[kind="primary"]:hover {
    background: #1d4ed8 !important;
    border-color: #1d4ed8 !important;
}
</style>""",
    unsafe_allow_html=True,
)

PAGE_CSS = """
<style>
  .stApp { background: #f5f7fb; }
  .block-container { max-width: 1400px; padding-top: 2rem; padding-bottom: 2rem; }
  .app-kicker {
    display: inline-flex; align-items: center; gap: .45rem;
    padding: .35rem .65rem; border: 1px solid #dbe5f3; border-radius: 999px;
    background: #edf4ff; color: #245da8; font-size: .74rem; font-weight: 750;
    letter-spacing: .045em; text-transform: uppercase;
  }
  .app-title { margin: .8rem 0 .35rem; color: #17263b; font-size: clamp(2rem, 5vw, 2.65rem); line-height: 1.12; font-weight: 800; letter-spacing: -.045em; }
  .app-subtitle { max-width: 670px; margin: 0 0 1.2rem; color: #5d6b80; font-size: 1.02rem; line-height: 1.65; }
  .intro-card {
    padding: 1.05rem 1.15rem; border: 1px solid #e0e7f0; border-radius: 14px;
    background: #fff; color: #425169; line-height: 1.6; font-size: .91rem;
  }
  .intro-card strong { color: #1b2e47; }
  .section-label { margin: 1.35rem 0 .5rem; color: #172033; font-size: 1.15rem; font-weight: 800; letter-spacing: .04em; text-transform: uppercase; }
  .disclaimer {
    margin-top: 1rem; padding: .9rem 1rem; border-left: 3px solid #d49a36;
    border-radius: 8px; background: #fff8e9; color: #6c542b; font-size: .82rem; line-height: 1.55;
  }
  .result-panel {
    padding: 1.25rem; border: 1px solid #dce5f0; border-radius: 16px;
    background: #fff; margin-top: .6rem;
  }
  .score-label { margin: 0; color: #67768c; font-size: .78rem; font-weight: 700; letter-spacing: .06em; text-transform: uppercase; }
  .score-value { margin: .2rem 0 .25rem; color: #1e5ca8; font-size: 2.5rem; font-weight: 800; letter-spacing: -.04em; font-variant-numeric: tabular-nums; }
  .score-caption { color: #65748a; font-size: .82rem; line-height: 1.5; }
  .metric-note { color: #6b7789; font-size: .78rem; }
  hr { border-color: #e0e7f0; }
  @media (max-width: 600px) { .block-container { padding-top: 1.3rem; } }

/* Results card */

.result-panel {
    padding: 1.5rem;
    margin-top: 0.7rem;
    margin-bottom: 1.2rem;
    background: #ffffff;
    border: 1px solid #dce5f0;
    border-radius: 16px;
    box-shadow: 0 3px 12px rgba(15, 23, 42, 0.04);
}

/* Model score */

.score-label {
    color: #64748b !important;
    font-size: 0.8rem;
    font-weight: 750;
    letter-spacing: 0.06em;
}

.score-value {
    color: #1e5ca8 !important;
    font-size: 2.8rem;
    font-weight: 850;
    font-variant-numeric: tabular-nums;
}

.score-caption {
    color: #475569 !important;
    font-size: 0.9rem;
    line-height: 1.6;
}

/* Metric cards */

[data-testid="stMetric"] {
    min-height: 112px;
    background: #ffffff !important;
    border: 1px solid #dce5f0;
    border-radius: 14px;
    padding: 16px !important;
}

[data-testid="stMetricLabel"],
[data-testid="stMetricLabel"] p,
[data-testid="stMetricLabel"] span {
    color: #475569 !important;
}

[data-testid="stMetricValue"],
[data-testid="stMetricValue"] div,
[data-testid="stMetricValue"] span {
    color: #1e40af !important;
    font-weight: 750 !important;
}

/* Results headings */

[data-testid="stMarkdownContainer"] h3,
[data-testid="stMarkdownContainer"] h4 {
    color: #172033 !important;
    font-weight: 750 !important;
}

div[data-testid="stButton"] > button[data-testid="stBaseButton-primary"],
.stButton > button[kind="primary"] {
    background-color: #2563eb !important;
    border: 1px solid #2563eb !important;
    color: #ffffff !important;
    -webkit-text-fill-color: #ffffff !important;
    font-weight: 750 !important;
    border-radius: 10px !important;
    min-height: 48px;
    transition: background-color 0.2s ease;
}

/* Ensure nested text is also white */

div[data-testid="stButton"] > button[data-testid="stBaseButton-primary"] *,
.stButton > button[kind="primary"] * {
    color: #ffffff !important;
    -webkit-text-fill-color: #ffffff !important;
}

/* Hover */

div[data-testid="stButton"] > button[data-testid="stBaseButton-primary"]:hover,
.stButton > button[kind="primary"]:hover {
    background-color: #1d4ed8 !important;
    border-color: #1d4ed8 !important;
}

/* Focus */

div[data-testid="stButton"] > button[data-testid="stBaseButton-primary"]:focus-visible {
    outline: 3px solid #93c5fd !important;
    outline-offset: 3px;
}

/* Capture duration information card */
.capture-duration-card {
    display: flex;
    align-items: center;
    justify-content: space-between;
    flex-wrap: wrap;
    gap: 12px;

    margin: 12px 0 18px;
    padding: 16px 20px;

    background: #111827;
    border: 1px solid #475569;
    border-left: 5px solid #38bdf8;
    border-radius: 12px;
}

.capture-duration-label {
    color: #cbd5e1 !important;
    font-size: 13px;
    font-weight: 800;
    letter-spacing: 1px;
}

.capture-duration-value {
    color: #ffffff !important;
    font-size: 18px;
    font-weight: 800;
}

/* Research classification notice */
.classification-notice {
    margin: 16px 0;
    padding: 18px 20px;

    background: #292235;
    border: 1px solid #a78bfa;
    border-left: 5px solid #a78bfa;
    border-radius: 12px;
}

.classification-notice-title {
    color: #ddd6fe !important;
    font-size: 17px;
    font-weight: 800;
    margin-bottom: 9px;
}

.classification-notice p {
    color: #f1f5f9 !important;
    font-size: 14px;
    line-height: 1.65;
    margin: 6px 0;
}

.welcome-hero {
    display: grid;
    grid-template-columns: minmax(0, 1.4fr) minmax(260px, 0.8fr);
    gap: 24px;
    align-items: stretch;

    padding: clamp(24px, 3vw, 38px);
    margin: 16px 0 22px;

    background: linear-gradient(135deg, #ffffff 0%, #edf5ff 100%);
    border: 1px solid #dce7f4;
    border-radius: 22px;

    box-shadow: 0 5px 22px rgba(15, 35, 65, 0.05);
}

.welcome-main {
    display: flex;
    flex-direction: column;
    align-items: flex-start;
    justify-content: center;
    min-width: 0;
}

.welcome-kicker {
    display: inline-flex;
    align-items: center;
    gap: 9px;

    padding: 7px 12px;
    border: 1px solid #cbdcf3;
    border-radius: 30px;

    background: #e5efff;
    color: #245da8 !important;

    font-size: 11px;
    font-weight: 800;
    letter-spacing: 1px;
}

.welcome-kicker span {
    font-size: 16px;
}

.welcome-hero .app-title {
    margin: 22px 0 12px;

    color: #14263d !important;
    font-size: clamp(2rem, 4.5vw, 3.1rem);
    line-height: 1.12;
    font-weight: 850;
    letter-spacing: -1.6px;
}

.welcome-hero .app-subtitle {
    max-width: 600px;
    margin: 0 0 20px;

    color: #52647a !important;
    font-size: 15px;
    line-height: 1.75;
}

.welcome-tags {
    display: flex;
    flex-wrap: wrap;
    gap: 8px;
}

.welcome-tags span {
    padding: 7px 11px;
    border-radius: 8px;

    background: #ffffff;
    border: 1px solid #dce6f2;

    color: #42566f;
    font-size: 11px;
    font-weight: 650;
}

/* Overview panel */

.welcome-overview {
    display: flex;
    flex-direction: column;
    justify-content: center;
    gap: 17px;

    padding: 24px;
    border-radius: 17px;

    background: linear-gradient(145deg, #14263d, #1d3b5e);
    box-shadow: 0 8px 20px rgba(20, 38, 61, 0.12);
}

.overview-eyebrow {
    color: #bfdbfe !important;
    font-size: 11px;
    font-weight: 800;
    letter-spacing: 1.3px;
}

.overview-metric {
    display: flex;
    flex-direction: column;
    gap: 4px;
}

.overview-metric strong {
    color: #ffffff !important;
    font-size: clamp(1.55rem, 2.5vw, 2rem);
    font-weight: 850;
    letter-spacing: -0.7px;
}

.overview-metric span {
    color: #cbd5e1 !important;
    font-size: 12px;
    line-height: 1.5;
}

.overview-divider {
    height: 1px;
    background: rgba(255, 255, 255, 0.18);
}

.overview-note {
    padding-top: 5px;
    color: #dbeafe !important;
    font-size: 12px;
    line-height: 1.7;
}

/* Three-step instructions */

.welcome-steps {
    display: grid;
    grid-template-columns: repeat(3, minmax(0, 1fr));
    gap: 13px;

    margin: 0 0 26px;
}

.welcome-step {
    display: flex;
    align-items: flex-start;
    gap: 12px;

    min-width: 0;
    padding: 17px;

    background: #ffffff;
    border: 1px solid #e0e7f0;
    border-radius: 14px;

    box-shadow: 0 2px 8px rgba(15, 23, 42, 0.025);
}

.welcome-step-number {
    display: flex;
    flex-shrink: 0;
    align-items: center;
    justify-content: center;

    width: 32px;
    height: 32px;
    border-radius: 9px;

    background: #eaf2ff;
    color: #2563eb;

    font-size: 12px;
    font-weight: 850;
}

.welcome-step strong {
    display: block;
    margin: 2px 0 5px;

    color: #1e293b;
    font-size: 13px;
    font-weight: 750;
}

.welcome-step p {
    margin: 0;

    color: #64748b;
    font-size: 12px;
    line-height: 1.55;
}

/* Responsive layout */

@media (max-width: 850px) {
    .welcome-hero {
        grid-template-columns: 1fr;
    }

    .welcome-steps {
        grid-template-columns: 1fr;
    }
}

@media (max-width: 500px) {
    .welcome-hero {
        padding: 20px;
        border-radius: 16px;
    }

    .welcome-overview {
        padding: 20px;
    }
}

/* Typing session heading */
.session-heading {
    margin: 22px 0 5px;
    color: #17263b !important;
    font-size: 1.3rem;
    font-weight: 850;
    letter-spacing: -0.3px;
}

.session-subtitle {
    margin: 0 0 16px;
    color: #52647a !important;
    font-size: 0.94rem;
    line-height: 1.65;
}
    </style>
    """
st.markdown(PAGE_CSS, unsafe_allow_html=True)

st.markdown(
    f"""
    <section class="welcome-hero">
        <div class="welcome-main">
            <div class="welcome-kicker">
                <span>⌨</span> RESEARCH PROTOTYPE
            </div>
            <h1 class="app-title">{APP_TITLE}</h1>
            <p class="app-subtitle">
                Explore keyboard timing patterns through a short,
                browser-based research activity. Type naturally and
                discover how your keyboard interactions are measured.
            </p>
            <div class="welcome-tags">
                <span>Browser-based assessment</span>
                <span>Timing analysis</span>
                <span>Research prototype</span>
            </div>
        </div>
        <aside class="welcome-overview">
            <div class="overview-eyebrow">
                ASSESSMENT OVERVIEW
            </div>
            <div class="overview-metric">
                <strong>200–1,000</strong>
                <span>Valid keystroke observations</span>
            </div>
            <div class="overview-divider"></div>
            <div class="overview-note">
                Type the provided passage naturally using a physical keyboard.
                Only numeric key-hold durations are sent for analysis; typed
                characters stay in the browser.
            </div>
        </aside>
    </section>

    <section class="welcome-steps" aria-label="Assessment steps">
        <article class="welcome-step">
            <span class="welcome-step-number">01</span>
            <div>
                <strong>Start the assessment</strong>
                <p>Begin when you are ready and use a physical keyboard.</p>
            </div>
        </article>
        <article class="welcome-step">
            <span class="welcome-step-number">02</span>
            <div>
                <strong>Type naturally</strong>
                <p>Enter the passage at your usual pace; no need to rush.</p>
            </div>
        </article>
        <article class="welcome-step">
            <span class="welcome-step-number">03</span>
            <div>
                <strong>Review your measurements</strong>
                <p>Analyse timing observations after reaching the minimum.</p>
            </div>
        </article>
    </section>
    """,
    unsafe_allow_html=True,
)

if "assessment_result" not in st.session_state:
    st.session_state.assessment_result = None
if "assessment_error" not in st.session_state:
    st.session_state.assessment_error = None
if "capture_component_key" not in st.session_state:
    st.session_state.capture_component_key = f"capture_{uuid.uuid4().hex}"

assessment = st.session_state.assessment_result

if assessment is None:
    st.markdown(
        '<div class="session-heading">Your typing session</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        """
        <p class="session-subtitle">
            Type the provided paragraph naturally. You can submit your
            assessment once 200 valid keystrokes have been captured.
            You may continue typing up to 1,000 keystrokes before submitting.
        </p>
        """,
        unsafe_allow_html=True,
    )

    if st.session_state.assessment_error:
        st.warning(st.session_state.assessment_error)
        if st.button("Try the typing test again", type="primary"):
            st.session_state.assessment_error = None
            st.session_state.capture_component_key = f"capture_{uuid.uuid4().hex}"
            st.rerun()
    else:
        try:
            capture_state = render_typing_capture(
                passage=PASSAGE,
                minimum_count=MIN_VALID_KEYSTROKES,
                maximum_count=MAX_VALID_KEYSTROKES,
                key=st.session_state.capture_component_key,
            )
        except Exception as exc:  # present an actionable message for setup/API issues
            st.error("The typing component could not be loaded.")
            st.code(
                "python -m pip install --upgrade -r requirements.txt\n"
                "streamlit run app.py",
                language="bash",
            )
            st.caption(f"Details: {exc}")
        else:
            capture_payload = getattr(capture_state, "capture_result", None)
            if isinstance(capture_payload, dict) and capture_payload.get("completed"):
                try:
                    capture_audit = validate_capture_payload(capture_payload)
                    raw_hold_times = capture_audit["hold_times"]
                    result = analyse_keystrokes(
                        raw_hold_times,
                        minimum_keystrokes=MIN_VALID_KEYSTROKES,
                        maximum_keystrokes=MAX_VALID_KEYSTROKES,
                    )
                    feature_audit = verify_analysis_result(
                        raw_hold_times,
                        result,
                    )
                    result.update(feature_audit)
                    training_audit = compare_with_training_distribution(
                        result["macd_seconds"]
                    )
                    result.update(training_audit)
                    received = capture_audit["received_count"]
                    accepted = result["valid_keystrokes"]
                    rejected = received - accepted
                    validity_percentage = (
                        (accepted / received) * 100
                        if received > 0
                        else 0.0
                    )
                    result.update(
                        {
                            "received_observations": received,
                            "accepted_observations": accepted,
                            "rejected_observations": rejected,
                            "validity_percentage": round(
                                validity_percentage, 2
                            ),
                            "capture_elapsed_seconds": capture_audit[
                                "elapsed_seconds"
                            ],
                        }
                    )
                    st.session_state.assessment_result = result
                except (InferenceError, FileNotFoundError, TypeError, ValueError) as exc:
                    st.session_state.assessment_error = str(exc)
                finally:
                    # The raw timing array is used only for this inference call.
                    st.session_state.pop(st.session_state.capture_component_key, None)
                st.rerun()
else:
    st.markdown(
        '<div class="section-label">Assessment Results</div>',
        unsafe_allow_html=True,
    )
    st.success(
        "Your typing assessment has been completed successfully. "
        "Your keyboard timing observations have been analysed."
    )
    st.markdown(
        f"""
        <div class="result-panel">
          <p class="score-label">
            EXPERIMENTAL LOGISTIC REGRESSION OUTPUT
          </p>
          <div class="score-value">{assessment["model_output_score"]:.4f}</div>
          <div class="score-caption">
            Raw logistic-regression output corresponding to encoded class 1
            in the research dataset. This is an experimental model score,
            not a clinically calibrated probability.
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("### Keystroke Analysis")
    first, second, third = st.columns(3)
    first.metric("MACD", f'{assessment["macd_seconds"]:.5f} s')
    second.metric("Valid keystrokes", f'{assessment["valid_keystrokes"]:,}')
    third.metric("MACD pairs", f'{assessment["macd_pairs"]:,}')

    st.markdown("### Typing Measurements")
    a, b, c = st.columns(3)
    a.metric("Average", f'{assessment["mean_hold_seconds"] * 1000:.1f} ms')
    b.metric("Median", f'{assessment["median_hold_seconds"] * 1000:.1f} ms')
    c.metric("Variation", f'{assessment["std_hold_seconds"] * 1000:.1f} ms')

    st.markdown("### Data Quality")
    st.info("Timing observations successfully analysed.")
    if assessment.get("capture_elapsed_seconds") is not None:
        duration_seconds = float(assessment["capture_elapsed_seconds"])
        st.markdown(
            f"""
            <div class="capture-duration-card">
                <span class="capture-duration-label">
                    CAPTURE DURATION
                </span>
                <strong class="capture-duration-value">
                    {duration_seconds:.2f} seconds
                </strong>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with st.expander("How are the typing features calculated?"):
        st.markdown(
            """
            **Mean Absolute Consecutive Difference (MACD)**

            MACD measures the average absolute difference between consecutive
            keyboard hold durations. The research implementation first
            identifies the 60th–100th percentile corridor. A consecutive pair
            contributes only when both durations fall inside this corridor.

            **Typing measurements**

            - Average hold: Mean key hold duration.
            - Median hold: Middle key hold duration.
            - Hold-time variation: Population standard deviation.
            - MACD pairs: Number of qualifying consecutive pairs.

            All timing measurements are expressed in seconds internally and
            displayed in milliseconds where appropriate.
            """
        )
        st.caption(
            f"Percentile corridor: "
            f"{assessment['percentile_low'] * 100:.0f}th–"
            f"{assessment['percentile_high'] * 100:.0f}th percentile."
        )
        st.caption(
            f"Verified observations: "
            f"{assessment['verified_observation_count']:,}"
        )

    st.markdown("### Training Data Comparison")
    distribution_metrics = (
        ("Minimum", "min"),
        ("P1", "p1"),
        ("P5", "p5"),
        ("Median", "median"),
        ("P95", "p95"),
        ("P99", "p99"),
        ("Maximum", "max"),
    )
    for start in range(0, len(distribution_metrics), 4):
        metric_columns = st.columns(4)
        for column, (label, percentile) in zip(
            metric_columns,
            distribution_metrics[start : start + 4],
        ):
            column.metric(
                label,
                f'{assessment[f"training_distribution_{percentile}"]:.5f}',
            )
    st.caption(
        "The measured MACD is "
        f'{assessment["training_distribution_position"].lower()}. '
        "This comparison does not establish that browser timing is "
        "equivalent to the research dataset."
    )
    if assessment["outside_training_range"]:
        st.warning(
            "The measured MACD is outside the feature range observed during model training. "
            "The model output may be unreliable for this sample."
        )

    st.markdown("### Important Medical Disclaimer")
    st.markdown(
        "<div class='disclaimer'>This research prototype cannot confirm or "
        "rule out Parkinson's disease. Browser-collected keyboard timings "
        "have not been validated as equivalent to the research dataset. "
        "Consult a qualified healthcare professional for health concerns.</div>",
        unsafe_allow_html=True,
    )
    st.write("")
    st.markdown("### Take the typing test again")
    if st.button("Take the typing test again", type="primary", use_container_width=True):
        st.session_state.assessment_result = None
        st.session_state.assessment_error = None
        st.session_state.capture_component_key = f"capture_{uuid.uuid4().hex}"
        st.rerun()

st.markdown(
    """
    <div class='disclaimer'>
        <strong>Privacy and data handling:</strong>
        The typing passage is displayed in the browser.
        Characters entered by the user are not included in
        the capture payload sent to the Python application.
        Only numeric key-hold durations and necessary capture
        metadata are transmitted for analysis.
        Derived assessment measurements and the experimental
        model score are retained in the current Streamlit session.
        The application does not intentionally write captured
        text or raw timing observations to a database or file.
    </div>
    """,
    unsafe_allow_html=True,
)
