from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
APP_PATH = ROOT / "app.py"
JS_PATH = ROOT / "app_components" / "typing_capture.js"


def test_application_enforces_minimum_and_maximum():
    app = APP_PATH.read_text(encoding="utf-8")
    js = JS_PATH.read_text(encoding="utf-8")

    assert "MIN_VALID_KEYSTROKES = 200" in app
    assert "MAX_VALID_KEYSTROKES = 1000" in app
    assert "minimum_count=MIN_VALID_KEYSTROKES" in app
    assert "maximum_count=MAX_VALID_KEYSTROKES" in app
    assert "maximum_keystrokes=MAX_VALID_KEYSTROKES" in app
    assert "holdTimes.length < maximumCount" in js
    assert "holdTimes.length > maximumCount" in js


def test_submission_requires_minimum_sample_without_matching_passage():
    js = JS_PATH.read_text(encoding="utf-8")
    html = (ROOT / "app_components" / "typing_capture.html").read_text(
        encoding="utf-8"
    )

    assert "function submitCapture()" in js
    assert "holdTimes.length < minimumCount" in js
    assert "minimumReached" in js
    assert 'id="submit-button" type="button" disabled' in html
    assert 'id="timing-count"' in html
