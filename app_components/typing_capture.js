export default function (component) {
  const { parentElement, setStateValue, data } = component;
  const passage = parentElement.querySelector("#passage-text");
  const input = parentElement.querySelector("#typing-input");
  const startButton = parentElement.querySelector("#start-button");
  const cancelButton = parentElement.querySelector("#cancel-button");
  const submitButton = parentElement.querySelector("#submit-button");
  const stateLabel = parentElement.querySelector("#capture-state");
  const targetLabel = parentElement.querySelector("#capture-target");
  const progressTrack = parentElement.querySelector("#progress-track");
  const progressFill = parentElement.querySelector("#progress-fill");
  const progressCount = parentElement.querySelector("#progress-count");
  const timingCount = parentElement.querySelector("#timing-count");
  const status = parentElement.querySelector("#capture-status");

  const minimumCount = Math.max(
    2,
    Number(data?.minimum_count) || 200
  );
  const maximumCount = Math.max(
    minimumCount,
    Number(data?.maximum_count) || 1000
  );
  const maxHoldSeconds = 5;
  const keyDownTimes = new Map();
  let holdTimes = [];
  let isCapturing = false;
  let isFinished = false;
  let startedAt = 0;

  passage.textContent = data?.passage || "";
  targetLabel.textContent = `${minimumCount}–${maximumCount} keystrokes`;
  progressTrack.setAttribute("aria-valuemax", "100");

  function updateProgress() {
    const count = holdTimes.length;
    const percentage = Math.min(
      100,
      Math.round((count / minimumCount) * 100)
    );
    const minimumReached = count >= minimumCount;

    progressFill.style.width = `${percentage}%`;
    progressCount.textContent =
      `${percentage}% of minimum required`;
    progressTrack.setAttribute(
      "aria-valuenow",
      String(percentage)
    );
    timingCount.textContent =
      `Valid keystrokes: ${count} / ${maximumCount} | Minimum: ${minimumCount}`;

    submitButton.disabled = !(
      isCapturing &&
      minimumReached &&
      count <= maximumCount
    );

    if (!isCapturing) return;

    if (count >= maximumCount) {
      input.disabled = true;
      status.textContent =
        `Maximum of ${maximumCount.toLocaleString()} keystrokes reached. ` +
        "You can now analyse your typing.";
    } else if (minimumReached) {
      status.textContent =
        "Minimum requirement reached! You can analyse your typing now or continue collecting observations.";
    } else {
      const remaining = minimumCount - count;
      status.textContent =
        `Keep typing naturally. ${remaining} more valid keystrokes needed before submission.`;
    }
  }

  function isCharacterKey(event) {
    return !event.isComposing && !event.metaKey && !event.ctrlKey && !event.altKey
      && ((typeof event.key === "string" && event.key.length === 1) || event.code === "Space");
  }

  function keyDown(event) {
    if (!isCapturing || event.repeat || !isCharacterKey(event)) return;
    const keyId = event.code || event.key;
    if (!keyDownTimes.has(keyId)) keyDownTimes.set(keyId, event.timeStamp);
  }

  function keyUp(event) {
    if (!isCapturing || !isCharacterKey(event)) return;
    const keyId = event.code || event.key;
    if (!keyDownTimes.has(keyId)) return;

    const durationSeconds = (event.timeStamp - keyDownTimes.get(keyId)) / 1000;
    keyDownTimes.delete(keyId);

    // Match the research pipeline's accepted timing domain: finite, >= 0 and < 5 s.
    if (
      Number.isFinite(durationSeconds) &&
      durationSeconds >= 0 &&
      durationSeconds < maxHoldSeconds &&
      holdTimes.length < maximumCount
    ) {
      holdTimes.push(durationSeconds);
      updateProgress();
      if (holdTimes.length === maximumCount) {
        status.textContent =
          `Maximum of ${maximumCount} valid timing events reached. ` +
          "Finish the paragraph and submit when ready.";
      }
    }
  }

  function handleWindowBlur() {
    // Do not pair a keydown before a focus loss with a keyup after focus returns.
    keyDownTimes.clear();
  }

  function submitCapture() {
    if (!isCapturing || isFinished) return;
    if (
      holdTimes.length < minimumCount ||
      holdTimes.length > maximumCount
    ) {
      updateProgress();
      return;
    }

    isCapturing = false;
    isFinished = true;
    keyDownTimes.clear();
    input.removeEventListener("keydown", keyDown);
    input.removeEventListener("keyup", keyUp);
    input.removeEventListener("input", updateProgress);
    window.removeEventListener("blur", handleWindowBlur);

    const elapsedSeconds = Math.max(0, (performance.now() - startedAt) / 1000);
    const timingOnlyPayload = {
      hold_times_s: holdTimes.slice(0, maximumCount),
      valid_keystrokes: Math.min(holdTimes.length, maximumCount),
      elapsed_seconds: elapsedSeconds,
      completed: true
    };

    // The text is checked locally and never included in the message to Python.
    input.value = "";
    input.disabled = true;
    startButton.disabled = true;
    cancelButton.disabled = true;
    submitButton.disabled = true;
    stateLabel.textContent = "Capture complete";
    parentElement.querySelector(".step-dot").classList.remove("active");
    status.textContent = "Timing capture complete. The typed text has been cleared from the input.";

    setStateValue("capture_result", timingOnlyPayload);
    // Drop the local timing array after handing off the numeric payload.
    holdTimes = [];
  }

  function startCapture() {
    if (isCapturing || isFinished) return;
    holdTimes = [];
    keyDownTimes.clear();
    isCapturing = true;
    startedAt = performance.now();
    input.value = "";
    input.disabled = false;
    startButton.disabled = true;
    cancelButton.disabled = false;
    stateLabel.textContent = "Assessment in progress";
    parentElement.querySelector(".step-dot").classList.add("active");
    status.textContent =
      "Type naturally. You can submit once 200 valid keystrokes have been captured.";
    updateProgress();
    input.addEventListener("keydown", keyDown);
    input.addEventListener("keyup", keyUp);
    input.addEventListener("input", updateProgress);
    window.addEventListener("blur", handleWindowBlur);
    input.focus();
  }

  function cancelCapture() {
    if (!isCapturing) return;
    isCapturing = false;
    keyDownTimes.clear();
    holdTimes = [];
    input.removeEventListener("keydown", keyDown);
    input.removeEventListener("keyup", keyUp);
    input.removeEventListener("input", updateProgress);
    window.removeEventListener("blur", handleWindowBlur);
    input.value = "";
    input.disabled = true;
    startButton.disabled = false;
    cancelButton.disabled = true;
    submitButton.disabled = true;
    stateLabel.textContent = "Assessment ready";
    parentElement.querySelector(".step-dot").classList.remove("active");
    status.textContent = "Test cancelled. No timing data was sent to Python.";
    updateProgress();
  }

  startButton.addEventListener("click", startCapture);
  cancelButton.addEventListener("click", cancelCapture);
  submitButton.addEventListener("click", submitCapture);
  updateProgress();

  return () => {
    // Remove button event listeners.
    startButton.removeEventListener("click", startCapture);
    cancelButton.removeEventListener("click", cancelCapture);
    submitButton.removeEventListener("click", submitCapture);

    // Remove typing event listeners.
    input.removeEventListener("keydown", keyDown);
    input.removeEventListener("keyup", keyUp);
    input.removeEventListener("input", updateProgress);

    // Remove window event listeners.
    window.removeEventListener("blur", handleWindowBlur);

    // Clear all locally retained timing data.
    keyDownTimes.clear();
    holdTimes = [];

    // Clear the typed text and disable the input.
    input.value = "";
    input.disabled = true;
  };
}
