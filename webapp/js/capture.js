export function createCapture(onProgress, onComplete, targetCount = 200) {
  const starts = new Map();
  const holds = [];
  function down(event) {
    if (event.repeat || starts.has(event.code)) return;
    starts.set(event.code, event.timeStamp);
  }
  function up(event) {
    if (!starts.has(event.code)) return;
    const hold = (event.timeStamp - starts.get(event.code)) / 1000;
    starts.delete(event.code);
    if (Number.isFinite(hold) && hold >= 0 && hold < 5) {
      holds.push(hold);
      onProgress(Math.min(holds.length / targetCount, 1), holds.length);
      if (holds.length >= targetCount) stop();
    }
  }
  function stop() {
    window.removeEventListener('keydown', down);
    window.removeEventListener('keyup', up);
    onComplete(holds.slice());
  }
  window.addEventListener('keydown', down);
  window.addEventListener('keyup', up);
  return { stop };
}
