import { createCapture } from './capture.js';
import { macd } from './macd.js';

const status = document.querySelector('#status');
const progress = document.querySelector('#progress');
const result = document.querySelector('#result');
const startButton = document.querySelector('#start');
let model;
fetch('../models/macd_model.json').then((r) => r.json()).then((m) => { model = m; startButton.disabled = false; status.textContent = 'Ready. No typed text is stored.'; });

startButton.addEventListener('click', () => {
  startButton.disabled = true;
  status.textContent = 'Type naturally. Only timing values are retained in this browser.';
  createCapture((fraction, count) => { progress.value = fraction; status.textContent = `${count} timing observations captured`; }, (holds) => {
    const score = macd(holds, model.corridor_low, model.corridor_high);
    const logit = model.intercept + model.coefficient * score;
    const probability = 1 / (1 + Math.exp(-logit));
    result.textContent = `Research screening score: ${(probability * 100).toFixed(1)}% (not a diagnosis).`;
    startButton.disabled = false;
  }, 200);
});
