export function quantile(values, q) {
  const sorted = [...values].sort((a, b) => a - b);
  const position = (sorted.length - 1) * q;
  const lower = Math.floor(position), upper = Math.ceil(position);
  if (lower === upper) return sorted[lower];
  return sorted[lower] + (sorted[upper] - sorted[lower]) * (position - lower);
}

export function macd(holdTimes, ql = 0.6, qh = 1.0) {
  const x = holdTimes.filter(Number.isFinite);
  if (x.length < 2) return NaN;
  const low = quantile(x, ql), high = quantile(x, qh);
  const diffs = [];
  for (let i = 0; i < x.length - 1; i += 1) {
    if (x[i] >= low && x[i] <= high && x[i + 1] >= low && x[i + 1] <= high) {
      diffs.push(Math.abs(x[i + 1] - x[i]));
    }
  }
  return diffs.length ? diffs.reduce((a, b) => a + b, 0) / diffs.length : NaN;
}
