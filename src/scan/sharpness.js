export function sharpness(gray, width, height) {
  let sum = 0, squared = 0, count = 0;
  for (let y = 1; y < height - 1; y++) for (let x = 1; x < width - 1; x++) {
    const i = y * width + x;
    const value = 4 * gray[i] - gray[i - 1] - gray[i + 1] - gray[i - width] - gray[i + width];
    sum += value;
    squared += value * value;
    count++;
  }
  return count ? squared / count - (sum / count) ** 2 : 0;
}

export class KeyframePicker {
  constructor(durationS, count) {
    this.interval = Math.max(durationS / Math.max(1, count), 1 / 60);
    this.bin = -1;
    this.best = -Infinity;
  }

  offer(timeS, score) {
    const bin = Math.floor(timeS / this.interval);
    const flush = bin !== this.bin && this.bin >= 0;
    if (bin !== this.bin) { this.bin = bin; this.best = -Infinity; }
    const keep = score > this.best;
    if (keep) this.best = score;
    return { flush, keep };
  }
}
