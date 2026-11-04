export function grayscale(rgba, width, height) {
  const gray = new Uint8Array(width * height);
  for (let i = 0, j = 0; i < gray.length; i++, j += 4) gray[i] = (rgba[j] * 77 + rgba[j + 1] * 150 + rgba[j + 2] * 29) >> 8;
  return gray;
}

export function detect(gray, width, height) {
  const cell = 18;
  const points = [];
  for (let cy = 5; cy < height - 5; cy += cell) for (let cx = 5; cx < width - 5; cx += cell) {
    let best = 1200, point = null;
    for (let y = cy; y < Math.min(cy + cell, height - 5); y += 2) for (let x = cx; x < Math.min(cx + cell, width - 5); x += 2) {
      let xx = 0, xy = 0, yy = 0;
      for (let dy = -2; dy <= 2; dy += 2) for (let dx = -2; dx <= 2; dx += 2) {
        const i = (y + dy) * width + x + dx;
        const gx = gray[i + 1] - gray[i - 1], gy = gray[i + width] - gray[i - width];
        xx += gx * gx; xy += gx * gy; yy += gy * gy;
      }
      const score = (xx + yy - Math.hypot(xx - yy, 2 * xy)) / 2;
      if (score > best) { best = score; point = { x, y }; }
    }
    if (point) points.push(point);
  }
  return points;
}

function patchError(a, b, width, x1, y1, x2, y2, limit) {
  let error = 0;
  for (let dy = -3; dy <= 3; dy++) for (let dx = -3; dx <= 3; dx++) {
    const difference = a[(y1 + dy) * width + x1 + dx] - b[(y2 + dy) * width + x2 + dx];
    error += difference * difference;
    if (error > limit) return error;
  }
  return error;
}

export function match(previous, current, features, width, height) {
  const candidates = [];
  for (const { x, y } of features) {
    let best = 49 * 600, x2 = 0, y2 = 0;
    for (let yy = Math.max(4, y - 32); yy <= Math.min(height - 5, y + 32); yy += 2) {
      for (let xx = Math.max(4, x - 32); xx <= Math.min(width - 5, x + 32); xx += 2) {
        const error = patchError(previous, current, width, x, y, xx, yy, best);
        if (error < best) { best = error; x2 = xx; y2 = yy; }
      }
    }
    if (best < 49 * 500 && Math.hypot(x2 - x, y2 - y) < 43) candidates.push({ x1: x, y1: y, x2, y2, error: best });
  }
  candidates.sort((a, b) => a.error - b.error);
  const pairs = [];
  for (const candidate of candidates) {
    if (pairs.every((pair) => Math.hypot(pair.x2 - candidate.x2, pair.y2 - candidate.y2) >= 5)) pairs.push(candidate);
  }
  return pairs.slice(0, 180);
}
