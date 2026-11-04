export function grayscale(rgba, width, height) {
  const gray = new Uint8Array(width * height);
  for (let i = 0, j = 0; i < gray.length; i++, j += 4) gray[i] = (rgba[j] * 77 + rgba[j + 1] * 150 + rgba[j + 2] * 29) >> 8;
  return gray;
}

export function detect(gray, width, height) {
  const cell = 16;
  const points = [];
  for (let cy = 5; cy < height - 5; cy += cell) for (let cx = 5; cx < width - 5; cx += cell) {
    let best = 1400, point = null;
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

function half(image, width, height) {
  const w = Math.floor(width / 2), h = Math.floor(height / 2);
  const pixels = new Uint8Array(w * h);
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
    const i = y * 2 * width + x * 2;
    pixels[y * w + x] = (image[i] + image[i + 1] + image[i + width] + image[i + width + 1]) >> 2;
  }
  return { pixels, width: w, height: h };
}

function patch(image, width, x, y, radius) {
  const values = [];
  let sum = 0;
  for (let dy = -radius; dy <= radius; dy++) for (let dx = -radius; dx <= radius; dx++) {
    const value = image[(y + dy) * width + x + dx];
    values.push(value);
    sum += value;
  }
  const mean = sum / values.length;
  let variance = 0;
  for (let i = 0; i < values.length; i++) { values[i] -= mean; variance += values[i] * values[i]; }
  return { values, variance };
}

function score(reference, image, width, x, y, radius) {
  let sum = 0, squared = 0, cross = 0, k = 0;
  for (let dy = -radius; dy <= radius; dy++) for (let dx = -radius; dx <= radius; dx++) {
    const value = image[(y + dy) * width + x + dx];
    sum += value;
    squared += value * value;
    cross += reference.values[k++] * value;
  }
  const variance = squared - sum * sum / reference.values.length;
  return variance > 100 && reference.variance > 100 ? 1 - cross / Math.sqrt(reference.variance * variance) : 2;
}

function search(reference, image, width, height, centerX, centerY, radius, patchRadius) {
  let best = { score: 2 };
  const minX = Math.max(patchRadius + 1, Math.round(centerX) - radius);
  const maxX = Math.min(width - patchRadius - 2, Math.round(centerX) + radius);
  const minY = Math.max(patchRadius + 1, Math.round(centerY) - radius);
  const maxY = Math.min(height - patchRadius - 2, Math.round(centerY) + radius);
  for (let y = minY; y <= maxY; y++) for (let x = minX; x <= maxX; x++) {
    const value = score(reference, image, width, x, y, patchRadius);
    if (value < best.score) best = { x, y, score: value };
  }
  return best;
}

export function match(previous, current, features, width, height, hint = [0, 0]) {
  const coarseA = half(previous, width, height), coarseB = half(current, width, height);
  const candidates = [];
  for (const { x, y } of features) {
    const cx = Math.round(x / 2), cy = Math.round(y / 2);
    if (cx < 5 || cy < 5 || cx >= coarseA.width - 5 || cy >= coarseA.height - 5) continue;
    const coarsePatch = patch(coarseA.pixels, coarseA.width, cx, cy, 2);
    if (coarsePatch.variance < 400) continue;
    const coarse = search(coarsePatch, coarseB.pixels, coarseB.width, coarseB.height, cx + hint[0] / 2, cy + hint[1] / 2, 27, 2);
    if (coarse.score > 0.75) continue;
    const source = patch(previous, width, x, y, 3);
    const fine = search(source, current, width, height, coarse.x * 2, coarse.y * 2, 5, 3);
    if (fine.score > 0.35) continue;
    const reverse = patch(current, width, fine.x, fine.y, 3);
    const back = search(reverse, previous, width, height, x, y, 4, 3);
    if (Math.hypot(back.x - x, back.y - y) > 1.5) continue;
    candidates.push({ x1: x, y1: y, x2: fine.x, y2: fine.y, error: fine.score });
  }
  candidates.sort((a, b) => a.error - b.error);
  const pairs = [];
  for (const candidate of candidates) {
    if (pairs.every((pair) => Math.hypot(pair.x2 - candidate.x2, pair.y2 - candidate.y2) >= 4)) pairs.push(candidate);
  }
  return pairs.slice(0, 240);
}
