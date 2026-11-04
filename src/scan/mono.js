import { TRANSFORMERS_URL, DEPTH_MODEL } from './vendor.js';

function halfToFloat(bits) {
  const sign = bits & 0x8000 ? -1 : 1, exponent = (bits >> 10) & 0x1f, fraction = bits & 0x3ff;
  if (!exponent) return sign * 2 ** -14 * (fraction / 1024);
  if (exponent === 31) return fraction ? NaN : sign * Infinity;
  return sign * 2 ** (exponent - 15) * (1 + fraction / 1024);
}

function resample(source, sourceWidth, sourceHeight, width, height) {
  if (sourceWidth === width && sourceHeight === height) return source;
  const out = new Float32Array(width * height);
  for (let y = 0; y < height; y++) for (let x = 0; x < width; x++) {
    const sx = Math.min(sourceWidth - 1.001, Math.max(0, (x + 0.5) * sourceWidth / width - 0.5));
    const sy = Math.min(sourceHeight - 1.001, Math.max(0, (y + 0.5) * sourceHeight / height - 0.5));
    const ix = Math.floor(sx), iy = Math.floor(sy), u = sx - ix, v = sy - iy, i = iy * sourceWidth + ix;
    out[y * width + x] = (source[i] * (1 - u) + source[i + 1] * u) * (1 - v)
      + (source[i + sourceWidth] * (1 - u) + source[i + sourceWidth + 1] * u) * v;
  }
  return out;
}

export async function loadDepthModel(onStatus) {
  const { pipeline, env, RawImage } = await import(TRANSFORMERS_URL);
  env.allowLocalModels = false;
  const adapter = typeof navigator !== 'undefined' && navigator.gpu ? await navigator.gpu.requestAdapter().catch(() => null) : null;
  const files = new Map();
  const estimator = await pipeline('depth-estimation', DEPTH_MODEL, {
    device: adapter ? 'webgpu' : 'wasm',
    dtype: adapter ? 'fp16' : 'q8',
    progress_callback: (event) => {
      if (event.status !== 'progress' || !event.total) return;
      files.set(event.file, [event.loaded, event.total]);
      const [loaded, total] = [...files.values()].reduce((sum, [a, b]) => [sum[0] + a, sum[1] + b], [0, 0]);
      onStatus?.({ message: `Downloading depth model ${Math.round(loaded / total * 100)}%` });
    },
  });
  const run = async ({ rgba, width, height }) => {
    const rgb = new Uint8ClampedArray(width * height * 3);
    for (let i = 0, j = 0; i < rgb.length; i += 3, j += 4) { rgb[i] = rgba[j]; rgb[i + 1] = rgba[j + 1]; rgb[i + 2] = rgba[j + 2]; }
    const output = await estimator(new RawImage(rgb, width, height, 3));
    const tensor = (Array.isArray(output) ? output[0] : output).predicted_depth;
    const [sourceHeight, sourceWidth] = tensor.dims.slice(-2);
    const data = tensor.type === 'float16' && tensor.data instanceof Uint16Array
      ? Float32Array.from(tensor.data, halfToFloat) : Float32Array.from(tensor.data);
    return resample(data, sourceWidth, sourceHeight, width, height);
  };
  return { run, device: adapter ? 'webgpu' : 'wasm' };
}

function solveLine(samples, prior) {
  let sw = 0, sx = 0, sy = 0, sxx = 0, sxy = 0;
  for (const { p, inverse, weight = 1 } of samples) {
    const w = weight * prior;
    sw += w; sx += w * p; sy += w * inverse; sxx += w * p * p; sxy += w * p * inverse;
  }
  const determinant = sw * sxx - sx * sx;
  if (Math.abs(determinant) < 1e-12) return null;
  return { scale: (sw * sxy - sx * sy) / determinant, shift: (sxx * sy - sx * sxy) / determinant };
}

// Depth Anything predicts affine-invariant inverse depth, so inverse depth ~= scale * prior + shift.
export function alignPrior(prior, width, height, anchors, seed = 11) {
  const samples = anchors.flatMap(({ x, y, depth, weight }) => {
    const ix = Math.round(x), iy = Math.round(y);
    if (!(depth > 0) || ix < 0 || iy < 0 || ix >= width || iy >= height) return [];
    const p = prior[iy * width + ix];
    return Number.isFinite(p) ? [{ p, inverse: 1 / depth, weight }] : [];
  });
  if (samples.length < 6) return null;
  let random = seed;
  const next = () => ((random = (random * 1664525 + 1013904223) >>> 0) / 2 ** 32);
  const inliersOf = ({ scale, shift }) => samples.filter(({ p, inverse }) => Math.abs(scale * p + shift - inverse) < inverse * 0.12);
  let best = [];
  for (let iteration = 0; iteration < 80; iteration++) {
    const a = samples[Math.floor(next() * samples.length)], b = samples[Math.floor(next() * samples.length)];
    if (Math.abs(a.p - b.p) < 1e-6) continue;
    const scale = (a.inverse - b.inverse) / (a.p - b.p);
    if (!(scale > 0)) continue;
    const inliers = inliersOf({ scale, shift: a.inverse - scale * a.p });
    if (inliers.length > best.length) best = inliers;
  }
  if (best.length < Math.max(6, samples.length * 0.35)) return null;
  let fit = solveLine(best, 1);
  if (!fit || !(fit.scale > 0)) return null;
  best = inliersOf(fit);
  fit = solveLine(best, 1) || fit;
  if (!(fit.scale > 0)) return null;
  const depth = new Float32Array(width * height);
  for (let i = 0; i < depth.length; i++) {
    const inverse = fit.scale * prior[i] + fit.shift;
    if (inverse > 1e-3) depth[i] = 1 / inverse;
  }
  return { ...fit, depth, inliers: best.length, ratio: best.length / samples.length };
}
