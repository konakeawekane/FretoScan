import { Baker } from './bake.js';
import { loadDepthModel } from './mono.js';

let baker, depthModel;

async function decode({ blob, width, height, pose, focal, landmarks, sharpness }, size) {
  const bitmap = await createImageBitmap(blob);
  const scale = Math.min(1, size / Math.max(bitmap.width, bitmap.height));
  const outWidth = Math.max(1, Math.round(bitmap.width * scale)), outHeight = Math.max(1, Math.round(bitmap.height * scale));
  const context = new OffscreenCanvas(outWidth, outHeight).getContext('2d', { willReadFrequently: true });
  context.drawImage(bitmap, 0, 0, outWidth, outHeight);
  bitmap.close();
  const rgba = context.getImageData(0, 0, outWidth, outHeight).data;
  const sx = outWidth / width, sy = outHeight / height;
  return { width: outWidth, height: outHeight, rgba, pose, focal: focal * sx, sharpness,
    landmarks: landmarks.map(({ id, x, y, world }) => ({ id, x: x * sx, y: y * sy, world })) };
}

self.onmessage = async ({ data }) => {
  try {
    if (data.type === 'start') {
      baker = new Baker(data.quality);
      depthModel = null;
      let warning;
      if (data.neural) {
        try { depthModel = await loadDepthModel((status) => self.postMessage({ type: 'status', ...status })); }
        catch (error) { warning = `Neural depth unavailable (${error.message}); using stereo only`; }
      }
      self.postMessage({ type: 'ready', neural: depthModel?.device || null, warning });
    } else if (data.type === 'frame') {
      const frame = await decode(data.frame, baker.settings.size);
      if (depthModel) frame.prior = await depthModel.run(frame);
      self.postMessage({ type: 'progress', metrics: baker.addFrame(frame) });
    } else if (data.type === 'finish') {
      const result = baker.finish();
      self.postMessage({ type: 'complete', ...result }, [result.positions.buffer, result.colors.buffer]);
    }
  } catch (error) {
    self.postMessage({ type: 'error', message: error.message, stack: error.stack });
  }
};
