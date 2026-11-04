import { Scanner } from './pipeline.js';

let scanner = new Scanner();

self.onmessage = ({ data }) => {
  if (data.type === 'reset') { scanner = new Scanner(data.fov); return; }
  const result = scanner.process(data);
  self.postMessage({ ...result, sessionId: data.sessionId, tMs: data.tMs, mediaTime: data.mediaTime });
};
