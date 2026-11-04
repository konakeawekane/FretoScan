import { Scanner } from './pipeline.js';

let scanner = new Scanner();

self.onmessage = ({ data }) => {
  if (data.type === 'reset') { scanner = new Scanner(data.fov); return; }
  const result = scanner.process(data);
  const message = { ...result, sessionId: data.sessionId };
  const buffers = result.positions ? [result.positions.buffer, result.colors.buffer] : [];
  self.postMessage(message, buffers);
};
