const MASTERS = new Set([0x18538067, 0x1654ae6b, 0xae, 0xe0, 0x1f43b675, 0xa0]);
const TRACK_ENTRY = 0xae, TRACK_NUMBER = 0xd7, TRACK_TYPE = 0x83, CODEC_ID = 0x86, CODEC_PRIVATE = 0x63a2;
const PIXEL_WIDTH = 0xb0, PIXEL_HEIGHT = 0xba, TIMECODE_SCALE = 0x2ad7b1, TIMECODE = 0xe7;
const SIMPLE_BLOCK = 0xa3, BLOCK = 0xa1;

function vint(bytes, offset, keepMarker) {
  const first = bytes[offset];
  if (!first) return null;
  let length = 1;
  while (!(first & (0x80 >> (length - 1)))) length++;
  if (offset + length > bytes.length) return null;
  let value = keepMarker ? first : first & (0xff >> length);
  let unknown = value === (0xff >> length);
  for (let i = 1; i < length; i++) {
    value = value * 256 + bytes[offset + i];
    if (bytes[offset + i] !== 0xff) unknown = false;
  }
  return { value, length, unknown: !keepMarker && unknown };
}

function unsigned(bytes, start, end) {
  let value = 0;
  for (let i = start; i < end; i++) value = value * 256 + bytes[i];
  return value;
}

function codecs(id, privateData) {
  if (id === 'V_VP8') return [{ codec: 'vp8' }];
  if (id === 'V_VP9') return ['vp09.00.10.08', 'vp09.00.41.08', 'vp09.01.10.08', 'vp09.02.10.10'].map((codec) => ({ codec }));
  if (id === 'V_AV1') return ['av01.0.04M.08', 'av01.0.08M.08', 'av01.0.12M.08'].map((codec) => ({ codec }));
  if (id === 'V_MPEG4/ISO/AVC' && privateData?.length > 3) {
    const hex = Array.from(privateData.subarray(1, 4), (value) => value.toString(16).padStart(2, '0')).join('');
    return [{ codec: `avc1.${hex}`, description: privateData }];
  }
  return [];
}

export function parseWebm(bytes) {
  const tracks = [];
  const samples = [];
  let track = null, scale = 1e6, cluster = 0;
  let offset = 0;
  while (offset < bytes.length) {
    const id = vint(bytes, offset, true);
    if (!id) break;
    const size = vint(bytes, offset + id.length, false);
    if (!size) break;
    const start = offset + id.length + size.length;
    const end = size.unknown ? bytes.length : Math.min(bytes.length, start + size.value);
    if (MASTERS.has(id.value)) {
      if (id.value === TRACK_ENTRY) tracks.push(track = { number: 0, type: 0, codecId: '', width: 0, height: 0 });
      offset = start;
      continue;
    }
    if (id.value === TIMECODE_SCALE) scale = unsigned(bytes, start, end);
    else if (id.value === TIMECODE) cluster = unsigned(bytes, start, end);
    else if (track && id.value === TRACK_NUMBER) track.number = unsigned(bytes, start, end);
    else if (track && id.value === TRACK_TYPE) track.type = unsigned(bytes, start, end);
    else if (track && id.value === CODEC_ID) track.codecId = new TextDecoder().decode(bytes.subarray(start, end)).replace(/\0+$/, '');
    else if (track && id.value === CODEC_PRIVATE) track.codecPrivate = bytes.slice(start, end);
    else if (track && id.value === PIXEL_WIDTH) track.width = unsigned(bytes, start, end);
    else if (track && id.value === PIXEL_HEIGHT) track.height = unsigned(bytes, start, end);
    else if (id.value === SIMPLE_BLOCK || id.value === BLOCK) {
      const number = vint(bytes, start, false);
      if (number && start + number.length + 3 <= end) {
        const at = start + number.length;
        const relative = (bytes[at] << 24 >> 16) | bytes[at + 1];
        const flags = bytes[at + 2];
        if (!(flags & 0x06)) {
          samples.push({
            track: number.value,
            timestamp: (cluster + relative) * scale / 1000,
            key: id.value === SIMPLE_BLOCK ? Boolean(flags & 0x80) : samples.length === 0,
            data: bytes.subarray(at + 3, end),
          });
        }
      }
    }
    offset = end;
  }
  const video = tracks.find((entry) => entry.type === 1) || tracks.find((entry) => entry.codecId.startsWith('V_'));
  if (!video) throw new Error('No video track in WebM');
  return {
    width: video.width,
    height: video.height,
    configs: codecs(video.codecId, video.codecPrivate),
    samples: samples.filter((sample) => sample.track === video.number),
  };
}
