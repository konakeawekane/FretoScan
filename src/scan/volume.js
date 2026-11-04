import { mul3, transpose3 } from './reconstruction.js';
import { cleanMesh, expand } from './mesh.js';

const corners = [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0], [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]];
const tetrahedra = [[0, 5, 1, 6], [0, 1, 2, 6], [0, 2, 3, 6], [0, 3, 7, 6], [0, 7, 4, 6], [0, 4, 5, 6]];
const edges = [[0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3]];
const RANGE = 1024;
const key = (x, y, z) => ((x + RANGE) * 2048 + (y + RANGE)) * 2048 + (z + RANGE);
const inside = (x, y, z) => x >= -RANGE && y >= -RANGE && z >= -RANGE && x < RANGE && y < RANGE && z < RANGE;

export class TsdfVolume {
  constructor(voxelSize = 0.035, truncation = 0.105, maxCells = 160000) {
    this.voxelSize = voxelSize;
    this.truncation = truncation;
    this.maxCells = maxCells;
    this.index = new Map();
    this.capacity = 0;
    this.size = 0;
    this.grow(Math.min(maxCells, 16384));
  }

  grow(capacity) {
    const copy = (Type, old, stride) => { const next = new Type(capacity * stride); if (old) next.set(old); return next; };
    this.coords = copy(Int32Array, this.coords, 3);
    this.values = copy(Float32Array, this.values, 1);
    this.weights = copy(Float32Array, this.weights, 1);
    this.colors = copy(Float32Array, this.colors, 3);
    this.capacity = capacity;
  }

  cell(x, y, z, create) {
    if (!inside(x, y, z)) return -1;
    const id = key(x, y, z);
    let index = this.index.get(id);
    if (index !== undefined || !create) return index ?? -1;
    if (this.size >= this.maxCells) return -1;
    if (this.size >= this.capacity) this.grow(Math.min(this.maxCells, this.capacity * 2));
    index = this.size++;
    this.coords.set([x, y, z], index * 3);
    this.index.set(id, index);
    return index;
  }

  integrate(depthMap, reference, focal) {
    const { width, height, pose } = reference;
    const pixels = depthMap.colors || reference.rgba;
    const channels = depthMap.colors ? 3 : 4;
    const inverse = transpose3(pose.R);
    const { depths, confidence, step } = depthMap;
    let integrated = 0;
    for (let oy = 0; oy < depthMap.height; oy++) for (let ox = 0; ox < depthMap.width; ox++) {
      const index = oy * depthMap.width + ox;
      const depth = depths[index], weight = confidence[index];
      if (!depth || weight < 0.15) continue;
      const x = ox * step, y = oy * step;
      const ray = [(x - width / 2) / focal, (height / 2 - y) / focal, 1];
      const length = Math.hypot(...ray);
      const direction = mul3(inverse, ray.map((value) => value / length));
      const distance = depth * length;
      const pixel = (y * width + x) * channels;
      for (let offset = -this.truncation; offset <= this.truncation + 1e-6; offset += this.voxelSize * 0.75) {
        const cell = this.cell(...pose.C.map((value, i) => Math.floor((value + direction[i] * (distance + offset)) / this.voxelSize)), true);
        if (cell < 0) continue;
        const previous = this.weights[cell];
        const portion = weight / (previous + weight);
        this.values[cell] += (Math.max(-1, Math.min(1, -offset / this.truncation)) - this.values[cell]) * portion;
        for (let channel = 0; channel < 3; channel++) {
          this.colors[cell * 3 + channel] += (pixels[pixel + channel] / 255 - this.colors[cell * 3 + channel]) * portion;
        }
        this.weights[cell] = Math.min(40, previous + weight);
      }
      integrated++;
    }
    return integrated;
  }

  extract(minWeight = 0.3) {
    const positions = [], colors = [], indices = [];
    const vertices = new Map();
    const cube = new Int32Array(8);
    const vertex = (ca, cb) => {
      const edge = ca < cb ? `${ca}:${cb}` : `${cb}:${ca}`;
      let id = vertices.get(edge);
      if (id !== undefined) return id;
      const va = this.values[ca], vb = this.values[cb];
      const blend = va / (va - vb);
      id = positions.length / 3;
      for (let i = 0; i < 3; i++) {
        const a = this.coords[ca * 3 + i], b = this.coords[cb * 3 + i];
        positions.push((a + 0.5 + blend * (b - a)) * this.voxelSize);
        colors.push(this.colors[ca * 3 + i] + blend * (this.colors[cb * 3 + i] - this.colors[ca * 3 + i]));
      }
      vertices.set(edge, id);
      return id;
    };
    for (let cell = 0; cell < this.size; cell++) {
      const x = this.coords[cell * 3], y = this.coords[cell * 3 + 1], z = this.coords[cell * 3 + 2];
      let valid = true, positive = 0;
      for (let c = 0; c < 8 && valid; c++) {
        const index = this.cell(x + corners[c][0], y + corners[c][1], z + corners[c][2], false);
        if (index < 0 || this.weights[index] < minWeight) valid = false;
        else { cube[c] = index; if (this.values[index] >= 0) positive++; }
      }
      if (!valid || positive === 0 || positive === 8) continue;
      for (const tetra of tetrahedra) {
        const points = [];
        for (const [a, b] of edges) {
          const ca = cube[tetra[a]], cb = cube[tetra[b]];
          if ((this.values[ca] < 0) === (this.values[cb] < 0)) continue;
          points.push(vertex(ca, cb));
        }
        if (points.length === 3) indices.push(points[0], points[1], points[2]);
        // Crossing edges arrive as 01-02-13-23 style orders; the quad cycle is p0, p1, p3, p2.
        if (points.length === 4) indices.push(points[0], points[1], points[3], points[0], points[3], points[2]);
      }
    }
    return { positions: Float32Array.from(positions), colors: Float32Array.from(colors), indices: Uint32Array.from(indices) };
  }

  mesh(limit = 18000) {
    return expand(cleanMesh(this.extract(), { limit, voxel: this.voxelSize }));
  }
}
