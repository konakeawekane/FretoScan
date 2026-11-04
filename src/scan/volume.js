import { mul3, transpose3 } from './reconstruction.js';

const corners = [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0], [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]];
const tetrahedra = [[0, 5, 1, 6], [0, 1, 2, 6], [0, 2, 3, 6], [0, 3, 7, 6], [0, 7, 4, 6], [0, 4, 5, 6]];
const edges = [[0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3]];
const key = (x, y, z) => `${x}:${y}:${z}`;

export class TsdfVolume {
  constructor(voxelSize = 0.035, truncation = 0.105, maxCells = 160000) {
    this.voxelSize = voxelSize;
    this.truncation = truncation;
    this.maxCells = maxCells;
    this.cells = new Map();
  }

  integrate(depthMap, reference, focal) {
    const { width, height, rgba, pose } = reference;
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
      const pixel = (y * width + x) * 4;
      for (let offset = -this.truncation; offset <= this.truncation + 1e-6; offset += this.voxelSize) {
        const world = pose.C.map((value, i) => value + direction[i] * (distance + offset));
        const coords = world.map((value) => Math.floor(value / this.voxelSize));
        const id = key(...coords);
        let cell = this.cells.get(id);
        if (!cell) {
          if (this.cells.size >= this.maxCells) continue;
          cell = { coords, value: 0, weight: 0, color: [0, 0, 0] };
          this.cells.set(id, cell);
        }
        const nextWeight = Math.min(20, cell.weight + weight);
        const portion = weight / (cell.weight + weight);
        cell.value += (-offset / this.truncation - cell.value) * portion;
        for (let channel = 0; channel < 3; channel++) cell.color[channel] += (rgba[pixel + channel] / 255 - cell.color[channel]) * portion;
        cell.weight = nextWeight;
      }
      integrated++;
    }
    return integrated;
  }

  mesh(limit = 18000) {
    const positions = [], colors = [];
    const emit = (vertices) => {
      for (const vertex of vertices) {
        positions.push(vertex.position[0], vertex.position[1], -vertex.position[2]);
        colors.push(...vertex.color);
      }
    };
    for (const cell of this.cells.values()) {
      if (positions.length / 9 >= limit) break;
      const [x, y, z] = cell.coords;
      const cube = corners.map(([dx, dy, dz]) => this.cells.get(key(x + dx, y + dy, z + dz)));
      if (cube.some((corner) => !corner || corner.weight < 0.3)) continue;
      if (cube.every((corner) => corner.value >= 0) || cube.every((corner) => corner.value <= 0)) continue;
      for (const tetra of tetrahedra) {
        const points = [];
        for (const [a, b] of edges) {
          const ia = tetra[a], ib = tetra[b], ca = cube[ia], cb = cube[ib];
          if ((ca.value < 0) === (cb.value < 0)) continue;
          const blend = ca.value / (ca.value - cb.value);
          const position = corners[ia].map((offset, i) => (cell.coords[i] + offset + blend * (corners[ib][i] - offset)) * this.voxelSize);
          const color = ca.color.map((value, i) => value + blend * (cb.color[i] - value));
          points.push({ position, color });
        }
        if (points.length === 3) emit(points);
        if (points.length === 4) { emit([points[0], points[1], points[2]]); emit([points[0], points[2], points[3]]); }
      }
    }
    return { positions: Float32Array.from(positions), colors: Float32Array.from(colors) };
  }
}
