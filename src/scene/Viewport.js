import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';

export class Viewport {
  constructor(container) {
    this.container = container;
    this.renderer = new THREE.WebGLRenderer({ antialias: true });
    this.renderer.setPixelRatio(Math.min(devicePixelRatio, 1.5));
    this.renderer.setClearColor(0x0b1018);
    this.renderer.outputColorSpace = THREE.SRGBColorSpace;
    this.renderer.autoClear = false;
    container.append(this.renderer.domElement);

    this.scene = new THREE.Scene();
    this.camera = new THREE.PerspectiveCamera(60, 1, 0.01, 100);
    this.camera.position.set(3, 2, 4);
    this.camera.lookAt(0, 0, 0);
    this.controls = new OrbitControls(this.camera, this.renderer.domElement);
    this.controls.enableDamping = true;
    this.controls.target.set(0, 0, -1);
    this.controls.update();
    this.grid = new THREE.GridHelper(10, 20, 0x3d7e86, 0x22343e);
    this.scene.add(this.grid);
    this.meshes = new THREE.Group();
    this.scene.add(this.meshes);
    this.material = new THREE.MeshBasicMaterial({ vertexColors: true, side: THREE.DoubleSide });
    this.selected = new Map();
    this.seen = new Set();
    this.triangles = 0;
    this.limit = 120000;
    this.eraseMode = false;
    this.cameraActive = false;
    this.raycaster = new THREE.Raycaster();
    this.pointer = new THREE.Vector2();
    this.renderer.domElement.addEventListener('pointerdown', (event) => {
      if (!this.eraseMode) return;
      this.drawing = true;
      this.renderer.domElement.setPointerCapture(event.pointerId);
      this.select(event);
    });
    this.renderer.domElement.addEventListener('pointermove', (event) => { if (this.drawing) this.select(event); });
    for (const type of ['pointerup', 'pointercancel']) this.renderer.domElement.addEventListener(type, () => { this.drawing = false; });

    this.background = new THREE.Scene();
    this.backgroundCamera = new THREE.OrthographicCamera(-1, 1, 1, -1, 0, 2);
    this.backgroundCamera.position.z = 1;
    this.backgroundPlane = new THREE.Mesh(new THREE.PlaneGeometry(2, 2), new THREE.MeshBasicMaterial());
    this.background.add(this.backgroundPlane);
    this.backgroundPlane.visible = false;
    this.trackingPoints = new THREE.Points(
      new THREE.BufferGeometry(),
      new THREE.PointsMaterial({ color: 0x9df7dd, size: 4, sizeAttenuation: false, depthTest: false }),
    );
    this.trackingPoints.visible = false;
    this.trackingPoints.renderOrder = 1;
    this.background.add(this.trackingPoints);

    this.resize = this.resize.bind(this);
    window.addEventListener('resize', this.resize);
    this.resize();
    this.lastFrame = 0;
    this.renderer.setAnimationLoop((now) => {
      if (now - this.lastFrame < 32) return;
      this.lastFrame = now;
      if (this.controls.enabled) this.controls.update();
      this.renderer.clear();
      if (this.backgroundPlane.visible) this.renderer.render(this.background, this.backgroundCamera);
      this.renderer.clearDepth();
      this.renderer.render(this.scene, this.camera);
    });
  }

  showCamera(video) {
    this.video = video;
    this.trackData = null;
    this.trackingPoints.visible = false;
    this.focalVideo = null;
    this.cameraActive = true;
    this.controls.enabled = false;
    this.texture = new THREE.VideoTexture(video);
    this.texture.colorSpace = THREE.SRGBColorSpace;
    this.backgroundPlane.material.map = this.texture;
    this.backgroundPlane.material.needsUpdate = true;
    this.backgroundPlane.visible = true;
    this.grid.visible = false;
    this.resize();
  }

  hideCamera() {
    this.cameraActive = false;
    this.trackData = null;
    this.backgroundPlane.visible = false;
    this.trackingPoints.visible = false;
    this.backgroundPlane.material.map = null;
    this.backgroundPlane.material.needsUpdate = true;
    this.texture?.dispose();
    this.texture = null;
    this.video = null;
    this.grid.visible = true;
    this.controls.target.copy(this.camera.position).add(new THREE.Vector3(0, 0, -2).applyQuaternion(this.camera.quaternion));
    this.controls.enabled = !this.eraseMode;
    this.controls.update();
  }

  setPose(pose, focal, frameWidth) {
    this.focalVideo = focal * this.video.videoWidth / frameWidth;
    const R = pose.R;
    const matrix = new THREE.Matrix4().set(
      R[0], R[3], -R[6], 0,
      R[1], R[4], -R[7], 0,
      -R[2], -R[5], R[8], 0,
      0, 0, 0, 1,
    );
    this.camera.quaternion.setFromRotationMatrix(matrix);
    this.camera.position.set(pose.C[0], pose.C[1], -pose.C[2]);
    this.resize();
  }

  setTracks(tracks, width, height, tracking) {
    if (!this.cameraActive) return;
    this.trackData = { tracks, width, height };
    this.trackingPoints.material.color.setHex(tracking === 'anchored' ? 0x9df7dd : tracking === 'relative' ? 0x8dd5ff : 0xffd68a);
    this.updateTracks();
  }

  updateTracks() {
    if (!this.trackData || !this.cameraActive) return;
    const { tracks, width, height } = this.trackData;
    const sx = this.backgroundPlane.scale.x, sy = this.backgroundPlane.scale.y;
    const stride = Math.max(2, Math.ceil(tracks.length / 300) * 2);
    const positions = [];
    for (let i = 0; i < tracks.length; i += stride) {
      positions.push((tracks[i] / width * 2 - 1) * sx, (1 - tracks[i + 1] / height * 2) * sy, 0.05);
    }
    this.trackingPoints.geometry.dispose();
    this.trackingPoints.geometry = new THREE.BufferGeometry();
    this.trackingPoints.geometry.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3));
    this.trackingPoints.visible = positions.length > 0;
  }

  addPatch(positions, colors, deduplicate = true) {
    const keptPositions = [], keptColors = [], keptKeys = [];
    for (let face = 0; face < positions.length / 9 && this.triangles + keptPositions.length / 9 < this.limit; face++) {
      const offset = face * 9;
      const x = (positions[offset] + positions[offset + 3] + positions[offset + 6]) / 3;
      const y = (positions[offset + 1] + positions[offset + 4] + positions[offset + 7]) / 3;
      const z = (positions[offset + 2] + positions[offset + 5] + positions[offset + 8]) / 3;
      const key = deduplicate ? [x, y, z].map((value) => Math.round(value / 0.06)).join(':') : `fused:${face}`;
      if (deduplicate && this.seen.has(key)) continue;
      if (deduplicate) this.seen.add(key);
      keptKeys.push(key);
      for (let i = 0; i < 9; i++) { keptPositions.push(positions[offset + i]); keptColors.push(colors[offset + i]); }
    }
    if (!keptPositions.length) return 0;
    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute('position', new THREE.Float32BufferAttribute(keptPositions, 3));
    geometry.setAttribute('color', new THREE.Float32BufferAttribute(keptColors, 3));
    const mesh = new THREE.Mesh(geometry, this.material);
    mesh.userData.colors = Float32Array.from(keptColors);
    mesh.userData.keys = keptKeys;
    this.meshes.add(mesh);
    const added = keptPositions.length / 9;
    this.triangles += added;
    return added;
  }

  setEraseMode(active) {
    this.eraseMode = active;
    this.controls.enabled = !active && !this.cameraActive;
    this.renderer.domElement.style.touchAction = active ? 'none' : 'auto';
  }

  select(event) {
    if (!this.triangles) return;
    const bounds = this.renderer.domElement.getBoundingClientRect();
    for (const [dx, dy] of [[0, 0], [-9, 0], [9, 0], [0, -9], [0, 9]]) {
      this.pointer.set(((event.clientX + dx - bounds.left) / bounds.width) * 2 - 1, -((event.clientY + dy - bounds.top) / bounds.height) * 2 + 1);
      this.raycaster.setFromCamera(this.pointer, this.camera);
      const hit = this.raycaster.intersectObjects(this.meshes.children, false)[0];
      if (!hit) continue;
      const selected = this.selected.get(hit.object) || new Set();
      if (selected.has(hit.faceIndex)) continue;
      selected.add(hit.faceIndex);
      this.selected.set(hit.object, selected);
      const attribute = hit.object.geometry.getAttribute('color');
      for (let vertex = 0; vertex < 3; vertex++) attribute.setXYZ(hit.faceIndex * 3 + vertex, 1, 0.22, 0.16);
      attribute.needsUpdate = true;
    }
    this.onSelectionChange?.(this.selectionCount);
  }

  get selectionCount() { return [...this.selected.values()].reduce((sum, faces) => sum + faces.size, 0); }

  clearSelection() {
    for (const mesh of this.selected.keys()) {
      const attribute = mesh.geometry.getAttribute('color');
      attribute.array.set(mesh.userData.colors);
      attribute.needsUpdate = true;
    }
    this.selected.clear();
    this.onSelectionChange?.(0);
  }

  deleteSelection() {
    for (const [mesh, selected] of this.selected) {
      const geometry = mesh.geometry;
      const position = geometry.getAttribute('position').array;
      const colors = mesh.userData.colors;
      const keptPositions = [], keptColors = [], keptKeys = [];
      for (let face = 0; face < position.length / 9; face++) {
        if (selected.has(face)) { this.seen.delete(mesh.userData.keys[face]); continue; }
        keptKeys.push(mesh.userData.keys[face]);
        for (let i = face * 9; i < face * 9 + 9; i++) { keptPositions.push(position[i]); keptColors.push(colors[i]); }
      }
      this.triangles -= selected.size;
      geometry.dispose();
      if (!keptPositions.length) this.meshes.remove(mesh);
      else {
        const next = new THREE.BufferGeometry();
        next.setAttribute('position', new THREE.Float32BufferAttribute(keptPositions, 3));
        next.setAttribute('color', new THREE.Float32BufferAttribute(keptColors, 3));
        mesh.geometry = next;
        mesh.userData.colors = Float32Array.from(keptColors);
        mesh.userData.keys = keptKeys;
      }
    }
    this.selected.clear();
    this.onSelectionChange?.(0);
    return this.triangles;
  }

  clearMesh() {
    this.clearSelection();
    for (const mesh of [...this.meshes.children]) { mesh.geometry.dispose(); this.meshes.remove(mesh); }
    this.triangles = 0;
    this.seen.clear();
  }

  replaceMesh(positions, colors) {
    this.clearMesh();
    const faces = Math.min(Math.floor(positions.length / 9), this.limit);
    if (!faces) return;
    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute('position', new THREE.BufferAttribute(positions.subarray(0, faces * 9), 3));
    geometry.setAttribute('color', new THREE.BufferAttribute(colors.subarray(0, faces * 9), 3));
    const mesh = new THREE.Mesh(geometry, this.material);
    mesh.userData.colors = Float32Array.from(colors.subarray(0, faces * 9));
    mesh.userData.keys = Array.from({ length: faces }, (_, index) => index);
    this.meshes.add(mesh);
    this.triangles = faces;
    const bounds = new THREE.Box3().setFromObject(this.meshes);
    if (bounds.isEmpty()) return;
    const center = bounds.getCenter(new THREE.Vector3());
    const radius = Math.max(bounds.getSize(new THREE.Vector3()).length() / 2, 0.2);
    const distance = radius / (Math.tan(THREE.MathUtils.degToRad(this.camera.fov / 2)) * Math.min(1, this.camera.aspect)) * 1.3;
    this.camera.position.copy(center).add(new THREE.Vector3(distance * 0.5, distance * 0.35, distance));
    this.camera.lookAt(center);
    this.controls.target.copy(center);
    this.controls.update();
  }

  async exportMesh(format) {
    this.clearSelection();
    if (format === 'obj') {
      const { OBJExporter } = await import('three/addons/exporters/OBJExporter.js');
      return new Blob([new OBJExporter().parse(this.meshes)], { type: 'text/plain' });
    }
    const { GLTFExporter } = await import('three/addons/exporters/GLTFExporter.js');
    const result = await new Promise((resolve, reject) => new GLTFExporter().parse(this.meshes, resolve, reject, { binary: true }));
    return new Blob([result], { type: 'model/gltf-binary' });
  }

  resize() {
    const width = this.container.clientWidth;
    const height = this.container.clientHeight;
    if (!width || !height) return;
    this.renderer.setSize(width, height, false);
    this.camera.aspect = width / height;
    if (this.video?.videoWidth) {
      const cover = Math.max(width / this.video.videoWidth, height / this.video.videoHeight);
      const focal = (this.focalVideo || Math.max(this.video.videoWidth, this.video.videoHeight) * 0.9) * cover;
      this.camera.fov = 2 * Math.atan(height / (2 * focal)) * 180 / Math.PI;
    }
    this.camera.updateProjectionMatrix();
    if (this.video?.videoWidth) {
      const ratio = (this.video.videoWidth / this.video.videoHeight) / (width / height);
      this.backgroundPlane.scale.set(Math.max(1, ratio), Math.max(1, 1 / ratio), 1);
      this.updateTracks();
    }
  }
}
