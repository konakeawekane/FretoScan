import * as THREE from 'three';

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
    this.grid = new THREE.GridHelper(10, 20, 0x3d7e86, 0x22343e);
    this.scene.add(this.grid);

    this.background = new THREE.Scene();
    this.backgroundCamera = new THREE.OrthographicCamera(-1, 1, 1, -1, 0, 2);
    this.backgroundCamera.position.z = 1;
    this.backgroundPlane = new THREE.Mesh(new THREE.PlaneGeometry(2, 2), new THREE.MeshBasicMaterial());
    this.background.add(this.backgroundPlane);
    this.backgroundPlane.visible = false;

    this.resize = this.resize.bind(this);
    window.addEventListener('resize', this.resize);
    this.resize();
    this.lastFrame = 0;
    this.renderer.setAnimationLoop((now) => {
      if (now - this.lastFrame < 32) return;
      this.lastFrame = now;
      this.renderer.clear();
      if (this.backgroundPlane.visible) this.renderer.render(this.background, this.backgroundCamera);
      this.renderer.clearDepth();
      this.renderer.render(this.scene, this.camera);
    });
  }

  showCamera(video) {
    this.video = video;
    this.texture = new THREE.VideoTexture(video);
    this.texture.colorSpace = THREE.SRGBColorSpace;
    this.backgroundPlane.material.map = this.texture;
    this.backgroundPlane.material.needsUpdate = true;
    this.backgroundPlane.visible = true;
    this.grid.visible = false;
    this.resize();
  }

  hideCamera() {
    this.backgroundPlane.visible = false;
    this.backgroundPlane.material.map = null;
    this.backgroundPlane.material.needsUpdate = true;
    this.texture?.dispose();
    this.texture = null;
    this.video = null;
    this.grid.visible = true;
  }

  resize() {
    const width = this.container.clientWidth;
    const height = this.container.clientHeight;
    if (!width || !height) return;
    this.renderer.setSize(width, height, false);
    this.camera.aspect = width / height;
    this.camera.updateProjectionMatrix();
    if (this.video?.videoWidth) {
      const ratio = (this.video.videoWidth / this.video.videoHeight) / (width / height);
      this.backgroundPlane.scale.set(Math.max(1, ratio), Math.max(1, 1 / ratio), 1);
    }
  }
}
