// scene.js — building three.js objects (points, meshes, floor clip, camera).

import * as THREE from 'three';

export function createPoints(positions, colors, size) {
  const geo = new THREE.BufferGeometry();
  geo.setAttribute('position', new THREE.BufferAttribute(positions, 3));
  geo.setAttribute('color', new THREE.BufferAttribute(colors, 3));
  const mat = new THREE.PointsMaterial({
    vertexColors: true,
    size,
    sizeAttenuation: true,
  });
  return new THREE.Points(geo, mat);
}

export function createMeshFromObj(parsed, opts = {}) {
  const { wireframe = false, color = null, transparent = false, opacity = 1.0 } = opts;
  const geo = new THREE.BufferGeometry();
  geo.setAttribute('position', new THREE.BufferAttribute(parsed.positions, 3));

  if (color) {
    const n = parsed.positions.length / 3;
    const c = new Float32Array(n * 3);
    for (let i = 0; i < n; i++) {
      c[i * 3] = color[0] / 255; c[i * 3 + 1] = color[1] / 255; c[i * 3 + 2] = color[2] / 255;
    }
    geo.setAttribute('color', new THREE.BufferAttribute(c, 3));
  } else {
    geo.setAttribute('color', new THREE.BufferAttribute(parsed.colors, 3));
  }

  const mat = new THREE.MeshBasicMaterial({
    vertexColors: !color,
    wireframe,
    side: THREE.DoubleSide,
    transparent,
    opacity,
  });
  return new THREE.Mesh(geo, mat);
}

export function fitCamera(camera, controls, group) {
  const box = new THREE.Box3().setFromObject(group);
  if (box.isEmpty()) return;
  const center = box.getCenter(new THREE.Vector3());
  const size = box.getSize(new THREE.Vector3());
  const radius = Math.max(size.x, size.y, size.z) * 0.6;
  const dist = Math.max(radius * 2.6, 8);
  const dir = new THREE.Vector3(1, 0.65, 1).normalize();
  camera.position.copy(center).addScaledVector(dir, dist);
  camera.near = Math.max(dist / 1000, 0.01);
  camera.far = dist * 100;
  camera.updateProjectionMatrix();
  controls.target.copy(center);
  controls.update();
}

export function buildFloorPlane(width, z) {
  const geo = new THREE.PlaneGeometry(width, width);
  const mat = new THREE.MeshBasicMaterial({
    color: 0x66ccff,
    transparent: true,
    opacity: 0.18,
    side: THREE.DoubleSide,
    depthWrite: false,
  });
  const mesh = new THREE.Mesh(geo, mat);
  mesh.position.z = z;
  return mesh;
}

export function applyClip(group, zCut) {
  const plane = new THREE.Plane(new THREE.Vector3(0, 0, 1), -zCut);
  group.traverse((o) => {
    if (o.isMesh || o.isPoints) {
      const mats = Array.isArray(o.material) ? o.material : [o.material];
      mats.forEach((m) => { m.clippingPlanes = [plane]; });
    }
  });
}

export function clearClip(group) {
  group.traverse((o) => {
    if (o.isMesh || o.isPoints) {
      const mats = Array.isArray(o.material) ? o.material : [o.material];
      mats.forEach((m) => { m.clippingPlanes = null; });
    }
  });
}