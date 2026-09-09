// app.js — main orchestrator for the SIH 26011 synthetic building test viewer.

import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import {
  fetchJSON, fetchBuffer, parsePLY, parseOBJ,
  classColor, SURFACE_COLORS, LOD1_COLOR,
} from './loader.js';
import {
  createPoints, createMeshFromObj, fitCamera, buildFloorPlane, applyClip, clearClip,
} from './scene.js';
import { renderBuildingInfo, renderMLInfo, renderFloorInfo, renderULPINHero } from './metadata.js';
import { populateSelect, setSelectIndex, buildFloorButtons, setActiveFloor, setStatus } from './controls.js';

const $ = (id) => document.getElementById(id);

// ------------------------------------------------------------------ state --

const state = {
  index: null,
  sceneIdx: 0,
  buildingIdx: 0,
  showLod1: true,
  showLod2: true,
  showLidar: false,
  showPred: false,
  overlay: false,
  context: false,
  pointSize: 2,
  floorIdx: 0,
  cacheLidar: {},
  cachePred: {},
  cacheObj: {},
  pointObjects: [],
  webglOk: true,
};

// ----------------------------------------------------------------- three.js --

const container = $('viewport');
const scene = new THREE.Scene();
scene.background = new THREE.Color(0x0a0e1a);

const camera = new THREE.PerspectiveCamera(50, 1, 0.01, 5000);
camera.position.set(60, 50, 70);

let renderer = null;
try {
  renderer = new THREE.WebGLRenderer({ antialias: true });
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
  renderer.localClippingEnabled = true;
  container.appendChild(renderer.domElement);
} catch (e) {
  state.webglOk = false;
  console.error('WebGL unavailable:', e.message);
  setStatus('WebGL unavailable — data loading still works, 3D view disabled.');
}

let controls = null;
if (renderer) {
  controls = new OrbitControls(camera, renderer.domElement);
  controls.enableDamping = true;
}

const bldg = new THREE.Group();
scene.add(bldg);
const grid = new THREE.GridHelper(160, 32, 0x334155, 0x1f2937);
scene.add(grid);

function resize() {
  const w = container.clientWidth || 800;
  const h = container.clientHeight || 600;
  camera.aspect = w / h;
  camera.updateProjectionMatrix();
  if (renderer) renderer.setSize(w, h);
}
window.addEventListener('resize', resize);

// ------------------------------------------------------------- data helpers --

const currentScene = () => state.index.scenes[state.sceneIdx];
const currentBuilding = () => currentScene().buildings[state.buildingIdx];

async function loadLidar(sceneInfo) {
  if (!state.cacheLidar[sceneInfo.id]) {
    const buf = await fetchBuffer('/' + sceneInfo.lidar_path);
    state.cacheLidar[sceneInfo.id] = parsePLY(buf);
  }
  return state.cacheLidar[sceneInfo.id];
}

async function loadPrediction(sceneInfo) {
  if (!sceneInfo.prediction) return null;
  if (!state.cachePred[sceneInfo.id]) {
    const buf = await fetchBuffer('/' + sceneInfo.prediction.path);
    state.cachePred[sceneInfo.id] = parsePLY(buf);
  }
  return state.cachePred[sceneInfo.id];
}

async function loadObj(key, path, colorMap) {
  if (!state.cacheObj[key]) {
    const text = await (await fetch('/' + path)).text();
    state.cacheObj[key] = parseOBJ(text, colorMap);
  }
  return state.cacheObj[key];
}

function indicesForBuilding(parsed, buildingId) {
  const bid = parsed.data.building_id;
  const out = [];
  for (let i = 0; i < bid.length; i++) if (bid[i] === buildingId) out.push(i);
  return out;
}

function allIndices(n) {
  const out = new Array(n);
  for (let i = 0; i < n; i++) out[i] = i;
  return out;
}

function collectPositions(parsed, indices) {
  const n = indices.length;
  const pos = new Float32Array(n * 3);
  for (let i = 0; i < n; i++) {
    const k = indices[i];
    pos[i * 3] = parsed.data.x[k];
    pos[i * 3 + 1] = parsed.data.y[k];
    pos[i * 3 + 2] = parsed.data.z[k];
  }
  return pos;
}

function colorsFrom(parsed, indices, classKey) {
  const n = indices.length;
  const c = new Float32Array(n * 3);
  const vals = parsed.data[classKey];
  for (let i = 0; i < n; i++) {
    const rgb = classColor(vals[indices[i]]);
    c[i * 3] = rgb[0] / 255;
    c[i * 3 + 1] = rgb[1] / 255;
    c[i * 3 + 2] = rgb[2] / 255;
  }
  return c;
}

// ------------------------------------------------------------- rendering --

let renderSeq = 0;

function clearBuilding() {
  while (bldg.children.length) bldg.remove(bldg.children[0]);
  state.pointObjects = [];
}

function visibleCategories() {
  const cats = [];
  if (state.showLod1) cats.push('lod1');
  if (state.showLod2) cats.push('lod2');
  if (state.showLidar) cats.push('lidar');
  if (state.showPred) cats.push('pred');
  return state.overlay ? cats : (cats.length ? [cats[0]] : []);
}

async function render() {
  const seq = ++renderSeq;
  clearBuilding();
  const sceneInfo = currentScene();
  const b = currentBuilding();
  const meta = b.metadata;
  const visible = visibleCategories();

  let lod1 = null;
  let lod2 = null;
  if (visible.includes('lod1') || visible.includes('lod2')) {
    [lod1, lod2] = await Promise.all([
      loadObj(`${sceneInfo.id}/${b.id}/lod1`, b.lod1_path, {}),
      loadObj(`${sceneInfo.id}/${b.id}/lod2`, b.lod2_path, SURFACE_COLORS),
    ]);
  }
  if (seq !== renderSeq) return;

  if (visible.includes('lod1') && lod1) {
    bldg.add(createMeshFromObj(lod1, { wireframe: true, color: LOD1_COLOR }));
  }
  if (visible.includes('lod2') && lod2) {
    bldg.add(createMeshFromObj(lod2, {}));
  }

  if (visible.includes('lidar') || visible.includes('pred')) {
    const lidar = await loadLidar(sceneInfo);
    const pred = state.showPred ? await loadPrediction(sceneInfo) : null;
    if (seq !== renderSeq) return;

    const indices = state.context ? allIndices(lidar.count) : indicesForBuilding(lidar, b.id);
    if (visible.includes('lidar')) {
      const pos = collectPositions(lidar, indices);
      const col = colorsFrom(lidar, indices, 'class_id');
      const pts = createPoints(pos, col, state.pointSize * 0.08);
      bldg.add(pts);
      state.pointObjects.push(pts);
    }
    if (visible.includes('pred') && pred) {
      const pos = collectPositions(pred, indices);
      const col = colorsFrom(pred, indices, 'predicted_class');
      const pts = createPoints(pos, col, state.pointSize * 0.08);
      bldg.add(pts);
      state.pointObjects.push(pts);
    }
  }
  if (seq !== renderSeq) return;

  if (state.floorIdx > 0 && meta && meta.floor_levels) {
    const z = meta.floor_levels[state.floorIdx - 1];
    const box = new THREE.Box3().setFromObject(bldg);
    const size = box.isEmpty() ? new THREE.Vector3(30, 30, 1) : box.getSize(new THREE.Vector3());
    const w = Math.max(size.x, size.y, 12) + 4;
    bldg.add(buildFloorPlane(w, z));
    applyClip(bldg, meta.floor_levels[state.floorIdx]);
  } else {
    clearClip(bldg);
  }

  if (state.webglOk) {
    fitCamera(camera, controls, bldg);
    renderer.render(scene, camera);
    setStatus(`scene ${sceneInfo.id} (${sceneInfo.split}) · building ${b.id} · ${meta.footprint_type} · ${meta.roof_type} · ${meta.floor_count} floors · 3D rendered`);
  } else {
    setStatus(`scene ${sceneInfo.id} (${sceneInfo.split}) · building ${b.id} · data loaded (3D view disabled)`);
  }
}

// --------------------------------------------------------------- selection --

async function onSceneChange(sceneIdx) {
  state.sceneIdx = sceneIdx;
  const info = currentScene();
  populateSelect($('building-select'), info.buildings,
    (b) => `B${b.id} · ${b.metadata.footprint_type} · ${b.metadata.roof_type}`);
  state.buildingIdx = 0;
  _updateCityGMLPanel(info);
  await onBuildingChange(0);
}

async function onBuildingChange(buildingIdx) {
  state.buildingIdx = buildingIdx;
  state.floorIdx = 0;
  const sceneInfo = currentScene();
  const b = currentBuilding();
  const meta = b.metadata;
  buildFloorButtons(meta, (i) => { state.floorIdx = i; setActiveFloor(meta, i); renderFloorInfo(meta, i); render(); }, state.floorIdx);
  renderULPINHero(sceneInfo.id, meta);
  renderBuildingInfo(meta, sceneInfo.id);
  const pred = sceneInfo.prediction;
  renderMLInfo(pred ? pred.metrics_by_building[String(b.id)] : null, !!pred);
  _updateCityGMLPanel(sceneInfo);
  await render();
}

function _updateCityGMLPanel(sceneInfo) {
  const btn = document.getElementById('btn-download-gml');
  const status = document.getElementById('citygml-status');
  if (!btn || !status) return;
  if (sceneInfo.citygml_path) {
    status.textContent = `CityGML ready: ${sceneInfo.citygml_path}`;
    status.style.color = 'var(--plateau)';
    btn.disabled = false;
    btn.onclick = () => {
      const a = document.createElement('a');
      a.href = '/' + sceneInfo.citygml_path;
      a.download = 'city_model.gml';
      a.click();
    };
  } else {
    status.textContent = 'CityGML not yet generated — run pipeline_plateau.py first.';
    status.style.color = '';
    btn.disabled = true;
    btn.onclick = null;
  }
}

// ------------------------------------------------------------------- init --

async function init() {
  setStatus('Loading index…');
  const idx = await fetchJSON('data/index.json');
  state.index = idx;

  populateSelect($('scene-select'), idx.scenes, (s) => `${s.id} (${s.split})`);

  const params = new URLSearchParams(location.search);
  const wantScene = params.get('scene');
  const wantBuilding = params.get('building');

  // optional query-param presets: lidar=1 pred=1 overlay=1 context=1 lod1=0 lod2=0 floor=3
  const applyToggle = (name, setter) => {
    const v = params.get(name);
    if (v !== null) {
      const on = v !== '0' && v !== 'false';
      setter(on);
      const cb = $('toggle-' + name);
      if (cb) cb.checked = on;
    }
  };
  applyToggle('lidar', (v) => { state.showLidar = v; });
  applyToggle('pred', (v) => { state.showPred = v; });
  applyToggle('overlay', (v) => { state.overlay = v; });
  applyToggle('context', (v) => { state.context = v; });
  applyToggle('lod1', (v) => { state.showLod1 = v; });
  applyToggle('lod2', (v) => { state.showLod2 = v; });

  let si = 0;
  if (wantScene) {
    const hit = idx.scenes.findIndex((s) => s.id === wantScene);
    if (hit >= 0) si = hit;
  }
  setSelectIndex($('scene-select'), si);
  await onSceneChange(si);

  if (wantBuilding) {
    const bi = currentScene().buildings.findIndex((b) => String(b.id) === wantBuilding);
    if (bi >= 0) { setSelectIndex($('building-select'), bi); await onBuildingChange(bi); }
  }

  const floorParam = params.get('floor');
  if (floorParam && metaFloorCount() >= 1) {
    const f = parseInt(floorParam, 10);
    const fl = currentBuilding().metadata.floor_count;
    if (f >= 1 && f <= fl) { state.floorIdx = f; setActiveFloor(currentBuilding().metadata, f); renderFloorInfo(currentBuilding().metadata, f); await render(); }
  }

  resize();
  if (state.webglOk) renderer.render(scene, camera);
}

function metaFloorCount() {
  try { return currentBuilding().metadata.floor_count || 0; } catch (e) { return 0; }
}

// ------------------------------------------------------------ event wiring --

$('scene-select').addEventListener('change', (e) => onSceneChange(parseInt(e.target.value, 10)));
$('building-select').addEventListener('change', (e) => onBuildingChange(parseInt(e.target.value, 10)));

$('prev-building').addEventListener('click', async () => {
  const n = currentScene().buildings.length;
  const next = (state.buildingIdx - 1 + n) % n;
  setSelectIndex($('building-select'), next);
  await onBuildingChange(next);
});
$('next-building').addEventListener('click', async () => {
  const n = currentScene().buildings.length;
  const next = (state.buildingIdx + 1) % n;
  setSelectIndex($('building-select'), next);
  await onBuildingChange(next);
});

$('toggle-lod1').addEventListener('change', (e) => { state.showLod1 = e.target.checked; render(); });
$('toggle-lod2').addEventListener('change', (e) => { state.showLod2 = e.target.checked; render(); });
$('toggle-lidar').addEventListener('change', (e) => { state.showLidar = e.target.checked; render(); });
$('toggle-pred').addEventListener('change', (e) => { state.showPred = e.target.checked; render(); });
$('toggle-overlay').addEventListener('change', (e) => { state.overlay = e.target.checked; render(); });
$('toggle-context').addEventListener('change', (e) => { state.context = e.target.checked; render(); });

$('point-size').addEventListener('input', (e) => {
  state.pointSize = parseFloat(e.target.value);
  state.pointObjects.forEach((p) => { p.material.size = state.pointSize * 0.08; });
});

$('reset-camera').addEventListener('click', () => {
  if (state.webglOk) { fitCamera(camera, controls, bldg); renderer.render(scene, camera); }
});

init().catch((e) => {
  console.error('init failed:', e);
  setStatus('Initialisation failed: ' + (e && e.message));
});