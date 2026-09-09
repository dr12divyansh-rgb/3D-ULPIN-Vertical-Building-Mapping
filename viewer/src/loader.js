// loader.js — fetch + parse helpers for the SIH 26011 test viewer.
//
// Pure parsing functions (parsePLY / parseOBJ) have no DOM / THREE dependency
// so they can be unit-tested in Node against the real dataset files.

export const CLASS_NAMES = ['ground', 'building', 'vegetation', 'road', 'vehicle', 'other'];

// RGB (0-255) ported from synthetic_city/core/labels.py
export const CLASS_COLORS = {
  ground: [120, 120, 120],
  building: [220, 110, 55],
  vegetation: [45, 160, 80],
  road: [40, 40, 40],
  vehicle: [70, 130, 200],
  other: [200, 200, 40],
};

// LOD2 surface group colors (OBJ 'g' groups), 0-255.
export const SURFACE_COLORS = {
  WallSurface: [217, 140, 77],
  RoofSurface: [217, 51, 51],
  GroundSurface: [115, 115, 115],
};

export const LOD1_COLOR = [89, 140, 217];

export function classColor(classId) {
  const name = CLASS_NAMES[classId] || 'other';
  return CLASS_COLORS[name];
}

// ---------------------------------------------------------------- fetching --

export async function fetchJSON(path) {
  const res = await fetch(path);
  if (!res.ok) throw new Error(`fetch ${path} -> ${res.status}`);
  return res.json();
}

export async function fetchText(path) {
  const res = await fetch(path);
  if (!res.ok) throw new Error(`fetch ${path} -> ${res.status}`);
  return res.text();
}

export async function fetchBuffer(path) {
  const res = await fetch(path);
  if (!res.ok) throw new Error(`fetch ${path} -> ${res.status}`);
  return res.arrayBuffer();
}

// -------------------------------------------------------------------- PLY --

const PLY_TYPE_SIZE = {
  float: 4, double: 8, uchar: 1, uint8: 1, char: 1,
  int: 4, int32: 4, uint: 4, short: 2, ushort: 2,
};

function readFloat(dv, o) { return dv.getFloat32(o, true); }
function readDouble(dv, o) { return dv.getFloat64(o, true); }
function readU8(dv, o) { return dv.getUint8(o); }
function readI8(dv, o) { return dv.getInt8(o); }
function readI32(dv, o) { return dv.getInt32(o, true); }
function readU32(dv, o) { return dv.getUint32(o, true); }
function readI16(dv, o) { return dv.getInt16(o, true); }
function readU16(dv, o) { return dv.getUint16(o, true); }

const PLY_READERS = {
  float: readFloat, double: readDouble, uchar: readU8, uint8: readU8, char: readI8,
  int: readI32, int32: readI32, uint: readU32, short: readI16, ushort: readU16,
};

// Find the byte offset just past the "end_header" line terminator.
function headerBodySplit(bytes) {
  const needle = new TextEncoder().encode('end_header');
  outer: for (let i = 0; i <= bytes.length - needle.length; i++) {
    for (let j = 0; j < needle.length; j++) {
      if (bytes[i + j] !== needle[j]) continue outer;
    }
    let k = i + needle.length;
    while (k < bytes.length && (bytes[k] === 10 || bytes[k] === 13)) k++;
    return { headerText: new TextDecoder().decode(bytes.slice(0, i)), bodyStart: k };
  }
  throw new Error('PLY: end_header not found');
}

export function parsePLY(arrayBuffer) {
  const bytes = new Uint8Array(arrayBuffer);
  const { headerText, bodyStart } = headerBodySplit(bytes);

  let format = 'binary_little_endian';
  let count = 0;
  const properties = []; // {name, type}

  for (const rawLine of headerText.split(/\r?\n/)) {
    const line = rawLine.trim();
    if (!line || line.startsWith('ply') || line.startsWith('comment') ||
        line.startsWith('obj_info') || line.startsWith('end_header')) continue;
    const parts = line.split(/\s+/);
    if (parts[0] === 'format') format = parts[1];
    else if (parts[0] === 'element' && parts[1] === 'vertex') count = parseInt(parts[2], 10);
    else if (parts[0] === 'property') properties.push({ name: parts[2], type: parts[1] });
  }

  const data = {};
  for (const p of properties) data[p.name] = new Float32Array(count);

  if (format === 'ascii') {
    const body = new TextDecoder().decode(bytes.slice(bodyStart));
    const lines = body.split(/\r?\n/);
    for (let i = 0; i < count && i < lines.length; i++) {
      const vals = lines[i].trim().split(/\s+/);
      properties.forEach((p, j) => { data[p.name][i] = parseFloat(vals[j]); });
    }
    return { count, properties, data };
  }

  if (format !== 'binary_little_endian') {
    throw new Error(`PLY: unsupported format ${format}`);
  }

  const dv = new DataView(arrayBuffer, bodyStart);
  const stride = properties.reduce((s, p) => s + (PLY_TYPE_SIZE[p.type] || 4), 0);
  for (let i = 0; i < count; i++) {
    let o = i * stride;
    for (const p of properties) {
      data[p.name][i] = PLY_READERS[p.type](dv, o);
      o += PLY_TYPE_SIZE[p.type];
    }
  }
  return { count, properties, data };
}

// -------------------------------------------------------------------- OBJ --

// colorMap: group name -> [r,g,b] 0-255. Returns triangle-expanded positions
// and per-vertex colors.
export function parseOBJ(text, colorMap = {}) {
  const verts = [];
  const positions = [];
  const colors = [];
  let currentGroup = 'default';

  const resolveIndex = (tok) => {
    let i = parseInt(tok.split('/')[0], 10);
    return i > 0 ? i - 1 : verts.length + i;
  };

  const emitFace = (idx) => {
    const col = colorMap[currentGroup] || [170, 170, 170];
    for (let t = 1; t < idx.length - 1; t++) {
      const tri = [idx[0], idx[t], idx[t + 1]];
      for (const vi of tri) {
        const v = verts[vi];
        positions.push(v[0], v[1], v[2]);
        colors.push(col[0] / 255, col[1] / 255, col[2] / 255);
      }
    }
  };

  for (const rawLine of text.split(/\r?\n/)) {
    const line = rawLine.trim();
    if (!line || line.startsWith('#')) continue;
    const parts = line.split(/\s+/);
    if (parts[0] === 'v') {
      verts.push([parseFloat(parts[1]), parseFloat(parts[2]), parseFloat(parts[3])]);
    } else if (parts[0] === 'g') {
      currentGroup = parts[1] || 'default';
    } else if (parts[0] === 'f') {
      emitFace(parts.slice(1).map(resolveIndex));
    }
  }

  return {
    positions: new Float32Array(positions),
    colors: new Float32Array(colors),
    vertexCount: verts.length,
  };
}