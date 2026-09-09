// metadata.js — renders the Building info and ML info panels.

const $ = (id) => document.getElementById(id);

const fmt = (v) => (typeof v === 'number' ? v.toFixed(2) : String(v));

export function computeULPIN(sceneId, buildingId) {
  const m = String(sceneId).match(/(\d+)/);
  const num = m ? parseInt(m[1], 10) : 0;
  return `ULPIN-26011-S${String(num).padStart(4, '0')}-B${String(buildingId).padStart(6, '0')}`;
}

export function renderULPINHero(sceneId, buildingMeta) {
  const ulpin = computeULPIN(sceneId, buildingMeta.building_id);
  const ulpinEl = $('ulpin-value');
  const attrsEl = $('ulpin-attrs');
  if (ulpinEl) ulpinEl.textContent = ulpin;
  if (attrsEl) {
    const chips = [
      `${buildingMeta.roof_type} roof`,
      `${buildingMeta.floor_count} floors`,
      `h=${fmt(buildingMeta.total_height)}m`,
      `${buildingMeta.footprint_type}`,
    ];
    attrsEl.innerHTML = chips.map(c => `<span class="ulpin-chip">${c}</span>`).join('');
  }
  return ulpin;
}

export function renderBuildingInfo(meta, sceneId) {
  const el = $('building-info');
  if (!el || !meta) return;
  const pos = meta.position ? `${meta.position[0].toFixed(1)}, ${meta.position[1].toFixed(1)}` : '—';
  const ulpin = computeULPIN(sceneId || 'scene_00000', meta.building_id);
  el.innerHTML = `
    <table>
      <tr><td>ULPIN</td><td style="font-family:monospace;font-size:11px;color:var(--plateau)">${ulpin}</td></tr>
      <tr><td>Building ID</td><td>${meta.building_id}</td></tr>
      <tr><td>CityGML LOD</td><td>LOD1 + LOD2</td></tr>
      <tr><td>Footprint type</td><td>${meta.footprint_type}</td></tr>
      <tr><td>Roof type</td><td>${meta.roof_type}</td></tr>
      <tr><td>Width</td><td>${fmt(meta.width)} m</td></tr>
      <tr><td>Length</td><td>${fmt(meta.length)} m</td></tr>
      <tr><td>Wall height</td><td>${fmt(meta.wall_height)} m</td></tr>
      <tr><td>Total height</td><td>${fmt(meta.total_height)} m</td></tr>
      <tr><td>Floors</td><td>${meta.floor_count}</td></tr>
      <tr><td>Footprint area</td><td>${fmt(meta.footprint_area)} m²</td></tr>
      <tr><td>Position</td><td>${pos}</td></tr>
      <tr><td>Rotation</td><td>${fmt(meta.rotation_deg)}°</td></tr>
    </table>`;
}

export function renderMLInfo(metricsEntry, predictionAvailable) {
  const el = $('ml-info');
  if (!el) return;
  if (!predictionAvailable) {
    el.textContent = 'AI prediction not available for this scene.';
    return;
  }
  if (!metricsEntry) {
    el.textContent = 'No stored ML metrics for this building (not evaluated on the test split).';
    return;
  }
  el.innerHTML = `
    <div class="tag">Predicted</div>
    <table>
      <tr><td>Building segmentation</td><td></td></tr>
      <tr><td>Precision</td><td>${metricsEntry.building_precision.toFixed(4)}</td></tr>
      <tr><td>Recall</td><td>${metricsEntry.building_recall.toFixed(4)}</td></tr>
      <tr><td>F1</td><td>${metricsEntry.building_f1.toFixed(4)}</td></tr>
      <tr><td>IoU</td><td>${metricsEntry.building_iou.toFixed(4)}</td></tr>
    </table>`;
}

export function renderFloorInfo(meta, floorIdx) {
  const el = $('floor-info');
  if (!el) return;
  if (!meta || floorIdx === 0) {
    el.textContent = 'All floors (no clipping).';
    return;
  }
  const lo = meta.floor_levels[floorIdx - 1];
  const hi = meta.floor_levels[floorIdx];
  el.textContent = `Synthetic floor level ${floorIdx}: Z ${lo.toFixed(2)} – ${hi.toFixed(2)} m ` +
    `(view clipped at ${hi.toFixed(2)} m). Not an interior floor plan.`;
}