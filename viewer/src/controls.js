// controls.js — DOM wiring: selectors, next/prev, floor buttons.

const $ = (id) => document.getElementById(id);

export function populateSelect(sel, items, labelFn) {
  sel.innerHTML = '';
  items.forEach((item, i) => {
    const o = document.createElement('option');
    o.value = String(i);
    o.textContent = labelFn(item);
    sel.appendChild(o);
  });
}

export function setSelectIndex(sel, index) {
  if (sel && index >= 0 && index < sel.options.length) sel.selectedIndex = index;
}

export function buildFloorButtons(meta, onFloor, current) {
  const wrap = $('floor-buttons');
  if (!wrap) return;
  wrap.innerHTML = '';
  const add = (label, idx) => {
    const b = document.createElement('button');
    b.type = 'button';
    b.textContent = label;
    b.className = 'floor-btn' + (idx === current ? ' active' : '');
    b.addEventListener('click', () => onFloor(idx));
    wrap.appendChild(b);
  };
  add('All', 0);
  const n = meta && meta.floor_count ? meta.floor_count : 0;
  for (let i = 1; i <= n; i++) add(`Floor ${i}`, i);
}

export function setActiveFloor(meta, current) {
  const wrap = $('floor-buttons');
  if (!wrap) return;
  const btns = wrap.querySelectorAll('.floor-btn');
  btns.forEach((b, i) => {
    b.classList.toggle('active', i === current);
  });
}

export function setStatus(text) {
  const el = $('status');
  if (el) el.textContent = text;
}