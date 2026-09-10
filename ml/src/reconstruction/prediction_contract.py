"""Dataset-agnostic prediction contract for SIH 26011.

Defines the common prediction representation consumed by all downstream
reconstruction stages, regardless of whether the source is STPLS3D, DALES,
or a future dataset.

Schema
------
scene_id        : str          -- human-readable scene name
source_dataset  : str          -- "STPLS3D" | "DALES" | "unknown"
source_ply      : str          -- absolute path that was read
n_points        : int
x, y, z         : np.ndarray float32 (N,)
pred_class      : np.ndarray int32   (N,)  -- class label per point
confidence      : np.ndarray float32 (N,)  -- softmax max; None if absent
gt_class        : np.ndarray int32   (N,)  -- None if absent

Class map (fixed for this project)
0=ground  1=building  2=vegetation  3=road  4=vehicle  5=other

DO NOT invent confidence values.  If the PLY has no confidence field,
`confidence` stays None and callers must not fabricate it downstream.
"""

from __future__ import annotations

import json
import struct
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

# ── PLY type table (covers every variant seen in public LiDAR datasets) ────────
_PLY_TYPES: dict[str, str] = {
    "float": "f", "float32": "f", "double": "d", "float64": "d",
    "char": "b",  "int8": "b",   "uchar": "B",  "uint8": "B",
    "short": "h", "int16": "h",  "ushort": "H", "uint16": "H",
    "int": "i",   "int32": "i",  "uint": "I",   "uint32": "I",
    "long": "l",  "int64": "l",  "ulong": "L",  "uint64": "L",
}

CLASS_NAMES = {0: "ground", 1: "building", 2: "vegetation",
               3: "road", 4: "vehicle", 5: "other"}
BUILDING_CLASS = 1


# ── Dataclass ──────────────────────────────────────────────────────────────────

@dataclass
class PredictionRecord:
    """Immutable container for one scene's prediction output."""

    scene_id: str
    source_dataset: str          # "STPLS3D" | "DALES" | "unknown"
    source_ply: str              # path string
    x: np.ndarray                # float32 (N,)
    y: np.ndarray                # float32 (N,)
    z: np.ndarray                # float32 (N,)
    pred_class: np.ndarray       # int32   (N,)
    confidence: "np.ndarray | None" = field(default=None)   # float32 (N,) or None
    gt_class:   "np.ndarray | None" = field(default=None)   # int32   (N,) or None

    # ── derived properties ──────────────────────────────────────────────────

    @property
    def n_points(self) -> int:
        return int(self.x.shape[0])

    @property
    def xyz(self) -> np.ndarray:
        return np.stack([self.x, self.y, self.z], axis=1)

    def building_mask(self, building_class: int = BUILDING_CLASS) -> np.ndarray:
        return self.pred_class == building_class

    def building_xyz(self, building_class: int = BUILDING_CLASS) -> np.ndarray:
        return self.xyz[self.building_mask(building_class)]

    def class_summary(self) -> dict[str, int]:
        unique, counts = np.unique(self.pred_class, return_counts=True)
        return {CLASS_NAMES.get(int(c), f"class_{c}"): int(n)
                for c, n in zip(unique, counts)}

    def to_metadata(self) -> dict:
        """Return a serialisable dict describing provenance (no arrays)."""
        return {
            "scene_id": self.scene_id,
            "source_dataset": self.source_dataset,
            "source_ply": self.source_ply,
            "n_points": self.n_points,
            "has_confidence": self.confidence is not None,
            "has_ground_truth": self.gt_class is not None,
            "class_summary": self.class_summary(),
        }


# ── PLY reader ─────────────────────────────────────────────────────────────────

def _read_ply_fields(path: Path) -> dict[str, np.ndarray]:
    """Minimal PLY reader for all common formats and type names."""
    raw = path.read_bytes()
    hend = raw.find(b"end_header")
    if hend == -1:
        raise ValueError(f"{path}: not a valid PLY (no end_header)")
    header = raw[:hend].decode("ascii", errors="replace")
    body   = raw[hend + len(b"end_header"):].lstrip(b"\r\n")

    fmt, n_verts, props = "ascii", 0, []
    for line in header.splitlines():
        parts = line.strip().split()
        if not parts:
            continue
        if parts[0] == "format" and len(parts) >= 2:
            fmt = parts[1]
        elif parts[0] == "element" and len(parts) >= 3 and parts[1] == "vertex":
            n_verts = int(parts[2])
        elif parts[0] == "property" and len(parts) >= 3 and parts[1] != "list":
            sc = _PLY_TYPES.get(parts[1])
            if sc:
                props.append((parts[2], sc))

    endian = ">" if fmt == "binary_big_endian" else "<"
    data = {name: np.empty(n_verts, dtype=np.dtype(endian + sc))
            for name, sc in props}

    if fmt == "ascii":
        lines = body.decode("ascii").strip().splitlines()
        for i, line in enumerate(lines[:n_verts]):
            for (name, sc), val in zip(props, line.split()):
                data[name][i] = float(val) if sc in "fd" else int(val)
    else:
        row_fmt  = struct.Struct(endian + "".join(sc for _, sc in props))
        row_size = row_fmt.size
        for i in range(n_verts):
            row = row_fmt.unpack_from(body, i * row_size)
            for (name, _), val in zip(props, row):
                data[name][i] = val

    return data


def _pick(data: dict, *candidates: str) -> "np.ndarray | None":
    for k in candidates:
        if k in data:
            return data[k]
    return None


# ── Public loader ──────────────────────────────────────────────────────────────

def load_prediction_ply(
    ply_path: "str | Path",
    scene_id: str | None = None,
    source_dataset: str = "unknown",
) -> PredictionRecord:
    """Load a prediction PLY into a PredictionRecord.

    The PLY must have x, y, z, predicted_class fields.
    confidence and gt_class are loaded if present; never fabricated.

    Args:
        ply_path:       Path to the prediction PLY file.
        scene_id:       Human-readable scene name; defaults to the file stem.
        source_dataset: "STPLS3D", "DALES", or "unknown".
    """
    ply_path = Path(ply_path)
    if not ply_path.exists():
        raise FileNotFoundError(f"Prediction PLY not found: {ply_path}")

    data = _read_ply_fields(ply_path)
    keys = set(data.keys())

    # XYZ (required)
    x_arr = _pick(data, "x", "X")
    y_arr = _pick(data, "y", "Y")
    z_arr = _pick(data, "z", "Z")
    if x_arr is None or y_arr is None or z_arr is None:
        raise ValueError(f"{ply_path}: missing x/y/z fields. Present: {sorted(keys)}")

    # predicted_class (required)
    cls_arr = _pick(data, "predicted_class", "class_id", "classification", "label")
    if cls_arr is None:
        raise ValueError(
            f"{ply_path}: missing predicted_class field. Present: {sorted(keys)}\n"
            "Expected field name: predicted_class, class_id, classification, or label."
        )

    # confidence (optional — not fabricated)
    conf_arr = _pick(data, "confidence", "prob", "score")

    # ground truth class (optional)
    gt_arr = _pick(data, "gt_class", "ground_truth", "true_class", "gt_label")

    return PredictionRecord(
        scene_id       = scene_id or ply_path.stem,
        source_dataset = source_dataset,
        source_ply     = str(ply_path.resolve()),
        x              = x_arr.astype(np.float32),
        y              = y_arr.astype(np.float32),
        z              = z_arr.astype(np.float32),
        pred_class     = cls_arr.astype(np.int32),
        confidence     = conf_arr.astype(np.float32) if conf_arr is not None else None,
        gt_class       = gt_arr.astype(np.int32)     if gt_arr  is not None else None,
    )


# ── Convenience: save metadata sidecar ─────────────────────────────────────────

def save_prediction_metadata(record: PredictionRecord, out_path: "str | Path") -> None:
    """Write a JSON sidecar recording PLY provenance (no arrays)."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(record.to_metadata(), f, indent=2)
