"""CityGML 2.0 exporter for SIH 26011 — PLATEAU-equivalent output.

Reads reconstruction_metadata.json + per-building OBJ files and writes
a valid CityGML 2.0 city model with:
  - LOD2 <bldg:Building> per building
  - Semantic surfaces: WallSurface, RoofSurface, GroundSurface
  - Building attributes: height, floors, roof type, footprint vertices
  - ULPIN identifier as gml:id and gml:name

Usage:
    from ml.src.export.citygml import export_scene
    export_scene(meta_json_path, lod2_dir, output_gml_path)
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from xml.etree import ElementTree as ET

from ml.src.export.ulpin import ulpin_from_ids

# ── XML namespaces ────────────────────────────────────────────────────────────
NS = {
    "core": "http://www.opengis.net/citygml/2.0",
    "bldg": "http://www.opengis.net/citygml/building/2.0",
    "gml":  "http://www.opengis.net/gml",
    "gen":  "http://www.opengis.net/citygml/generics/2.0",
    "xsi":  "http://www.w3.org/2001/XMLSchema-instance",
}
_SCHEMA = (
    "http://www.opengis.net/citygml/2.0 "
    "http://schemas.opengis.net/citygml/2.0/cityGMLBase.xsd "
    "http://www.opengis.net/citygml/building/2.0 "
    "http://schemas.opengis.net/citygml/building/2.0/building.xsd"
)

for prefix, uri in NS.items():
    ET.register_namespace(prefix, uri)


def _tag(ns_key: str, local: str) -> str:
    return f"{{{NS[ns_key]}}}{local}"


# ── OBJ parser ────────────────────────────────────────────────────────────────

def _parse_obj(path: Path) -> dict[str, list]:
    """Parse OBJ file; return {group_name: [[(x,y,z), ...], ...]} of face vertex lists."""
    verts: list[tuple] = []
    groups: dict[str, list] = {}
    current = "default"
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if parts[0] == "v":
                verts.append((float(parts[1]), float(parts[2]), float(parts[3])))
            elif parts[0] == "g":
                current = parts[1] if len(parts) > 1 else "default"
                groups.setdefault(current, [])
            elif parts[0] == "f":
                idx = [int(p.split("/")[0]) - 1 for p in parts[1:]]
                face_verts = [verts[i] for i in idx]
                groups.setdefault(current, []).append(face_verts)
    return groups


# ── GML geometry helpers ──────────────────────────────────────────────────────

def _pos_list(face_verts: list[tuple]) -> str:
    """Convert face vertex list to GML posList (closed ring, YXZ → XYZ)."""
    ring = list(face_verts) + [face_verts[0]]
    coords = " ".join(f"{x:.4f} {y:.4f} {z:.4f}" for x, y, z in ring)
    return coords


def _polygon_element(uid: str, face_verts: list[tuple]) -> ET.Element:
    poly = ET.Element(_tag("gml", "Polygon"), attrib={_tag("gml", "id"): uid})
    ext = ET.SubElement(poly, _tag("gml", "exterior"))
    ring = ET.SubElement(ext, _tag("gml", "LinearRing"))
    pl = ET.SubElement(ring, _tag("gml", "posList"))
    pl.attrib["srsDimension"] = "3"
    pl.text = _pos_list(face_verts)
    return poly


def _surface_member(uid_prefix: str, faces: list, parent: ET.Element, idx_start: int) -> int:
    for i, face in enumerate(faces):
        sm = ET.SubElement(parent, _tag("gml", "surfaceMember"))
        sm.append(_polygon_element(f"{uid_prefix}-F{idx_start + i}", face))
    return idx_start + len(faces)


# ── Per-building CityGML element ──────────────────────────────────────────────

_SURFACE_ELEMENT = {
    "WallSurface":   "bldg:WallSurface",
    "RoofSurface":   "bldg:RoofSurface",
    "GroundSurface": "bldg:GroundSurface",
}
_SURFACE_TAG = {
    "WallSurface":   _tag("bldg", "WallSurface"),
    "RoofSurface":   _tag("bldg", "RoofSurface"),
    "GroundSurface": _tag("bldg", "GroundSurface"),
}


def _building_element(b: dict, scene_id: str, obj_dir: Path) -> ET.Element:
    ulpin = ulpin_from_ids(scene_id, b["building_id"])
    obj_path = obj_dir / b["obj"]
    groups = _parse_obj(obj_path) if obj_path.exists() else {}

    bldg_el = ET.Element(_tag("bldg", "Building"))
    bldg_el.attrib[_tag("gml", "id")] = ulpin

    # ── gml:name ──
    name_el = ET.SubElement(bldg_el, _tag("gml", "name"))
    name_el.text = ulpin

    # ── measuredHeight ──
    h_el = ET.SubElement(bldg_el, _tag("bldg", "measuredHeight"))
    h_el.attrib["uom"] = "m"
    h_el.text = f"{b.get('height_pred', 0.0):.3f}"

    # ── storeysAboveGround (estimate: ~3m per floor) ──
    floors = max(1, round(b.get("height_pred", 3.0) / 3.0))
    st_el = ET.SubElement(bldg_el, _tag("bldg", "storeysAboveGround"))
    st_el.text = str(floors)

    # ── generic attributes ──
    def _str_attr(name: str, value: str) -> None:
        a = ET.SubElement(bldg_el, _tag("gen", "stringAttribute"), attrib={"name": name})
        ET.SubElement(a, _tag("gen", "value")).text = value

    def _dbl_attr(name: str, value: float) -> None:
        a = ET.SubElement(bldg_el, _tag("gen", "doubleAttribute"), attrib={"name": name})
        ET.SubElement(a, _tag("gen", "value")).text = f"{value:.4f}"

    _str_attr("roofType", b.get("roof_type", "unknown"))
    _str_attr("footprintSource", b.get("footprint_source", "unknown"))
    _str_attr("dataSource", "SIH26011-SyntheticLiDAR")
    _str_attr("project", "ULPIN-3D")
    _dbl_attr("baseZ", b.get("base_z", 0.0))
    _dbl_attr("topZ", b.get("top_z", 0.0))
    _dbl_attr("eaveZ", b.get("eave_z", b.get("top_z", 0.0)))
    _dbl_attr("peakZ", b.get("peak_z", b.get("top_z", 0.0)))
    _dbl_attr("footprintVertices", float(b.get("footprint_n_vertices", 0)))

    # ── lod2Solid — all faces combined ──
    all_faces = [f for faces in groups.values() for f in faces]
    if all_faces:
        lod2_solid = ET.SubElement(bldg_el, _tag("bldg", "lod2Solid"))
        solid = ET.SubElement(lod2_solid, _tag("gml", "Solid"))
        exterior = ET.SubElement(solid, _tag("gml", "exterior"))
        cs = ET.SubElement(exterior, _tag("gml", "CompositeSurface"))
        _surface_member(f"{ulpin}-ALL", all_faces, cs, 0)

    # ── boundedBy — semantic surfaces ──
    for surf_name in ("WallSurface", "RoofSurface", "GroundSurface"):
        faces = groups.get(surf_name, [])
        if not faces:
            continue
        bb = ET.SubElement(bldg_el, _tag("bldg", "boundedBy"))
        surf_el = ET.SubElement(bb, _SURFACE_TAG[surf_name])
        surf_el.attrib[_tag("gml", "id")] = f"{ulpin}-{surf_name.upper()[:4]}"
        ms_wrap = ET.SubElement(surf_el, _tag("bldg", "lod2MultiSurface"))
        ms = ET.SubElement(ms_wrap, _tag("gml", "MultiSurface"))
        _surface_member(f"{ulpin}-{surf_name}", faces, ms, 0)

    return bldg_el


# ── Scene-level export ────────────────────────────────────────────────────────

def export_scene(
    meta_json: Path,
    lod2_dir: Path,
    out_gml: Path,
    srs_name: str = "urn:ogc:def:crs:OGC:1.3:CRS84",
) -> list[dict]:
    """Export one scene's reconstruction as CityGML 2.0.

    Returns the list of ULPIN registry records (one per building).
    """
    with open(meta_json, encoding="utf-8") as f:
        meta = json.load(f)

    scene_id = meta["scene"]
    buildings = meta["buildings"]

    # Bounding box across all buildings
    all_x, all_y, all_z = [], [], []
    for b in buildings:
        for field, lst in [("base_z", all_z), ("top_z", all_z)]:
            if field in b:
                lst.append(b[field])

    # ── Root element ──
    root = ET.Element(_tag("core", "CityModel"))
    root.attrib[_tag("xsi", "schemaLocation")] = _SCHEMA

    # gml:description
    desc = ET.SubElement(root, _tag("gml", "description"))
    desc.text = f"SIH 26011 ULPIN Project - 3D city model (PLATEAU-equivalent) - scene {scene_id}"

    # gml:name
    nm = ET.SubElement(root, _tag("gml", "name"))
    nm.text = f"ULPIN-26011-{scene_id}"

    # cityObjectMember per building
    registry_records = []
    for b in buildings:
        member = ET.SubElement(root, _tag("core", "cityObjectMember"))
        bldg_el = _building_element(b, scene_id, lod2_dir)
        member.append(bldg_el)

        ulpin = ulpin_from_ids(scene_id, b["building_id"])
        registry_records.append({
            "ulpin": ulpin,
            "scene_id": scene_id,
            "building_id": b["building_id"],
            "roof_type": b.get("roof_type", ""),
            "height_pred_m": round(b.get("height_pred", 0.0), 3),
            "footprint_vertices": b.get("footprint_n_vertices", 0),
            "watertight": b.get("mesh_watertight", False),
        })

    # ── Write ──
    out_gml.parent.mkdir(parents=True, exist_ok=True)
    tree = ET.ElementTree(root)
    try:
        ET.indent(tree, space="  ")  # Python ≥ 3.9
    except AttributeError:
        pass  # Python < 3.9 — still valid XML, just unindented
    with open(out_gml, "wb") as f:
        f.write(b'<?xml version="1.0" encoding="UTF-8"?>\n')
        tree.write(f, encoding="utf-8", xml_declaration=False)

    print(f"  CityGML → {out_gml}  ({len(buildings)} buildings)")
    return registry_records
