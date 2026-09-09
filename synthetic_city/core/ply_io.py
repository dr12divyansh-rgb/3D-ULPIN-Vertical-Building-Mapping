"""Minimal, dependency-free binary PLY reader/writer for our fixed point-cloud schema.

The schema is intentionally fixed so that every artifact in the dataset is
interpretable by the same code (Blender side and the venv side alike):

    x, y, z                    float32
    red, green, blue           uint8
    class_id                   uint8
    building_id                int32   (-1 for non-building points)
    intensity                  uint8   (synthetic, see docs)
    return_number              uint8   (synthetic, see docs)

Only standard library modules are used so this file runs unchanged inside
Blender's bundled Python as well as the project venv.
"""

from __future__ import annotations

import struct
from pathlib import Path

# (name, ply type string, struct char, struct format char)
_PROPERTIES = [
    ("x", "float", "f"),
    ("y", "float", "f"),
    ("z", "float", "f"),
    ("red", "uchar", "B"),
    ("green", "uchar", "B"),
    ("blue", "uchar", "B"),
    ("class_id", "uchar", "B"),
    ("building_id", "int", "i"),
    ("intensity", "uchar", "B"),
    ("return_number", "uchar", "B"),
]

_POINT_STRUCT = struct.Struct("<" + "".join(c for _, _, c in _PROPERTIES))

_PLY_TYPE_TO_STRUCT = {"float": "f", "double": "d", "uchar": "B", "uint8": "B",
                       "int": "i", "int32": "i", "short": "h", "ushort": "H"}


def write_pointcloud(path, xyz, class_id, building_id, intensity=None, return_number=None):
    """Write a binary little-endian PLY file.

    xyz       : sequence of (x, y, z) floats
    class_id  : sequence of ints
    building_id : sequence of ints
    intensity : optional sequence of ints (default 0)
    return_number : optional sequence of ints (default 1)
    """
    xyz = list(xyz)
    n = len(xyz)
    class_id = list(class_id)
    building_id = list(building_id)
    if intensity is None:
        intensity = [0] * n
    if return_number is None:
        return_number = [1] * n
    assert len(class_id) == n and len(building_id) == n
    assert len(intensity) == n and len(return_number) == n

    from .labels import CLASS_COLORS, class_name

    lines = [
        "ply",
        "format binary_little_endian 1.0",
        "comment Synthetic LiDAR-like point cloud (SIH 26011)",
        f"element vertex {n}",
    ]
    for name, ply_type, _ in _PROPERTIES:
        lines.append(f"property {ply_type} {name}")
    lines.append("end_header")
    header = "\n".join(lines) + "\n"

    buf = bytearray()
    for i in range(n):
        x, y, z = xyz[i]
        cid = int(class_id[i])
        bid = int(building_id[i])
        r, g, b = CLASS_COLORS[class_name(cid)]
        buf += _POINT_STRUCT.pack(
            float(x), float(y), float(z),
            int(r), int(g), int(b),
            cid, bid,
            int(intensity[i]), int(return_number[i]),
        )

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        f.write(header.encode("ascii"))
        f.write(bytes(buf))
    return path


def read_pointcloud(path):
    """Read a PLY file written by this module into a dict of property -> list.

    Also returns the header element count. Works for both ascii and
    binary_little_endian PLY files, but the pipeline only ever writes binary.
    """
    path = Path(path)
    with open(path, "rb") as f:
        raw = f.read()

    header_end = raw.find(b"end_header")
    if header_end == -1:
        raise ValueError(f"{path}: not a valid PLY (no end_header)")
    header_text = raw[:header_end].decode("ascii")
    body = raw[header_end + len(b"end_header"):]
    # skip the newline (and any whitespace) that terminates the header
    i = 0
    while i < len(body) and body[i:i + 1] in (b"\n", b"\r", b" ", b"\t"):
        i += 1
    body = body[i:]

    lines = header_text.splitlines()
    if lines[0].strip() != "ply":
        raise ValueError(f"{path}: not a PLY file")

    fmt = "ascii"
    count = 0
    props = []  # list of (name, struct_char)
    for line in lines[1:]:
        parts = line.strip().split()
        if not parts:
            continue
        if parts[0] == "format":
            fmt = parts[1]
        elif parts[0] == "element":
            if parts[1] == "vertex":
                count = int(parts[2])
        elif parts[0] == "property":
            ptype, pname = parts[1], parts[2]
            props.append((pname, _PLY_TYPE_TO_STRUCT[ptype]))

    if fmt == "ascii":
        # not produced by this pipeline, but cheap to support for robustness
        text = body.decode("ascii")
        data = {name: [] for name, _ in props}
        for line in text.strip().splitlines():
            vals = line.split()
            for (name, sc), val in zip(props, vals):
                if sc in "fd":
                    data[name].append(float(val))
                else:
                    data[name].append(int(val))
        return data, count

    if fmt != "binary_little_endian":
        raise ValueError(f"{path}: unsupported PLY format {fmt}")

    fmt_str = "<" + "".join(sc for _, sc in props)
    size = struct.calcsize(fmt_str)
    if len(body) < count * size:
        raise ValueError(f"{path}: truncated binary PLY body")
    unpack = struct.Struct(fmt_str).unpack_from
    data = {name: [] for name, _ in props}
    offset = 0
    for _ in range(count):
        row = unpack(body, offset)
        offset += size
        for (name, _sc), val in zip(props, row):
            data[name].append(val)
    return data, count
