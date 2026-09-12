"""
Turn a sequence of protein structures into a movie (.mp4) of a membrane protein.

The input is an ordered list of structure files (.pdb / .cif / .ent):
  * the FIRST is treated as an all-atom (atomistic) structure,
  * every SUBSEQUENT file is treated as a Martini-style coarse-grained (CG)
    structure (one ATOM per bead, standard residue names + BB/SC1..SC3 labels).

A coarse-grained lipid bilayer (grey beads) is generated from the first
structure's geometry and fades in as the movie moves past the atomistic frame,
so the protein sits in a membrane for the CG portion. The protein is coloured by
Martini bead *size class* when it can be inferred from the residue/bead names
(tiny=red, small=blue, regular/large=yellow — same scheme as
atomistic_to_martini_cg.py); any protein bead that can't be classified falls
back to light blue, as does the all-atom first frame.

Pipeline (mirrors the repo's cg_conversion_prep.py + render_conversion.pml + ffmpeg flow):
  1. parse the first structure, build a grey CG membrane slab (membrane.pdb)
  2. write movie_meta.json (file list, colour lookup, render settings)
  3. write + run render.pml under headless PyMOL -> PNG frames
  4. stitch frames into an .mp4 with ffmpeg

Requires `pymol` and `ffmpeg` on PATH.

Usage:
    python scripts/cg_structures_to_movie.py atomistic.pdb cg_001.pdb cg_002.pdb ... -o movie.mp4
    python scripts/cg_structures_to_movie.py frames/*.pdb -o traj.mp4 --fps 30 --frames-per-structure 4
    python scripts/cg_structures_to_movie.py protein.cif cg_*.pdb -o out.mp4 --no-membrane --keep-frames
"""
import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
from Bio.PDB.PDBParser import PDBParser
from Bio.PDB.MMCIFParser import MMCIFParser

from cg_membrane_experiments.constants import (
    MARTINI_RESIDUE_BEADS,
    MODIFIED_RESIDUES_TO_STANDARD,
)

# Martini bead size classes (by LJ sigma) — kept in sync with
# atomistic_to_martini_cg.py: tiny=red, small=blue, regular/large=yellow.
_SIZE_CLASS_BY_BEAD_TYPE = {
    "TC": "TINY", "ION": "TINY",              # tiny apolar / ions  (sigma ~0.34 nm)
    "SC": "SMAL",                             # apolar small        (sigma ~0.41 nm)
    "BB": "REGU", "C": "REGU", "N": "REGU",   # backbone / regular  (sigma ~0.47 nm)
    "Q": "REGU", "W": "REGU",
}


def _size_class(bead_type):
    """Martini size class for a bead type (defaults to REGU)."""
    return _SIZE_CLASS_BY_BEAD_TYPE.get(bead_type, "REGU")

# segID (size class) -> PyMOL colour name
SIZE_COLORS = {"TINY": "magenta", "SMAL": "magenta", "REGU": "magenta"}
FALLBACK_PROTEIN_COLOR = "magenta"   # protein bead we couldn't size / all-atom frame
MEMBRANE_COLOR = "grey70"

# CG lipid residues (the sim writes the bilayer as chain 'L', resname 'LIP');
# these are shown as the grey stick membrane. Water/ions are hidden.
LIPID_RESNAMES = {"LIP", "POPC", "DPPC", "POPE", "POPG", "POPS", "POPA",
                  "DOPC", "DLPC", "DMPC", "CHOL"}
SOLVENT_RESNAMES = {"SOL", "W", "WAT", "HOH", "TIP3", "NA", "CL", "ION",
                    "NA+", "CL-", "K", "K+"}


def has_lipids(path):
    """True if the structure already contains a CG lipid bilayer (chain L / LIP)."""
    model = _read_structure(path)
    for chain in model.get_chains():
        if chain.id == "L":
            return True
    for res in model.get_residues():
        if res.resname.strip() in LIPID_RESNAMES:
            return True
    return False


def _read_structure(path):
    """Parse a .pdb/.cif/.ent file, returning the first model."""
    path = str(path)
    parser = (MMCIFParser(QUIET=True)
              if path.lower().endswith((".cif", ".mmcif"))
              else PDBParser(QUIET=True))
    return parser.get_structure("s", path)[0]


def build_resname_bead_lookup():
    """(resname, bead_label) -> size class 'TINY'/'SMAL'/'REGU' for every residue.

    Includes modified-residue aliases so CG files using e.g. MSE/SEC still map.
    Returns (lookup_list, amino_acid_names) as JSON-serialisable structures.
    """
    lookup = []          # list of [resname, bead_label, size_class]
    aa_names = set()
    for resname, bead_defs in MARTINI_RESIDUE_BEADS.items():
        aa_names.add(resname)
        for bead_type, bead_label, _charge, _atoms in bead_defs:
            lookup.append([resname, bead_label, _size_class(bead_type)])
    # modified-residue aliases point at their standard residue's beads
    for mod, std in MODIFIED_RESIDUES_TO_STANDARD.items():
        bead_defs = MARTINI_RESIDUE_BEADS.get(std)
        if bead_defs is None:
            continue
        aa_names.add(mod)
        for bead_type, bead_label, _charge, _atoms in bead_defs:
            lookup.append([mod, bead_label, _size_class(bead_type)])
    return lookup, sorted(aa_names)


def build_membrane(first_structure_path, spacing=8.0, thickness=40.0,
                   padding=15.0, exclusion=5.0):
    """Generate a grey CG lipid bilayer (stick lipids) around the first structure.

    Assumes the membrane protein is oriented with its transmembrane axis along Z
    (the project's convention; centre it with OPM/PPM before generating frames).
    Each XY grid point gets one lipid per leaflet: a head bead at the outer face
    and a tail bead near the midplane, bonded into a stick that points inward, so
    the two rows of rods bracket the protein and read as an embedded bilayer.
    Grid points overlapping the protein footprint are dropped so lipids hug (not
    bury) the protein. Returns PDB HETATM + CONECT lines, or [] if empty.
    """
    model = _read_structure(first_structure_path)
    coords = np.array([a.get_coord() for a in model.get_atoms()])  # Angstrom
    if coords.size == 0:
        return []

    centre = coords.mean(axis=0)
    z0 = centre[2]  # bilayer midplane ~ protein centre for a TM protein
    half = thickness / 2.0
    tail_z = min(4.0, half * 0.15)  # tail bead sits just off the midplane

    xmin, ymin = coords[:, 0].min() - padding, coords[:, 1].min() - padding
    xmax, ymax = coords[:, 0].max() + padding, coords[:, 1].max() + padding
    xs = np.arange(xmin, xmax + spacing, spacing)
    ys = np.arange(ymin, ymax + spacing, spacing)
    gx, gy = np.meshgrid(xs, ys, indexing="ij")
    grid_xy = np.stack([gx.ravel(), gy.ravel()], axis=1)

    # protein atoms within the membrane slab, used to punch the protein's hole
    in_slab = np.abs(coords[:, 2] - z0) <= half
    prot_xy = coords[in_slab][:, :2]
    if len(prot_xy):
        keep = np.ones(len(grid_xy), bool)
        for s in range(0, len(grid_xy), 2000):
            chunk = grid_xy[s:s + 2000]
            d = np.sqrt(((chunk[:, None, :] - prot_xy[None, :, :]) ** 2).sum(2))
            keep[s:s + 2000] = d.min(1) >= exclusion
        grid_xy = grid_xy[keep]

    # one lipid = (head bead, tail bead) rod; upper leaflet points down, lower up
    lipids = []
    for x, y in grid_xy:
        lipids.append(((x, y, z0 + half), (x, y, z0 + tail_z)))   # upper
        lipids.append(((x, y, z0 - half), (x, y, z0 - tail_z)))   # lower

    atom_lines, conect_lines, serial, resid = [], [], 0, 0
    for head, tail in lipids:
        resid += 1
        head_serial = serial + 1
        for name, (x, y, z) in (("H", head), ("T", tail)):
            serial += 1
            atom_lines.append(
                f"HETATM{serial:>5}  {name:<3}LIP M{resid % 10000:>4}    "
                f"{x:8.3f}{y:8.3f}{z:8.3f}"
                f"{1.00:6.2f}{0.00:6.2f}      MEMB C"
            )
        conect_lines.append(f"CONECT{head_serial:>5}{head_serial + 1:>5}")
    return atom_lines + conect_lines + ["END"]


# ── PyMOL render script (static; reads movie_meta.json from its own dir) ──────
RENDER_PML = r'''
reinitialize
bg_color white
set orthoscopic, 1
set ray_shadows, 0
set ray_opaque_background, 1
set specular, 0.25

python
import json, os
from pymol import cmd

meta = json.load(open("movie_meta.json"))   # PyMOL runs with cwd = frames dir

files      = meta["files"]                 # ordered; files[0] = atomistic
membrane   = meta["membrane"]              # path or None
size_colors = meta["size_colors"]          # {segid: color}
fallback   = meta["fallback_color"]
mem_color  = meta["membrane_color"]
fpe        = meta["frames_per_structure"]
dpf        = meta["deg_per_frame"]
width      = meta["width"]
height     = meta["height"]
use_ray    = meta["ray"]
frames_dir = meta["frames_dir"]
sphere     = meta["sphere_scale"]
mem_alpha  = meta["membrane_transparency"]   # settled lipid transparency

# (resname, bead_label) -> size class ; + amino-acid name set for fallback
lookup = {(r, l): s for r, l, s in meta["lookup"]}
aa = set(meta["amino_acids"])
lipids = set(meta["lipid_resnames"])
solvent = set(meta["solvent_resnames"])

def classify(resn, name, chain):
    key = (resn, name.strip())
    if key in lookup:
        return lookup[key]      # protein bead -> TINY / SMAL / REGU
    if resn in aa:
        return "PROT"           # protein bead, size unknown -> fallback colour
    if resn in lipids or chain == "L":
        return "MEMB"           # lipid bilayer already in the file -> grey sticks
    return "SOLV"               # water / ions -> hidden

os.makedirs(frames_dir, exist_ok=True)

# --- load every structure as its own object, style it ---
objs = []
for i, f in enumerate(files):
    obj = "f%03d" % i
    cmd.load(f, obj)
    cmd.hide("everything", obj)
    objs.append(obj)
    if i == 0:
        # atomistic reference: light-blue cartoon + sticks
        cmd.show("cartoon", obj)
        cmd.show("sticks", obj)
        cmd.set("stick_radius", 0.14, obj)
        cmd.color(fallback, obj)
    else:
        # tag every bead with a class in its segID, then style by class
        cmd.alter(obj, "segi = classify(resn, name, chain)",
                  space={"classify": classify})
        cmd.rebuild()
        cmd.hide("everything", obj)
        cmd.hide("everything", "%s and segi SOLV" % obj)   # water / ions stay hidden
        # protein beads -> spheres coloured by Martini size class
        prot = "%s and (segi PROT or segi TINY or segi SMAL or segi REGU)" % obj
        cmd.show("spheres", prot)
        cmd.set("sphere_scale", sphere, obj)
        cmd.color(fallback, "%s and segi PROT" % obj)
        for segid, color in size_colors.items():
            cmd.color(color, "%s and segi %s" % (obj, segid))
        # lipid bilayer already in the file -> grey sticks (the real membrane).
        # Bond consecutive beads within each lipid so it renders as sticks even
        # when the file has no CONECT records (distance-thresholded to avoid
        # linking separate tails).
        mem = "%s and segi MEMB" % obj
        mem_atoms = cmd.get_model(mem).atom
        lip_groups = {}
        for at in mem_atoms:
            lip_groups.setdefault((at.chain, at.resi), []).append(
                (at.index, at.coord))
        for beads in lip_groups.values():
            for (ia, ca), (ib, cb) in zip(beads, beads[1:]):
                if sum((ca[k] - cb[k]) ** 2 for k in range(3)) <= 7.0 ** 2:
                    cmd.bond("%s and index %d" % (obj, ia),
                             "%s and index %d" % (obj, ib))
        cmd.show("sticks", mem)
        cmd.set("stick_radius", 0.4, obj)
        cmd.color(mem_color, mem)
        cmd.set("stick_transparency", mem_alpha, mem)   # see protein through it

# --- generated grey membrane: lipids as sticks (rods pointing inward) ---
have_mem = bool(membrane) and os.path.exists(membrane)
if have_mem:
    cmd.load(membrane, "membrane")
    cmd.hide("everything", "membrane")
    cmd.show("sticks", "membrane")
    cmd.set("stick_radius", 0.6, "membrane")
    cmd.color(mem_color, "membrane")
    cmd.set("stick_transparency", mem_alpha, "membrane")   # see protein through it

# --- camera: transmembrane (Z) axis vertical, slight tilt so it reads as 3D ---
cmd.viewport(width, height)     # set aspect so zoom frames the landscape correctly
cmd.disable("all")
cmd.enable(objs[0])
if have_mem:
    cmd.enable("membrane")
cmd.reset()
cmd.turn("x", -75)          # ~edge-on with a 15-deg tilt to see into the bilayer
cmd.zoom("all", 8)
spin_origin = cmd.centerofmass("all")   # fixed pivot for the world-Z spin
if have_mem:
    cmd.disable("membrane")

counter = [0]
def render_frame():
    if use_ray:
        cmd.ray(width, height)
    else:
        cmd.draw(width, height)
    cmd.png(os.path.join(frames_dir, "frame_%04d.png" % counter[0]), dpi=100)
    # spin the whole scene about world Z (the membrane normal) so the bilayer
    # stays level regardless of the camera tilt
    cmd.rotate("z", dpf, "all", camera=0, origin=spin_origin)
    counter[0] += 1

# --- play the sequence: atomistic frame, then CG frames (membrane fades in) ---
for i, obj in enumerate(objs):
    cmd.disable("all")
    cmd.enable(obj)
    if have_mem and i >= 1:
        cmd.enable("membrane")
    for k in range(fpe):
        if i == 1:
            # fade the membrane in across the first CG structure's frames,
            # from invisible down to its settled transparency (mem_alpha),
            # whether it is the synthetic object or the file's own lipid beads
            t = mem_alpha + (1.0 - mem_alpha) * (1.0 - (k + 1) / float(fpe))
            if have_mem:
                cmd.set("stick_transparency", t, "membrane")
            cmd.set("stick_transparency", t, "%s and segi MEMB" % obj)
        render_frame()

print("MOVIE_FRAMES_DONE", counter[0])
python end
'''


def main():
    ap = argparse.ArgumentParser(
        description="Render a sequence of structures (first atomistic, rest CG) "
                    "into an .mp4 of a membrane protein.")
    ap.add_argument("inputs", nargs="+",
                    help="Ordered structure files; first = atomistic, rest = CG")
    ap.add_argument("-o", "--output", default="cg_movie.mp4",
                    help="Output .mp4 path (default: cg_movie.mp4)")
    ap.add_argument("--fps", type=int, default=24, help="Movie frame rate")
    ap.add_argument("--frames-per-structure", type=int, default=12,
                    help="Rendered frames each structure is held for")
    ap.add_argument("--deg-per-frame", type=float, default=1.0,
                    help="Continuous spin per frame (deg, about membrane normal)")
    ap.add_argument("--width", type=int, default=1024,
                    help="Frame width in pixels (default 1024)")
    ap.add_argument("--height", type=int, default=768,
                    help="Frame height in pixels (default 768; 1024x768 = 4:3 landscape)")
    ap.add_argument("--sphere-scale", type=float, default=0.5,
                    help="CG protein bead sphere scale")
    ap.add_argument("--no-membrane", action="store_true",
                    help="Do not generate a CG membrane")
    ap.add_argument("--membrane-thickness", type=float, default=40.0,
                    help="Bilayer thickness in Angstrom (default 40)")
    ap.add_argument("--membrane-transparency", type=float, default=0.5,
                    help="Lipid stick transparency 0=opaque..1=invisible "
                         "(default 0.5, so protein beads show through)")
    ap.add_argument("--fast", action="store_true",
                    help="Use draw() instead of ray() (faster, lower quality)")
    ap.add_argument("--keep-frames", action="store_true",
                    help="Keep the intermediate PNG frames")
    args = ap.parse_args()

    for tool in ("pymol", "ffmpeg"):
        if shutil.which(tool) is None:
            ap.error(f"'{tool}' not found on PATH — required for rendering.")

    inputs = [Path(p) for p in args.inputs]
    missing = [str(p) for p in inputs if not p.is_file()]
    if missing:
        ap.error("Input file(s) not found: " + ", ".join(missing))
    if len(inputs) < 2:
        print("Note: only one structure given — the movie will be a single "
              "held/rotating frame.", file=sys.stderr)

    out_path = Path(args.output).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    work = out_path.with_suffix("")              # e.g. movie/ alongside movie.mp4
    frames_dir = Path(str(work) + "_frames")
    frames_dir.mkdir(parents=True, exist_ok=True)

    # Membrane: prefer the lipid bilayer already in the CG files (keeps the movie
    # aligned with what PyMOL shows). Only synthesise one when the CG structures
    # have no lipids — and build it from the first CG structure so it sits in the
    # same coordinate frame as the CG protein (not the recentred atomistic frame).
    membrane_path = None
    first_cg = inputs[1] if len(inputs) > 1 else inputs[0]
    if args.no_membrane:
        pass
    elif has_lipids(first_cg):
        print("Detected a lipid bilayer in the CG structures — rendering those "
              "beads as the grey membrane (no synthetic membrane added).")
    else:
        mem_lines = build_membrane(first_cg, thickness=args.membrane_thickness)
        if mem_lines:
            membrane_path = frames_dir / "membrane.pdb"
            membrane_path.write_text("\n".join(mem_lines) + "\n")
            n_beads = sum(1 for l in mem_lines if l.startswith("HETATM"))
            print(f"No lipids in CG files — built a synthetic grey membrane: "
                  f"{n_beads} beads -> {membrane_path}")

    lookup, aa_names = build_resname_bead_lookup()
    meta = {
        "files": [str(p.resolve()) for p in inputs],
        "membrane": str(membrane_path) if membrane_path else None,
        "size_colors": SIZE_COLORS,
        "fallback_color": FALLBACK_PROTEIN_COLOR,
        "membrane_color": MEMBRANE_COLOR,
        "frames_per_structure": args.frames_per_structure,
        "deg_per_frame": args.deg_per_frame,
        "width": args.width,
        "height": args.height,
        "ray": not args.fast,
        "frames_dir": str(frames_dir),
        "sphere_scale": args.sphere_scale,
        "lookup": lookup,
        "amino_acids": aa_names,
        "lipid_resnames": sorted(LIPID_RESNAMES),
        "solvent_resnames": sorted(SOLVENT_RESNAMES),
        "membrane_transparency": args.membrane_transparency,
    }
    (frames_dir / "movie_meta.json").write_text(json.dumps(meta, indent=2))
    pml_path = frames_dir / "render.pml"
    pml_path.write_text(RENDER_PML)

    # render frames under headless PyMOL (cwd = frames_dir so meta.json resolves)
    print(f"Rendering frames with PyMOL ({args.width}x{args.height}, "
          f"{'ray' if not args.fast else 'draw'})...")
    proc = subprocess.run(["pymol", "-cq", "render.pml"],
                          cwd=str(frames_dir), capture_output=True, text=True)
    sys.stdout.write(proc.stdout)
    if "MOVIE_FRAMES_DONE" not in proc.stdout:
        sys.stderr.write(proc.stderr)
        ap.error("PyMOL did not finish rendering — see output above.")

    n_frames = len(list(frames_dir.glob("frame_*.png")))
    if n_frames == 0:
        ap.error("No frames were rendered.")

    # stitch to mp4 (pad odd dimensions so libx264 accepts them)
    print(f"Encoding {n_frames} frames -> {out_path}")
    ff = subprocess.run(
        ["ffmpeg", "-y", "-framerate", str(args.fps),
         "-i", str(frames_dir / "frame_%04d.png"),
         "-c:v", "libx264", "-pix_fmt", "yuv420p",
         "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2", str(out_path)],
        capture_output=True, text=True)
    if ff.returncode != 0:
        sys.stderr.write(ff.stderr)
        ap.error("ffmpeg failed to encode the movie.")

    print(f"Wrote {out_path}")
    if not args.keep_frames:
        shutil.rmtree(frames_dir, ignore_errors=True)
    else:
        print(f"Kept frames + render.pml in {frames_dir}")


if __name__ == "__main__":
    main()
