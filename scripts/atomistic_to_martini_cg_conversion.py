"""
Convert an atomistic protein structure (.ent/.pdb) into a Martini 3-based
coarse-grained (CG) structure file for overlay in PyMOL.

Each residue is reduced to 1-4 beads (BB + 0-3 sidechain beads) using the exact
same mapping the proteogram pipeline uses for its CG force model
(`MARTINI_RESIDUE_BEADS` in src/cg_membrane_experiments/constants.py). Every bead is
placed at the centroid of its heavy atoms, so the CG beads sit inside the
atomistic structure in the *same coordinate frame* — loading both files into
PyMOL overlays them directly, no alignment needed.

Output:
  - a CG PDB where each bead is one ATOM record (name = BB/SC1/SC2/SC3,
    resname/chain/resSeq preserved, B-factor = bead charge), with CONECT
    records for backbone (BB-BB) and intra-residue (BB-SC-SC-SC) bonds.
  - (optional, --pymol) a .pml script that loads the original + CG structures
    and styles the CG beads as spheres over the atomistic cartoon.

Usage:
    python scripts/atomistic_to_martini_cg.py input.ent
    python scripts/atomistic_to_martini_cg.py input.pdb -o out_cg.pdb --pymol

Bead mapping and force-field basis: see docs/martini_md_simulation_methodology.md
and proteogram/v2/martini_nonbonded_forces.py.
"""
import argparse
import warnings
from pathlib import Path

import numpy as np
from Bio.PDB.PDBParser import PDBParser
from Bio.PDB.PDBExceptions import PDBConstructionWarning

from cg_membrane_experiments.constants import (
    MARTINI_RESIDUE_BEADS,
    MODIFIED_RESIDUES_TO_STANDARD,
)

warnings.filterwarnings("ignore", category=PDBConstructionWarning)


def compute_bead_position(residue, atom_names):
    """Return centroid (Å) of atom_names in residue, falling back to CA/N/C.

    Mirrors MartiniNonBondedForceModel._compute_bead_position so the CG beads
    line up exactly with the ones used in the energy pipeline.
    """
    coords = [residue[name].get_vector().get_array()
              for name in atom_names if name in residue]
    if coords:
        return np.mean(coords, axis=0)
    for fallback in ("CA", "N", "C"):
        if fallback in residue:
            return residue[fallback].get_vector().get_array()
    raise KeyError(
        f"Residue {residue.resname} {residue.id} has no usable atoms "
        f"(tried {atom_names} + CA/N/C fallbacks)"
    )


def structure_to_beads(pdb_path):
    """Parse a PDB/ENT file into a flat list of CG bead dicts.

    Each bead: {serial, name, resname, chain, resseq, xyz (Å), charge,
                bb_serial (serial of this residue's BB bead)}.
    Returns (beads, bb_serials_by_chain) where the latter maps chain_id -> list
    of (resseq, bb_serial) in order, used to build backbone CONECT records.
    """
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("protein", str(pdb_path))
    model = structure[0]

    beads = []
    bb_by_chain = {}
    serial = 0

    for chain in model.get_chains():
        for res in chain:
            raw = res.resname.strip()
            resname = MODIFIED_RESIDUES_TO_STANDARD.get(raw, raw)
            bead_defs = MARTINI_RESIDUE_BEADS.get(resname)
            if bead_defs is None:
                continue  # HETATM, solvent, non-standard residue — skip
            if "CA" not in res:
                continue  # no alpha carbon — BB position unreliable, skip

            resseq = res.id[1]
            bb_serial = None
            for bead_type, bead_label, charge, atom_names in bead_defs:
                serial += 1
                xyz = compute_bead_position(res, atom_names)
                if bead_label == "BB":
                    bb_serial = serial
                beads.append({
                    "serial": serial,
                    "name": bead_label,
                    "resname": resname,
                    "chain": chain.id if chain.id.strip() else "A",
                    "resseq": resseq,
                    "xyz": xyz,
                    "charge": charge,
                    "bead_type": bead_type,
                    "bb_serial": bb_serial,
                })
            bb_by_chain.setdefault(chain.id, []).append((resseq, bb_serial))

    if not beads:
        raise ValueError(f"No recognised protein residues found in {pdb_path}")
    return beads, bb_by_chain


def build_bonds(beads, bb_by_chain):
    """Return a list of (serial_a, serial_b) CG bonds for CONECT records.

    Intra-residue: chain the beads in definition order (BB-SC1-SC2-SC3).
    Backbone: BB(i)-BB(i+1) for sequence-adjacent residues in the same chain.
    """
    bonds = []

    # intra-residue: consecutive beads sharing the same BB serial
    by_res = {}
    for b in beads:
        by_res.setdefault((b["chain"], b["resseq"], b["bb_serial"]), []).append(b["serial"])
    for serials in by_res.values():
        serials.sort()
        for a, c in zip(serials, serials[1:]):
            bonds.append((a, c))

    # backbone: BB-BB between sequence-adjacent residues
    for residues in bb_by_chain.values():
        residues.sort(key=lambda t: t[0])
        for (seq_a, bb_a), (seq_b, bb_b) in zip(residues, residues[1:]):
            if seq_b == seq_a + 1:  # only bond true neighbours (skip crystal gaps)
                bonds.append((bb_a, bb_b))
    return bonds


# Martini bead size classes (by LJ sigma), stamped into the PDB segID column
# so PyMOL can color beads by size: tiny=red, small=blue, regular/large=yellow.
_SIZE_CLASS_BY_BEAD_TYPE = {
    "TC": "TINY", "ION": "TINY",   # tiny apolar / ions   (sigma ~0.34 nm)
    "SC": "SMAL",                   # apolar small         (sigma ~0.41 nm)
    "BB": "REGU", "C": "REGU",      # backbone / regular   (sigma ~0.47 nm)
    "N": "REGU", "Q": "REGU", "W": "REGU",
}


def _size_class(bead_type):
    """PDB segID marking the bead's Martini size class (defaults to REGU)."""
    return _SIZE_CLASS_BY_BEAD_TYPE.get(bead_type, "REGU")


def _pdb_element(bead_type, charge):
    """Pseudo-element for PDB col 77-78, chosen so PyMOL's default coloring
    separates bead classes: N (blue) for +, O (red) for -, S (yellow) for
    backbone, C (grey) for neutral sidechains."""
    if charge > 0:
        return "N"
    if charge < 0:
        return "O"
    if bead_type == "BB":
        return "S"
    return "C"


def write_cg_pdb(beads, bonds, out_path):
    """Write beads + bonds to a CG PDB file."""
    lines = [
        "REMARK   Martini 3-based coarse-grained representation",
        "REMARK   Generated by scripts/utilities/atomistic_to_martini_cg.py",
        "REMARK   Beads = heavy-atom centroids; B-factor column = bead charge",
    ]
    for b in beads:
        x, y, z = b["xyz"]
        name = b["name"]
        # atom-name field (cols 13-16): pad short names starting at col 14
        name_field = f"{name:<4}" if len(name) >= 4 else f" {name:<3}"
        element = _pdb_element(b["bead_type"], b["charge"])
        segid = _size_class(b["bead_type"])  # cols 73-76: size class for coloring
        lines.append(
            f"ATOM  {b['serial']:>5} {name_field} {b['resname']:>3} "
            f"{b['chain']:1}{b['resseq']:>4}    "
            f"{x:8.3f}{y:8.3f}{z:8.3f}"
            f"{1.00:6.2f}{b['charge']:6.2f}      {segid:<4}{element:>2}"
        )
    for a, c in bonds:
        lines.append(f"CONECT{a:>5}{c:>5}")
    lines.append("END")
    Path(out_path).write_text("\n".join(lines) + "\n")


def write_pymol_script(orig_path, cg_path, pml_path):
    """Write a .pml that overlays the CG beads on the atomistic structure."""
    orig_obj = Path(orig_path).stem
    cg_obj = Path(cg_path).stem
    script = f"""\
# Overlay Martini CG beads on the atomistic structure.
# Run with:  pymol {Path(pml_path).name}
load {Path(orig_path).name}, {orig_obj}
load {Path(cg_path).name}, {cg_obj}

# atomistic reference: opaque grey cartoon + sticks
hide everything, {orig_obj}
show cartoon, {orig_obj}
show sticks, {orig_obj}
color grey80, {orig_obj}
set cartoon_transparency, 0.0, {orig_obj}
set stick_radius, 0.12, {orig_obj}

# CG beads: spheres sized to the Martini bead radius, bonds as sticks
hide everything, {cg_obj}
show spheres, {cg_obj}
set sphere_scale, 0.45, {cg_obj}
set stick_radius, 0.3, {cg_obj}
# color by Martini bead size class (segID): tiny=red, small=blue, regular/large=yellow
color yellow, {cg_obj} and segi REGU
color blue,   {cg_obj} and segi SMAL
color red,    {cg_obj} and segi TINY

bg_color white
zoom
"""
    Path(pml_path).write_text(script)


def main():
    ap = argparse.ArgumentParser(
        description="Convert an atomistic .ent/.pdb to a Martini 3-based CG "
                    "structure for overlay in PyMOL.")
    ap.add_argument("input", help="Input atomistic structure (.ent or .pdb)")
    ap.add_argument("-o", "--output",
                    help="Output CG PDB path (default: <input>_cg.pdb)")
    ap.add_argument("--pymol", action="store_true",
                    help="Also write a .pml overlay script next to the output")
    args = ap.parse_args()

    in_path = Path(args.input)
    if not in_path.is_file():
        ap.error(f"Input file not found: {in_path}")
    out_path = Path(args.output) if args.output else in_path.with_name(in_path.stem + "_cg.pdb")

    beads, bb_by_chain = structure_to_beads(in_path)
    bonds = build_bonds(beads, bb_by_chain)
    write_cg_pdb(beads, bonds, out_path)

    n_res = len({(b["chain"], b["resseq"]) for b in beads})
    print(f"Wrote {len(beads)} beads ({n_res} residues, {len(bonds)} bonds) -> {out_path}")

    if args.pymol:
        pml_path = out_path.with_suffix(".pml")
        write_pymol_script(in_path, out_path, pml_path)
        print(f"Wrote PyMOL overlay script -> {pml_path}")
        print(f"  View with:  cd {out_path.parent}  &&  pymol {pml_path.name}")


if __name__ == "__main__":
    main()
