# View a CG multimer in a membrane (output of cg_md_simulation_for_membrane_protein.py)
# Usage:  pymol <md-output>.pdb scripts/view_cg_membrane.pml
# Works on any PDB written by the MD script (minimized, nvt, npt, early_prod,
# repacked, stability, stability_frame_NNNN).

python
from pymol import cmd
obj = cmd.get_object_list()[0]

# Protein beads have no CONECT records and are ~2.7-3.5 A apart (too far for
# PyMOL's automatic bonding), so bond BB-SC1-SC2-SC3 within each residue and
# BB-BB between consecutive residues of the same chain.
beads = []
cmd.iterate(f"{obj} and not resn LIP+NA+CL+SOL+W",
            "beads.append((chain, resv, name, index))", space={"beads": beads})
by_res = {}
for chain, resv, name, idx in beads:
    by_res.setdefault((chain, resv), {})[name] = idx
for (chain, resv), names in by_res.items():
    order = [n for n in ("BB", "SC1", "SC2", "SC3") if n in names]
    for a, b in zip(order, order[1:]):
        cmd.bond(f"{obj} and index {names[a]}", f"{obj} and index {names[b]}")
    nxt = by_res.get((chain, resv + 1))
    if "BB" in names and nxt and "BB" in nxt:
        cmd.bond(f"{obj} and index {names['BB']}", f"{obj} and index {nxt['BB']}")

# One object per protein chain, plus membrane and ions, so each can be toggled
# in the object panel.
for ch in cmd.get_chains(f"{obj} and not resn LIP+NA+CL+SOL+W"):
    cmd.create(f"chain_{ch}", f"{obj} and chain {ch} and not resn LIP+NA+CL+SOL+W")
cmd.create("membrane", f"{obj} and resn LIP")
cmd.create("ions", f"{obj} and resn NA+CL")
cmd.disable(obj)
python end

hide everything
bg_color white
set orthoscopic, 1

# protein: beads as spheres + bonds as sticks, one color per chain
show spheres, chain_*
show sticks, chain_*
set sphere_scale, 0.7, chain_*
set stick_radius, 0.4, chain_*
util.color_chains("chain_*")

# membrane: transparent grey lipid sticks, PO4 headgroup beads as spheres
show sticks, membrane
set stick_radius, 0.3, membrane
set stick_transparency, 0.7, membrane
color grey60, membrane
show spheres, membrane and name PO4
set sphere_scale, 0.5, membrane
set sphere_transparency, 0.2, membrane
color orange, membrane and name PO4

# ions hidden by default:  show spheres, ions

# periodic box, viewed edge-on with the membrane normal (Z) pointing up
show cell, membrane
reset
turn x, -90
zoom membrane or chain_*, 5
