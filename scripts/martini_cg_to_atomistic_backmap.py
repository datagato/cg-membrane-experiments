"""
Backmap a Martini 3-based coarse-grained (CG) protein structure back to an
all-atom (atomistic) structure, with an optional energy-minimization step to
remove steric clashes and idealize local geometry.

This is the *reverse* of scripts/tmp/atomistic_to_martini_cg_conversion.py.
Backmapping is an inherently under-determined problem: each CG bead is the
centroid of several heavy atoms (see MARTINI_RESIDUE_BEADS in
src/cg_membrane_experiments/constants.py), so the atomistic detail collapsed into a bead
cannot be recovered exactly. This script reconstructs a *plausible* atomistic
structure and then (optionally) relaxes it with a real force field:

  1. Backbone  - N, CA, C are rebuilt per residue from the BB-bead trace using
                 the idealized backbone internal coordinates in
                 BACKBONE_POSITIONS, oriented by the neighboring BB beads.
                 (CA is placed at the BB bead; BB is the centroid of N/CA/C/O,
                 so this is an approximation the later minimization cleans up.)
  2. Sidechains - grown from PDBFixer's residue templates (findMissingAtoms /
                 addMissingAtoms), which also adds the carbonyl O and terminal
                 OXT. Hydrogens are added by PDBFixer (reconstruction output) or
                 by the force field (minimization path).
  3. Minimize  - (optional, --minimize) AMBER + implicit-solvent energy
                 minimization in OpenMM to remove clashes and fix local
                 geometry, mirroring the atomistic model in
                 proteogram/v2/atomistic_nonbonded_forces.py.

A validation step re-derives CG beads from the rebuilt atomistic structure
(same centroid mapping the forward script uses) and reports the bead-position
RMSD against the input CG beads, quantifying backmapping fidelity.

SIDECHAIN CONFORMATIONS:
  - By default, sidechains come from PDBFixer's template rotamer, which ignores
    the residue's SC bead positions. The SC beads carry the refined
    sidechain-orientation signal from the CG simulation, so `--sc-guided` adds a
    restrained-minimization pass: a CustomCentroidBondForce pulls each bead's
    heavy-atom *centroid* toward its CG bead (the same grouping the forward
    mapping uses), rotating sidechains onto their refined SC positions, then
    softens the restraint to relax force-field strain. This is a restrained-EM
    prototype (cf. Wassenaar et al. backward/initram), not a rotamer-library
    search; it markedly lowers SC-bead RMSD versus the unguided path.

KNOWN APPROXIMATIONS (documented; candidates for a future, higher-fidelity
version):
  - The BB-BB CG bond (~0.35 nm) is slightly shorter than a real CA-CA spacing
    (~0.38 nm), so the reconstructed chain is marginally compressed before
    minimization.
  - Gaps (non-consecutive residue numbers) are treated as separate backbone
    segments; missing residues are NOT filled in.

Usage:
    python scripts/martini_cg_to_atomistic_backmap.py input_cg.pdb
    python scripts/martini_cg_to_atomistic_backmap.py input_cg.pdb --minimize
    python scripts/martini_cg_to_atomistic_backmap.py input_cg.pdb \
        -o rebuilt.pdb --minimize --min-iterations 1000

Bead mapping and force-field basis: see docs/martini_md_simulation_methodology.md.
"""
import argparse
import math
import warnings
from pathlib import Path

import numpy as np
from Bio.PDB.PDBParser import PDBParser
from Bio.PDB.PDBExceptions import PDBConstructionWarning

from cg_membrane_experiments.constants import (
    MARTINI_RESIDUE_BEADS,
    BACKBONE_POSITIONS,
    MODIFIED_RESIDUES_TO_STANDARD,
)

warnings.filterwarnings("ignore", category=PDBConstructionWarning)

# Backbone atoms we reconstruct explicitly; PDBFixer adds O, OXT, sidechain, H.
_RECON_BACKBONE = ("N", "CA", "C")


# --------------------------------------------------------------------------- #
# 1. Parse the CG structure into per-residue bead sets
# --------------------------------------------------------------------------- #
def parse_cg_structure(cg_path):
    """Parse a CG PDB into an ordered list of residue dicts.

    Each residue: {chain, resseq, resname, beads}, where beads maps a bead
    label (BB, SC1, SC2, SC3) to its xyz position in Angstrom. Non-protein
    residues (water W beads, ions, anything not in MARTINI_RESIDUE_BEADS) are
    skipped, as are residues missing a BB bead (backbone anchor).
    """
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("cg", str(cg_path))
    model = structure[0]

    residues = []
    for chain in model.get_chains():
        chain_id = chain.id if chain.id.strip() else "A"
        for res in chain:
            raw = res.resname.strip()
            resname = MODIFIED_RESIDUES_TO_STANDARD.get(raw, raw)
            if resname not in MARTINI_RESIDUE_BEADS:
                continue  # solvent, ions, non-standard — not part of the protein
            beads = {atom.name.strip(): np.asarray(atom.coord, dtype=np.float64)
                     for atom in res}
            if "BB" not in beads:
                continue  # no backbone anchor — cannot place this residue
            residues.append({
                "chain": chain_id,
                "resseq": res.id[1],
                "resname": resname,
                "beads": beads,
            })

    if not residues:
        raise ValueError(
            f"No CG protein residues (with BB beads) found in {cg_path}. "
            f"Is this a Martini CG structure with BB/SC bead names?")
    return residues


# --------------------------------------------------------------------------- #
# 2. Reconstruct backbone N/CA/C from the BB-bead trace
# --------------------------------------------------------------------------- #
def _local_frame(ca, prev_ca, next_ca):
    """Orthonormal frame (u, v, w) placing the idealized residue in space.

    +u  (local +x) points along CA(i)->CA(i+1), matching the idealized C atom
        which sits at +x in BACKBONE_POSITIONS.
    +v  (local +y) lies in the CA(i-1)/CA(i)/CA(i+1) plane toward CA(i-1),
        matching the idealized N atom which sits at +x-offset,+y.
    +w  = u x v.
    Falls back to arbitrary orthogonal axes at chain termini / isolated beads.
    """
    def _norm(x):
        n = np.linalg.norm(x)
        return x / n if n > 1e-6 else None

    # Primary axis: toward the next CA (or back from the previous one).
    u = None
    if next_ca is not None:
        u = _norm(next_ca - ca)
    if u is None and prev_ca is not None:
        u = _norm(ca - prev_ca)
    if u is None:
        u = np.array([1.0, 0.0, 0.0])

    # In-plane reference: toward the previous CA (or away from the next one).
    ref = None
    if prev_ca is not None:
        ref = prev_ca - ca
    elif next_ca is not None:
        ref = -(next_ca - ca)
    if ref is None:
        ref = np.array([0.0, 1.0, 0.0])

    v = _norm(ref - np.dot(ref, u) * u)
    if v is None:  # ref parallel to u — pick any orthogonal direction
        trial = np.array([0.0, 0.0, 1.0])
        v = _norm(trial - np.dot(trial, u) * u)
        if v is None:
            v = _norm(np.array([0.0, 1.0, 0.0]) - u[1] * u)
    w = np.cross(u, v)
    return u, v, w


def reconstruct_backbone(residues):
    """Add reconstructed N/CA/C coordinates (Angstrom) to each residue dict.

    Residues are grouped per chain and split into segments of consecutive
    residue numbers; frames use in-segment neighbors so chain gaps don't create
    spurious geometry. Populates residue['atoms'] = {name: xyz}.
    """
    # Group indices by chain, preserving input order.
    by_chain = {}
    for idx, res in enumerate(residues):
        by_chain.setdefault(res["chain"], []).append(idx)

    for chain_idx in by_chain.values():
        # Sort by residue number and split into consecutive segments.
        chain_idx.sort(key=lambda i: residues[i]["resseq"])
        segments, seg = [], [chain_idx[0]]
        for i in chain_idx[1:]:
            if residues[i]["resseq"] == residues[seg[-1]]["resseq"] + 1:
                seg.append(i)
            else:
                segments.append(seg)
                seg = [i]
        segments.append(seg)

        for seg in segments:
            for pos, ridx in enumerate(seg):
                res = residues[ridx]
                ca = res["beads"]["BB"]
                prev_ca = residues[seg[pos - 1]]["beads"]["BB"] if pos > 0 else None
                next_ca = (residues[seg[pos + 1]]["beads"]["BB"]
                           if pos < len(seg) - 1 else None)
                u, v, w = _local_frame(ca, prev_ca, next_ca)

                ideal = BACKBONE_POSITIONS.get(res["resname"],
                                               BACKBONE_POSITIONS["UNK"])
                ideal = dict((name, np.asarray(xyz, dtype=np.float64))
                             for name, xyz in ideal)
                atoms = {}
                for name in _RECON_BACKBONE:
                    lx, ly, lz = ideal[name]
                    atoms[name] = ca + lx * u + ly * v + lz * w
                # Anchor CA exactly at the BB bead (local origin).
                atoms["CA"] = ca.copy()
                res["atoms"] = atoms
    return residues


# --------------------------------------------------------------------------- #
# 3. Write the reconstructed backbone as a PDB for PDBFixer
# --------------------------------------------------------------------------- #
def write_backbone_pdb(residues, out_path):
    """Write reconstructed N/CA/C atoms to a backbone-only PDB (TER per chain)."""
    lines = [
        "REMARK   Reconstructed backbone (N/CA/C) from Martini CG BB-bead trace",
        "REMARK   Generated by scripts/tmp/martini_cg_to_atomistic_backmap.py",
    ]
    serial = 0
    prev_chain = None
    # Preserve chain grouping; residues already carry chain + resseq.
    ordered = sorted(range(len(residues)),
                     key=lambda i: (residues[i]["chain"], residues[i]["resseq"]))
    for ridx in ordered:
        res = residues[ridx]
        if prev_chain is not None and res["chain"] != prev_chain:
            serial += 1
            lines.append(f"TER   {serial:>5}")
        prev_chain = res["chain"]
        for name in _RECON_BACKBONE:
            serial += 1
            x, y, z = res["atoms"][name]
            element = name[0]  # N / C
            name_field = f" {name:<3}"
            lines.append(
                f"ATOM  {serial:>5} {name_field} {res['resname']:>3} "
                f"{res['chain']:1}{res['resseq']:>4}    "
                f"{x:8.3f}{y:8.3f}{z:8.3f}{1.00:6.2f}{0.00:6.2f}"
                f"          {element:>2}")
    lines.append(f"TER   {serial + 1:>5}")
    lines.append("END")
    Path(out_path).write_text("\n".join(lines) + "\n")


# --------------------------------------------------------------------------- #
# 4. Grow sidechains + hydrogens with PDBFixer
# --------------------------------------------------------------------------- #
def grow_atoms(backbone_pdb, add_hydrogens=True, ph=7.0):
    """Run PDBFixer to build missing sidechain heavy atoms (+ O, OXT, H).

    Returns (topology, positions) from the completed structure. Missing residues
    are explicitly disabled so chain gaps are not filled with new residues.
    """
    from pdbfixer import PDBFixer

    fixer = PDBFixer(filename=str(backbone_pdb))
    fixer.findMissingResidues()
    fixer.missingResidues = {}          # do not graft in gap residues
    fixer.findNonstandardResidues()
    fixer.replaceNonstandardResidues()
    fixer.findMissingAtoms()            # every sidechain atom + O/OXT is "missing"
    fixer.addMissingAtoms()
    if add_hydrogens:
        fixer.addMissingHydrogens(ph)
    return fixer.topology, fixer.positions


# --------------------------------------------------------------------------- #
# 4b. Secondary-structure assignment (for SS-preserving restraints)
# --------------------------------------------------------------------------- #
# Ideal backbone dihedrals (radians) per SS class. Helix and generic beta.
_SS_PHI_PSI = {
    "H": (math.radians(-57.0), math.radians(-47.0)),   # alpha-helix
    "E": (math.radians(-120.0), math.radians(130.0)),  # beta-strand (generic)
}


def _classify_phi_psi(phi, psi):
    """Classify a (phi, psi) pair (radians) as H (helix), E (strand), or C."""
    p, s = math.degrees(phi), math.degrees(psi)
    if -100.0 <= p <= -30.0 and -80.0 <= s <= -5.0:
        return "H"
    if -170.0 <= p <= -40.0 and 90.0 <= s <= 170.0:
        return "E"
    return "C"


def assign_ss_from_reference(residues, ref_path):
    """Attach res['ss'] (H/E/C) using phi/psi basins of a reference structure.

    Keyed by (chain, resseq), so the reference is typically the pre-CG atomistic
    structure. No DSSP binary required (uses BioPython phi/psi). Returns the
    number of residues assigned H or E.
    """
    from Bio.PDB.Polypeptide import PPBuilder

    if str(ref_path).lower().endswith((".cif", ".mmcif")):
        from Bio.PDB.MMCIFParser import MMCIFParser
        parser = MMCIFParser(QUIET=True)
    else:
        parser = PDBParser(QUIET=True)
    structure = parser.get_structure("ref", str(ref_path))
    ss_map = {}
    for pp in PPBuilder().build_peptides(structure[0]):
        for res, (phi, psi) in zip(pp, pp.get_phi_psi_list()):
            if phi is None or psi is None:
                continue
            chain = res.parent.id if res.parent.id.strip() else "A"
            ss_map[(chain, res.id[1])] = _classify_phi_psi(phi, psi)

    n_struct = 0
    for res in residues:
        res["ss"] = ss_map.get((res["chain"], res["resseq"]), "C")
        n_struct += res["ss"] in ("H", "E")
    return n_struct


def assign_ss_from_trace(residues, min_run=3):
    """Attach res['ss'] (H/C) heuristically from the BB-bead trace geometry.

    Fallback when no reference is available. Uses the BB-BB-BB pseudo-angle
    (helix ~96 deg, coil ~127 deg, sheet ~134 deg — the same values the CG model
    assigns per DSSP) with run-length smoothing so isolated residues aren't
    called helix. Sheets are NOT reliably separable from coil by angle alone, so
    this only marks helices. Prefer assign_ss_from_reference when possible.
    Returns the number of residues marked H.
    """
    by_chain = {}
    for idx, res in enumerate(residues):
        by_chain.setdefault(res["chain"], []).append(idx)

    for res in residues:
        res["ss"] = "C"

    for chain_idx in by_chain.values():
        chain_idx.sort(key=lambda i: residues[i]["resseq"])
        # Split into consecutive-resseq segments.
        segments, seg = [], [chain_idx[0]]
        for i in chain_idx[1:]:
            if residues[i]["resseq"] == residues[seg[-1]]["resseq"] + 1:
                seg.append(i)
            else:
                segments.append(seg)
                seg = [i]
        segments.append(seg)

        for seg in segments:
            raw = [False] * len(seg)
            for p in range(1, len(seg) - 1):
                a = residues[seg[p - 1]]["beads"]["BB"]
                b = residues[seg[p]]["beads"]["BB"]
                c = residues[seg[p + 1]]["beads"]["BB"]
                v1, v2 = a - b, c - b
                cosang = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-9)
                ang = math.degrees(math.acos(np.clip(cosang, -1.0, 1.0)))
                raw[p] = ang < 112.0  # closer to helix (96) than coil (127)
            # Run-length filter: keep only helical runs of >= min_run.
            p = 0
            while p < len(seg):
                if raw[p]:
                    q = p
                    while q < len(seg) and raw[q]:
                        q += 1
                    if q - p >= min_run:
                        for r in range(p, q):
                            residues[seg[r]]["ss"] = "H"
                    p = q
                else:
                    p += 1
    return sum(r["ss"] == "H" for r in residues)


# --------------------------------------------------------------------------- #
# 5. Optional energy minimization (OpenMM + AMBER, implicit solvent)
# --------------------------------------------------------------------------- #
def minimize(topology, positions, forcefield_files, max_iterations, tolerance_kj):
    """Energy-minimize the all-atom structure to remove clashes.

    Rebuilds hydrogens consistent with the force field via Modeller, then runs
    L-BFGS minimization in vacuum-with-implicit-solvent (NoCutoff). Returns
    (topology, positions) for the relaxed structure.
    """
    from openmm.app import ForceField, Modeller, Simulation, NoCutoff, HBonds
    from openmm import LangevinMiddleIntegrator
    from openmm.unit import (kelvin, picosecond, femtoseconds, nanometer,
                             kilojoule_per_mole)

    forcefield = ForceField(*forcefield_files)
    modeller = Modeller(topology, positions)
    # Replace any existing H with force-field-consistent hydrogens.
    modeller.delete([a for a in modeller.topology.atoms() if a.element is not None
                     and a.element.symbol == "H"])
    modeller.addHydrogens(forcefield, pH=7.0)

    system = forcefield.createSystem(
        modeller.topology, nonbondedMethod=NoCutoff, constraints=HBonds)
    integrator = LangevinMiddleIntegrator(
        300 * kelvin, 1.0 / picosecond, 2.0 * femtoseconds)
    simulation = Simulation(modeller.topology, system, integrator)
    simulation.context.setPositions(modeller.positions)
    # OpenMM's minimizer tolerance is a force (gradient) tolerance, kJ/mol/nm.
    simulation.minimizeEnergy(
        maxIterations=max_iterations,
        tolerance=tolerance_kj * kilojoule_per_mole / nanometer)
    state = simulation.context.getState(getPositions=True, getEnergy=True)
    energy = state.getPotentialEnergy().value_in_unit(kilojoule_per_mole)
    print(f"  Minimized potential energy: {energy:,.1f} kJ/mol")
    return modeller.topology, state.getPositions()


# --------------------------------------------------------------------------- #
# 5b. SC-bead-guided restrained minimization
# --------------------------------------------------------------------------- #
def _collect_restraint_groups(topology, residues, restrain_bb):
    """Map each CG bead to (target_xyz_nm, [atom indices]) in the given topology.

    Groups a bead's heavy atoms (per MARTINI_RESIDUE_BEADS) by name so their
    *geometric* centroid can be restrained toward the CG bead position — the
    same atom grouping the forward CG mapping uses. Beads with no matching atoms
    present (or BB when restrain_bb is False) are skipped.
    """
    idx_by_res = {}
    for atom in topology.atoms():
        res = atom.residue
        chain_id = res.chain.id if res.chain.id.strip() else "A"
        idx_by_res.setdefault((chain_id, int(res.id)), {})[atom.name] = atom.index

    groups = []
    for res in residues:
        amap = idx_by_res.get((res["chain"], res["resseq"]))
        if not amap:
            continue
        for _bt, label, _q, atom_names in MARTINI_RESIDUE_BEADS[res["resname"]]:
            if label not in res["beads"]:
                continue
            if label == "BB" and not restrain_bb:
                continue
            indices = [amap[n] for n in atom_names if n in amap]
            if not indices:
                continue
            groups.append((res["beads"][label] * 0.1, indices, label))  # A -> nm
    return groups


def _add_ss_dihedral_restraints(system, topology, residues, k_ss):
    """Restrain backbone phi/psi of H/E residues toward ideal SS dihedrals.

    Adds one periodicity-safe CustomTorsionForce (k_ss*(1-cos(theta-theta0)))
    covering both phi (C[i-1]-N-CA-C) and psi (N-CA-C-N[i+1]); each torsion
    carries its own target. Requires res['ss'] to be set. Returns the number of
    torsion restraints added.
    """
    from openmm import CustomTorsionForce

    idx = {}
    for atom in topology.atoms():
        r = atom.residue
        chain = r.chain.id if r.chain.id.strip() else "A"
        idx.setdefault((chain, int(r.id)), {})[atom.name] = atom.index

    force = CustomTorsionForce("k_ss*(1 - cos(theta - theta0))")
    force.addGlobalParameter("k_ss", k_ss)
    force.addPerTorsionParameter("theta0")
    force.setUsesPeriodicBoundaryConditions(False)

    count = 0
    for res in residues:
        ss = res.get("ss", "C")
        if ss not in _SS_PHI_PSI:
            continue
        phi0, psi0 = _SS_PHI_PSI[ss]
        cur = idx.get((res["chain"], res["resseq"]))
        prev = idx.get((res["chain"], res["resseq"] - 1))
        nxt = idx.get((res["chain"], res["resseq"] + 1))
        if not cur or not all(n in cur for n in ("N", "CA", "C")):
            continue
        if prev and "C" in prev:  # phi
            force.addTorsion(prev["C"], cur["N"], cur["CA"], cur["C"], [phi0])
            count += 1
        if nxt and "N" in nxt:    # psi
            force.addTorsion(cur["N"], cur["CA"], cur["C"], nxt["N"], [psi0])
            count += 1
    if count:
        system.addForce(force)
    return count


def minimize_sc_guided(topology, positions, residues, forcefield_files,
                       restraint_k, max_iterations, tolerance_kj,
                       restrain_bb=True, final_k_frac=0.1,
                       preserve_ss=False, ss_k=200.0):
    """Minimize with harmonic restraints pulling bead-atom centroids to their
    CG bead targets, so rebuilt sidechains honor the refined SC bead positions.

    Two phases: (1) full restraint strength rotates sidechains onto their SC
    beads; (2) reduced restraint (final_k_frac x k) relieves force-field strain
    and removes residual clashes while keeping sidechains near their targets.
    Restraints act on each bead's *geometric* atom-group centroid via a
    CustomCentroidBondForce (unit atom weights). Returns (topology, positions).
    """
    from openmm.app import ForceField, Modeller, Simulation, NoCutoff, HBonds
    from openmm import LangevinMiddleIntegrator, CustomCentroidBondForce
    from openmm.unit import (kelvin, picosecond, femtoseconds, nanometer,
                             kilojoule_per_mole)

    forcefield = ForceField(*forcefield_files)
    modeller = Modeller(topology, positions)
    modeller.delete([a for a in modeller.topology.atoms() if a.element is not None
                     and a.element.symbol == "H"])
    modeller.addHydrogens(forcefield, pH=7.0)

    system = forcefield.createSystem(
        modeller.topology, nonbondedMethod=NoCutoff, constraints=HBonds)

    # Centroid restraints toward CG bead targets (indices from the H-added topology).
    groups = _collect_restraint_groups(modeller.topology, residues, restrain_bb)
    restraint = CustomCentroidBondForce(
        1, "0.5*k_sc*((x1-x0)^2 + (y1-y0)^2 + (z1-z0)^2)")
    restraint.addGlobalParameter("k_sc", restraint_k)
    for p in ("x0", "y0", "z0"):
        restraint.addPerBondParameter(p)
    restraint.setUsesPeriodicBoundaryConditions(False)
    n_sc = 0
    for target_nm, indices, label in groups:
        gid = restraint.addGroup([int(i) for i in indices], [1.0] * len(indices))
        restraint.addBond([gid], [float(target_nm[0]), float(target_nm[1]),
                                  float(target_nm[2])])
        n_sc += (label != "BB")
    system.addForce(restraint)

    # Secondary-structure phi/psi restraints (kept at full strength through both
    # phases so helices/strands survive the geometry-relaxation phase).
    n_ss = 0
    if preserve_ss:
        n_ss = _add_ss_dihedral_restraints(
            system, modeller.topology, residues, ss_k)

    integrator = LangevinMiddleIntegrator(
        300 * kelvin, 1.0 / picosecond, 2.0 * femtoseconds)
    simulation = Simulation(modeller.topology, system, integrator)
    simulation.context.setPositions(modeller.positions)
    tol = tolerance_kj * kilojoule_per_mole / nanometer

    # Phase 1: strong restraints — snap sidechains onto their SC beads.
    simulation.minimizeEnergy(maxIterations=max_iterations, tolerance=tol)
    # Phase 2: soften restraints — relax geometry / clashes, keep guidance.
    simulation.context.setParameter("k_sc", restraint_k * final_k_frac)
    simulation.minimizeEnergy(maxIterations=max_iterations, tolerance=tol)

    state = simulation.context.getState(getPositions=True, getEnergy=True)
    energy = state.getPotentialEnergy().value_in_unit(kilojoule_per_mole)

    # Restraint-free single-point energy at the final coordinates. The energy
    # above includes the (positive) restraint penalties and is only comparable
    # within one run; zeroing the restraint force constants and re-evaluating
    # gives the pure force-field energy — the interpretable measure of whether
    # the structure is clash-free and well-packed. Positions are unchanged.
    simulation.context.setParameter("k_sc", 0.0)
    if preserve_ss and n_ss > 0:
        simulation.context.setParameter("k_ss", 0.0)
    ff_energy = (simulation.context.getState(getEnergy=True)
                 .getPotentialEnergy().value_in_unit(kilojoule_per_mole))

    print(f"  SC-guided restraints: {len(groups)} beads "
          f"({n_sc} sidechain), k = {restraint_k:g} -> "
          f"{restraint_k * final_k_frac:g} kJ/mol/nm^2")
    if preserve_ss:
        print(f"  SS phi/psi restraints: {n_ss} torsions, k_ss = {ss_k:g} kJ/mol")
    print(f"  Minimized potential energy (incl. residual restraints): "
          f"{energy:,.1f} kJ/mol")
    n_res = max(len(residues), 1)
    print(f"  Force-field energy (restraints removed): {ff_energy:,.1f} kJ/mol "
          f"({ff_energy / n_res:.1f} per residue)")
    return modeller.topology, state.getPositions()


# --------------------------------------------------------------------------- #
# 6. Validation: re-derive CG beads and report RMSD to the input
# --------------------------------------------------------------------------- #
def _bead_centroid(res_atoms, atom_names):
    """Centroid (A) of atom_names present in res_atoms, CA/N/C fallback."""
    coords = [res_atoms[n] for n in atom_names if n in res_atoms]
    if coords:
        return np.mean(coords, axis=0)
    for fb in ("CA", "N", "C"):
        if fb in res_atoms:
            return res_atoms[fb]
    return None


def validate_backmap(residues, topology, positions):
    """Report bead-position RMSD between input CG beads and beads re-derived
    from the rebuilt atomistic structure (same centroid mapping as the forward
    script). Returns (bb_rmsd, all_rmsd) in Angstrom."""
    from openmm.unit import angstrom

    pos = np.asarray(positions.value_in_unit(angstrom), dtype=np.float64)
    # Collect rebuilt atoms per (chain, resseq).
    rebuilt = {}
    for atom in topology.atoms():
        res = atom.residue
        chain_id = res.chain.id if res.chain.id.strip() else "A"
        key = (chain_id, int(res.id))
        rebuilt.setdefault(key, {})[atom.name] = pos[atom.index]

    bb_sq, bb_n, all_sq, all_n = 0.0, 0, 0.0, 0
    for res in residues:
        key = (res["chain"], res["resseq"])
        atoms = rebuilt.get(key)
        if not atoms:
            continue
        for bead_type, label, _charge, atom_names in MARTINI_RESIDUE_BEADS[res["resname"]]:
            if label not in res["beads"]:
                continue
            new = _bead_centroid(atoms, atom_names)
            if new is None:
                continue
            d2 = float(np.sum((new - res["beads"][label]) ** 2))
            all_sq += d2
            all_n += 1
            if label == "BB":
                bb_sq += d2
                bb_n += 1
    bb_rmsd = np.sqrt(bb_sq / bb_n) if bb_n else float("nan")
    all_rmsd = np.sqrt(all_sq / all_n) if all_n else float("nan")
    return bb_rmsd, all_rmsd


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #
def write_atomistic_pdb(topology, positions, out_path):
    from openmm.app import PDBFile
    with open(out_path, "w") as fh:
        PDBFile.writeFile(topology, positions, fh, keepIds=True)


def main():
    ap = argparse.ArgumentParser(
        description="Backmap a Martini CG protein structure to all-atom, with "
                    "optional energy minimization.")
    ap.add_argument("input", help="Input CG structure (.pdb) with BB/SC beads")
    ap.add_argument("-o", "--output",
                    help="Output atomistic PDB (default: <input>_atomistic.pdb; "
                         "with --minimize also writes <stem>_min.pdb)")
    ap.add_argument("--minimize", action="store_true",
                    help="Energy-minimize the rebuilt structure (OpenMM + AMBER)")
    ap.add_argument("--sc-guided", action="store_true",
                    help="SC-bead-guided restrained minimization: bias sidechain "
                         "atom centroids toward their CG SC beads, then relax "
                         "(implies --minimize)")
    ap.add_argument("--sc-restraint-k", type=float, default=20000.0,
                    help="Restraint force constant for --sc-guided, kJ/mol/nm^2 "
                         "(default 20000; softened to 10%% in the relax phase)")
    ap.add_argument("--no-restrain-bb", action="store_true",
                    help="With --sc-guided, restrain only SC beads (leave the "
                         "backbone free during guided minimization)")
    ap.add_argument("--preserve-ss", action="store_true",
                    help="Restrain backbone phi/psi of helix/strand residues to "
                         "ideal SS dihedrals during minimization (implies "
                         "--sc-guided). Prevents helices unraveling on backmap.")
    ap.add_argument("--ss-ref",
                    help="Reference structure (e.g. the pre-CG atomistic PDB) to "
                         "assign secondary structure from, keyed by chain+resseq. "
                         "If omitted, SS is detected from the BB-bead trace "
                         "(helices only; less reliable).")
    ap.add_argument("--ss-k", type=float, default=200.0,
                    help="Force constant for phi/psi SS restraints, kJ/mol "
                         "(default 200)")
    ap.add_argument("--forcefield", nargs="+",
                    default=["amber14-all.xml", "implicit/gbn2.xml"],
                    help="OpenMM force field XMLs for minimization "
                         "(default: amber14-all.xml implicit/gbn2.xml)")
    ap.add_argument("--min-iterations", type=int, default=0,
                    help="Max minimization iterations per phase "
                         "(0 = until converged; default 0)")
    ap.add_argument("--min-tolerance", type=float, default=10.0,
                    help="Minimization force tolerance in kJ/mol/nm (default 10)")
    ap.add_argument("--ph", type=float, default=7.0,
                    help="pH for hydrogen placement (default 7.0)")
    args = ap.parse_args()

    in_path = Path(args.input)
    if not in_path.is_file():
        ap.error(f"Input file not found: {in_path}")
    out_path = (Path(args.output) if args.output
                else in_path.with_name(in_path.stem + "_atomistic.pdb"))

    print(f"Reading CG structure: {in_path}")
    residues = parse_cg_structure(in_path)
    n_res = len(residues)
    n_beads = sum(len(r["beads"]) for r in residues)
    print(f"  {n_res} protein residues, {n_beads} beads")

    reconstruct_backbone(residues)

    # Intermediate backbone PDB next to the output (kept for inspection).
    backbone_pdb = out_path.with_name(out_path.stem + "_backbone.pdb")
    write_backbone_pdb(residues, backbone_pdb)
    print(f"  Wrote reconstructed backbone -> {backbone_pdb}")

    print("Growing sidechains + hydrogens with PDBFixer...")
    topology, positions = grow_atoms(backbone_pdb, add_hydrogens=True, ph=args.ph)
    n_atoms = topology.getNumAtoms()
    write_atomistic_pdb(topology, positions, out_path)
    print(f"  Wrote atomistic structure ({n_atoms} atoms) -> {out_path}")

    bb_rmsd, all_rmsd = validate_backmap(residues, topology, positions)
    print(f"  Backmap fidelity (pre-min): BB-bead RMSD = {bb_rmsd:.3f} A, "
          f"all-bead RMSD = {all_rmsd:.3f} A")

    if args.preserve_ss or args.sc_guided:
        if args.preserve_ss:
            if args.ss_ref:
                n_ss = assign_ss_from_reference(residues, args.ss_ref)
                print(f"  Secondary structure from {args.ss_ref}: "
                      f"{n_ss} helix/strand residues")
            else:
                n_ss = assign_ss_from_trace(residues)
                print(f"  Secondary structure from BB-trace geometry: "
                      f"{n_ss} helix residues (no reference given)")
        print("SC-bead-guided restrained minimization (OpenMM)...")
        min_top, min_pos = minimize_sc_guided(
            topology, positions, residues, args.forcefield,
            restraint_k=args.sc_restraint_k,
            max_iterations=args.min_iterations,
            tolerance_kj=args.min_tolerance,
            restrain_bb=not args.no_restrain_bb,
            preserve_ss=args.preserve_ss,
            ss_k=args.ss_k)
        min_out = out_path.with_name(out_path.stem + "_scguided.pdb")
        write_atomistic_pdb(min_top, min_pos, min_out)
        print(f"  Wrote SC-guided structure -> {min_out}")
        bb_rmsd_m, all_rmsd_m = validate_backmap(residues, min_top, min_pos)
        print(f"  Backmap fidelity (SC-guided): BB-bead RMSD = {bb_rmsd_m:.3f} A, "
              f"all-bead RMSD = {all_rmsd_m:.3f} A")
    elif args.minimize:
        print("Energy-minimizing (OpenMM)...")
        min_top, min_pos = minimize(
            topology, positions, args.forcefield,
            args.min_iterations, args.min_tolerance)
        min_out = out_path.with_name(out_path.stem + "_min.pdb")
        write_atomistic_pdb(min_top, min_pos, min_out)
        print(f"  Wrote minimized structure -> {min_out}")
        bb_rmsd_m, all_rmsd_m = validate_backmap(residues, min_top, min_pos)
        print(f"  Backmap fidelity (post-min): BB-bead RMSD = {bb_rmsd_m:.3f} A, "
              f"all-bead RMSD = {all_rmsd_m:.3f} A")

    print("Done.")


if __name__ == "__main__":
    main()
