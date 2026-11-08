"""Standalone CG-MD simulation script for membrane proteins using OpenMM + Martini 3.

Coarse-grained (CG) molecular dynamics for membrane proteins, based on the
project's MartiniNonBondedForceModel (martini_nonbonded_forces.py) and extended
to include a POPC/DPPC/POPE lipid bilayer for membrane embedding.

Pipeline:
  1. PDB → multi-bead CG protein   (1–4 beads/residue, same scheme as
     MartiniNonBondedForceModel; all chains, chain/gap-aware bonds and angles,
     disulfide bridges from SSBOND records or SG–SG distances)
  2. Lipid bilayer construction     (Martini 3-inspired POPC/DPPC/POPE, 12 beads/lipid;
     XY grid with protein footprint excluded)
  3. CG water solvation             (W beads, ~4 H₂O each; cubic grid outside bilayer)
  4. Ion placement                  (Na⁺/Cl⁻, net-charge neutralisation + 0.15 M NaCl)
  5. Energy minimisation            (up to 3 attempts with jitter on NaN)
  6. NVT equilibration  (~310 K)    (barostat off; moderate BB restraints on TM helices;
     optional intra-helix ENM; 10 fs timestep)
  7. NPT equilibration (semi-iso)   (MonteCarloMembraneBarostat, XY isotropic, Z free;
     same restraints updated each chunk for box rescaling)
  8. Early production               (20 fs; gradual BB restraint release k→0 in 5 stages;
     intra-helix ENM kept if active)
  9. Repacking — simulated annealing (NVT, 310 ↔ sa_t_high K; multiple replicas;
     fixed secondary structure; ENM springs kept; best replica selected)
 10. Stability production           (constant T/P, semi-isotropic; 1–5 µs;
     PDB/DCD output)

Orientation: the input protein must have its transmembrane helix/helices along the
Z axis. Either pre-orient it externally (OPM/PPM, CHARMM-GUI Membrane Builder, or
PPM2 — most accurate, they account for hydrophobicity and tilt), OR pass the TM
helix residue ranges via --tm-residues and the script will align the helix axis to
Z and center the TM region at the bilayer midplane (pure Python; good for clear TM
bundles). The script then builds the bilayer at z=0.

Multimers / complexes (e.g. a TCR–CD3 assembly): every chain in the input is
converted, and no bond or backbone angle is ever formed across a chain boundary
or an unresolved gap. Disulfide bridges — including inter-chain ones, such as the
TCR α/β connecting-peptide bridge — are built automatically from SSBOND records
or SG–SG distances (--no-disulfides to skip). Output PDBs keep the original
chain ids, residue numbers and insertion codes.

What is NOT automatic for a complex: nothing but the CG non-bonded terms holds
two chains together once the positional restraints are released in step 8, and
the intra-helix ENM does not stabilise all-β Ig folds. For an antibody/TCR-like
ectodomain use --enm-sheet (cross-strand springs inside each chain) and, if the
assembly must stay intact for the whole run, --enm-interchain (interface springs,
which pin interfaces at their input geometry — so not for studying association).
List the TM helix of every subunit in --tm-residues.

Usage:
    python cg_md_simulation_for_membrane_protein.py my_protein.pdb
    python cg_md_simulation_for_membrane_protein.py my_protein.pdb --tm-residues A:34-58,A:70-92
    python cg_md_simulation_for_membrane_protein.py my_protein.pdb --lipid DPPC --gpu
    python cg_md_simulation_for_membrane_protein.py my_protein.pdb --npt-steps 100000 --save-dcd
    python cg_md_simulation_for_membrane_protein.py tcr_cd3.pdb \
        --tm-residues A:207-232,B:249-274,D:106-127,E:112-133 \
        --use-enm --enm-sheet --enm-interchain
"""

import argparse
import gc
import math
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
from Bio.PDB.PDBParser import PDBParser
from Bio.PDB.MMCIFParser import MMCIFParser

from openmm.app import (
    DCDReporter,
    Element,
    Modeller,
    PDBFile,
    Simulation,
    StateDataReporter,
    Topology,
)
from openmm import (
    CustomBondForce,
    CustomExternalForce,
    CustomNonbondedForce,
    Discrete2DFunction,
    HarmonicAngleForce,
    HarmonicBondForce,
    LangevinMiddleIntegrator,
    MonteCarloBarostat,
    MonteCarloMembraneBarostat,
    Platform,
    System,
    Vec3,
)
from openmm.unit import (
    bar,
    dalton,
    femtoseconds,
    kelvin,
    kilojoules_per_mole,
    nanometer,
    nanometers,
    picosecond,
    radian,
)


# ---------------------------------------------------------------------------
# Embedded Martini 3 constants (from proteogram/common/constants.py)
# ---------------------------------------------------------------------------

_COULOMB_K = 138.935456 / 15.0    # kJ·nm/(mol·e²), ε_r = 15
_CUTOFF_NM = 1.1                   # nm — LJ and RF cutoff
_EPS_WATER = 80.0
_EPS_PROT  = 15.0
_RF_K = (_EPS_WATER - _EPS_PROT) / (2 * _EPS_WATER + _EPS_PROT) / _CUTOFF_NM ** 3
_RF_C = (3 * _EPS_WATER)          / (2 * _EPS_WATER + _EPS_PROT) / _CUTOFF_NM

_BEAD_MASS_DA = 72.0
_WATER_MASS   = 72.0
_ION_MASS_NA  = 22.99
_ION_MASS_CL  = 35.45
_SALT_CONC_M  = 0.15

# Bead type → (sigma nm, epsilon kJ/mol)
_BEAD_TYPE_PARAMS = {
    'BB':  (0.47, 4.06),
    'C':   (0.47, 3.39),
    'SC':  (0.41, 2.35),
    'N':   (0.47, 3.52),
    'Q':   (0.47, 5.95),
    'TC':  (0.34, 1.51),
    'W':   (0.47, 4.65),  # W self-ε from ITP (matches _PAIR_EPS W–W)
    'ION': (0.354, 1.18),
}

_BEAD_TYPE_INDEX = {
    'BB': 0, 'C': 1, 'SC': 2, 'N': 3, 'Q': 4, 'TC': 5, 'W': 6,
    'ION': 7, 'ION_NA': 7, 'ION_CL': 7,
}

# Explicit pairwise LJ σ/ε from martini_v3.0.0.itp [nonbond_params]
# Row/column order: BB, C, SC, N, Q, TC, W, ION
_PAIR_SIGMA = np.array([
    [0.470, 0.470, 0.430, 0.470, 0.470, 0.398, 0.465, 0.395],
    [0.470, 0.470, 0.430, 0.470, 0.470, 0.395, 0.465, 0.465],
    [0.430, 0.430, 0.410, 0.430, 0.430, 0.365, 0.425, 0.484],
    [0.470, 0.470, 0.430, 0.470, 0.470, 0.395, 0.465, 0.395],
    [0.470, 0.470, 0.430, 0.470, 0.470, 0.401, 0.465, 0.405],
    [0.398, 0.395, 0.365, 0.395, 0.401, 0.340, 0.393, 0.505],
    [0.465, 0.465, 0.425, 0.465, 0.465, 0.393, 0.470, 0.385],
    [0.395, 0.465, 0.484, 0.395, 0.405, 0.505, 0.385, 0.354],
], dtype=np.float64)

_PAIR_EPS = np.array([
    [ 4.060, 2.790, 2.160, 3.390, 5.148, 1.450,  4.330,  9.642],
    [ 2.790, 3.390, 2.920, 3.240, 2.787, 2.310,  2.420,  2.946],
    [ 2.160, 2.920, 2.350, 2.770, 1.961, 1.910,  1.800,  2.506],
    [ 3.390, 3.240, 2.770, 3.520, 4.477, 2.110,  3.290,  8.247],
    [ 5.148, 2.787, 1.961, 4.477, 5.950, 1.168,  5.960,  6.360],
    [ 1.450, 2.310, 1.910, 2.110, 1.168, 1.510,  1.120,  1.818],
    [ 4.330, 2.420, 1.800, 3.290, 5.960, 1.120,  4.650, 11.460],
    [ 9.642, 2.946, 2.506, 8.247, 6.360, 1.818, 11.460,  1.180],
], dtype=np.float64)

# Protein bond/angle parameters
_BB_BB_K  = 3800.0;  _BB_BB_R0 = 0.35
_BB_SC_K  = 3800.0;  _BB_SC_R0 = 0.27
_SC_SC_K  = 2500.0;  _SC_SC_R0 = 0.27
_BB_ANG_K     = 40.0
_BB_ANG_COIL  = math.radians(127.0)
_BB_ANG_HELIX = math.radians(96.0)
_BB_ANG_SHEET = math.radians(134.0)
_BB_ENM_MAX_DIST_NM = 0.9        # max intra-helix BB–BB distance for ENM spring
_ENM_MAX_DIST_NM    = 0.9        # default cutoff for sheet / inter-chain ENM springs
_ENM_SHEET_MIN_SEP  = 3          # min |i−j| for intra-chain sheet ENM springs
_ENM_IC_K           = 200.0      # default inter-chain ENM spring constant

# Disulfide bridges (CYS SC1–SC1).  Stiff bond, as in martinize2; r0 is taken
# from the input geometry so the bridge carries no strain at t=0, clamped to a
# sane range for the CB+SG centroid bead.
_SS_BOND_K        = 5000.0       # kJ/mol/nm²
_SS_BOND_R0_MIN   = 0.20         # nm
_SS_BOND_R0_MAX   = 0.45         # nm
_SS_SG_MAX_DIST_A = 2.5          # Å — SG–SG distance that defines a bridge

# A peptide BB–BB bond is made only between residues that are BOTH contiguous in
# numbering and close in space; anything else is treated as a chain break
# (unresolved loop), so missing regions are not fused by a stiff 0.35 nm bond.
_BB_BOND_MAX_NM = 0.55           # nm

# --tm-residues sanity checks: each range must be mostly present in the input,
# helical in the input (CG cannot fold a coil into a helix), and hydrophobic.
_TM_MIN_COVERAGE         = 0.5           # fraction of range residues present
_TM_MIN_HELIX_FRACTION   = 0.5           # fraction of present residues helical
_TM_MIN_HYDROPHOBIC_FRAC = 0.5           # fraction of present residues in _TM_HYDROPHOBIC
_TM_HYDROPHOBIC          = set('AILMFVWC')

# Protein residue bead definitions: (bead_type, bead_label, charge, [atom_names])
_RESIDUE_BEADS = {
    'GLY': [('BB', 'BB',  0.0, ['N', 'CA', 'C', 'O'])],
    'ALA': [('BB', 'BB',  0.0, ['N', 'CA', 'C', 'O']),
            ('TC', 'SC1', 0.0, ['CB'])],
    'VAL': [('BB', 'BB',  0.0, ['N', 'CA', 'C', 'O']),
            ('SC', 'SC1', 0.0, ['CB', 'CG1', 'CG2'])],
    'LEU': [('BB', 'BB',  0.0, ['N', 'CA', 'C', 'O']),
            ('SC', 'SC1', 0.0, ['CB', 'CG', 'CD1', 'CD2'])],
    'ILE': [('BB', 'BB',  0.0, ['N', 'CA', 'C', 'O']),
            ('SC', 'SC1', 0.0, ['CB', 'CG1', 'CG2', 'CD1'])],
    'PRO': [('BB', 'BB',  0.0, ['N', 'CA', 'C', 'O']),
            ('SC', 'SC1', 0.0, ['CB', 'CG', 'CD'])],
    'MET': [('BB', 'BB',  0.0, ['N', 'CA', 'C', 'O']),
            ('C',  'SC1', 0.0, ['CB', 'CG', 'SD', 'CE'])],
    'CYS': [('BB', 'BB',  0.0, ['N', 'CA', 'C', 'O']),
            ('TC', 'SC1', 0.0, ['CB', 'SG'])],
    'SER': [('BB', 'BB',  0.0, ['N', 'CA', 'C', 'O']),
            ('N',  'SC1', 0.0, ['CB', 'OG'])],
    'THR': [('BB', 'BB',  0.0, ['N', 'CA', 'C', 'O']),
            ('N',  'SC1', 0.0, ['CB', 'OG1', 'CG2'])],
    'ASN': [('BB', 'BB',  0.0, ['N', 'CA', 'C', 'O']),
            ('N',  'SC1', 0.0, ['CB', 'CG', 'OD1', 'ND2'])],
    'GLN': [('BB', 'BB',  0.0, ['N', 'CA', 'C', 'O']),
            ('N',  'SC1', 0.0, ['CB', 'CG', 'CD', 'OE1', 'NE2'])],
    'ASP': [('BB', 'BB',  0.0, ['N', 'CA', 'C', 'O']),
            ('Q',  'SC1', -1.0, ['CB', 'CG', 'OD1', 'OD2'])],
    'GLU': [('BB', 'BB',  0.0, ['N', 'CA', 'C', 'O']),
            ('Q',  'SC1', -1.0, ['CB', 'CG', 'CD', 'OE1', 'OE2'])],
    'LYS': [('BB', 'BB',  0.0, ['N', 'CA', 'C', 'O']),
            ('SC', 'SC1', 0.0, ['CB', 'CG', 'CD']),
            ('Q',  'SC2', +1.0, ['CE', 'NZ'])],
    'ARG': [('BB', 'BB',  0.0, ['N', 'CA', 'C', 'O']),
            ('N',  'SC1', 0.0, ['CB', 'CG', 'CD']),
            ('Q',  'SC2', +1.0, ['NE', 'CZ', 'NH1', 'NH2'])],
    'HIS': [('BB', 'BB',  0.0, ['N', 'CA', 'C', 'O']),
            ('TC', 'SC1', 0.0, ['CB', 'CG']),
            ('TC', 'SC2', +0.5, ['ND1', 'CD2', 'CE1', 'NE2'])],
    'PHE': [('BB', 'BB',  0.0, ['N', 'CA', 'C', 'O']),
            ('SC', 'SC1', 0.0, ['CB', 'CG', 'CD1', 'CD2']),
            ('SC', 'SC2', 0.0, ['CE1', 'CE2', 'CZ'])],
    'TYR': [('BB', 'BB',  0.0, ['N', 'CA', 'C', 'O']),
            ('SC', 'SC1', 0.0, ['CB', 'CG', 'CD1', 'CD2']),
            ('SC', 'SC2', 0.0, ['CE1', 'CE2', 'CZ', 'OH'])],
    'TRP': [('BB', 'BB',  0.0, ['N', 'CA', 'C', 'O']),
            ('SC', 'SC1', 0.0, ['CB', 'CG', 'CD1', 'NE1']),
            ('SC', 'SC2', 0.0, ['CD2', 'CE2', 'CZ2', 'CH2']),
            ('TC', 'SC3', 0.0, ['CE3', 'CZ3'])],
}

_MODIFIED_TO_STANDARD = {
    'MSE': 'MET', 'FME': 'MET', 'CXM': 'MET',
    'M3L': 'LYS', 'MLY': 'LYS', 'MLZ': 'LYS', 'KCX': 'LYS',
    'CSO': 'CYS', 'CME': 'CYS', 'SEC': 'CYS',
    'SEP': 'SER', 'TPO': 'THR', 'PTR': 'TYR',
    'HYP': 'PRO', 'CGU': 'GLU', 'NEP': 'HIS',
}


# ---------------------------------------------------------------------------
# Lipid CG definitions (Martini 3-inspired)
# ---------------------------------------------------------------------------
# Each lipid is an ordered list of: (label, bead_type, charge, [dx, dy, dz from PO4 nm])
# Upper-leaflet template; lower leaflet negates dz.
# Bonds: list of (i, j, k_bond kJ/mol/nm², r0 nm)
# Angles: list of (i, j, k, k_angle kJ/mol/rad², theta0 degrees)

# POPC — palmitoyl-oleoyl-phosphatidylcholine (Martini 3, 12 beads)
# NC3(Q,+1), PO4(Q,-1), GL1(N), GL2(N), C1A-C4A(C), C1B-D2B-C3B-C4B(C)
_POPC_BEADS = [
    ('NC3', 'Q', +1.0, [ 0.00,  0.00, +0.40]),   # 0  choline
    ('PO4', 'Q', -1.0, [ 0.00,  0.00,  0.00]),   # 1  phosphate (z-reference)
    ('GL1', 'N',  0.0, [ 0.00,  0.00, -0.47]),   # 2  glycerol 1
    ('GL2', 'N',  0.0, [ 0.37,  0.00, -0.47]),   # 3  glycerol 2
    ('C1A', 'C',  0.0, [ 0.00,  0.00, -0.87]),   # 4  sn-1 palmitoyl
    ('C2A', 'C',  0.0, [ 0.00,  0.00, -1.22]),   # 5
    ('C3A', 'C',  0.0, [ 0.00,  0.00, -1.57]),   # 6
    ('C4A', 'C',  0.0, [ 0.00,  0.00, -1.92]),   # 7
    ('C1B', 'C',  0.0, [ 0.37,  0.00, -0.87]),   # 8  sn-2 oleoyl
    ('D2B', 'C',  0.0, [ 0.37,  0.00, -1.22]),   # 9  unsaturated (C4h → mapped C)
    ('C3B', 'C',  0.0, [ 0.37,  0.00, -1.57]),   # 10
    ('C4B', 'C',  0.0, [ 0.37,  0.00, -1.92]),   # 11
]
_POPC_BONDS = [
    (0, 1, 3800, 0.40), (1, 2, 3800, 0.47), (2, 3, 3800, 0.37),
    (2, 4, 3800, 0.47), (4, 5, 3800, 0.47), (5, 6, 3800, 0.47), (6, 7, 3800, 0.47),
    (3, 8, 3800, 0.47), (8, 9, 3800, 0.47), (9, 10, 3800, 0.47), (10, 11, 3800, 0.47),
]
_POPC_ANGLES = [
    (0, 1, 2, 25, 180), (1, 2, 3, 25, 120), (1, 2, 4, 25, 120),
    (2, 4, 5, 40, 180), (4, 5, 6, 40, 180), (5, 6, 7, 40, 180),
    (2, 3, 8, 40, 180), (3, 8, 9, 40, 180),
    (8, 9, 10, 40, 120),   # cis double bond — bent
    (9, 10, 11, 40, 180),
]

# DPPC — dipalmitoyl-phosphatidylcholine (fully saturated)
_DPPC_BEADS = [
    ('NC3', 'Q', +1.0, [ 0.00,  0.00, +0.40]),
    ('PO4', 'Q', -1.0, [ 0.00,  0.00,  0.00]),
    ('GL1', 'N',  0.0, [ 0.00,  0.00, -0.47]),
    ('GL2', 'N',  0.0, [ 0.37,  0.00, -0.47]),
    ('C1A', 'C',  0.0, [ 0.00,  0.00, -0.87]),
    ('C2A', 'C',  0.0, [ 0.00,  0.00, -1.22]),
    ('C3A', 'C',  0.0, [ 0.00,  0.00, -1.57]),
    ('C4A', 'C',  0.0, [ 0.00,  0.00, -1.92]),
    ('C1B', 'C',  0.0, [ 0.37,  0.00, -0.87]),
    ('C2B', 'C',  0.0, [ 0.37,  0.00, -1.22]),   # saturated
    ('C3B', 'C',  0.0, [ 0.37,  0.00, -1.57]),
    ('C4B', 'C',  0.0, [ 0.37,  0.00, -1.92]),
]
_DPPC_BONDS = _POPC_BONDS
_DPPC_ANGLES = [
    (0, 1, 2, 25, 180), (1, 2, 3, 25, 120), (1, 2, 4, 25, 120),
    (2, 4, 5, 40, 180), (4, 5, 6, 40, 180), (5, 6, 7, 40, 180),
    (2, 3, 8, 40, 180), (3, 8, 9, 40, 180), (8, 9, 10, 40, 180), (9, 10, 11, 40, 180),
]

# POPE — palmitoyl-oleoyl-phosphatidylethanolamine (smaller PE head)
_POPE_BEADS = [
    ('NH3', 'Q', +1.0, [ 0.00,  0.00, +0.33]),   # ethanolamine
    ('PO4', 'Q', -1.0, [ 0.00,  0.00,  0.00]),
    ('GL1', 'N',  0.0, [ 0.00,  0.00, -0.47]),
    ('GL2', 'N',  0.0, [ 0.37,  0.00, -0.47]),
    ('C1A', 'C',  0.0, [ 0.00,  0.00, -0.87]),
    ('C2A', 'C',  0.0, [ 0.00,  0.00, -1.22]),
    ('C3A', 'C',  0.0, [ 0.00,  0.00, -1.57]),
    ('C4A', 'C',  0.0, [ 0.00,  0.00, -1.92]),
    ('C1B', 'C',  0.0, [ 0.37,  0.00, -0.87]),
    ('D2B', 'C',  0.0, [ 0.37,  0.00, -1.22]),
    ('C3B', 'C',  0.0, [ 0.37,  0.00, -1.57]),
    ('C4B', 'C',  0.0, [ 0.37,  0.00, -1.92]),
]
_POPE_BONDS  = _POPC_BONDS
_POPE_ANGLES = _POPC_ANGLES

_LIPID_DEFS = {
    'POPC': (_POPC_BEADS, _POPC_BONDS, _POPC_ANGLES),
    'DPPC': (_DPPC_BEADS, _DPPC_BONDS, _DPPC_ANGLES),
    'POPE': (_POPE_BEADS, _POPE_BONDS, _POPE_ANGLES),
}


# ---------------------------------------------------------------------------
# Default simulation parameters
# ---------------------------------------------------------------------------

DEFAULT_TEMPERATURE      = 310.15   # K (37 °C)
DEFAULT_PRESSURE         = 1.01325  # bar
DEFAULT_SURFACE_TENSION  = 0.0      # bar·nm (tension-free NPT, not NPγT)
DEFAULT_FRICTION         = 1.0      # 1/ps
DEFAULT_EQUIL_TIMESTEP   = 10.0     # fs — conservative during equilibration
DEFAULT_PROD_TIMESTEP    = 20.0     # fs — standard Martini 3 production timestep
DEFAULT_NVT_STEPS        = 25000    # 250 ps at 10 fs/step
DEFAULT_NPT_STEPS        = 50000    # 500 ps at 10 fs/step
DEFAULT_PROD_STEPS       = 500000   # 10 ns at 20 fs/step (kept for back-compat)
DEFAULT_REPORT_INTERVAL  = 5000
DEFAULT_K_NVT            = 200.0    # kJ/mol/nm² — moderate NVT/NPT BB restraint
DEFAULT_EARLY_PROD_STEPS = 100_000  # 2 ns at 20 fs/step — gradual restraint release
DEFAULT_STABILITY_STEPS  = 50_000_000   # 1 µs at 20 fs/step
DEFAULT_SA_T_HIGH_K      = 335.0    # K — high temperature for SA cycles
DEFAULT_SA_STEPS_HALF    = 25_000   # steps per SA heating/cooling half-cycle
DEFAULT_SA_CYCLES        = 5        # SA cycles per replica
DEFAULT_N_REPLICAS       = 3        # number of SA replicas
DEFAULT_Z_PO4_NM         = 2.0      # nm from bilayer midplane to PO4 head group
DEFAULT_WATER_LAYER_NM   = 1.5      # nm of CG water above/below bilayer
DEFAULT_XY_PADDING_NM    = 2.5      # nm extra bilayer beyond protein XY footprint
_GRID_SPACING_XY         = 0.80     # nm — lipid XY grid (≈ sqrt(0.64 nm², POPC APL)
_WATER_SPACING           = 0.527    # nm — W–W LJ minimum = 2^(1/6) × 0.47


# ---------------------------------------------------------------------------
# Step 1: PDB → CG protein beads
# ---------------------------------------------------------------------------

def _bead_centroid(residue, atom_names: list) -> np.ndarray:
    coords = [residue[n].get_vector().get_array()
              for n in atom_names if n in residue]
    if coords:
        return np.mean(coords, axis=0)
    for fallback in ('CA', 'N', 'C'):
        if fallback in residue:
            return residue[fallback].get_vector().get_array()
    raise KeyError(f"{residue.resname} {residue.id}: no usable atoms")


def _phi_psi_ss_angles(model, residue_seq_info: list,
                       residue_icodes: list = None) -> list:
    """DSSP-free secondary-structure angles from backbone phi/psi basins.

    Fallback for when the mkdssp binary is unavailable: classifies each residue
    as helix / strand / coil from its Ramachandran basin (BioPython, no external
    binary) and returns the matching backbone angle. Assumes the input structure
    already carries the intended fold (e.g. a crystal or predicted model), which
    is the case for this pipeline's inputs. Returns None if phi/psi cannot be
    computed at all, so the caller can decide how to warn.
    """
    from Bio.PDB.Polypeptide import PPBuilder

    ss_by_res = {}
    for pp in PPBuilder().build_peptides(model):
        for res, (phi, psi) in zip(pp, pp.get_phi_psi_list()):
            if phi is None or psi is None:
                continue
            p, s = math.degrees(phi), math.degrees(psi)
            if -100.0 <= p <= -30.0 and -80.0 <= s <= -5.0:
                ss = 'H'
            elif -170.0 <= p <= -40.0 and 90.0 <= s <= 170.0:
                ss = 'E'
            else:
                ss = 'C'
            ss_by_res[(res.parent.id, res.id[1], res.id[2])] = ss

    if not ss_by_res:
        return None
    angle_of = {'H': _BB_ANG_HELIX, 'E': _BB_ANG_SHEET, 'C': _BB_ANG_COIL}
    icodes = residue_icodes or [' '] * len(residue_seq_info)
    return [angle_of.get(ss_by_res.get((chain_id, seq_num, icode or ' '), 'C'),
                         _BB_ANG_COIL)
            for (chain_id, seq_num), icode in zip(residue_seq_info, icodes)]


def _dssp_angles(model, pdb_path: str, residue_seq_info: list,
                 residue_icodes: list = None) -> list:
    """Per-residue backbone angles from secondary structure.

    Tries DSSP first; if the mkdssp binary is missing (common), falls back to a
    phi/psi-basin assignment rather than silently treating every residue as coil
    — an all-coil assignment applies NO helix/sheet restraint, so helices
    unravel during the CG run (and --use-enm becomes a no-op, since it keys off
    the 96 deg helix angle). Only if both routes fail do we default to all-coil,
    and then loudly.
    """
    try:
        from Bio.PDB.DSSP import DSSP
        dssp = DSSP(model, pdb_path)
        angles = []
        icodes = residue_icodes or [' '] * len(residue_seq_info)
        for (chain_id, seq_num), icode in zip(residue_seq_info, icodes):
            key = (chain_id, (' ', seq_num, icode or ' '))
            if key in dssp:
                ss = dssp[key][2]
                if ss in ('H', 'G', 'I'):
                    angles.append(_BB_ANG_HELIX)
                elif ss in ('E', 'B'):
                    angles.append(_BB_ANG_SHEET)
                else:
                    angles.append(_BB_ANG_COIL)
            else:
                angles.append(_BB_ANG_COIL)
        return angles
    except Exception as exc:
        print(f"  WARNING: DSSP unavailable ({type(exc).__name__}: {exc}); "
              f"assigning secondary structure from backbone phi/psi instead.")
        angles = _phi_psi_ss_angles(model, residue_seq_info, residue_icodes)
        if angles is None:
            print("  WARNING: phi/psi assignment also failed — ALL residues "
                  "will be treated as coil. Helices/sheets will NOT be "
                  "restrained and secondary structure will unravel.")
            return [_BB_ANG_COIL] * len(residue_seq_info)
        n_h = sum(abs(a - _BB_ANG_HELIX) < 1e-6 for a in angles)
        n_e = sum(abs(a - _BB_ANG_SHEET) < 1e-6 for a in angles)
        print(f"  phi/psi secondary structure: {n_h} helix, {n_e} strand, "
              f"{len(angles) - n_h - n_e} coil residues.")
        return angles


def _parse_ssbond_records(pdb_path: str) -> list:
    """SSBOND records from a PDB file → [((ch, resseq, icode), (ch, resseq, icode))].

    mmCIF input carries no SSBOND records, so bridges are found geometrically
    instead (see _find_disulfides).
    """
    if str(pdb_path).lower().endswith(('.cif', '.mmcif')):
        return []
    pairs_out = []
    try:
        with open(pdb_path) as fh:
            for line in fh:
                if not line.startswith('SSBOND'):
                    continue
                try:
                    k1 = (line[15], int(line[17:21]), line[21])
                    k2 = (line[29], int(line[31:35]), line[35])
                except (IndexError, ValueError):
                    continue
                pairs_out.append((k1, k2))
    except OSError:
        return []
    return pairs_out


def _find_disulfides(model, pdb_path: str, res_index_by_key: dict) -> list:
    """Locate disulfide bridges → sorted [(residue_idx_i, residue_idx_j), ...].

    Two sources are unioned:
      * SSBOND records in the input PDB header
      * SG–SG distances ≤ _SS_SG_MAX_DIST_A between CYS residues, which also
        covers mmCIF input, predicted models, and header-stripped files

    Inter-chain bridges are included: they are what covalently ties multimers
    together (e.g. the TCR α/β connecting-peptide bridge, or a CD3 dimer).
    """
    found: set = set()

    for k1, k2 in _parse_ssbond_records(pdb_path):
        ri, rj = res_index_by_key.get(k1), res_index_by_key.get(k2)
        if ri is not None and rj is not None and ri != rj:
            found.add((min(ri, rj), max(ri, rj)))

    sg_atoms: list = []
    for chain in model.get_chains():
        for res in chain:
            raw = res.resname.strip()
            if _MODIFIED_TO_STANDARD.get(raw, raw) != 'CYS' or 'SG' not in res:
                continue
            ri = res_index_by_key.get((chain.id, res.id[1], res.id[2]))
            if ri is not None:
                sg_atoms.append((ri, res['SG'].get_vector().get_array()))

    for a in range(len(sg_atoms)):
        ri, pi = sg_atoms[a]
        for b in range(a + 1, len(sg_atoms)):
            rj, pj = sg_atoms[b]
            if ri != rj and float(np.linalg.norm(pi - pj)) <= _SS_SG_MAX_DIST_A:
                found.add((min(ri, rj), max(ri, rj)))

    return sorted(found)


def convert_pdb_to_cg_beads(pdb_path: str, detect_disulfides: bool = True) -> dict:
    """Parse PDB → Martini CG bead data (same scheme as MartiniNonBondedForceModel).

    Every chain in the first model is converted.  Chain identity is preserved:
    peptide BB–BB bonds and backbone angles are never formed across a chain
    boundary or an unresolved gap (see bb_linked), and disulfide bridges —
    including inter-chain ones — are added as stiff SC1–SC1 bonds.

    Returns a dict with keys:
        beads, residue_names, residue_bead_ranges, bb_bead_indices,
        residue_seq_info, residue_icodes, bb_linked, bond_list, ss_bonds,
        bb_angles, openmm_exclusions
    """
    if str(pdb_path).lower().endswith(('.cif', '.mmcif')):
        parser = MMCIFParser(QUIET=True)
    else:
        parser = PDBParser(QUIET=True)
    structure = parser.get_structure("protein", pdb_path)
    model = next(structure.get_models())

    beads_data: list          = []
    residue_names: list       = []
    residue_bead_ranges: list = []
    bb_bead_indices: list     = []
    residue_seq_info: list    = []
    residue_icodes: list      = []

    for chain in model.get_chains():
        for res in chain:
            raw = res.resname.strip()
            resname = _MODIFIED_TO_STANDARD.get(raw, raw)
            bead_defs = _RESIDUE_BEADS.get(resname)
            if bead_defs is None or 'CA' not in res:
                continue
            ri = len(residue_names)
            residue_names.append(resname)
            residue_seq_info.append((chain.id, res.id[1]))
            residue_icodes.append(res.id[2])
            first_bead = len(beads_data)
            for bead_type, bead_label, charge, atom_names in bead_defs:
                pos_ang = _bead_centroid(res, atom_names)
                beads_data.append({
                    'residue_idx': ri,
                    'bead_type':   bead_type,
                    'bead_label':  bead_label,
                    'charge':      charge,
                    'pos_nm':      pos_ang / 10.0,   # Å → nm
                })
            bb_bead_indices.append(first_bead)
            residue_bead_ranges.append((first_bead, len(beads_data)))

    if not beads_data:
        raise ValueError(f"No recognised residues in {pdb_path}")

    N = len(residue_names)
    bead_pos = np.array([b['pos_nm'] for b in beads_data], dtype=np.float64)

    # Which consecutive residues are genuinely peptide-bonded: same chain,
    # contiguous numbering, and a plausible BB–BB distance.  Anything else is a
    # chain break — a different subunit, or an unresolved loop — and gets no
    # bond and no backbone angle, so fragments are not fused by a stiff bond.
    bb_linked: list = []
    breaks: list    = []
    for ri in range(N - 1):
        ch_i, rs_i = residue_seq_info[ri]
        ch_j, rs_j = residue_seq_info[ri + 1]
        if ch_i != ch_j:
            bb_linked.append(False)
            continue
        seq_gap    = rs_j - rs_i
        contiguous = (seq_gap == 1) or (
            seq_gap == 0 and residue_icodes[ri] != residue_icodes[ri + 1])
        dist = float(np.linalg.norm(bead_pos[bb_bead_indices[ri + 1]]
                                    - bead_pos[bb_bead_indices[ri]]))
        linked = contiguous and dist <= _BB_BOND_MAX_NM
        bb_linked.append(linked)
        if not linked:
            breaks.append((ch_i, rs_i, rs_j, dist))

    # Bond list: intra-residue SC chain + inter-residue BB–BB (bonded pairs only)
    bond_list: list = []
    for ri in range(N):
        lo, hi = residue_bead_ranges[ri]
        idxs = list(range(lo, hi))
        for k in range(len(idxs) - 1):
            bond_list.append((idxs[k], idxs[k + 1]))
    for ri in range(N - 1):
        if bb_linked[ri]:
            bond_list.append((bb_bead_indices[ri], bb_bead_indices[ri + 1]))

    chain_ids = sorted({ci for ci, _ in residue_seq_info})
    print(f"  Chains: {len(chain_ids)} "
          f"({', '.join(repr(c) for c in chain_ids)})")
    if breaks:
        print(f"  {len(breaks)} intra-chain break(s) left unbonded "
              f"(missing residues or non-peptide geometry):")
        for ch, rs_i, rs_j, dist in breaks[:8]:
            print(f"    chain {ch!r}: {rs_i} → {rs_j}  (BB–BB {dist:.2f} nm)")
        if len(breaks) > 8:
            print(f"    ... and {len(breaks) - 8} more")

    # Disulfide bridges — stiff SC1–SC1 bonds, intra- and inter-chain
    res_index_by_key = {
        (residue_seq_info[ri][0], residue_seq_info[ri][1], residue_icodes[ri]): ri
        for ri in range(N)
    }
    ss_bonds: list = []
    if detect_disulfides:
        used: set = set()
        n_inter = 0
        for ri, rj in _find_disulfides(model, pdb_path, res_index_by_key):
            if ri in used or rj in used:
                print(f"  WARNING: ignoring extra disulfide for an already-bridged "
                      f"cysteine ({residue_seq_info[ri]} – {residue_seq_info[rj]})")
                continue
            lo_i, hi_i = residue_bead_ranges[ri]
            lo_j, hi_j = residue_bead_ranges[rj]
            # SC1 (CB+SG centroid) if present, else fall back to BB
            bi = lo_i + 1 if hi_i - lo_i > 1 else lo_i
            bj = lo_j + 1 if hi_j - lo_j > 1 else lo_j
            r0_meas = float(np.linalg.norm(bead_pos[bi] - bead_pos[bj]))
            r0 = min(max(r0_meas, _SS_BOND_R0_MIN), _SS_BOND_R0_MAX)
            if abs(r0 - r0_meas) > 1e-9:
                print(f"  WARNING: bridge {residue_seq_info[ri]} – "
                      f"{residue_seq_info[rj]} has an SC1–SC1 distance of "
                      f"{r0_meas:.2f} nm; clamped to {r0:.2f} nm, so the bond "
                      f"starts strained (check the side-chain geometry)")
            ss_bonds.append((bi, bj, r0))
            used.update((ri, rj))
            if residue_seq_info[ri][0] != residue_seq_info[rj][0]:
                n_inter += 1
        if ss_bonds:
            print(f"  Disulfides: {len(ss_bonds)} bridge(s), {n_inter} inter-chain "
                  f"(k={_SS_BOND_K:.0f} kJ/mol/nm²)")
        else:
            print("  Disulfides: none detected")

    # 1-2 / 1-3 exclusion pairs
    neighbors: dict = defaultdict(set)
    for i, j in bond_list:
        neighbors[i].add(j); neighbors[j].add(i)
    excl: set = set()
    B_prot = len(beads_data)
    for b in range(B_prot):
        for n1 in list(neighbors[b]):
            excl.add((min(b, n1), max(b, n1)))
            for n2 in list(neighbors[n1]):
                if n2 != b:
                    excl.add((min(b, n2), max(b, n2)))
    # Disulfides are excluded as 1-2 pairs only — the stiff bond holds them at
    # ~0.3 nm, well inside the SC–SC LJ minimum, but 1-3 neighbours across the
    # bridge are far enough apart to keep their normal non-bonded interaction.
    for (bi, bj, _) in ss_bonds:
        excl.add((min(bi, bj), max(bi, bj)))

    bb_angles = _dssp_angles(model, pdb_path, residue_seq_info, residue_icodes)

    return {
        'beads':               beads_data,
        'residue_names':       residue_names,
        'residue_bead_ranges': residue_bead_ranges,
        'bb_bead_indices':     bb_bead_indices,
        'residue_seq_info':    residue_seq_info,
        'residue_icodes':      residue_icodes,
        'bb_linked':           bb_linked,
        'bond_list':           bond_list,
        'ss_bonds':            ss_bonds,
        'bb_angles':           bb_angles,
        'openmm_exclusions':   list(excl),
    }


# ---------------------------------------------------------------------------
# Step 2: Bilayer construction
# ---------------------------------------------------------------------------

def build_cg_bilayer(
    protein_pos_nm: np.ndarray,
    box_lengths: np.ndarray,
    lipid_type: str = 'POPC',
    z_po4: float = DEFAULT_Z_PO4_NM,
    xy_spacing: float = _GRID_SPACING_XY,
    exclusion_radius_nm: float = 0.6,
) -> tuple:
    """Build a CG lipid bilayer on an XY grid around the protein.

    PO4 heads are placed on an XY grid at z = mid_z ± z_po4.
    Grid positions within exclusion_radius_nm of any protein bead
    (restricted to protein beads in the bilayer z-range) are removed.

    Returns:
        (lipid_beads, lipid_bonds, lipid_angles)
        lipid_bonds: list of (i, j, k_bond, r0) — local lipid-bead indices
        lipid_angles: list of (i, j, k, k_ang, theta0_rad) — local indices
    """
    bead_defs, bond_defs, angle_defs = _LIPID_DEFS[lipid_type]
    Lx, Ly  = box_lengths[0], box_lengths[1]
    mid_z   = box_lengths[2] / 2.0

    # Protein beads in the bilayer z-range → define exclusion zone in XY
    z_rel         = protein_pos_nm[:, 2] - mid_z
    in_bilayer    = np.abs(z_rel) < (z_po4 + 0.5)
    prot_xy       = protein_pos_nm[in_bilayer, :2]

    xs       = np.arange(0.5 * xy_spacing, Lx, xy_spacing)
    ys       = np.arange(0.5 * xy_spacing, Ly, xy_spacing)
    gx, gy   = np.meshgrid(xs, ys, indexing='ij')
    grid_xy  = np.stack([gx.ravel(), gy.ravel()], axis=1)   # (G, 2)

    if len(prot_xy) > 0:
        diff   = grid_xy[:, np.newaxis, :] - prot_xy[np.newaxis, :, :]
        min_d  = np.sqrt((diff ** 2).sum(axis=2)).min(axis=1)
        keep   = grid_xy[min_d >= exclusion_radius_nm]
    else:
        keep   = grid_xy

    # Upper-leaflet bead offsets (z > 0 from PO4 reference)
    offsets_raw = np.array([b[3] for b in bead_defs], dtype=np.float64)  # (N, 3)

    lipid_beads:  list = []
    lipid_bonds:  list = []
    lipid_angles: list = []
    lipid_idx         = 0

    for sign, z_abs in [(+1, mid_z + z_po4), (-1, mid_z - z_po4)]:
        offsets = offsets_raw.copy()
        offsets[:, 2] *= sign   # flip z for lower leaflet

        for gxy in keep:
            po4_ref = np.array([gxy[0], gxy[1], z_abs])
            first   = len(lipid_beads)
            for k, (label, btype, charge, _) in enumerate(bead_defs):
                lipid_beads.append({
                    'lipid_idx':  lipid_idx,
                    'bead_label': label,
                    'bead_type':  btype,
                    'charge':     charge,
                    'pos_nm':     po4_ref + offsets[k],
                })
            for (i, j, kbond, r0) in bond_defs:
                lipid_bonds.append((first + i, first + j, kbond, r0))
            for (i, j, k, kang, th0) in angle_defs:
                lipid_angles.append((
                    first + i, first + j, first + k,
                    kang, math.radians(th0),
                ))
            lipid_idx += 1

    n_grid = len(keep)
    print(f"  Bilayer: {n_grid} grid points × 2 leaflets = {lipid_idx} lipids "
          f"({len(lipid_beads)} beads)")
    return lipid_beads, lipid_bonds, lipid_angles


# ---------------------------------------------------------------------------
# Step 3: CG solvation and ion placement
# ---------------------------------------------------------------------------

def _build_solvent_box(
    all_placed_nm: np.ndarray,
    box_lengths: np.ndarray,
    z_bilayer_outer: float,
    spacing: float = _WATER_SPACING,
    min_clash_nm: float = 0.53,
) -> np.ndarray:
    """Place W beads on a cubic grid outside the bilayer, removing clashes."""
    axes = [np.arange(0.0, box_lengths[d], spacing) for d in range(3)]
    gx, gy, gz = np.meshgrid(*axes, indexing='ij')
    grid = np.stack([gx.ravel(), gy.ravel(), gz.ravel()], axis=1)

    mid_z   = box_lengths[2] / 2.0
    z_abs   = np.abs(grid[:, 2] - mid_z)
    outside = z_abs > z_bilayer_outer
    grid    = grid[outside]

    valid = np.ones(len(grid), dtype=bool)
    chunk = 2000
    for start in range(0, len(grid), chunk):
        slc   = grid[start:start + chunk]
        diff  = slc[:, np.newaxis, :] - all_placed_nm[np.newaxis, :, :]
        min_d = np.sqrt((diff ** 2).sum(axis=2)).min(axis=1)
        valid[start:start + chunk] = min_d >= min_clash_nm

    water_pos = grid[valid]
    n_excluded = (~outside).sum() + (~valid).sum()
    print(f"  Solvation: {len(water_pos)} W beads placed "
          f"({n_excluded} grid points excluded)")
    return water_pos


def _add_ions(
    water_pos: np.ndarray,
    net_charge: float,
    box_lengths: np.ndarray,
    salt_conc_M: float = _SALT_CONC_M,
) -> tuple:
    """Replace water beads with Na⁺/Cl⁻ for charge neutralisation + 0.15 M NaCl."""
    rng    = np.random.default_rng(42)
    order  = rng.permutation(len(water_pos))
    used   = 0
    na_idx: list = []
    cl_idx: list = []

    net_int = round(net_charge)
    for _ in range(max(0, -net_int)):
        na_idx.append(int(order[used])); used += 1
    for _ in range(max(0, net_int)):
        cl_idx.append(int(order[used])); used += 1

    vol_l   = float(np.prod(box_lengths)) * 1e-24
    n_pairs = int(round(salt_conc_M * vol_l * 6.02214076e23))
    n_water = len(water_pos)
    for _ in range(n_pairs):
        if used < n_water:
            na_idx.append(int(order[used])); used += 1
        if used < n_water:
            cl_idx.append(int(order[used])); used += 1

    ion_set = set(na_idx + cl_idx)
    keep    = np.array([i for i in range(n_water) if i not in ion_set])
    final_w = water_pos[keep]           if len(keep)    else np.empty((0, 3))
    na_pos  = water_pos[na_idx]         if na_idx       else np.empty((0, 3))
    cl_pos  = water_pos[cl_idx]         if cl_idx       else np.empty((0, 3))

    print(f"  Ions: {len(na_idx)} Na⁺, {len(cl_idx)} Cl⁻  "
          f"(net charge={net_int:+d}, {n_pairs} NaCl pairs)")
    return final_w, na_pos, cl_pos


# ---------------------------------------------------------------------------
# Optional: CG water pre-equilibration before membrane embedding
# ---------------------------------------------------------------------------

def _pick_chain_id(preferred: str, used: set) -> str:
    """A single-character chain id for solvent/lipid that the protein isn't using."""
    for cid in (preferred, *'XYZWVUTSRQPONMLKJIHGFEDCBA0123456789'):
        if cid not in used:
            used.add(cid)
            return cid
    return preferred


def _add_protein_to_topology(topology, protein, element, all_atoms: list) -> set:
    """Add the CG protein to an OpenMM Topology as one chain per input chain.

    Source chain ids, residue numbers and insertion codes are carried through, so
    output PDB/DCD frames map straight back onto the input structure.  For a
    multimer this is what keeps the subunits distinguishable (a single merged
    chain would make e.g. TCRα and CD3ε indistinguishable downstream).
    Disulfide bridges are added as topology bonds too, so viewers draw them.

    Returns the set of chain ids used by the protein.
    """
    res_names  = protein['residue_names']
    res_ranges = protein['residue_bead_ranges']
    res_seq    = protein['residue_seq_info']
    icodes     = protein.get('residue_icodes') or [' '] * len(res_names)
    beads      = protein['beads']

    chain_ids: set = set()
    chain   = None
    cur_cid = None
    for ri, rname in enumerate(res_names):
        cid = res_seq[ri][0]
        if chain is None or cid != cur_cid:
            label = cid if str(cid).strip() else 'P'
            chain = topology.addChain(label)
            chain_ids.add(label)
            cur_cid = cid
        icode   = icodes[ri] if str(icodes[ri]).strip() else ' '
        res_omm = topology.addResidue(rname, chain, str(res_seq[ri][1]), icode)
        lo_r, hi_r = res_ranges[ri]
        for b_idx in range(lo_r, hi_r):
            all_atoms.append(
                topology.addAtom(beads[b_idx]['bead_label'], element, res_omm))

    for i, j in protein['bond_list']:
        topology.addBond(all_atoms[i], all_atoms[j])
    for (bi, bj, _) in protein.get('ss_bonds', []):
        topology.addBond(all_atoms[bi], all_atoms[bj])
    return chain_ids


def _add_ss_bonds_to_force(bond_force, protein) -> int:
    """Add disulfide bridges (stiff SC1–SC1 bonds) to a HarmonicBondForce."""
    ss_bonds = protein.get('ss_bonds') or []
    for (bi, bj, r0) in ss_bonds:
        bond_force.addBond(bi, bj, r0 * nanometer,
                           _SS_BOND_K * kilojoules_per_mole / nanometer**2)
    return len(ss_bonds)


def _pre_equilibrate_protein_cg(
    protein: dict,
    temperature: float,
    pre_equil_steps: int,
    equil_timestep: float,
    use_gpu: bool,
) -> None:
    """CG NVT pre-equilibration in a water-only box before membrane embedding.

    Lets extended loops and disordered termini of partially unfolded proteins
    settle before the bilayer is placed around them.  The protein dict's
    bead positions are updated in-place, centred at the origin so the main
    pipeline's box-frame translation works unchanged.

    Protocol:
      1. Build a cubic CG water box with 1.5 nm buffer around the protein.
      2. Add neutralising ions only (no bulk salt — box is temporary).
      3. Energy minimise (up to 2 attempts with jitter on NaN).
      4. NVT MD for pre_equil_steps steps — no barostat, no BB restraints
         (protein is free to collapse toward lower-energy conformations).
      5. Extract protein bead positions; centre at origin.
    """
    prot_beads = protein['beads']
    prot_bonds = protein['bond_list']
    bb_local   = protein['bb_bead_indices']
    res_names  = protein['residue_names']
    bb_angs    = protein['bb_angles']
    bb_linked  = protein['bb_linked']
    N      = len(res_names)
    B_prot = len(prot_beads)

    # ── Cubic water box around protein ───────────────────────────────────
    pos_nm = np.array([b['pos_nm'] for b in prot_beads], dtype=np.float64)
    buf    = 1.5
    lo = pos_nm.min(axis=0) - buf
    hi = pos_nm.max(axis=0) + buf
    side = float((hi - lo).max())          # cubic: max edge length
    box_lengths = np.array([side, side, side])

    # Translate protein to box-frame centre
    prot_pos_box = pos_nm - lo + (box_lengths - (hi - lo)) / 2.0

    # Solvate: z_bilayer_outer=0 → exclude no region → fill entire box
    water_pos = _build_solvent_box(prot_pos_box, box_lengths, z_bilayer_outer=0.0)

    # Neutralise only (no bulk NaCl; this box is discarded after pre-equil)
    net_charge = sum(b['charge'] for b in prot_beads)
    water_pos, na_pos, cl_pos = _add_ions(water_pos, net_charge, box_lengths,
                                           salt_conc_M=0.0)
    n_w = len(water_pos); n_na = len(na_pos); n_cl = len(cl_pos)

    # ── OpenMM Topology ──────────────────────────────────────────────────
    topology = Topology()
    carbon   = Element.getBySymbol('C')
    all_atoms: list = []

    prot_cids = _add_protein_to_topology(topology, protein, carbon, all_atoms)

    if n_w > 0:
        sol_chain = topology.addChain(_pick_chain_id('S', prot_cids))
        for _ in range(n_w):
            r = topology.addResidue('SOL', sol_chain)
            all_atoms.append(topology.addAtom('W', carbon, r))
    if n_na + n_cl > 0:
        ion_chain = topology.addChain(_pick_chain_id('I', prot_cids))
        for _ in range(n_na):
            r = topology.addResidue('NA', ion_chain); all_atoms.append(
                topology.addAtom('NA', carbon, r))
        for _ in range(n_cl):
            r = topology.addResidue('CL', ion_chain); all_atoms.append(
                topology.addAtom('CL', carbon, r))

    bvx = Vec3(side, 0, 0) * nanometer
    bvy = Vec3(0, side, 0) * nanometer
    bvz = Vec3(0, 0, side) * nanometer
    topology.setPeriodicBoxVectors((bvx, bvy, bvz))

    pieces = [prot_pos_box]
    if n_w  > 0: pieces.append(water_pos)
    if n_na > 0: pieces.append(na_pos)
    if n_cl > 0: pieces.append(cl_pos)
    positions = [Vec3(*p) * nanometer for p in np.vstack(pieces)]

    # ── OpenMM System ────────────────────────────────────────────────────
    system = System()
    system.setDefaultPeriodicBoxVectors(bvx, bvy, bvz)
    for _ in range(B_prot + n_w + n_na + n_cl):
        system.addParticle(_BEAD_MASS_DA * dalton)

    bond_force = HarmonicBondForce()
    for i, j in prot_bonds:
        li = prot_beads[i]['bead_label']; lj = prot_beads[j]['bead_label']
        is_bb_bb = li == 'BB' and lj == 'BB'
        is_bb_sc = (li == 'BB') != (lj == 'BB')
        if is_bb_bb: kb, r0 = _BB_BB_K, _BB_BB_R0
        elif is_bb_sc: kb, r0 = _BB_SC_K, _BB_SC_R0
        else: kb, r0 = _SC_SC_K, _SC_SC_R0
        bond_force.addBond(i, j, r0 * nanometer,
                           kb * kilojoules_per_mole / nanometer**2)
    _add_ss_bonds_to_force(bond_force, protein)
    system.addForce(bond_force)

    angle_force = HarmonicAngleForce()
    for ri in range(1, N - 1):
        if not (bb_linked[ri - 1] and bb_linked[ri]):
            continue
        angle_force.addAngle(bb_local[ri-1], bb_local[ri], bb_local[ri+1],
                             bb_angs[ri] * radian,
                             _BB_ANG_K * kilojoules_per_mole / radian**2)
    system.addForce(angle_force)

    n_types = _PAIR_SIGMA.shape[0]
    lj = CustomNonbondedForce(
        "4*eps_tab(itype1,itype2)*"
        "((sig_tab(itype1,itype2)/r)^12-(sig_tab(itype1,itype2)/r)^6);"
    )
    lj.addTabulatedFunction('sig_tab',
        Discrete2DFunction(n_types, n_types, _PAIR_SIGMA.T.flatten().tolist()))
    lj.addTabulatedFunction('eps_tab',
        Discrete2DFunction(n_types, n_types, _PAIR_EPS.T.flatten().tolist()))
    lj.addPerParticleParameter('itype')
    lj.setNonbondedMethod(CustomNonbondedForce.CutoffPeriodic)
    lj.setCutoffDistance(_CUTOFF_NM * nanometer)

    es = CustomNonbondedForce(
        f"{_COULOMB_K:.8f}*charge1*charge2*(1/r+{_RF_K:.8f}*r^2-{_RF_C:.8f});"
    )
    es.addPerParticleParameter('charge')
    es.setNonbondedMethod(CustomNonbondedForce.CutoffPeriodic)
    es.setCutoffDistance(_CUTOFF_NM * nanometer)

    w_idx = _type_idx('W'); ion_idx = _type_idx('ION')
    for b in prot_beads:
        lj.addParticle([_type_idx(b['bead_type'])]); es.addParticle([b['charge']])
    for _ in range(n_w):
        lj.addParticle([w_idx]); es.addParticle([0.0])
    for _ in range(n_na):
        lj.addParticle([ion_idx]); es.addParticle([+1.0])
    for _ in range(n_cl):
        lj.addParticle([ion_idx]); es.addParticle([-1.0])
    for i, j in protein['openmm_exclusions']:
        lj.addExclusion(i, j); es.addExclusion(i, j)
    system.addForce(lj); system.addForce(es)
    # Barostat starts disabled (freq=0) → this is a pure NVT pre-equil
    system.addForce(MonteCarloBarostat(1.0 * bar, temperature * kelvin, 0))

    # ── Minimise (2 attempts) ────────────────────────────────────────────
    cur_pos = positions
    init_nm = np.array([[float(p[0].value_in_unit(nanometer)),
                         float(p[1].value_in_unit(nanometer)),
                         float(p[2].value_in_unit(nanometer))]
                        for p in positions], dtype=np.float64)
    rng = np.random.default_rng(7)
    for attempt in range(2):
        sim = _new_sim(topology, system, cur_pos, None,
                       temperature, equil_timestep, use_gpu)
        failed = False
        try:
            sim.minimizeEnergy(maxIterations=1000)
            state   = sim.context.getState(getPositions=True, enforcePeriodicBox=True)
            pos_arr = np.asarray(
                state.getPositions(asNumpy=True).value_in_unit(nanometer))
            del state
            if np.any(np.isnan(pos_arr)):
                failed = True
            else:
                cur_pos = [Vec3(*p) * nanometer for p in pos_arr]
        except Exception:
            failed = True
        _teardown(sim)
        if not failed:
            break
        if attempt < 1:
            jitted  = init_nm + rng.uniform(-0.05, 0.05, init_nm.shape)
            cur_pos = [Vec3(*p) * nanometer for p in jitted]

    # ── NVT equilibration ────────────────────────────────────────────────
    sim = _new_sim(topology, system, cur_pos, None,
                   temperature, equil_timestep, use_gpu)
    sim.context.setVelocitiesToTemperature(temperature * kelvin)
    sim.reporters.append(StateDataReporter(
        sys.stdout, max(pre_equil_steps // 5, 1000),
        step=True, potentialEnergy=True, temperature=True, separator='\t',
    ))
    sim.step(pre_equil_steps)
    state   = sim.context.getState(getPositions=True, enforcePeriodicBox=True)
    pos_arr = np.asarray(
        state.getPositions(asNumpy=True).value_in_unit(nanometer), dtype=np.float64)
    del state
    _teardown(sim)

    # ── Update protein dict in-place (centred at origin) ─────────────────
    prot_final = pos_arr[:B_prot]
    prot_final -= prot_final.mean(axis=0)
    for b_idx, b in enumerate(prot_beads):
        b['pos_nm'] = prot_final[b_idx]

    print(f"  CG water pre-equilibration done: {pre_equil_steps:,} steps "
          f"({pre_equil_steps * equil_timestep / 1e6:.3f} ns). "
          f"Protein COM centred at origin for membrane embedding.")


# ---------------------------------------------------------------------------
# Step 4: Build OpenMM Topology + System
# ---------------------------------------------------------------------------

def _type_idx(btype: str) -> float:
    return float(_BEAD_TYPE_INDEX.get(btype, _BEAD_TYPE_INDEX['C']))


def build_openmm_system(
    protein: dict,
    lipid_beads: list,
    lipid_bonds: list,
    lipid_angles: list,
    water_pos: np.ndarray,
    na_pos: np.ndarray,
    cl_pos: np.ndarray,
    box_lengths: np.ndarray,
    temperature: float,
) -> tuple:
    """Assemble OpenMM Topology + System for the full CG membrane system.

    Particle order: protein beads | lipid beads | W beads | Na⁺ | Cl⁻

    Returns:
        (topology, system, positions, bb_global_indices)
    """
    prot_beads = protein['beads']
    prot_bonds = protein['bond_list']
    bb_local   = protein['bb_bead_indices']
    res_names  = protein['residue_names']
    bb_angs    = protein['bb_angles']
    bb_linked  = protein['bb_linked']
    N          = len(res_names)

    B_prot  = len(prot_beads)
    B_lip   = len(lipid_beads)
    n_w     = len(water_pos)
    n_na    = len(na_pos)
    n_cl    = len(cl_pos)

    # ── Topology ─────────────────────────────────────────────────────────
    topology = Topology()
    carbon   = Element.getBySymbol('C')
    all_atoms: list = []

    prot_cids = _add_protein_to_topology(topology, protein, carbon, all_atoms)

    if B_lip > 0:
        lip_chain = topology.addChain(_pick_chain_id('L', prot_cids))
        prev_lid  = None
        lip_res   = None
        for lb in lipid_beads:
            if lb['lipid_idx'] != prev_lid:
                lip_res  = topology.addResidue('LIP', lip_chain)
                prev_lid = lb['lipid_idx']
            all_atoms.append(topology.addAtom(lb['bead_label'], carbon, lip_res))
        for (i, j, _, _) in lipid_bonds:
            topology.addBond(all_atoms[B_prot + i], all_atoms[B_prot + j])

    if n_w > 0:
        sol_chain = topology.addChain(_pick_chain_id('S', prot_cids))
        for _ in range(n_w):
            r = topology.addResidue('SOL', sol_chain)
            all_atoms.append(topology.addAtom('W', carbon, r))

    if n_na + n_cl > 0:
        ion_chain = topology.addChain(_pick_chain_id('I', prot_cids))
        for _ in range(n_na):
            r = topology.addResidue('NA', ion_chain)
            all_atoms.append(topology.addAtom('NA', carbon, r))
        for _ in range(n_cl):
            r = topology.addResidue('CL', ion_chain)
            all_atoms.append(topology.addAtom('CL', carbon, r))

    Lx, Ly, Lz = box_lengths
    bvx = Vec3(Lx, 0, 0) * nanometer
    bvy = Vec3(0, Ly, 0) * nanometer
    bvz = Vec3(0, 0, Lz) * nanometer
    topology.setPeriodicBoxVectors((bvx, bvy, bvz))

    # ── Positions ────────────────────────────────────────────────────────
    pieces = [np.array([b['pos_nm'] for b in prot_beads])]
    if B_lip > 0:
        pieces.append(np.array([b['pos_nm'] for b in lipid_beads]))
    if n_w > 0:
        pieces.append(water_pos)
    if n_na > 0:
        pieces.append(na_pos)
    if n_cl > 0:
        pieces.append(cl_pos)
    positions = [Vec3(*p) * nanometer for p in np.vstack(pieces)]

    # ── System ───────────────────────────────────────────────────────────
    system = System()
    system.setDefaultPeriodicBoxVectors(bvx, bvy, bvz)
    for _ in range(B_prot + B_lip):
        system.addParticle(_BEAD_MASS_DA * dalton)
    for _ in range(n_w):
        system.addParticle(_WATER_MASS * dalton)
    for _ in range(n_na):
        system.addParticle(_ION_MASS_NA * dalton)
    for _ in range(n_cl):
        system.addParticle(_ION_MASS_CL * dalton)

    # Harmonic bonds — protein
    bond_force = HarmonicBondForce()
    for i, j in prot_bonds:
        li = prot_beads[i]['bead_label']
        lj = prot_beads[j]['bead_label']
        is_bb_bb = li == 'BB' and lj == 'BB'
        is_bb_sc = (li == 'BB') != (lj == 'BB')
        if is_bb_bb:
            kb, r0 = _BB_BB_K, _BB_BB_R0
        elif is_bb_sc:
            kb, r0 = _BB_SC_K, _BB_SC_R0
        else:
            kb, r0 = _SC_SC_K, _SC_SC_R0
        bond_force.addBond(i, j, r0 * nanometer,
                           kb * kilojoules_per_mole / nanometer**2)

    # Harmonic bonds — disulfide bridges (intra- and inter-chain)
    _add_ss_bonds_to_force(bond_force, protein)

    # Harmonic bonds — lipid
    for (i, j, kb, r0) in lipid_bonds:
        bond_force.addBond(
            B_prot + i, B_prot + j,
            r0 * nanometer,
            kb * kilojoules_per_mole / nanometer**2,
        )
    system.addForce(bond_force)

    # Backbone angles — protein (skipped across chain boundaries and gaps)
    angle_force = HarmonicAngleForce()
    for ri in range(1, N - 1):
        if not (bb_linked[ri - 1] and bb_linked[ri]):
            continue
        angle_force.addAngle(
            bb_local[ri - 1], bb_local[ri], bb_local[ri + 1],
            bb_angs[ri] * radian,
            _BB_ANG_K * kilojoules_per_mole / radian**2,
        )

    # Angles — lipid
    for (i, j, k, kang, th0) in lipid_angles:
        angle_force.addAngle(
            B_prot + i, B_prot + j, B_prot + k,
            th0 * radian,
            kang * kilojoules_per_mole / radian**2,
        )
    system.addForce(angle_force)

    # LJ-12-6 via Discrete2DFunction (explicit Martini 3 pair parameters)
    n_types  = _PAIR_SIGMA.shape[0]
    sig_flat = _PAIR_SIGMA.T.flatten().tolist()
    eps_flat = _PAIR_EPS.T.flatten().tolist()
    lj = CustomNonbondedForce(
        "4*eps_tab(itype1,itype2)*"
        "((sig_tab(itype1,itype2)/r)^12 - (sig_tab(itype1,itype2)/r)^6);"
    )
    lj.addTabulatedFunction('sig_tab', Discrete2DFunction(n_types, n_types, sig_flat))
    lj.addTabulatedFunction('eps_tab', Discrete2DFunction(n_types, n_types, eps_flat))
    lj.addPerParticleParameter('itype')
    lj.setNonbondedMethod(CustomNonbondedForce.CutoffPeriodic)
    lj.setCutoffDistance(_CUTOFF_NM * nanometer)

    # Reaction-field Coulomb
    es = CustomNonbondedForce(
        f"{_COULOMB_K:.8f}*charge1*charge2*(1/r + {_RF_K:.8f}*r^2 - {_RF_C:.8f});"
    )
    es.addPerParticleParameter('charge')
    es.setNonbondedMethod(CustomNonbondedForce.CutoffPeriodic)
    es.setCutoffDistance(_CUTOFF_NM * nanometer)

    for b in prot_beads:
        lj.addParticle([_type_idx(b['bead_type'])]); es.addParticle([b['charge']])
    for lb in lipid_beads:
        lj.addParticle([_type_idx(lb['bead_type'])]); es.addParticle([lb['charge']])
    w_idx = _type_idx('W'); ion_idx = _type_idx('ION')
    for _ in range(n_w):
        lj.addParticle([w_idx]); es.addParticle([0.0])
    for _ in range(n_na):
        lj.addParticle([ion_idx]); es.addParticle([+1.0])
    for _ in range(n_cl):
        lj.addParticle([ion_idx]); es.addParticle([-1.0])

    # Protein 1-2/1-3 exclusions
    for i, j in protein['openmm_exclusions']:
        lj.addExclusion(i, j); es.addExclusion(i, j)

    # Lipid 1-2/1-3 exclusions
    lip_nbrs: dict = defaultdict(set)
    for (i, j, _, _) in lipid_bonds:
        lip_nbrs[i].add(j); lip_nbrs[j].add(i)
    lip_excl: set = set()
    for b in range(B_lip):
        for n1 in list(lip_nbrs[b]):
            lip_excl.add((min(b, n1), max(b, n1)))
            for n2 in list(lip_nbrs[n1]):
                if n2 != b:
                    lip_excl.add((min(b, n2), max(b, n2)))
    for i, j in lip_excl:
        lj.addExclusion(B_prot + i, B_prot + j)
        es.addExclusion(B_prot + i, B_prot + j)

    system.addForce(lj)
    system.addForce(es)

    # Isotropic barostat initially disabled (freq=0); replaced by membrane
    # barostat in NPT step.
    system.addForce(MonteCarloBarostat(1.0 * bar, temperature * kelvin, 0))

    bb_global = list(bb_local)   # protein BB bead global indices (protein first)
    print(f"  System: {B_prot} protein + {B_lip} lipid + {n_w} W + "
          f"{n_na} Na⁺ + {n_cl} Cl⁻  ({B_prot+B_lip+n_w+n_na+n_cl} total beads)")
    return topology, system, positions, bb_global


# ---------------------------------------------------------------------------
# BB-bead restraints (mirrors atomistic script's CA restraint approach)
# ---------------------------------------------------------------------------

def add_bb_restraints(
    system,
    bb_global_indices: list,
    positions: list,
    k: float = 500.0,
) -> tuple:
    """Add harmonic Cartesian restraints on protein backbone (BB) beads.

    Returns (restraint_force, bb_indices) so reference positions can be updated
    each chunk during NPT to track barostat coordinate rescaling.
    """
    if not bb_global_indices:
        return None, []
    restraint = CustomExternalForce(
        "0.5*k*((x-x0)^2 + (y-y0)^2 + (z-z0)^2)"
    )
    restraint.addGlobalParameter('k', k * kilojoules_per_mole / nanometer**2)
    restraint.addPerParticleParameter('x0')
    restraint.addPerParticleParameter('y0')
    restraint.addPerParticleParameter('z0')
    for idx in bb_global_indices:
        ref = list(positions[idx].value_in_unit(nanometers))
        restraint.addParticle(idx, ref)
    system.addForce(restraint)
    print(f"  BB restraints on {len(bb_global_indices)} beads (k={k} kJ/mol/nm²)")
    return restraint, bb_global_indices


def add_enm_springs(
    system,
    protein: dict,
    positions: list,
    k_enm: float = 500.0,
    max_seq_sep: int = 4,
    intra_helix: bool = True,
    sheet: bool = False,
    interchain: bool = False,
    cutoff_nm: float = _ENM_MAX_DIST_NM,
    k_interchain: float = _ENM_IC_K,
) -> list:
    """Add harmonic ENM springs between BB beads.  Three opt-in spring sets:

      intra_helix  pairs (i, j) inside one helix run, |i − j| ∈ [2, max_seq_sep]
                   and r < _BB_ENM_MAX_DIST_NM.  Stiffens helix geometry.
      sheet        pairs of β-residues in the same chain with |i − j| ≥
                   _ENM_SHEET_MIN_SEP and r < cutoff_nm.  Cross-strand springs
                   are what hold an Ig-like β-sandwich together — the backbone
                   angle term alone does not, so all-β domains (antibody/TCR
                   ectodomains) deform without this.
      interchain   BB pairs in *different* chains with r < cutoff_nm.  Keeps a
                   multimer assembled once the positional restraints are
                   released and through the annealing stage.  It pins the
                   interface at its input geometry, so do NOT use it when the
                   question is association, dissociation or interface
                   rearrangement — only to hold a known assembly together.

    Springs are harmonic in distance, so unlike positional restraints they need
    no reference update when the barostat rescales coordinates.  Inter-chain
    springs get their own force and their own constant (k_interchain).

    Helix and sheet runs never cross a chain break (protein['bb_linked']).

    Returns the list of CustomBondForce objects added to the system.
    """
    bb_local  = protein['bb_bead_indices']
    bb_angles = protein['bb_angles']
    res_seq   = protein['residue_seq_info']
    N = len(bb_angles)
    bb_linked = protein.get('bb_linked')
    if bb_linked is None:
        bb_linked = [res_seq[i][0] == res_seq[i + 1][0] for i in range(N - 1)]

    pos_nm = np.array(
        [list(positions[idx].value_in_unit(nanometers)) for idx in range(len(positions))],
        dtype=np.float64,
    )
    bb_idx = np.asarray(bb_local, dtype=int)
    bb_pos = pos_nm[bb_idx]                      # (N, 3) BB bead coordinates
    chains = np.asarray([res_seq[ri][0] for ri in range(N)], dtype=object)

    springs: dict    = {}      # (bead_i, bead_j) → r0, intra-chain
    ic_springs: dict = {}      # (bead_i, bead_j) → r0, inter-chain

    def _add(store, ri, rj, r0):
        bi, bj = int(bb_idx[ri]), int(bb_idx[rj])
        store.setdefault((min(bi, bj), max(bi, bj)), r0)

    # ── intra-helix ──────────────────────────────────────────────────────
    helix_runs: list = []
    if intra_helix:
        is_helix = [abs(a - _BB_ANG_HELIX) < 0.05 for a in bb_angles]
        i = 0
        while i < N:
            if is_helix[i]:
                j = i + 1
                while j < N and is_helix[j] and bb_linked[j - 1]:
                    j += 1
                if j - i >= 4:
                    helix_runs.append((i, j))
                i = j
            else:
                i += 1

        for (h_lo, h_hi) in helix_runs:
            for ri in range(h_lo, h_hi):
                for rj in range(ri + 2, min(ri + max_seq_sep + 1, h_hi)):
                    r0 = float(np.linalg.norm(bb_pos[rj] - bb_pos[ri]))
                    if r0 < _BB_ENM_MAX_DIST_NM:
                        _add(springs, ri, rj, r0)
    n_helix = len(springs)

    # ── β-sheet (intra-chain, cross-strand included) ─────────────────────
    if sheet:
        sheet_res = np.array(
            [ri for ri in range(N) if abs(bb_angles[ri] - _BB_ANG_SHEET) < 0.05],
            dtype=int,
        )
        for pos_a, ri in enumerate(sheet_res):
            partners = sheet_res[pos_a + 1:]
            if partners.size == 0:
                continue
            d    = np.linalg.norm(bb_pos[partners] - bb_pos[ri], axis=1)
            mask = ((partners - ri) >= _ENM_SHEET_MIN_SEP) & (d < cutoff_nm) & \
                   (chains[partners] == chains[ri])
            for rj, r0 in zip(partners[mask], d[mask]):
                _add(springs, int(ri), int(rj), float(r0))
    n_sheet = len(springs) - n_helix

    # ── inter-chain interface ────────────────────────────────────────────
    if interchain:
        for ri in range(N - 1):
            partners = np.arange(ri + 1, N)
            d    = np.linalg.norm(bb_pos[partners] - bb_pos[ri], axis=1)
            mask = (d < cutoff_nm) & (chains[partners] != chains[ri])
            for rj, r0 in zip(partners[mask], d[mask]):
                _add(ic_springs, ri, int(rj), float(r0))

    forces: list = []
    if springs:
        f = CustomBondForce("0.5*k_enm*(r-r0)^2")
        f.addGlobalParameter('k_enm', k_enm * kilojoules_per_mole / nanometer**2)
        f.addPerBondParameter('r0')
        for (bi, bj), r0 in springs.items():
            f.addBond(bi, bj, [r0])
        system.addForce(f)
        forces.append(f)
    if ic_springs:
        f = CustomBondForce("0.5*k_enm_ic*(r-r0)^2")
        f.addGlobalParameter('k_enm_ic',
                             k_interchain * kilojoules_per_mole / nanometer**2)
        f.addPerBondParameter('r0')
        for (bi, bj), r0 in ic_springs.items():
            f.addBond(bi, bj, [r0])
        system.addForce(f)
        forces.append(f)

    if intra_helix:
        print(f"  Intra-helix ENM: {n_helix} springs over {len(helix_runs)} "
              f"helix run(s) (k={k_enm:.0f} kJ/mol/nm²)")
    if sheet:
        print(f"  β-sheet ENM: {n_sheet} springs "
              f"(|i-j| ≥ {_ENM_SHEET_MIN_SEP}, r < {cutoff_nm:.2f} nm, "
              f"k={k_enm:.0f} kJ/mol/nm²)")
    if interchain:
        print(f"  Inter-chain ENM: {len(ic_springs)} interface springs "
              f"(r < {cutoff_nm:.2f} nm, k={k_interchain:.0f} kJ/mol/nm²) — "
              f"interface is pinned to its input geometry")
    return forces


# ---------------------------------------------------------------------------
# Simulation helpers
# ---------------------------------------------------------------------------

def _get_platform(use_gpu: bool) -> Platform:
    if use_gpu:
        try:
            return Platform.getPlatformByName('CUDA')
        except Exception:
            print("WARNING: CUDA not available, falling back to CPU.")
    return Platform.getPlatformByName('CPU')


def _new_sim(
    topology,
    system,
    positions,
    box_vectors,
    temperature: float,
    timestep_fs: float,
    use_gpu: bool,
) -> Simulation:
    platform = _get_platform(use_gpu)
    props = {'CudaPrecision': 'mixed'} if platform.getName() == 'CUDA' else {}
    integrator = LangevinMiddleIntegrator(
        temperature * kelvin,
        DEFAULT_FRICTION / picosecond,
        timestep_fs * femtoseconds,
    )
    sim = Simulation(topology, system, integrator, platform, props)
    if box_vectors is not None:
        sim.context.setPeriodicBoxVectors(*box_vectors)
    sim.context.setPositions(positions)
    return sim


def _get_state(sim: Simulation) -> tuple:
    state  = sim.context.getState(
        getPositions=True, getEnergy=True, enforcePeriodicBox=True
    )
    pos    = state.getPositions()
    raw_bv = state.getPeriodicBoxVectors()
    # Normalize to plain Vec3*nm — state.getPeriodicBoxVectors() can return
    # Vec3-of-Quantity objects; if stored in topology as-is, PDBFile.writeHeader
    # later calls float() on the components and gets back another Quantity
    # ("Quantity.__float__ returned non-float (type Quantity)").
    bv = tuple(
        Vec3(float(v[0].value_in_unit(nanometer)),
             float(v[1].value_in_unit(nanometer)),
             float(v[2].value_in_unit(nanometer))) * nanometer
        for v in raw_bv
    )
    energy = state.getPotentialEnergy().value_in_unit(kilojoules_per_mole)
    del state
    return pos, bv, energy


def _teardown(sim):
    sim.reporters.clear()
    del sim
    for _ in range(3):
        gc.collect()


def _write_cg_pdb(
    topology,
    positions,
    path: Path,
    strip_water: bool = True,
    strip_ions: bool = False,
    box_lengths_nm=None,
) -> None:
    """Write CG PDB, optionally removing SOL (W) beads and/or ions.

    PDBFile.writeHeader internally calls computeLengthsAndAngles which, in some
    OpenMM + Python 3.11 combinations, returns nested Quantity objects that
    cause "Quantity.__float__ returned non-float (type Quantity)" when %f tries
    to format them.  We bypass writeHeader entirely: write CRYST1 manually from
    the plain-float box_lengths_nm array, then call PDBFile.writeModel +
    PDBFile.writeFooter directly.
    """
    strip_names: set = set()
    if strip_water:
        strip_names.add('SOL')
    if strip_ions:
        strip_names.update({'NA', 'CL'})

    write_top = topology
    write_pos = positions
    new_top   = False   # True when write_top is a fresh Modeller topology

    if strip_names:
        mod    = Modeller(topology, positions)
        to_del = [a for a in topology.atoms()
                  if a.residue.name.upper() in strip_names]
        if to_del:
            mod.delete(to_del)   # creates a new topology stored in mod.topology
            write_top = mod.topology
            write_pos = mod.positions
            new_top   = True

    # Clear box vectors on the write topology so PDBFile.writeModel does not
    # try to use them (it doesn't need them for ATOM records).
    # If write_top is still the caller's topology (no atoms stripped), save and
    # restore so the outer simulation state is not affected.
    saved_bv = None
    if not new_top:
        saved_bv = write_top.getPeriodicBoxVectors()
    write_top.setPeriodicBoxVectors(None)

    with open(path, 'w') as fh:
        if box_lengths_nm is not None:
            Lx = float(box_lengths_nm[0]) * 10.0   # nm → Å
            Ly = float(box_lengths_nm[1]) * 10.0
            Lz = float(box_lengths_nm[2]) * 10.0
            fh.write("CRYST1%9.3f%9.3f%9.3f%7.2f%7.2f%7.2f P 1           1 \n"
                     % (Lx, Ly, Lz, 90.0, 90.0, 90.0))
        # keepIds=True → original chain ids / residue numbers / insertion codes
        # survive into the output, so multimer subunits stay identifiable.
        PDBFile.writeModel(write_top, write_pos, fh, keepIds=True)
        PDBFile.writeFooter(write_top, fh)

    if saved_bv is not None:
        write_top.setPeriodicBoxVectors(saved_bv)

    print(f"  Saved: {path}")


def _remove_force_by_type(system, force_class):
    """Remove the first force of a given class from system (backward scan)."""
    for fi in range(system.getNumForces() - 1, -1, -1):
        if isinstance(system.getForce(fi), force_class):
            system.removeForce(fi)
            return


# ---------------------------------------------------------------------------
# Transmembrane orientation (align TM helix axis with Z, center TM at midplane)
# ---------------------------------------------------------------------------

def parse_tm_ranges(spec: str) -> list:
    """Parse a TM-residue spec into [(chain, start, end), ...].

    Format: comma-separated "CHAIN:START-END" segments, e.g.
    "A:34-58,A:70-92". Chain may be omitted ("34-58") and defaults to 'A'.
    """
    ranges = []
    for seg in spec.split(','):
        seg = seg.strip()
        if not seg:
            continue
        chain, _, span = seg.rpartition(':')
        chain = chain.strip() or 'A'
        lo, _, hi = span.partition('-')
        ranges.append((chain, int(lo), int(hi)))
    return ranges


class TMRangeError(ValueError):
    """--tm-residues ranges failed the sanity checks in check_tm_ranges."""


def check_tm_ranges(protein: dict, tm_ranges: list, strict: bool = True) -> None:
    """Check each --tm-residues range before the protein is oriented on it.

    A range is flagged if it is
      * missing:     fewer than _TM_MIN_COVERAGE of its residues are in the input
                     (wrong chain id, or numbering that doesn't match the file,
                     e.g. UniProt numbering for a PDB with author numbering);
      * not helical: fewer than _TM_MIN_HELIX_FRACTION of its residues are helix
                     in the input secondary structure. The CG model keeps the
                     input secondary structure, so a coil or strand stays one;
      * polar:       fewer than _TM_MIN_HYDROPHOBIC_FRAC of its residues are
                     hydrophobic (AILMFVWC).
    Prints one summary line per range. Raises TMRangeError if any range is
    flagged and strict is True; otherwise prints a warning and returns.
    """
    from Bio.PDB.Polypeptide import index_to_one, three_to_index

    residue_names = protein['residue_names']
    residue_seq_info = protein['residue_seq_info']
    bb_angles = protein['bb_angles']
    chains = sorted({ch for ch, _ in residue_seq_info})

    print(f"  Checking {len(tm_ranges)} --tm-residues range(s):")
    failures = []
    for chain, lo, hi in tm_ranges:
        label = f"{chain}:{lo}-{hi}"
        if chain not in chains:
            print(f"    {label:<14} chain {chain!r} not in structure")
            failures.append(f"{label}: chain {chain!r} is not in the structure "
                            f"(chains: {', '.join(chains)})")
            continue

        idxs = [ri for ri, (ch, rs) in enumerate(residue_seq_info)
                if ch == chain and lo <= rs <= hi]
        n_expected = hi - lo + 1
        coverage = len(idxs) / n_expected if n_expected > 0 else 0.0
        if coverage < _TM_MIN_COVERAGE:
            present = [rs for ch, rs in residue_seq_info if ch == chain]
            print(f"    {label:<14} {len(idxs)}/{n_expected} residues present")
            failures.append(f"{label}: missing — only {len(idxs)}/{n_expected} "
                            f"residues are in the structure (chain {chain} has "
                            f"residues {min(present)}-{max(present)})")
            continue

        seq = ''.join(index_to_one(three_to_index(residue_names[ri]))
                      for ri in idxs)
        helix_frac = sum(abs(bb_angles[ri] - _BB_ANG_HELIX) < 1e-6
                         for ri in idxs) / len(idxs)
        hydrophobic_frac = sum(aa in _TM_HYDROPHOBIC for aa in seq) / len(seq)

        problems = []
        if helix_frac < _TM_MIN_HELIX_FRACTION:
            problems.append(f"not helical ({helix_frac:.0%} helix)")
        if hydrophobic_frac < _TM_MIN_HYDROPHOBIC_FRAC:
            problems.append(f"mostly polar ({hydrophobic_frac:.0%} hydrophobic)")
        status = "; ".join(problems) if problems else "ok"
        print(f"    {label:<14} {len(idxs)}/{n_expected} present, "
              f"{helix_frac:.0%} helix, {hydrophobic_frac:.0%} hydrophobic  "
              f"{seq}  [{status}]")
        if problems:
            failures.append(f"{label} ({seq}): {'; '.join(problems)}")

    if not failures:
        return
    lines = (["--tm-residues check failed:"]
             + [f"  - {f}" for f in failures]
             + ["Ranges use the residue numbering in the input file, which often "
                "differs from UniProt numbering. TM helices must already be "
                "helical in the input; the CG model does not fold them."])
    if strict:
        raise TMRangeError("\n".join(lines + ["Pass --skip-tm-checks to run anyway."]))
    print("  WARNING: " + "\n  ".join(lines))


def _rotation_align_vectors(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Rotation matrix R such that R @ a is parallel to b (Rodrigues)."""
    a = a / (np.linalg.norm(a) + 1e-12)
    b = b / (np.linalg.norm(b) + 1e-12)
    v = np.cross(a, b)
    c = float(np.dot(a, b))
    s = float(np.linalg.norm(v))
    if s < 1e-8:                       # already (anti)parallel
        if c > 0:
            return np.eye(3)
        # 180°: rotate about any axis perpendicular to a
        perp = np.array([1.0, 0.0, 0.0])
        if abs(a[0]) > 0.9:
            perp = np.array([0.0, 1.0, 0.0])
        axis = np.cross(a, perp)
        axis /= np.linalg.norm(axis)
        return 2.0 * np.outer(axis, axis) - np.eye(3)
    vx = np.array([[0, -v[2], v[1]],
                   [v[2], 0, -v[0]],
                   [-v[1], v[0], 0]])
    return np.eye(3) + vx + vx @ vx * ((1.0 - c) / (s * s))


def _tm_bb_indices_by_helix(protein: dict, tm_ranges: list) -> list:
    """For each TM range, the indices (into protein['beads']) of its BB beads."""
    seq_info = protein['residue_seq_info']         # [(chain, resseq), ...] per residue
    per_helix = []
    for chain, lo, hi in tm_ranges:
        idxs = []
        for bi, bead in enumerate(protein['beads']):
            if bead['bead_label'] != 'BB':
                continue
            ch, rs = seq_info[bead['residue_idx']]
            if ch == chain and lo <= rs <= hi:
                idxs.append(bi)
        per_helix.append(idxs)
    return per_helix


def orient_tm_along_z(prot_pos_nm: np.ndarray, protein: dict,
                      tm_ranges: list) -> tuple:
    """Rotate bead positions so the TM helix axis is along +Z, and center the
    TM region at z=0 (bilayer midplane); XY is centered on the protein COM.

    The membrane normal is the hemisphere-averaged first principal axis of each
    TM helix's BB beads (handles multi-pass bundles, including antiparallel
    helices, since axes are folded to a common hemisphere before averaging).
    Returns (new_positions_nm, n_tm_residues).
    """
    per_helix = _tm_bb_indices_by_helix(protein, tm_ranges)
    all_tm = [i for h in per_helix for i in h]
    if not all_tm:
        raise ValueError(
            "No BB beads matched the given --tm-residues ranges "
            "(check chain IDs and residue numbering).")

    # Per-helix principal axis (SVD), folded to a common hemisphere, averaged.
    ref_axis = None
    axes = []
    for idxs in per_helix:
        if len(idxs) < 3:
            continue                                   # too short for a stable axis
        pts = prot_pos_nm[idxs]
        pts = pts - pts.mean(axis=0)
        axis = np.linalg.svd(pts, full_matrices=False)[2][0]
        if ref_axis is None:
            ref_axis = axis
        if np.dot(axis, ref_axis) < 0:                 # fold to common hemisphere
            axis = -axis
        axes.append(axis)
    if not axes:                                       # fall back to combined PCA
        pts = prot_pos_nm[all_tm]
        pts = pts - pts.mean(axis=0)
        normal = np.linalg.svd(pts, full_matrices=False)[2][0]
    else:
        normal = np.mean(axes, axis=0)
    normal /= np.linalg.norm(normal)

    R = _rotation_align_vectors(normal, np.array([0.0, 0.0, 1.0]))
    rotated = prot_pos_nm @ R.T

    rotated[:, :2] -= rotated[:, :2].mean(axis=0)      # XY on protein COM
    rotated[:, 2] -= rotated[all_tm, 2].mean()         # Z on TM-region midpoint

    n_tm_res = len({(protein['residue_seq_info'][protein['beads'][i]['residue_idx']])
                    for i in all_tm})
    return rotated, n_tm_res


# ---------------------------------------------------------------------------
# Main CG membrane MD pipeline
# ---------------------------------------------------------------------------

def run_cg_membrane_md(
    pdb_path: str,
    output_dir: str = None,
    lipid_type: str = 'POPC',
    temperature: float = DEFAULT_TEMPERATURE,
    pressure: float = DEFAULT_PRESSURE,
    surface_tension: float = DEFAULT_SURFACE_TENSION,
    padding_nm: float = DEFAULT_XY_PADDING_NM,
    water_layer_nm: float = DEFAULT_WATER_LAYER_NM,
    z_po4_nm: float = DEFAULT_Z_PO4_NM,
    equil_timestep: float = DEFAULT_EQUIL_TIMESTEP,
    prod_timestep: float = DEFAULT_PROD_TIMESTEP,
    nvt_steps: int = DEFAULT_NVT_STEPS,
    npt_steps: int = DEFAULT_NPT_STEPS,
    early_prod_steps: int = DEFAULT_EARLY_PROD_STEPS,
    stability_steps: int = DEFAULT_STABILITY_STEPS,
    report_interval: int = DEFAULT_REPORT_INTERVAL,
    snapshot_interval: int = None,
    use_gpu: bool = False,
    strip_water_in_output: bool = True,
    save_dcd: bool = False,
    refine: bool = False,
    pre_equil_steps: int = 50000,
    k_nvt: float = DEFAULT_K_NVT,
    use_enm: bool = False,
    enm_k: float = 500.0,
    enm_sheet: bool = False,
    enm_interchain: bool = False,
    enm_cutoff: float = _ENM_MAX_DIST_NM,
    enm_interchain_k: float = _ENM_IC_K,
    detect_disulfides: bool = True,
    n_replicas: int = DEFAULT_N_REPLICAS,
    sa_cycles: int = DEFAULT_SA_CYCLES,
    sa_t_high: float = DEFAULT_SA_T_HIGH_K,
    sa_steps_per_half: int = DEFAULT_SA_STEPS_HALF,
    tm_residues: list = None,
    check_tm: bool = True,
) -> None:
    pdb_path = Path(pdb_path)
    out_dir  = Path(output_dir) if output_dir else pdb_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    stem     = pdb_path.stem
    _t0      = time.time()

    # ------------------------------------------------------------------
    # Step 1: PDB → CG protein
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Step 1: Converting protein to Martini CG representation")
    print("=" * 60)
    _t1 = time.time()
    protein = convert_pdb_to_cg_beads(str(pdb_path),
                                      detect_disulfides=detect_disulfides)
    N_res   = len(protein['residue_names'])
    B_prot  = len(protein['beads'])
    print(f"  {N_res} residues → {B_prot} CG beads  "
          f"({len(protein['bond_list'])} bonds)")

    if tm_residues:
        check_tm_ranges(protein, tm_residues, strict=check_tm)

    if refine:
        print(f"\n  [Refine] CG water pre-equilibration "
              f"({pre_equil_steps:,} steps / "
              f"{pre_equil_steps * equil_timestep / 1e6:.3f} ns) ...")
        _pre_equilibrate_protein_cg(
            protein, temperature, pre_equil_steps, equil_timestep, use_gpu
        )

    # Centre protein at origin (= bilayer midplane); then translate to box frame
    prot_pos = np.array([b['pos_nm'] for b in protein['beads']])
    if tm_residues:
        # Align TM helix axis with Z and put the TM region (not the whole-protein
        # COM) at the bilayer midplane — essential when large extramembrane
        # domains would otherwise pull the membrane off the TM segment.
        prot_pos, n_tm = orient_tm_along_z(prot_pos, protein, tm_residues)
        print(f"  Oriented TM helices along Z from {len(tm_residues)} range(s) "
              f"({n_tm} residues); TM region centred at z=0.")
    else:
        prot_pos -= prot_pos.mean(axis=0)   # centred at (0, 0, 0)

    prot_xy_extent = prot_pos[:, :2].max(axis=0) - prot_pos[:, :2].min(axis=0)
    box_xy = np.maximum(prot_xy_extent + 2 * padding_nm, 8.0)

    # z_bilayer_outer: outermost head-group z (PO4 + NC3 offset ≈ 0.40 nm)
    z_bilayer_outer = z_po4_nm + 0.45
    # Size the box in Z to contain BOTH the bilayer+water AND the protein's full
    # z-extent plus a water layer. Otherwise a tall protein (large extramembrane
    # domain) pokes through the periodic boundary into the opposite leaflet and
    # tears the bilayer apart. max|z| is used because the protein is centred on
    # its TM region (z=0), so extramembrane domains are asymmetric in z.
    prot_z_half = float(np.abs(prot_pos[:, 2]).max())
    box_z   = 2.0 * (max(z_bilayer_outer, prot_z_half) + water_layer_nm)
    mid_z   = box_z / 2.0

    # Translate to box frame: protein centred at (Lx/2, Ly/2, Lz/2)
    prot_pos[:, 0] += box_xy[0] / 2.0
    prot_pos[:, 1] += box_xy[1] / 2.0
    prot_pos[:, 2] += mid_z
    for b_idx, b in enumerate(protein['beads']):
        b['pos_nm'] = prot_pos[b_idx]

    box_lengths = np.array([box_xy[0], box_xy[1], box_z])
    print(f"  Box: {box_lengths[0]:.2f} × {box_lengths[1]:.2f} × {box_lengths[2]:.2f} nm")
    print(f"  Time: {time.time() - _t1:.1f} s")

    # ------------------------------------------------------------------
    # Step 2: Bilayer
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print(f"Step 2: Building {lipid_type} CG bilayer")
    print("=" * 60)
    _t2 = time.time()
    lipid_beads, lipid_bonds, lipid_angles = build_cg_bilayer(
        protein_pos_nm=prot_pos,
        box_lengths=box_lengths,
        lipid_type=lipid_type,
        z_po4=z_po4_nm,
    )
    print(f"  Time: {time.time() - _t2:.1f} s")

    # ------------------------------------------------------------------
    # Step 3: CG water + ions
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Step 3: CG solvation and ion placement")
    print("=" * 60)
    _t3 = time.time()
    lip_pos     = (np.array([b['pos_nm'] for b in lipid_beads])
                   if lipid_beads else np.empty((0, 3)))
    all_placed  = np.vstack([prot_pos, lip_pos]) if len(lip_pos) else prot_pos
    water_pos   = _build_solvent_box(all_placed, box_lengths, z_bilayer_outer)

    net_charge  = sum(b['charge'] for b in protein['beads'])
    net_charge += sum(lb['charge'] for lb in lipid_beads)   # POPC/DPPC: net 0
    water_pos, na_pos, cl_pos = _add_ions(water_pos, net_charge, box_lengths)
    print(f"  Time: {time.time() - _t3:.1f} s")

    # ------------------------------------------------------------------
    # Step 4: OpenMM system
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Step 4: Building OpenMM system")
    print("=" * 60)
    _t4 = time.time()
    topology, system, positions, bb_global = build_openmm_system(
        protein=protein,
        lipid_beads=lipid_beads,
        lipid_bonds=lipid_bonds,
        lipid_angles=lipid_angles,
        water_pos=water_pos,
        na_pos=na_pos,
        cl_pos=cl_pos,
        box_lengths=box_lengths,
        temperature=temperature,
    )
    print(f"  Time: {time.time() - _t4:.1f} s")

    # Running state
    cur_pos = positions
    cur_bv  = None

    def _mk_sim(ts_fs):
        return _new_sim(topology, system, cur_pos, cur_bv,
                        temperature, ts_fs, use_gpu)

    # ------------------------------------------------------------------
    # Step 5: Energy minimisation (up to 3 attempts with position jitter)
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Step 5: Energy minimisation")
    print("=" * 60)
    _t5 = time.time()

    init_pos_nm = np.array(
        [p.value_in_unit(nanometers) for p in positions], dtype=np.float64
    )
    rng = np.random.default_rng(42)

    for attempt in range(3):
        sim = _mk_sim(equil_timestep)
        failed = False
        try:
            sim.minimizeEnergy(maxIterations=2000)
            state  = sim.context.getState(getPositions=True, enforcePeriodicBox=True)
            pos_nm = np.asarray(
                state.getPositions(asNumpy=True).value_in_unit(nanometer)
            )
            del state
            if np.any(np.isnan(pos_nm)):
                failed = True
            else:
                cur_pos = [Vec3(*p) * nanometer for p in pos_nm]
        except Exception as exc:
            if 'NaN' not in str(exc) and 'nan' not in str(exc).lower():
                _teardown(sim); raise
            failed = True
        _teardown(sim)
        if not failed:
            break
        if attempt < 2:
            amp = 0.05 * (attempt + 1)
            print(f"  Attempt {attempt+1} NaN — jittering ±{amp:.2f} nm and retrying")
            jitted  = init_pos_nm + rng.uniform(-amp, amp, init_pos_nm.shape)
            cur_pos = [Vec3(*p) * nanometer for p in jitted]
        else:
            raise RuntimeError("Energy minimisation failed after 3 attempts")

    # Full convergence pass after successful coarse minimisation
    sim = _mk_sim(equil_timestep)
    sim.minimizeEnergy(maxIterations=0, tolerance=10.0)
    cur_pos, cur_bv, post_e = _get_state(sim)
    _teardown(sim)
    print(f"  Post-minimisation energy: {post_e:.1f} kJ/mol")
    _write_cg_pdb(topology, cur_pos,
                  out_dir / f"{stem}_cg_minimized.pdb",
                  strip_water=strip_water_in_output,
                  box_lengths_nm=box_lengths)
    print(f"  Time: {(time.time() - _t5) / 60:.1f} min")

    # ------------------------------------------------------------------
    # Step 6: NVT equilibration  (~310 K)
    #   Barostat disabled (freq=0).  Moderate BB restraints on TM helix
    #   backbone beads prevent drift while lipids and water equilibrate.
    #   Optional ENM springs (intra-helix, β-sheet, inter-chain) are added here
    #   and kept for the rest of the run; they are distance-based, so the
    #   barostat can rescale coordinates without invalidating them.
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print(f"Step 6: NVT equilibration ({nvt_steps:,} steps / "
          f"{nvt_steps * equil_timestep / 1e6:.3f} ns)")
    print("=" * 60)
    _t6 = time.time()

    restraint, restrained_bb = add_bb_restraints(
        system, bb_global, cur_pos, k=k_nvt
    )
    if use_enm or enm_sheet or enm_interchain:
        add_enm_springs(
            system, protein, cur_pos,
            k_enm=enm_k,
            intra_helix=use_enm,
            sheet=enm_sheet,
            interchain=enm_interchain,
            cutoff_nm=enm_cutoff,
            k_interchain=enm_interchain_k,
        )

    sim = _mk_sim(equil_timestep)
    sim.context.setVelocitiesToTemperature(temperature * kelvin)
    sim.reporters.append(StateDataReporter(
        sys.stdout, report_interval, step=True,
        potentialEnergy=True, temperature=True, separator='\t',
    ))
    sim.step(nvt_steps)
    cur_pos, cur_bv, nvt_e = _get_state(sim)
    _teardown(sim)

    _write_cg_pdb(topology, cur_pos,
                  out_dir / f"{stem}_cg_nvt.pdb",
                  strip_water=strip_water_in_output,
                  box_lengths_nm=box_lengths)
    print(f"  NVT final energy: {nvt_e:.1f} kJ/mol")
    print(f"  Time: {(time.time() - _t6) / 60:.1f} min")

    # ------------------------------------------------------------------
    # Step 7: NPT equilibration (semi-isotropic)
    #   Replace disabled isotropic barostat with MonteCarloMembraneBarostat
    #   (XY isotropic + Z free).  BB restraints and ENM remain active;
    #   reference positions updated each chunk to track barostat rescaling.
    #   Equilibrate until membrane area-per-lipid looks sane.
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print(f"Step 7: NPT equilibration ({npt_steps:,} steps / "
          f"{npt_steps * equil_timestep / 1e6:.3f} ns)")
    print("=" * 60)
    _t7 = time.time()

    _remove_force_by_type(system, MonteCarloBarostat)
    system.addForce(MonteCarloMembraneBarostat(
        pressure * bar,
        surface_tension * bar * nanometer,
        temperature * kelvin,
        MonteCarloMembraneBarostat.XYIsotropic,
        MonteCarloMembraneBarostat.ZFree,
        25,
    ))

    sim = _mk_sim(equil_timestep)
    sim.context.setVelocitiesToTemperature(temperature * kelvin)
    sim.reporters.append(StateDataReporter(
        sys.stdout, report_interval, step=True,
        potentialEnergy=True, temperature=True, volume=True, separator='\t',
    ))

    def _npt_chunk_runner(stage_steps):
        """Step sim for stage_steps, updating BB reference positions each chunk."""
        chunk = max(stage_steps // 5, report_interval)
        done  = 0
        while done < stage_steps:
            c = min(chunk, stage_steps - done)
            sim.step(c)
            done += c
            if restraint is not None and restrained_bb:
                s  = sim.context.getState(getPositions=True, enforcePeriodicBox=True)
                cp = s.getPositions()
                del s
                for k2, aidx in enumerate(restrained_bb):
                    ref = list(cp[aidx].value_in_unit(nanometers))
                    restraint.setParticleParameters(k2, aidx, ref)
                restraint.updateParametersInContext(sim.context)

    _npt_chunk_runner(npt_steps)
    cur_pos, cur_bv, npt_e = _get_state(sim)

    if cur_bv is not None:
        system.setDefaultPeriodicBoxVectors(*cur_bv)
        topology.setPeriodicBoxVectors(cur_bv)
        box_lengths = np.array([
            cur_bv[0][0].value_in_unit(nanometer),
            cur_bv[1][1].value_in_unit(nanometer),
            cur_bv[2][2].value_in_unit(nanometer),
        ])

    _teardown(sim)
    _write_cg_pdb(topology, cur_pos,
                  out_dir / f"{stem}_cg_npt.pdb",
                  strip_water=strip_water_in_output,
                  box_lengths_nm=box_lengths)
    print(f"  NPT final energy: {npt_e:.1f} kJ/mol")
    print(f"  Time: {(time.time() - _t7) / 60:.1f} min")

    # ------------------------------------------------------------------
    # Step 8: Early production — gradual restraint release
    #   Switch to production timestep (20 fs).  Stage BB restraints from
    #   k_nvt → 0 in five equal sub-stages so helices can repack without
    #   large positional jumps.  Intra-helix ENM (if active) is kept to
    #   preserve individual helix geometry during inter-helix adjustment.
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print(f"Step 8: Early production / restraint release "
          f"({early_prod_steps:,} steps / "
          f"{early_prod_steps * prod_timestep / 1e6:.3f} ns)")
    print("=" * 60)
    _t8 = time.time()

    sim = _mk_sim(prod_timestep)
    sim.context.setVelocitiesToTemperature(temperature * kelvin)
    sim.reporters.append(StateDataReporter(
        sys.stdout, report_interval, step=True,
        potentialEnergy=True, temperature=True, separator='\t',
    ))

    k_release_stages = [k_nvt * f for f in (0.8, 0.5, 0.2, 0.05, 0.0)]
    stage_steps_ep   = early_prod_steps // len(k_release_stages)
    for k_val in k_release_stages:
        print(f"  k={k_val:>6.1f} kJ/mol/nm²,  {stage_steps_ep:,} steps")
        if restraint is not None:
            sim.context.setParameter(
                'k', k_val * kilojoules_per_mole / nanometer**2
            )
        ep_chunk = max(stage_steps_ep // 5, report_interval)
        done = 0
        while done < stage_steps_ep:
            c = min(ep_chunk, stage_steps_ep - done)
            sim.step(c)
            done += c

    cur_pos, cur_bv, ep_e = _get_state(sim)
    _teardown(sim)
    if restraint is not None:
        _remove_force_by_type(system, CustomExternalForce)
        restraint = None

    _write_cg_pdb(topology, cur_pos,
                  out_dir / f"{stem}_cg_early_prod.pdb",
                  strip_water=strip_water_in_output,
                  box_lengths_nm=box_lengths)
    print(f"  Early production final energy: {ep_e:.1f} kJ/mol")
    print(f"  Time: {(time.time() - _t8) / 60:.1f} min")

    # ------------------------------------------------------------------
    # Step 9: Repacking — simulated annealing with multiple replicas
    #   Membrane barostat is disabled (NVT) during SA to avoid T/P coupling
    #   conflicts while temperature cycles between base and sa_t_high.
    #   Secondary structure is preserved by backbone angle forces already in
    #   the system.  ENM springs (if any) are kept; no strong positional
    #   restraints remain.  All replicas start from the same
    #   early-production snapshot; best (lowest energy) is selected.
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    sa_total_steps = n_replicas * sa_cycles * 2 * sa_steps_per_half
    print(f"Step 9: Repacking  "
          f"({n_replicas} replicas × {sa_cycles} cycles × "
          f"2 × {sa_steps_per_half:,} steps = {sa_total_steps:,} steps / "
          f"{sa_total_steps * prod_timestep / 1e6:.2f} ns)")
    print("=" * 60)
    _t9 = time.time()

    _remove_force_by_type(system, MonteCarloMembraneBarostat)

    sa_start_pos = cur_pos
    sa_start_bv  = cur_bv
    best_pos, best_bv, best_e = None, None, float('inf')

    for rep in range(n_replicas):
        print(f"\n  Replica {rep + 1}/{n_replicas}")
        sa_sim = _new_sim(topology, system, sa_start_pos, sa_start_bv,
                          temperature, prod_timestep, use_gpu)
        sa_sim.context.setVelocitiesToTemperature(temperature * kelvin)
        sa_sim.reporters.append(StateDataReporter(
            sys.stdout, max(sa_steps_per_half // 5, report_interval),
            step=True, potentialEnergy=True, temperature=True, separator='\t',
        ))

        for cycle in range(sa_cycles):
            sa_sim.integrator.setTemperature(sa_t_high * kelvin)
            sa_sim.step(sa_steps_per_half)
            sa_sim.integrator.setTemperature(temperature * kelvin)
            sa_sim.step(sa_steps_per_half)
            print(f"    SA cycle {cycle + 1}/{sa_cycles} complete")

        rep_pos, rep_bv, rep_e = _get_state(sa_sim)
        _teardown(sa_sim)
        print(f"  Replica {rep + 1} final energy: {rep_e:.1f} kJ/mol")
        if rep_e < best_e:
            best_e, best_pos, best_bv = rep_e, rep_pos, rep_bv

    print(f"\n  Best replica energy: {best_e:.1f} kJ/mol")
    cur_pos = best_pos
    cur_bv  = best_bv
    if cur_bv is not None:
        system.setDefaultPeriodicBoxVectors(*cur_bv)
        topology.setPeriodicBoxVectors(cur_bv)
        box_lengths = np.array([
            cur_bv[0][0].value_in_unit(nanometer),
            cur_bv[1][1].value_in_unit(nanometer),
            cur_bv[2][2].value_in_unit(nanometer),
        ])

    _write_cg_pdb(topology, cur_pos,
                  out_dir / f"{stem}_cg_repacked.pdb",
                  strip_water=strip_water_in_output,
                  box_lengths_nm=box_lengths)
    print(f"  Time: {(time.time() - _t9) / 60:.1f} min")

    # ------------------------------------------------------------------
    # Step 10: Stability production (constant T/P, 1–5 µs)
    #   Re-enable semi-isotropic membrane barostat.  Best-packed replica
    #   runs at constant T/P for the full stability_steps duration.
    #   No positional restraints; intra-helix ENM may remain active.
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print(f"Step 10: Stability production ({stability_steps:,} steps / "
          f"{stability_steps * prod_timestep / 1e6:.3f} ns)")
    print("=" * 60)
    _t10 = time.time()

    system.addForce(MonteCarloMembraneBarostat(
        pressure * bar,
        surface_tension * bar * nanometer,
        temperature * kelvin,
        MonteCarloMembraneBarostat.XYIsotropic,
        MonteCarloMembraneBarostat.ZFree,
        25,
    ))

    sim = _new_sim(topology, system, cur_pos, cur_bv,
                   temperature, prod_timestep, use_gpu)
    sim.context.setVelocitiesToTemperature(temperature * kelvin)

    stab_report = max(stability_steps // 20, report_interval)
    sim.reporters.append(StateDataReporter(
        sys.stdout, stab_report, step=True,
        potentialEnergy=True, temperature=True, volume=True, separator='\t',
    ))
    if save_dcd:
        dcd_path = out_dir / f"{stem}_cg_stability.dcd"
        sim.reporters.append(DCDReporter(str(dcd_path), stab_report))
        print(f"  DCD trajectory: {dcd_path}")

    if snapshot_interval:
        done, frame, stab_e = 0, 0, 0.0
        while done < stability_steps:
            c = min(snapshot_interval, stability_steps - done)
            sim.step(c)
            done += c
            frame += 1
            cur_pos, cur_bv, stab_e = _get_state(sim)
            snap = out_dir / f"{stem}_cg_stability_frame_{frame:04d}.pdb"
            _write_cg_pdb(topology, cur_pos, snap,
                          strip_water=strip_water_in_output,
                          box_lengths_nm=box_lengths)
            print(f"  Frame {frame:4d} ({done:>8,} steps): {stab_e:.1f} kJ/mol")
    else:
        sim.step(stability_steps)
        cur_pos, cur_bv, stab_e = _get_state(sim)
        _write_cg_pdb(topology, cur_pos,
                      out_dir / f"{stem}_cg_stability.pdb",
                      strip_water=strip_water_in_output,
                      box_lengths_nm=box_lengths)

    _teardown(sim)
    print(f"  Stability final energy: {stab_e:.1f} kJ/mol")
    print(f"  Time: {(time.time() - _t10) / 60:.1f} min")

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("CG membrane MD simulation complete.")
    print(f"Total time: {(time.time() - _t0) / 60:.1f} min")
    print(f"Output directory: {out_dir}")
    print(f"  {stem}_cg_minimized.pdb   — after energy minimisation")
    print(f"  {stem}_cg_nvt.pdb         — after NVT equilibration")
    print(f"  {stem}_cg_npt.pdb         — after NPT equilibration")
    print(f"  {stem}_cg_early_prod.pdb  — after gradual restraint release")
    print(f"  {stem}_cg_repacked.pdb    — best SA replica after repacking")
    if snapshot_interval:
        n_frames = stability_steps // snapshot_interval
        print(f"  {stem}_cg_stability_frame_NNNN.pdb — {n_frames} snapshot(s)")
    else:
        print(f"  {stem}_cg_stability.pdb — final stability structure")
    if save_dcd:
        print(f"  {stem}_cg_stability.dcd — stability trajectory")
    print("=" * 60)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "CG-MD pipeline for membrane proteins (OpenMM + Martini 3-inspired). "
            "Pre-orient the input PDB with TM helices along Z before running."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('pdb',
                        help='Input PDB file (orient TM helix along Z first)')
    parser.add_argument('--output-dir', '-o', default=None,
                        help='Output directory (default: same as input PDB)')
    parser.add_argument('--lipid', default='POPC', choices=list(_LIPID_DEFS),
                        help='Lipid type for the bilayer')
    parser.add_argument('--temperature', type=float, default=DEFAULT_TEMPERATURE,
                        help='Simulation temperature (K)')
    parser.add_argument('--pressure', type=float, default=DEFAULT_PRESSURE,
                        help='Simulation pressure (bar)')
    parser.add_argument('--surface-tension', type=float, default=DEFAULT_SURFACE_TENSION,
                        help='Bilayer surface tension (bar·nm); 0 = tension-free NPT')
    parser.add_argument('--padding', type=float, default=DEFAULT_XY_PADDING_NM,
                        help='XY bilayer extension beyond protein footprint (nm)')
    parser.add_argument('--water-layer', type=float, default=DEFAULT_WATER_LAYER_NM,
                        help='CG water layer thickness above/below bilayer (nm)')
    parser.add_argument('--z-po4', type=float, default=DEFAULT_Z_PO4_NM,
                        help='PO4 z-distance from bilayer midplane (nm)')
    parser.add_argument('--equil-timestep', type=float, default=DEFAULT_EQUIL_TIMESTEP,
                        help='Equilibration timestep (fs)')
    parser.add_argument('--prod-timestep', type=float, default=DEFAULT_PROD_TIMESTEP,
                        help='Production timestep (fs); 20 fs is standard for Martini 3')
    parser.add_argument('--nvt-steps', type=int, default=DEFAULT_NVT_STEPS,
                        help='NVT equilibration steps')
    parser.add_argument('--npt-steps', type=int, default=DEFAULT_NPT_STEPS,
                        help='NPT equilibration steps')
    parser.add_argument('--early-prod-steps', type=int,
                        default=DEFAULT_EARLY_PROD_STEPS,
                        help='Early-production gradual restraint release steps')
    parser.add_argument('--stability-steps', type=int,
                        default=DEFAULT_STABILITY_STEPS,
                        help='Stability production steps (default 1 µs at 20 fs)')
    parser.add_argument('--report-interval', type=int, default=DEFAULT_REPORT_INTERVAL,
                        help='StateDataReporter output interval (steps)')
    parser.add_argument('--gpu', action='store_true',
                        help='Use GPU (CUDA) acceleration')
    parser.add_argument('--keep-water', action='store_true',
                        help='Keep W beads in output PDB files')
    parser.add_argument('--snapshot-interval', type=int, default=None, metavar='N',
                        help='Save a numbered PDB snapshot every N stability steps')
    parser.add_argument('--save-dcd', action='store_true',
                        help='Save DCD trajectory during stability production')
    parser.add_argument('--k-nvt', type=float, default=DEFAULT_K_NVT,
                        help='BB restraint strength during NVT/NPT (kJ/mol/nm²)')
    parser.add_argument('--use-enm', action='store_true',
                        help='Add intra-helix ENM springs to stiffen TM helix '
                             'geometry during equilibration and early production')
    parser.add_argument('--enm-k', type=float, default=500.0,
                        help='ENM spring constant (kJ/mol/nm²) for intra-helix '
                             'and beta-sheet springs')
    parser.add_argument('--enm-sheet', action='store_true',
                        help='Add cross-strand beta-sheet ENM springs (intra-chain). '
                             'Recommended for all-beta domains such as antibody/TCR '
                             'Ig folds, which the backbone angle term alone does '
                             'not keep together')
    parser.add_argument('--enm-interchain', action='store_true',
                        help='Add ENM springs across chain-chain interfaces, so a '
                             'multimer stays assembled after BB restraints are '
                             'released and through annealing. Pins interfaces at '
                             'their input geometry — do not use when studying '
                             'association/dissociation')
    parser.add_argument('--enm-cutoff', type=float, default=_ENM_MAX_DIST_NM,
                        metavar='NM',
                        help='Distance cutoff (nm) for --enm-sheet and '
                             '--enm-interchain springs')
    parser.add_argument('--enm-interchain-k', type=float, default=_ENM_IC_K,
                        metavar='K',
                        help='Spring constant (kJ/mol/nm²) for --enm-interchain '
                             'springs; weaker than --enm-k by default')
    parser.add_argument('--no-disulfides', action='store_true',
                        help='Do not build disulfide bridges from SSBOND records '
                             'or SG-SG distances (they are built by default, '
                             'including inter-chain bridges)')
    parser.add_argument('--n-replicas', type=int, default=DEFAULT_N_REPLICAS,
                        help='Number of SA replicas for repacking phase')
    parser.add_argument('--sa-cycles', type=int, default=DEFAULT_SA_CYCLES,
                        help='Simulated annealing cycles per replica')
    parser.add_argument('--sa-t-high', type=float, default=DEFAULT_SA_T_HIGH_K,
                        help='High temperature for SA cycles (K)')
    parser.add_argument('--sa-steps-per-half', type=int,
                        default=DEFAULT_SA_STEPS_HALF,
                        help='Steps per SA heating/cooling half-cycle')
    parser.add_argument('--tm-residues', default=None, metavar='RANGES',
                        help="Transmembrane helix residue ranges to orient along "
                             "Z, e.g. 'A:34-58,A:70-92'. The helix axis is aligned "
                             "to Z and the TM region centred at the bilayer "
                             "midplane. If omitted, the input must already be "
                             "oriented with its TM helices along Z.")
    parser.add_argument('--skip-tm-checks', action='store_true',
                        help="Run even if a --tm-residues range is missing from "
                             "the structure, not helical in the input, or mostly "
                             "polar (these are reported as warnings instead)")
    parser.add_argument('--refine', action='store_true',
                        help='Enable CG NVT pre-equilibration in water before '
                             'membrane embedding (useful for predicted/disordered '
                             'structures)')
    parser.add_argument('--pre-equil-steps', type=int, default=50000,
                        metavar='N',
                        help='NVT steps for water pre-equilibration '
                             '(only with --refine; default 50 000 = 500 ps). '
                             'Increase to 200 000–500 000 for heavily disordered '
                             'structures.')

    args = parser.parse_args()
    try:
        run_cg_membrane_md(
            pdb_path=args.pdb,
            output_dir=args.output_dir,
            lipid_type=args.lipid,
            temperature=args.temperature,
            pressure=args.pressure,
            surface_tension=args.surface_tension,
            padding_nm=args.padding,
            water_layer_nm=args.water_layer,
            z_po4_nm=args.z_po4,
            equil_timestep=args.equil_timestep,
            prod_timestep=args.prod_timestep,
            nvt_steps=args.nvt_steps,
            npt_steps=args.npt_steps,
            early_prod_steps=args.early_prod_steps,
            stability_steps=args.stability_steps,
            report_interval=args.report_interval,
            snapshot_interval=args.snapshot_interval,
            use_gpu=args.gpu,
            strip_water_in_output=not args.keep_water,
            save_dcd=args.save_dcd,
            refine=args.refine,
            pre_equil_steps=args.pre_equil_steps,
            k_nvt=args.k_nvt,
            use_enm=args.use_enm,
            enm_k=args.enm_k,
            enm_sheet=args.enm_sheet,
            enm_interchain=args.enm_interchain,
            enm_cutoff=args.enm_cutoff,
            enm_interchain_k=args.enm_interchain_k,
            detect_disulfides=not args.no_disulfides,
            n_replicas=args.n_replicas,
            sa_cycles=args.sa_cycles,
            sa_t_high=args.sa_t_high,
            sa_steps_per_half=args.sa_steps_per_half,
            tm_residues=parse_tm_ranges(args.tm_residues) if args.tm_residues else None,
            check_tm=not args.skip_tm_checks,
        )
    except TMRangeError as exc:
        parser.exit(2, f"\nERROR: {exc}\n")


if __name__ == '__main__':
    main()
