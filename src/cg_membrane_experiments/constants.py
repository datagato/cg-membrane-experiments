"""Martini 3-based coarse-graining constants shared by the scripts.

Copied from proteogram/common/constants.py
(https://github.com/datagato/proteogram, commit 444cd5d) so the scripts don't
depend on the full proteogram package.
"""

MODIFIED_RESIDUES_TO_STANDARD = {
            # Methionine variants
            'MSE': 'MET',  # Selenomethionine
            'FME': 'MET',  # N-formylmethionine
            'CXM': 'MET',  # N-carboxymethionine
            # Lysine variants
            'M3L': 'LYS',  # N6,N6,N6-trimethyllysine
            'MLY': 'LYS',  # N6-methyllysine
            'MLZ': 'LYS',  # N6-methyllysine (alt code)
            'KCX': 'LYS',  # Lysine NZ-carboxylic acid
            'ALY': 'LYS',  # N6-acetyllysine
            'LLP': 'LYS',  # Lysine-pyridoxal-5-phosphate
            # Cysteine variants
            'CSO': 'CYS',  # S-hydroxycysteine
            'CME': 'CYS',  # S,S-(2-hydroxyethyl)thiocysteine
            'OCS': 'CYS',  # Cysteinesulfonic acid
            'SEC': 'CYS',  # Selenocysteine
            'SMC': 'CYS',  # S-methylcysteine
            'CSD': 'CYS',  # 3-sulfoalanine (treated as Cys)
            # Serine/Threonine variants
            'SEP': 'SER',  # Phosphoserine
            'TPO': 'THR',  # Phosphothreonine
            # Tyrosine variants
            'PTR': 'TYR',  # Phosphotyrosine
            'TYS': 'TYR',  # Sulfotyrosine
            # Proline variants
            'HYP': 'PRO',  # 4-hydroxyproline
            # Aspartate variants
            'BHD': 'ASP',  # (3S)-3-hydroxy-L-aspartic acid (beta-hydroxyaspartic acid)
            # Glutamate/Glutamine variants
            'CGU': 'GLU',  # Gamma-carboxyglutamate
            'PCA': 'GLN',  # Pyroglutamate (from Gln)
            # Histidine variants
            'NEP': 'HIS',  # N1-phosphohistidine
            'HIC': 'HIS',  # 4-methyl-histidine
        }


BACKBONE_ATOMS = ["N", "CA", "C"]

BACKBONE_POSITIONS = {
    "ALA": [
        ("N", [-0.525, 1.363, 0.000]),
        ("CA", [0.000, 0.000, 0.000]),
        ("C", [1.526, -0.000, -0.000]),
    ],
    "ARG": [
        ("N", [-0.524, 1.362, -0.000]),
        ("CA", [0.000, 0.000, 0.000]),
        ("C", [1.525, -0.000, -0.000]),
    ],
    "ASN": [
        ("N", [-0.536, 1.357, 0.000]),
        ("CA", [0.000, 0.000, 0.000]),
        ("C", [1.526, -0.000, -0.000]),
    ],
    "ASP": [
        ("N", [-0.525, 1.362, -0.000]),
        ("CA", [0.000, 0.000, 0.000]),
        ("C", [1.527, 0.000, -0.000]),
    ],
    "CYS": [
        ("N", [-0.522, 1.362, -0.000]),
        ("CA", [0.000, 0.000, 0.000]),
        ("C", [1.524, 0.000, 0.000]),
    ],
    "GLN": [
        ("N", [-0.526, 1.361, -0.000]),
        ("CA", [0.000, 0.000, 0.000]),
        ("C", [1.526, 0.000, 0.000]),
    ],
    "GLU": [
        ("N", [-0.528, 1.361, 0.000]),
        ("CA", [0.000, 0.000, 0.000]),
        ("C", [1.526, -0.000, -0.000]),
    ],
    "GLY": [
        ("N", [-0.572, 1.337, 0.000]),
        ("CA", [0.000, 0.000, 0.000]),
        ("C", [1.517, -0.000, -0.000]),
    ],
    "HIS": [
        ("N", [-0.527, 1.360, 0.000]),
        ("CA", [0.000, 0.000, 0.000]),
        ("C", [1.525, 0.000, 0.000]),
    ],
    "ILE": [
        ("N", [-0.493, 1.373, -0.000]),
        ("CA", [0.000, 0.000, 0.000]),
        ("C", [1.527, -0.000, -0.000]),
    ],
    "LEU": [
        ("N", [-0.520, 1.363, 0.000]),
        ("CA", [0.000, 0.000, 0.000]),
        ("C", [1.525, -0.000, -0.000]),
    ],
    "LYS": [
        ("N", [-0.526, 1.362, -0.000]),
        ("CA", [0.000, 0.000, 0.000]),
        ("C", [1.526, 0.000, 0.000]),
    ],
    "MET": [
        ("N", [-0.521, 1.364, -0.000]),
        ("CA", [0.000, 0.000, 0.000]),
        ("C", [1.525, 0.000, 0.000]),
    ],
    "PHE": [
        ("N", [-0.518, 1.363, 0.000]),
        ("CA", [0.000, 0.000, 0.000]),
        ("C", [1.524, 0.000, -0.000]),
    ],
    "PRO": [
        ("N", [-0.566, 1.351, -0.000]),
        ("CA", [0.000, 0.000, 0.000]),
        ("C", [1.527, -0.000, 0.000]),
    ],
    "SER": [
        ("N", [-0.529, 1.360, -0.000]),
        ("CA", [0.000, 0.000, 0.000]),
        ("C", [1.525, -0.000, -0.000]),
    ],
    "THR": [
        ("N", [-0.517, 1.364, 0.000]),
        ("CA", [0.000, 0.000, 0.000]),
        ("C", [1.526, 0.000, -0.000]),
    ],
    "TRP": [
        ("N", [-0.521, 1.363, 0.000]),
        ("CA", [0.000, 0.000, 0.000]),
        ("C", [1.525, -0.000, 0.000]),
    ],
    "TYR": [
        ("N", [-0.522, 1.362, 0.000]),
        ("CA", [0.000, 0.000, 0.000]),
        ("C", [1.524, -0.000, -0.000]),
    ],
    "VAL": [
        ("N", [-0.494, 1.373, -0.000]),
        ("CA", [0.000, 0.000, 0.000]),
        ("C", [1.527, -0.000, -0.000]),
    ],
    "UNK": [
        ("N", [0.000, 0.000, 0.000]),
        ("CA", [0.000, 0.000, 0.000]),
        ("C", [0.000, 0.000, 0.000]),
    ],
}

# Per-residue bead definitions: (bead_type, bead_label, charge_e, [atom_names])
# BB bead is always first. Missing atoms fall back to CA automatically.
MARTINI_RESIDUE_BEADS = {
    'GLY': [
        ('BB',  'BB',  0.0,  ['N', 'CA', 'C', 'O']),
    ],
    'ALA': [
        ('BB',  'BB',  0.0,  ['N', 'CA', 'C', 'O']),
        ('TC',  'SC1', 0.0,  ['CB']),                      # TC1 in Martini 3
    ],
    'VAL': [
        ('BB',  'BB',  0.0,  ['N', 'CA', 'C', 'O']),
        ('SC',  'SC1', 0.0,  ['CB', 'CG1', 'CG2']),        # SC2 in Martini 3
    ],
    'LEU': [
        ('BB',  'BB',  0.0,  ['N', 'CA', 'C', 'O']),
        ('SC',  'SC1', 0.0,  ['CB', 'CG', 'CD1', 'CD2']),  # SC3 in Martini 3
    ],
    'ILE': [
        ('BB',  'BB',  0.0,  ['N', 'CA', 'C', 'O']),
        ('SC',  'SC1', 0.0,  ['CB', 'CG1', 'CG2', 'CD1']), # SC4 in Martini 3
    ],
    'PRO': [
        ('BB',  'BB',  0.0,  ['N', 'CA', 'C', 'O']),
        ('SC',  'SC1', 0.0,  ['CB', 'CG', 'CD']),           # SC3 in Martini 3
    ],
    'MET': [
        ('BB',  'BB',  0.0,  ['N', 'CA', 'C', 'O']),
        ('C',   'SC1', 0.0,  ['CB', 'CG', 'SD', 'CE']),
    ],
    'CYS': [
        ('BB',  'BB',  0.0,  ['N', 'CA', 'C', 'O']),
        ('TC',  'SC1', 0.0,  ['CB', 'SG']),                 # TC4v in Martini 3
    ],
    'SER': [
        ('BB',  'BB',  0.0,  ['N', 'CA', 'C', 'O']),
        ('N',   'SC1', 0.0,  ['CB', 'OG']),
    ],
    'THR': [
        ('BB',  'BB',  0.0,  ['N', 'CA', 'C', 'O']),
        ('N',   'SC1', 0.0,  ['CB', 'OG1', 'CG2']),
    ],
    'ASN': [
        ('BB',  'BB',  0.0,  ['N', 'CA', 'C', 'O']),
        ('N',   'SC1', 0.0,  ['CB', 'CG', 'OD1', 'ND2']),
    ],
    'GLN': [
        ('BB',  'BB',  0.0,  ['N', 'CA', 'C', 'O']),
        ('N',   'SC1', 0.0,  ['CB', 'CG', 'CD', 'OE1', 'NE2']),
    ],
    'ASP': [
        ('BB',  'BB',  0.0,  ['N', 'CA', 'C', 'O']),
        ('Q',   'SC1', -1.0, ['CB', 'CG', 'OD1', 'OD2']),
    ],
    'GLU': [
        ('BB',  'BB',  0.0,  ['N', 'CA', 'C', 'O']),
        ('Q',   'SC1', -1.0, ['CB', 'CG', 'CD', 'OE1', 'OE2']),
    ],
    'LYS': [
        ('BB',  'BB',  0.0,  ['N', 'CA', 'C', 'O']),
        ('SC',  'SC1', 0.0,  ['CB', 'CG', 'CD']),           # SC3 aliphatic linker in Martini 3
        ('Q',   'SC2', +1.0, ['CE', 'NZ']),
    ],
    'ARG': [
        ('BB',  'BB',  0.0,  ['N', 'CA', 'C', 'O']),
        ('N',   'SC1', 0.0,  ['CB', 'CG', 'CD']),
        ('Q',   'SC2', +1.0, ['NE', 'CZ', 'NH1', 'NH2']),
    ],
    'HIS': [
        ('BB',  'BB',  0.0,  ['N', 'CA', 'C', 'O']),
        ('TC',  'SC1', 0.0,  ['CB', 'CG']),
        ('TC',  'SC2', +0.5, ['ND1', 'CD2', 'CE1', 'NE2']),
    ],
    'PHE': [
        ('BB',  'BB',  0.0,  ['N', 'CA', 'C', 'O']),
        ('SC',  'SC1', 0.0,  ['CB', 'CG', 'CD1', 'CD2']),   # SC4 ring bead in Martini 3
        ('SC',  'SC2', 0.0,  ['CE1', 'CE2', 'CZ']),          # SC4 ring bead in Martini 3
    ],
    'TYR': [
        ('BB',  'BB',  0.0,  ['N', 'CA', 'C', 'O']),
        ('SC',  'SC1', 0.0,  ['CB', 'CG', 'CD1', 'CD2']),   # SC4 ring bead in Martini 3
        ('SC',  'SC2', 0.0,  ['CE1', 'CE2', 'CZ', 'OH']),   # SC4 ring bead in Martini 3
    ],
    'TRP': [
        ('BB',  'BB',  0.0,  ['N', 'CA', 'C', 'O']),
        ('SC',  'SC1', 0.0,  ['CB', 'CG', 'CD1', 'NE1']),   # SC4 indole in Martini 3
        ('SC',  'SC2', 0.0,  ['CD2', 'CE2', 'CZ2', 'CH2']), # SC4 indole in Martini 3
        ('TC',  'SC3', 0.0,  ['CE3', 'CZ3']),                # TC4 small ring in Martini 3
    ],
}
