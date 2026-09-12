# cg-membrane-experiments

Coarse-grained (CG) molecular dynamics experiments for membrane proteins, using a
Martini 3-based force model in [OpenMM](https://openmm.org). The scripts cover the
full round trip:

```
atomistic PDB/CIF ──► CG beads ──► CG membrane MD ──► backmapped atomistic PDB
                          │                │
                          └──────┬─────────┘
                                 ▼
                          .mp4 movie (PyMOL + ffmpeg)
```

## Repository layout

```
src/cg_membrane_experiments/
└── constants.py                             # Martini bead mapping + backbone geometry shared by the scripts
scripts/
├── atomistic_to_martini_cg_conversion.py    # atomistic -> CG bead structure (+ PyMOL overlay)
├── cg_md_simulation_for_membrane_protein.py # CG membrane MD pipeline (OpenMM)
├── martini_cg_to_atomistic_backmap.py       # CG -> atomistic, optional minimization
├── cg_structures_to_movie.py                # structure sequence -> .mp4
└── data/tcr_experiments/
    ├── 1OGA.cif                             # JM22 TCR / HLA-A2-flu peptide complex
    └── 2NW2.cif                             # ELS4 TCR (1.4 Å)
```

## Installation

### Requirements

- Python ≥ 3.11
- [uv](https://docs.astral.sh/uv/)
- Only needed for some scripts:
  - `pymol` and `ffmpeg` on your `PATH`, for `cg_structures_to_movie.py`
    (macOS: `brew install pymol ffmpeg`)
  - `mkdssp`, which the MD script uses to assign secondary structure. Without it,
    the script estimates secondary structure from backbone φ/ψ angles and prints a
    warning.

### Set up the environment

```bash
git clone <this-repo-url>
cd cg-membrane-experiments
uv sync
```

This creates `.venv/` and installs NumPy, Biopython, OpenMM and PDBFixer, plus
this repo's small `cg_membrane_experiments` package in editable mode. That package
holds `constants.py`: the Martini bead mapping (`MARTINI_RESIDUE_BEADS`), the
modified-residue aliases and the idealized backbone geometry. The conversion,
backmap and movie scripts import these values, which were copied from
[`proteogram`](https://github.com/datagato/proteogram). The MD script keeps its
own copy of the constants and does not use the package.

For NVIDIA GPU runs (Linux/Windows), install the CUDA extra:

```bash
uv sync --extra cuda12
```

Run any script with `uv run` (no need to activate the venv):

```bash
uv run scripts/<script>.py --help
```

## Scripts

### 1. `atomistic_to_martini_cg_conversion.py`: atomistic to CG

Reduces each residue to 1–4 Martini beads (BB plus 0–3 sidechain beads). Each bead
sits at the centroid of its heavy atoms, so the CG model stays in the same
coordinate frame as the input and overlays it directly in PyMOL.

```bash
uv run scripts/atomistic_to_martini_cg_conversion.py input.pdb
uv run scripts/atomistic_to_martini_cg_conversion.py input.pdb -o out_cg.pdb --pymol
```

| Option | Description |
|---|---|
| `input` | Atomistic structure (`.pdb` or `.ent`; see the note on CIF below) |
| `-o, --output` | Output CG PDB (default `<input>_cg.pdb`) |
| `--pymol` | Also write a `.pml` overlay script (beads colored by size class: tiny=red, small=blue, regular=yellow) |

**Outputs:** a CG PDB with one `ATOM` per bead and a `CONECT` record per bond. The
B-factor column holds the bead charge and the segID column holds the bead size
class. With `--pymol`, a `.pml` script is also written; open it with
`pymol <name>.pml`.

> **CIF input:** this script reads PDB format only. To convert a CIF first:
> ```bash
> uv run python -c "from Bio.PDB import MMCIFParser, PDBIO; \
> s = MMCIFParser(QUIET=True).get_structure('x', 'scripts/data/tcr_experiments/1OGA.cif'); \
> io = PDBIO(); io.set_structure(s); io.save('1OGA.pdb')"
> ```

### 2. `cg_md_simulation_for_membrane_protein.py`: CG membrane MD

A self-contained CG-MD pipeline in OpenMM. It takes a `.pdb` or `.cif` file and runs:

1. PDB → multi-bead CG protein (all chains, gap-aware bonds, disulfides)
2. Lipid bilayer construction (POPC / DPPC / POPE)
3. CG water solvation
4. Na⁺/Cl⁻ neutralization + 0.15 M NaCl
5. Energy minimization
6. NVT equilibration (10 fs, backbone restraints)
7. NPT equilibration (semi-isotropic membrane barostat)
8. Early production (20 fs, gradual restraint release)
9. Repacking by simulated annealing (multiple replicas; best one kept)
10. Stability production (1 µs by default)

**Orientation matters.** The bilayer is built at z = 0, so the transmembrane (TM)
helices must lie along Z. You can either pre-orient the structure (OPM/PPM,
CHARMM-GUI) or pass `--tm-residues` and let the script align and center it.

```bash
# Pre-oriented input
uv run scripts/cg_md_simulation_for_membrane_protein.py my_protein.pdb

# Let the script orient the TM helices
uv run scripts/cg_md_simulation_for_membrane_protein.py my_protein.pdb \
    --tm-residues A:34-58,A:70-92

# TCR–CD3 complex: keep Ig domains and chain interfaces intact
uv run scripts/cg_md_simulation_for_membrane_protein.py tcr_cd3.pdb \
    --tm-residues A:207-232,B:249-274,D:106-127,E:112-133 \
    --use-enm --enm-sheet --enm-interchain

# Quick smoke test (a few seconds on CPU)
uv run scripts/cg_md_simulation_for_membrane_protein.py \
    scripts/data/tcr_experiments/2NW2.cif -o runs/2NW2_test \
    --nvt-steps 200 --npt-steps 200 --early-prod-steps 200 --stability-steps 200 \
    --n-replicas 1 --sa-cycles 1 --sa-steps-per-half 100 --report-interval 100
```

Commonly used options (run with `--help` for the full list and defaults):

| Option | Description |
|---|---|
| `-o, --output-dir` | Output directory (default: same directory as the input) |
| `--lipid {POPC,DPPC,POPE}` | Bilayer lipid (default POPC) |
| `--tm-residues` | TM helix ranges to align along Z, e.g. `A:34-58,B:10-30` |
| `--gpu` | Use the CUDA platform. If CUDA isn't available, including on macOS, the script falls back to CPU |
| `--stability-steps` | Length of the final production run (default 50,000,000 steps = 1 µs) |
| `--save-dcd`, `--snapshot-interval N` | Save a trajectory or numbered PDB snapshots during stability production |
| `--keep-water` | Keep W beads in the output PDBs |
| `--use-enm`, `--enm-sheet`, `--enm-interchain` | Elastic network springs for TM helices, β-sheets, and chain interfaces |
| `--no-disulfides` | Skip building disulfide bridges |
| `--refine` | Pre-equilibrate in water before membrane embedding (for predicted or disordered models) |
| `--n-replicas`, `--sa-cycles`, `--sa-t-high` | Simulated-annealing repacking controls |

**Outputs** (written to the output directory; `<stem>` is the input file name):

| File | Stage |
|---|---|
| `<stem>_cg_minimized.pdb` | After energy minimization |
| `<stem>_cg_nvt.pdb` | After NVT equilibration |
| `<stem>_cg_npt.pdb` | After NPT equilibration |
| `<stem>_cg_early_prod.pdb` | After restraint release |
| `<stem>_cg_repacked.pdb` | Best simulated-annealing replica |
| `<stem>_cg_stability.pdb` | Final structure |
| `<stem>_cg_stability.dcd` | Trajectory (with `--save-dcd`) |
| `<stem>_cg_stability_frame_NNNN.pdb` | Snapshots (with `--snapshot-interval`) |

> **Runtime:** with the default settings, the stability stage alone is 50 million
> steps. Use a GPU for real runs, and small step counts when testing.

### 3. `martini_cg_to_atomistic_backmap.py`: CG to atomistic

Rebuilds a plausible all-atom structure from a CG PDB:

1. It places the backbone (N, CA, C) from the BB-bead trace.
2. PDBFixer adds sidechains and hydrogens.
3. Optionally, OpenMM energy-minimizes the result (AMBER14 + GBn2 implicit solvent).

Every run ends with a bead RMSD check against the input beads, which shows how
closely the rebuilt structure matches.

```bash
uv run scripts/martini_cg_to_atomistic_backmap.py input_cg.pdb
uv run scripts/martini_cg_to_atomistic_backmap.py input_cg.pdb --minimize

# Pull sidechains toward the CG SC beads and keep helices/strands intact,
# taking secondary structure from the original atomistic model
uv run scripts/martini_cg_to_atomistic_backmap.py runs/2NW2/2NW2_cg_stability.pdb \
    --preserve-ss --ss-ref scripts/data/tcr_experiments/2NW2.cif
```

| Option | Description |
|---|---|
| `-o, --output` | Output PDB (default `<input>_atomistic.pdb`) |
| `--minimize` | Energy-minimize the rebuilt structure |
| `--sc-guided` | Restrained minimization that moves sidechains toward their SC bead positions (implies `--minimize`) |
| `--preserve-ss` | Restrain helix/strand φ/ψ during minimization (implies `--sc-guided`) |
| `--ss-ref` | Reference structure to take secondary structure from (otherwise it is estimated from the BB trace, which finds helices only) |
| `--forcefield` | OpenMM force field XMLs (default `amber14-all.xml implicit/gbn2.xml`) |
| `--min-iterations`, `--min-tolerance`, `--ph` | Minimization and protonation settings |

**Outputs** (for input `foo_cg.pdb` with the default output name):

| File | Written when |
|---|---|
| `foo_cg_atomistic.pdb` | Always: rebuilt all-atom structure |
| `foo_cg_atomistic_backbone.pdb` | Always: intermediate N/CA/C backbone |
| `foo_cg_atomistic_min.pdb` | `--minimize` |
| `foo_cg_atomistic_scguided.pdb` | `--sc-guided` or `--preserve-ss` |

### 4. `cg_structures_to_movie.py`: movie rendering

Renders an ordered list of structures into an `.mp4`. The first file is shown as an
atomistic cartoon with sticks, and the rest as CG bead spheres. A grey lipid
bilayer fades in after the first frame. If the CG files have no lipids, the script
generates one. PyMOL renders the frames headlessly and ffmpeg encodes them.

```bash
uv run scripts/cg_structures_to_movie.py \
    scripts/data/tcr_experiments/2NW2.cif \
    runs/2NW2/2NW2_cg_minimized.pdb runs/2NW2/2NW2_cg_npt.pdb runs/2NW2/2NW2_cg_stability.pdb \
    -o movies/2NW2.mp4

# Fast preview: draw instead of ray-trace, keep the PNG frames
uv run scripts/cg_structures_to_movie.py protein.cif cg_*.pdb -o preview.mp4 \
    --fast --keep-frames
```

| Option | Description |
|---|---|
| `-o, --output` | Output `.mp4` (default `cg_movie.mp4`) |
| `--fps`, `--frames-per-structure`, `--deg-per-frame` | Timing and rotation |
| `--width`, `--height` | Frame size (default 1024×768) |
| `--no-membrane`, `--membrane-thickness`, `--membrane-transparency` | Membrane display |
| `--sphere-scale` | CG bead size |
| `--fast` | Use PyMOL `draw` instead of `ray` (faster, lower quality) |
| `--keep-frames` | Keep `<output>_frames/` (PNGs, `render.pml`, `movie_meta.json`) |

## Example end-to-end workflow

```bash
uv sync

# 1. CG MD (orient first, or pass --tm-residues for a real membrane protein)
uv run scripts/cg_md_simulation_for_membrane_protein.py \
    scripts/data/tcr_experiments/2NW2.cif -o runs/2NW2 --stability-steps 500000

# 2. Backmap the final CG structure
uv run scripts/martini_cg_to_atomistic_backmap.py runs/2NW2/2NW2_cg_stability.pdb \
    --preserve-ss --ss-ref scripts/data/tcr_experiments/2NW2.cif

# 3. Render a movie of the trajectory stages
uv run scripts/cg_structures_to_movie.py scripts/data/tcr_experiments/2NW2.cif \
    runs/2NW2/2NW2_cg_{minimized,nvt,npt,early_prod,repacked,stability}.pdb \
    -o movies/2NW2.mp4
```

## License

MIT. See [LICENSE](LICENSE).
