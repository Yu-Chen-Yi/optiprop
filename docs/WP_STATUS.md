# OptiProp near-field workbench delivery status

The workbench is implemented as fifteen dependency-ordered work packages.

| WP | Deliverable | Status |
|---:|---|---|
| 01 | Canonical `Grid2D` / `Field2D` | Complete |
| 02 | Propagator contract and sampling reports | Complete |
| 03 | ASM and band-limited ASM | Complete |
| 04 | Fresnel propagation | Complete |
| 05 | Rayleigh-Sommerfeld FFT/direct | Complete |
| 06 | Layer protocol and `OpticalSystem` | Complete |
| 07 | Incident sources, thin elements, and interfaces | Complete |
| 08 | Safe canonical NPZ/MAT I/O and explicit legacy mapping | Complete |
| 09 | Strict Zemax ZBF v0/v1 integration | Complete |
| 10 | Versioned project schema and asset integrity | Complete |
| 11 | PySide6 project/source/layer models and main window | Complete |
| 12 | Property and import-mapping editors | Complete |
| 13 | Background simulation, cancellation, progress, and cache | Complete |
| 14 | Amplitude/phase/intensity/profile/XZ result views | Complete |
| 15 | Entry points, packaging, documentation, and release tests | Complete |

## Launch

```powershell
conda activate torcwa_app
pip install -e ".[gui]"
optiprop-gui
```

The GUI dependency is optional so importing the numerical `optiprop` package
continues to work on servers without Qt. Headless CI uses
`QT_QPA_PLATFORM=offscreen` for GUI smoke tests.

## Project and field formats

- Projects: versioned UTF-8 JSON (`.oprop` or `.json`) with stable UUIDs,
  relative asset references, size/SHA-256 checks, migration, and atomic save.
- Fields: canonical NPZ/MAT and Zemax ZBF; legacy NPZ/MAT requires an explicit
  axis/unit/key mapping, so the application never silently guesses units or
  transposes arrays.
- ZBF: v0/v1 input and v1 output, scalar or `Ex`/`Ey`, normalized to SI inside
  the application.

## Release verification

Run the complete suite from the repository root:

```powershell
conda run -n torcwa_app python -m pytest
python -m build
```

The wheel must be tested from an isolated directory for both numerical import
and the offscreen GUI startup path before publishing.
