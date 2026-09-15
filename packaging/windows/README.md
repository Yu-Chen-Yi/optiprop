# Windows portable release tooling

These tools package the application entry point `optiprop.workbench.app:main(argv=None)`.
The portable executable is `OptiProp.exe`, windowed, onedir, x64, CPU-only.
The ZIP contains `OptiProp/OptiProp.exe`, its `_internal/` runtime, project documentation and notices.

## Build environment contract

Build on Windows x64 with an isolated venv, defaulting to `tmp/build-venv/Scripts/python.exe`.
The shared development/Conda interpreter is **not** a packaging environment. Finish dependency
installation first, before running any build or dependency check. These scripts never run pip,
activate a Conda environment, delete a previous build, or silently fall back to another interpreter.

The isolated CPU build environment uses Python 3.12, CPU `torch==2.5.1+cpu` from the official CPU
wheel index, and `pyinstaller==6.20.0`, plus the project's runtime dependencies. Versions actually
used are recorded in the report/manifest; guardrails reject CUDA/ROCm wheels and installed
Qt/GPU-extra distributions, not just a machine where `cuda.is_available()` happens to be false.
`include-system-site-packages` must be false, and required libraries must resolve inside the venv.
No Qt, Tk, PIL.ImageQt/ImageTk or GPU-only runtime is allowed into the frozen payload. CPU torch's
own Python `torch.cuda` compatibility modules are allowed; they are not CUDA DLLs.

Create a **new, isolated CPU build environment** and install dependencies as a separate step:

```powershell
py -3.12 -m venv tmp/build-venv
& .\tmp\build-venv\Scripts\python.exe -m pip install --index-url https://download.pytorch.org/whl/cpu 'torch==2.5.1+cpu'
& .\tmp\build-venv\Scripts\python.exe -m pip install 'pyinstaller==6.20.0' .
```

Do not run concurrent dependency installations in the same environment, or build while an
installation is running. Install the project without the legacy `gui` or `gpu` extras.
The existing venv is never provisioned or modified by this tooling. These are baseline constraints,
not a full reproducible lockfile; dependency-manifest.json records all resolved build versions.

## Build and verify

From the repository root, **after dependency installation and application tests have completed**:

```powershell
.\packaging\windows\build.ps1 -CheckEnvironment
.\packaging\windows\build.ps1
```

Optional flags: `-BuildPython C:\isolated-venv\Scripts\python.exe`, `-QaTimeout 300`.
Use the wrapper or `python -I packaging/windows/bootstrap.py` so preflight ignores inherited
Python paths/user site packages. Do not run the spec directly: it requires the staged metadata
directory supplied by `build.py`. No cross-compilation or Linux binary is produced.

Each attempt uses new directories; failures and their logs remain available for inspection:

- `tmp/packaging-build/<run>/`: PyInstaller analysis, isolated cache, notices and build log.
- `dist/windows/<run>/OptiProp/`: complete portable folder.
- `dist/windows/<run>/OptiProp-<version>-windows-x64.zip`: release candidate ZIP.
- `tmp/packaging-qa/OptiProp QA 測試-<id>/`: extracted ZIP, scratch profile/CWD, actual self-test
  outputs, launch/process logs and `qa-report.json` (also written on QA failure).
- Only after QA succeeds: adjacent `.zip.sha256`, `build-report.json`, and
  `tmp/packaging-latest.json`. The latter points to the most recent **successful** build;
  a later failed attempt does not make that older build current for changed source.

The wrapper fails if environment validation, analysis, payload inspection, extraction/hash checks,
or frozen numerical self-test fails. QA verifies x64 Windows GUI PE headers, bundled Python DLLs,
static assets, JSON examples, notices, absence of Qt/GPU runtimes, every recorded payload hash,
and `--self-test --no-browser --port 0 --output-dir <scratch> --session-file <scratch>` exit code 0
with a PNG signature and nonempty NPZ, MAT and ZBF exports. `self-test.json` must report CPU torch,
success and a completed IncidentSource/Binary2LensLayer/ASM project. The application's
self-test is responsible for real source/Binary2/ASM numerical assertions; this tool neither mocks
that application work nor substitutes a synthetic success. `--self-test` must leave its smoke
outputs under the requested directory. A success without those files is rejected.

QA also starts a second frozen process with `--no-browser --port 0 --session-file`, reads its
`{url, origin, token}` session, fetches nonempty HTML/CSS/JavaScript with the expected MIME types,
Content Security Policy and `nosniff` headers, and reads the authenticated bootstrap with all
four examples. It checks unauthenticated API denial, then sends authenticated `POST /api/shutdown`
and requires clean exit. Session tokens are not included in the published build report. This
checks server/data packaging, not visual browser behavior or every frontend interaction.

The QA child runs the extracted EXE via its absolute path from an empty CWD, with no inherited
Python, Conda, CUDA, Qt or developer PATH. Temporary profiles/output directories and a path
containing spaces/non-ASCII characters test relocation. This is useful local isolation, not proof
of operation on a pristine Windows VM: the host still has its Windows/system runtimes installed.
Before release, additionally validate a supported clean Windows VM and normal
browser launch. Build tooling does not publish, upload, sign, commit or change CI.

Re-run QA for an existing candidate (using a Python interpreter only as the external QA driver):

```powershell
& .\tmp\build-venv\Scripts\python.exe .\packaging\windows\qa.py .\dist\windows\RUN\OptiProp-1.0.8-windows-x64.zip
```

Packaging unit tests require only Python's standard library and can run on Ubuntu or Windows:

```text
python -m unittest discover -s tests/packaging -v
```

## Payload and licensing

`collect_data_files` explicitly allows only `optiprop/workbench/static/*` (including subdirectories),
`optiprop/workbench/examples/*.json`, and `optiprop/examples/*.json`. No repository-wide data
collection is used; large Example/, results, experimental, source-field and development directories
are not input data. Dependency hooks retain runtime native libraries and source they actually need
(for example torch's source inspection). Matplotlib's hook is explicitly configured with `Agg`.

LICENSE and README.md are copied at build time. Installed wheel notices and the Python runtime
license are copied into LICENSES/. THIRD-PARTY-NOTICES.md and dependency-manifest.json identify
all build-environment distributions (including build-only tools); source-manifest.json fingerprints
the application source at build start (source contents are not copied); analysis-modules.json and
bundle-manifest.json distinguish frozen modules and actual files. Review these notices before
redistribution; OptiProp's MIT license does not replace third-party licenses.

Official implementation references:
[PyInstaller spec files](https://pyinstaller.org/en/stable/spec-files.html),
[data collection hooks](https://pyinstaller.org/en/stable/hooks.html#PyInstaller.utils.hooks.collect_data_files),
[Matplotlib hook configuration](https://pyinstaller.org/en/stable/hooks-config.html#matplotlib-hooks),
[windowed-process standard streams](https://pyinstaller.org/en/stable/common-issues-and-pitfalls.html#sys-stdin-sys-stdout-and-sys-stderr-in-noconsole-windowed-applications-windows-only).
