# OptiProp portable for Windows x64

Extract the entire ZIP into a writable folder, then open **OptiProp.exe**.
Keep `_internal/` beside the executable: this is a portable folder, not a single-file executable.
Python, CPU PyTorch and other required runtime libraries are included. No Python, Qt, CUDA,
GPU driver, package installation, or administrator privileges are required for application use.
The interface opens in your default browser and is served by a local localhost HTTP server.
Do not expose its port to other computers. Internet access is not required for local work.

For a controlled launch from PowerShell:

```powershell
& .\OptiProp.exe --no-browser --port 0 --output-dir 'C:\my optics\results' --session-file 'C:\my optics\session.json'
```

Read the generated session file for the local server address. For a numerical/PNG/export check:

```powershell
$check = Start-Process -FilePath .\OptiProp.exe -ArgumentList '--self-test --no-browser --port 0 --output-dir smoke-output --session-file smoke-session.json' -WindowStyle Hidden -Wait -PassThru
$check.ExitCode
```

Exit code 0 means the application's self-test succeeded. The windowed EXE does not print into
your terminal. Startup diagnostics are appended to `%LOCALAPPDATA%\OptiProp\logs\launcher.log`;
set `OPTIPROP_PACKAGING_LOG` to choose another file. Results belong in your selected output
directory, never in `_internal/`.

This CPU-only Windows distribution does not remove CUDA support from source installations.
Linux/Ubuntu users should use the Python package/source launcher, not this Windows EXE.
See README.md for application usage, LICENSE for OptiProp's MIT license, THIRD-PARTY-NOTICES.md
and LICENSES/ for third-party terms, dependency-manifest.json for dependency versions, and
bundle-manifest.json for payload hashes. A signature or clean-machine certification is not
implied by the local packaging QA. Your Windows security policy may require approving an
unsigned download.
