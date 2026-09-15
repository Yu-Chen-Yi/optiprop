"""Frozen entry point; application behavior belongs to optiprop.workbench.app."""

import multiprocessing
import traceback


if __name__ == "__main__":
    multiprocessing.freeze_support()
    try:
        from optiprop.workbench.app import main

        raise SystemExit(main())
    except Exception:
        # Windowed processes have no terminal; runtime_hook supplies a log stream.
        traceback.print_exc()
        raise SystemExit(1)
