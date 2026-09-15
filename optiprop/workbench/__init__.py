"""Qt-free local browser workbench. The numerical package needs no GUI toolkit."""


def launch(argv=None):
    from .app import main
    return main(argv)


__all__ = ["launch"]
