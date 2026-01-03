"""Output module."""

# Lazy import to avoid syphon version check crash
def get_syphon_output():
    from .syphon_server import SyphonOutput
    return SyphonOutput
