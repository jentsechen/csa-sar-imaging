"""Common interface for imaging (focusing) algorithms.

Adding an algorithm:
  1. Subclass ImagingAlgorithm in sarsim/imaging/<name>.py; do every
     scene-independent precomputation (phase filters, ...) in __init__ so it
     runs once per batch, and implement focus().
  2. Decorate it with @register("<name>"); it is then selectable via
     `python -m sarsim.pipeline --algo <name>`.
  3. Validate it on a single point target (resolution / PSLR / ISLR) before
     the multi-scene batch.

FFT-based algorithms (CSA, RDA, omega-K) are a natural fit for CuPy/cuFFT;
per-pixel summations such as back-projection should instead follow the
thread-per-pixel RawKernel pattern of sarsim/echo.py.
"""
ALGORITHMS = {}


def register(name):
    def deco(cls):
        cls.name = name
        ALGORITHMS[name] = cls
        return cls
    return deco


class ImagingAlgorithm:
    name = None

    def __init__(self, xp, ax, par):
        """xp: numpy or cupy; ax: sarsim.params.build_imaging_axes(par); par: input_par dict."""
        self.xp = xp

    def focus(self, echo):
        """(n_row, n_col) complex echo -> (n_row, n_col) complex focused image, same array module."""
        raise NotImplementedError
