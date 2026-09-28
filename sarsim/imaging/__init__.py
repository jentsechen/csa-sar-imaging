"""Imaging algorithms. Importing this package registers every algorithm in
base.ALGORITHMS; add new modules to the import list below."""
from .base import ALGORITHMS, ImagingAlgorithm, register  # noqa: F401
from . import csa  # noqa: F401
