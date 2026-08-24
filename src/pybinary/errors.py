"""Exception hierarchy.

Layout problems are raised at *class definition* time; parse and build problems
at call time.
"""

__all__ = ["PyBinaryError", "LayoutError", "ParseError", "BuildError"]


class PyBinaryError(Exception):
    """Base class for every error raised by pybinary."""


class LayoutError(PyBinaryError, TypeError):
    """A structure declaration is invalid. Raised when the class is created."""


class ParseError(PyBinaryError, ValueError):
    """A buffer could not be decoded against a structure."""


class BuildError(PyBinaryError, ValueError):
    """An instance could not be encoded."""
