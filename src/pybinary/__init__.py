"""pybinary — declarative binary structures that compile to specialized codecs.

    from pybinary import Binary, Const, Bytes, Array, u16, u32

    class Header(Binary, endian="<"):
        magic:   Const[b"BLOB"]
        version: u16
        n:       u32
        items:   Bytes[n]
        tags:    Array[u16, n]

    h = Header.unpack(buf)      # bytes fields are plain bytes
    raw = h.pack()

Pass ``copy=False`` to ``unpack`` for zero-copy parsing: ``Bytes`` fields come
back as memoryviews over the caller's buffer, which is then kept alive.
"""

from .base import Binary, BinaryMeta
from .errors import BuildError, LayoutError, ParseError, PyBinaryError
from .fields import FieldSpec
from .types import (
    Array, Bits, Bytes, Checksum, Const, If, Padding, Pointer, Str, Switch,
    f32, f64, i8, i16, i32, i64, u8, u16, u32, u64,
)

__version__ = "0.1.0"

__all__ = [
    "Binary", "BinaryMeta", "FieldSpec",
    "PyBinaryError", "LayoutError", "ParseError", "BuildError",
    "u8", "u16", "u32", "u64",
    "i8", "i16", "i32", "i64",
    "f32", "f64",
    "Const", "Bytes", "Str", "Array", "Padding", "Bits", "If", "Switch", "Checksum", "Pointer",
]
