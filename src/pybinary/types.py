"""Field types used in structure annotations.

Everything here is a *marker*: these classes are never instantiated, they only
appear in annotations and are interpreted by :mod:`pybinary.collect`.
"""

from __future__ import annotations

__all__ = [
    "u8", "u16", "u32", "u64",
    "i8", "i16", "i32", "i64",
    "f32", "f64",
    "Const", "Bytes", "Str", "Array", "Padding", "Bits", "If", "Switch", "Checksum", "Pointer",
]


class _Marker:
    """Head of a parametric-field spec tuple; identity is the tag."""

    __slots__ = ("label",)

    def __init__(self, label: str) -> None:
        self.label = label

    def __repr__(self) -> str:
        return f"<pybinary:{self.label}>"


CONST = _Marker("Const")
BYTES = _Marker("Bytes")
STR = _Marker("Str")
ARRAY = _Marker("Array")
PADDING = _Marker("Padding")
BITS = _Marker("Bits")
IF = _Marker("If")
SWITCH = _Marker("Switch")
CHECKSUM = _Marker("Checksum")
POINTER = _Marker("Pointer")


class _Scalar:
    """Base for fixed-width numeric fields."""

    char: str = ""
    size: int = 0
    py: type = int

    def __new__(cls, *args: object, **kwargs: object) -> _Scalar:
        raise TypeError(
            f"{cls.__name__} is a field type, not a value; use it as an annotation"
        )


class u8(_Scalar):
    char, size, py = "B", 1, int


class u16(_Scalar):
    char, size, py = "H", 2, int


class u32(_Scalar):
    char, size, py = "I", 4, int


class u64(_Scalar):
    char, size, py = "Q", 8, int


class i8(_Scalar):
    char, size, py = "b", 1, int


class i16(_Scalar):
    char, size, py = "h", 2, int


class i32(_Scalar):
    char, size, py = "i", 4, int


class i64(_Scalar):
    char, size, py = "q", 8, int


class f32(_Scalar):
    char, size, py = "f", 4, float


class f64(_Scalar):
    char, size, py = "d", 8, float


# --------------------------------------------------------------------------
# Parametric field types.
#
# Every ``__class_getitem__`` below returns the subscript argument *untouched*
# inside a plain tuple. That is load-bearing, not stylistic: under PEP 649 these
# run during annotationlib's fake-globals pass, and the arguments for references
# to other fields are placeholder objects that annotationlib rewrites into
# ForwardRefs on the way out. Inspecting or converting them here makes it
# collapse the whole annotation into a single opaque ForwardRef instead.
# --------------------------------------------------------------------------


class Const:
    """A literal byte sequence: ``magic: Const[b"BLOB"]``.

    Verified on parse, emitted on build, not stored as an attribute.
    """

    def __class_getitem__(cls, item):
        return (CONST, item)


class Bytes:
    """Raw bytes: ``Bytes[4]``, ``Bytes[n]``, ``Bytes[n * 2]``, ``Bytes[...]``.

    ``...`` means "the rest of the buffer".
    """

    def __class_getitem__(cls, item):
        return (BYTES, item)


class Str:
    """Text: ``Str[n]`` or ``Str[n, "ascii"]``. Defaults to utf-8."""

    def __class_getitem__(cls, item):
        return (STR, item)


class Array:
    """A sequence: ``Array[u16, n]``, ``Array[Payload, 4]``."""

    def __class_getitem__(cls, item):
        return (ARRAY, item)


class Padding:
    """``n`` skipped bytes, zero-filled on build: ``_pad: Padding[3]``."""

    def __class_getitem__(cls, item):
        return (PADDING, item)


class Bits:
    """A sub-byte field: ``ver: Bits[4]``, or ``_rsv: Bits[3, 0]``.

    Consecutive ``Bits`` fields form a run that must total a whole number of
    bytes. The run is read MSB-first as a big-endian bit string *regardless of
    the structure's endian*, matching how RFC-style wire diagrams number bits.

    The two-argument form declares a constant: it is verified on parse, written
    on build, and not stored as an attribute.
    """

    def __class_getitem__(cls, item):
        return (BITS, item)


class If:
    """A field that is only present when a condition holds.

    ``extra: If["version > 1", u32]`` -- the attribute is ``None`` when the
    condition is false.

    Write conditions with ``&``, ``|`` and ``~`` rather than ``and``/``or``/
    ``not``: Python evaluates those eagerly while the annotation is being
    built, so pybinary never sees the full expression. Quoting the condition
    also works and is always safe.
    """

    def __class_getitem__(cls, item):
        return (IF, item)


class Switch:
    """A variant chosen by an earlier field.

    ``body: Switch["kind", {1: Ping, 2: Pong, ...: Raw}]``

    Keys are integers; ``...`` is the fallback for any unlisted value, and a
    value of ``None`` means the variant occupies no bytes and the attribute is
    ``None``. Without a fallback, an unknown discriminator is a ``ParseError``.

    Variant classes must be defined *before* the class that references them.
    """

    def __class_getitem__(cls, item):
        return (SWITCH, item)


class Checksum:
    """A verified, computed integer: ``crc: Checksum[u32, zlib.crc32, magic]``.

    Covers every byte from the start of the named earlier field up to the
    checksum itself. Verified on parse and computed on build, so -- like
    ``Const`` -- it is not stored as an attribute.

    The function is any callable taking a bytes-like object and returning an
    int; the result is masked to the field's width. It may be handed a
    ``memoryview``, so it must not assume ``bytes``.
    """

    def __class_getitem__(cls, item):
        return (CHECKSUM, item)


class Pointer:
    """A field stored elsewhere: ``name: Pointer["off", Str["n"]]``.

    The offset is counted from the start of *this record*, and the cursor does
    not advance -- the field that follows a pointer reads from where the
    pointer began. At the top level a record starts at 0, so a record-relative
    offset is also a file offset.

    ``Pointer["off", T, "nullable"]`` treats an offset of 0 as "absent" and
    yields ``None``.

    On build, an offset written as a bare field name used by exactly one
    pointer is filled in automatically: the target is appended after the record
    and the offset patched in place. Any other offset expression must agree
    with where the target actually lands.
    """

    def __class_getitem__(cls, item):
        return (POINTER, item)
