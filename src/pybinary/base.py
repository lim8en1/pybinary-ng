"""The :class:`Binary` base class and its metaclass."""

from __future__ import annotations

from ._compat import read_annotations
from .collect import STRUCT_MARKER, build_fields
from .compile import as_view, compile_codec, field_width
from .errors import LayoutError, ParseError
from .fields import FieldSpec

__all__ = ["Binary", "BinaryMeta"]

_ENDIAN = {
    "<": "<", ">": ">", "!": "!", "=": "=",
    "little": "<", "big": ">", "network": "!", "native": "=",
}


def _normalize_endian(value: str) -> str:
    try:
        return _ENDIAN[value]
    except (KeyError, TypeError):
        raise LayoutError(
            f"endian={value!r} is not valid; use one of {sorted(_ENDIAN)}"
        ) from None



_SCALAR_NAMES = {}


def _scalar_name(obj: object) -> str:
    """Render a scalar field class as the name it was declared with."""
    if not _SCALAR_NAMES:
        from .types import _Scalar

        _SCALAR_NAMES.update(
            (c.char, c.__name__) for c in _Scalar.__subclasses__()
        )
    if isinstance(obj, str):
        return _SCALAR_NAMES.get(obj, obj)
    return getattr(obj, "__name__", repr(obj))


def _declaration(f: FieldSpec) -> str:
    """Reconstruct the annotation a field was declared with."""
    kind = f.kind
    if kind == "scalar":
        return _scalar_name(f.char)
    if kind == "const":
        # Show a magic number as hex rather than as mojibake bytes; printable
        # ASCII magics stay in the b"..." form they were written in.
        if all(0x20 <= b < 0x7F for b in f.const):
            return f"Const[{f.const!r}]"
        return f'Const["{f.const.hex(" ")}"]'
    if kind == "padding":
        return f"Padding[{f.count}]"
    if kind == "bits":
        return f"Bits[{f.width}]" if f.stored else f"Bits[{f.width}, {f.const}]"
    if kind == "nested":
        return f.nested.__name__
    if kind in ("bytes", "str"):
        name = "Bytes" if kind == "bytes" else "Str"
        if f.expr is not None:
            arg = repr(f.expr)
        elif f.count is not None:
            arg = str(f.count)
        else:
            arg = "..."
        if kind == "str" and f.encoding not in (None, "utf-8"):
            arg = f"{arg}, {f.encoding!r}"
        return f"{name}[{arg}]"
    if kind == "array":
        elem = _scalar_name(f.elem)
        arg = repr(f.expr) if f.expr is not None else str(f.count)
        return f"Array[{elem}, {arg}]"
    return kind


class BinaryMeta(type):
    """Reads the class body's annotations and attaches a compiled codec.

    Fields must be known *before* ``type.__new__`` so that ``__slots__`` can be
    injected, which is why annotations are pulled from the namespace's annotate
    function rather than from the finished class.
    """

    def __new__(mcls, name, bases, ns, endian=None, **kwargs):
        parents = [b for b in bases if isinstance(b, BinaryMeta)]
        if not parents:
            # The Binary base itself.
            ns.setdefault("__slots__", ())
            return super().__new__(mcls, name, bases, ns, **kwargs)

        if "__slots__" in ns:
            raise LayoutError(
                f"{name}: do not declare __slots__; pybinary generates it from the fields"
            )

        holders = [b for b in parents if STRUCT_MARKER in b.__dict__]
        if len(holders) > 1:
            raise LayoutError(
                f"{name}: inherits fields from more than one structure "
                f"({', '.join(b.__name__ for b in holders)})"
            )
        inherited: list[FieldSpec] = list(holders[0].__pybinary_fields__) if holders else []

        if endian is None:
            endian = next(
                (b.__pybinary_endian__ for b in parents
                 if getattr(b, "__pybinary_endian__", None)), "<"
            )
        endian = _normalize_endian(endian)

        raw, sources = read_annotations(ns)
        fields = build_fields(name, raw, sources, inherited, endian)

        own = fields[len(inherited):]
        ns["__slots__"] = tuple(f.name for f in own if f.stored)
        ns[STRUCT_MARKER] = tuple(fields)

        cls = super().__new__(mcls, name, bases, ns, **kwargs)
        cls.__pybinary_endian__ = endian
        compile_codec(cls, fields, endian)
        return cls

    def __init__(cls, name, bases, ns, endian=None, **kwargs):
        super().__init__(name, bases, ns, **kwargs)

    def __repr__(cls) -> str:
        size = getattr(cls, "__struct_size__", None)
        shape = f"{size} bytes" if size is not None else "variable"
        return f"<structure {cls.__name__} ({shape})>"


class Binary(metaclass=BinaryMeta):
    """Base class for declarative binary structures.

    Subclasses declare fields as annotations; the codec is generated when the
    class is created::

        class Header(Binary, endian="<"):
            magic: Const[b"BLOB"]
            n: u32
            items: Bytes[n]
    """

    __slots__ = ()
    __pybinary_endian__ = "<"
    __struct_size__ = None
    __needs_view__ = False

    # unpack / unpack_from / pack are generated per subclass; these stubs
    # document the API and are what a bare Binary subclass would inherit.

    @classmethod
    def unpack(cls, buf, *, copy: bool = True):
        """Decode ``buf`` in full. Raises if any bytes are left over.

        ``copy=False`` returns ``Bytes`` fields as memoryviews over ``buf``.
        """
        mv = as_view(buf)
        obj, off = (cls._unpack_copy if copy else cls._unpack_view)(mv, 0)
        if off != len(mv):
            raise ParseError(
                f"{cls.__name__}: {len(mv) - off} trailing byte(s) after a complete record"
            )
        return obj

    @classmethod
    def unpack_from(cls, buf, offset: int = 0, *, copy: bool = True):
        """Decode one record at ``offset``; returns ``(instance, end_offset)``."""
        mv = as_view(buf)
        return (cls._unpack_copy if copy else cls._unpack_view)(mv, offset)

    def pack(self) -> bytes:
        """Encode to a new ``bytes`` object."""
        out = bytearray()
        self._pack_into(out)
        return bytes(out)

    def pack_into(self, out: bytearray) -> bytearray:
        """Append the encoding to an existing ``bytearray``."""
        self._pack_into(out)
        return out

    @classmethod
    def fields(cls) -> tuple[FieldSpec, ...]:
        """The structure's fields, in declaration order."""
        return getattr(cls, STRUCT_MARKER, ())

    @classmethod
    def describe(cls) -> str:
        """A table of byte offsets, sizes and declarations, for checking a
        layout against a hex dump.

        ``?`` marks a position or width that is only known at parse time.
        """
        fields = cls.fields()
        total = cls.__struct_size__
        head = (
            f"{cls.__name__}  endian={cls.__pybinary_endian__!r}  "
            f"({total} bytes)" if total is not None
            else f"{cls.__name__}  endian={cls.__pybinary_endian__!r}  (variable)"
        )
        rows = [("off", "size", "field", "type")]
        at: int | None = 0
        bits = 0  # accumulated within the current run of Bits fields
        for f in fields:
            if f.kind == "bits":
                # every member of a run reports the byte the run starts at
                rows.append((
                    "?" if at is None else str(at), f"{f.width}b",
                    f.name, _declaration(f),
                ))
                bits += f.width
                if bits % 8 == 0:
                    at = None if at is None else at + bits // 8
                    bits = 0
                continue
            width = field_width(f)
            rows.append((
                "?" if at is None else str(at),
                "?" if width is None else str(width),
                f.name,
                _declaration(f),
            ))
            at = None if (at is None or width is None) else at + width

        w0 = max(len(r[0]) for r in rows)
        w1 = max(len(r[1]) for r in rows)
        w2 = max(len(r[2]) for r in rows)
        lines = [head]
        lines += [
            f"  {a:>{w0}}  {b:>{w1}}  {c:<{w2}}  {d}" for a, b, c, d in rows
        ]
        return "\n".join(lines)
