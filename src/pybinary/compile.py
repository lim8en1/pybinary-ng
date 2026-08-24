"""Code generation.

A structure's field list is turned into Python source for a specialized codec,
which is compiled once when the class is created. Runs of consecutive
fixed-width fields collapse into a single ``struct.Struct`` call; only
variable-length fields cost an interpreted step.

The generated source is kept on ``cls.__codec_source__``.
"""

from __future__ import annotations

import ast
import struct

from .errors import BuildError, LayoutError, ParseError
from .fields import FieldSpec
from .types import _Scalar

__all__ = ["compile_codec", "as_view", "MISSING"]


def as_view(buf) -> memoryview:
    """Normalize any buffer to a 1-D, 1-byte-per-item memoryview."""
    mv = buf if type(buf) is memoryview else memoryview(buf)
    if mv.ndim != 1 or mv.itemsize != 1:
        mv = mv.cast("B")
    return mv


class _Missing:
    __slots__ = ()

    def __repr__(self) -> str:
        return "<missing>"


MISSING = _Missing()

_ARRAY_CACHE_LIMIT = 512


def _array_factory(endian: str, char: str):
    """Return a memoized ``count -> Struct`` factory for one array field."""
    cache: dict[int, struct.Struct] = {}

    def factory(count: int) -> struct.Struct:
        s = cache.get(count)
        if s is None:
            if len(cache) >= _ARRAY_CACHE_LIMIT:
                cache.clear()
            s = cache[count] = struct.Struct(f"{endian}{count}{char}")
        return s

    return factory


# --------------------------------------------------------------------------
# Runtime helpers referenced by generated code.
# --------------------------------------------------------------------------


def _short(cls_name, field, off, need, lim):
    raise ParseError(
        f"{cls_name}.{field}: needs {need} bytes at offset {off}, "
        f"but only {max(lim - off, 0)} remain"
    )


def _const_err(cls_name, field, got, want, off):
    raise ParseError(
        f"{cls_name}.{field}: expected {want!r}, got {bytes(got)!r} at offset {off}"
    )


def _len_err(cls_name, field, got, want, expr):
    raise BuildError(
        f"{cls_name}.{field}: holds {got} bytes but the declared length "
        f"{expr!r} evaluates to {want}"
    )


def _count_err(cls_name, field, got, want, expr):
    raise BuildError(
        f"{cls_name}.{field}: holds {got} elements but the declared count "
        f"{expr!r} evaluates to {want}"
    )


def _missing(cls_name, field):
    raise TypeError(f"{cls_name}() missing required field {field!r}")


def _type_err(cls_name, field, got, want):
    raise BuildError(
        f"{cls_name}.{field}: expected a {want.__name__} instance, got "
        f"{type(got).__name__}. The declared type fixes the wire layout, so a "
        f"subclass would encode fields the reader will not decode."
    )


def _ptr_err(cls_name, field, target, lim):
    raise ParseError(
        f"{cls_name}.{field}: pointer target {target} is outside the buffer "
        f"(0..{lim})"
    )


def _ptr_pack_err(cls_name, field, got, want, expr):
    raise BuildError(
        f"{cls_name}.{field}: the offset {expr!r} evaluates to {got}, but the "
        f"target lands at {want}"
    )


def _ptr_null_err(cls_name, field, got):
    raise BuildError(
        f"{cls_name}.{field}: the field is None, so its offset must be 0, got {got}"
    )


def _sum_err(cls_name, field, got, want):
    raise ParseError(
        f"{cls_name}.{field}: checksum is {got:#x} but the covered bytes hash "
        f"to {want:#x}"
    )


def _switch_err(cls_name, field, disc, off):
    raise ParseError(
        f"{cls_name}.{field}: no case for discriminator {disc!r} at offset {off}, "
        f"and no '...' fallback"
    )


def _switch_build_err(cls_name, field, disc):
    raise BuildError(
        f"{cls_name}.{field}: no case for discriminator {disc!r}, and no '...' fallback"
    )


def _switch_none(cls_name, field, disc):
    raise BuildError(
        f"{cls_name}.{field}: discriminator {disc!r} selects a payload, but the "
        f"field is None"
    )


def _switch_extra(cls_name, field, disc):
    raise BuildError(
        f"{cls_name}.{field}: discriminator {disc!r} selects no payload, so the "
        f"field must be None"
    )


def _if_err(cls_name, field, cond):
    if cond:
        raise BuildError(
            f"{cls_name}.{field}: the field's condition is true, so a value is "
            f"required, but it is None"
        )
    raise BuildError(
        f"{cls_name}.{field}: the field's condition is false, so it must be None, "
        f"but it holds a value"
    )


def _bits_err(cls_name, field, got, width):
    raise BuildError(
        f"{cls_name}.{field}: {got} does not fit in {width} bit(s) "
        f"(0..{(1 << width) - 1})"
    )


def _bitconst_err(cls_name, field, got, want):
    raise ParseError(f"{cls_name}.{field}: expected the {want} bit pattern, got {got}")


def _trailing(cls_name, off, lim):
    raise ParseError(
        f"{cls_name}: {lim - off} trailing byte(s) after a complete record"
    )


def _readable(value):
    """Make memoryview fields legible in reprs."""
    return bytes(value) if isinstance(value, memoryview) else value


# --------------------------------------------------------------------------
# Source builder
# --------------------------------------------------------------------------


class _Source:
    def __init__(self) -> None:
        self.lines: list[str] = []

    def __call__(self, indent: int, text: str = "") -> None:
        self.lines.append("    " * indent + text if text else "")

    def text(self) -> str:
        return "\n".join(self.lines) + "\n"


def _referenced_names(fields: list[FieldSpec]) -> set[str]:
    """Field names that must be hoisted into bare locals in generated code.

    Every inlined expression counts -- lengths, counts, conditions,
    discriminators and pointer offsets -- and wrapping kinds are walked
    recursively. A Checksum's ``start`` is deliberately excluded: it needs a
    byte position, not a value, so hoisting it would cost the hot path a store
    for nothing.
    """
    names: set[str] = set()
    for f in fields:
        for expr in f.expressions():
            for node in ast.walk(ast.parse(expr, mode="eval")):
                if isinstance(node, ast.Name):
                    names.add(node.id)
        if f.inner is not None:
            names |= _referenced_names([f.inner])
        if f.children():
            names |= _referenced_names(list(f.children()))
    return names


def _check_pointer_layout(
    clsname: str, fields: list[FieldSpec], computed: dict[str, FieldSpec]
) -> None:
    """Reject layouts a pointer cannot be encoded in."""
    if not any(f.kind == "pointer" for f in fields):
        return
    for f in fields:
        if f.kind in ("bytes", "str") and f.count is None and f.expr is None:
            raise LayoutError(
                f"{clsname}.{f.name}: a rest-of-buffer field cannot coexist with a "
                f"pointer -- it would swallow the pointed-to data on parse"
            )
    derivable = _derivable(fields)
    both = set(computed) & set(derivable)
    if both:
        name = sorted(both)[0]
        raise LayoutError(
            f"{clsname}.{name}: {name!r} is both a length and a pointer offset; "
            f"pybinary cannot compute both. Give one of them an expression."
        )
    # A checksum computed in pass 1 would hash the unpatched placeholder.
    index = {f.name: i for i, f in enumerate(fields)}
    for f in fields:
        if f.kind != "checksum":
            continue
        lo, hi = index[f.start], index[f.name]
        for off_name in computed:
            if lo <= index[off_name] < hi:
                raise LayoutError(
                    f"{clsname}.{f.name}: the checksum covers {off_name!r}, which is "
                    f"a pointer offset patched after the record is built, so the "
                    f"value hashed on build could never match the one read back"
                )


def _computed_offsets(fields: list[FieldSpec]) -> dict[str, FieldSpec]:
    """Pointer offsets ``pack()`` fills in, keyed by the offset field's name.

    Mirrors ``_derivable``: only a bare field name qualifies, and only when
    exactly one pointer uses it. Anything else stays the caller's job.
    """
    users: dict[str, list[FieldSpec]] = {}
    for f in fields:
        if f.kind == "pointer" and f.off_expr.isidentifier():
            users.setdefault(f.off_expr, []).append(f)
    return {n: uses[0] for n, uses in users.items() if len(uses) == 1}


def _positioned_names(fields: list[FieldSpec]) -> set[str]:
    """Field names whose byte position must be recorded into a ``_p_<name>``.

    Checksums need where their covered span begins; pointers need where their
    offset scalar was written so it can be patched. Both are positions, not
    values, which is why these are tracked separately from ``referenced``.
    """
    names: set[str] = set()
    for f in fields:
        if f.start:
            names.add(f.start)
        if f.inner is not None:
            names |= _positioned_names([f.inner])
        if f.children():
            names |= _positioned_names(list(f.children()))
    return names


def _derivable(fields: list[FieldSpec]) -> dict[str, FieldSpec]:
    """Length fields that ``__init__`` can fill in from the field they size.

    Only bare-name references qualify, and only when exactly one field uses
    them -- anything else is ambiguous and stays required.
    """
    users: dict[str, list[FieldSpec]] = {}

    def visit(f: FieldSpec) -> None:
        if f.expr and f.expr.isidentifier():
            users.setdefault(f.expr, []).append(f)
        # Pointer/Checksum payloads are unconditionally present, so a length
        # inside them is still derivable. If/Switch payloads are not: deriving
        # from one would call len(None) whenever the branch is not taken.
        if f.inner is not None and f.kind in ("pointer", "checksum"):
            visit(f.inner)

    for f in fields:
        visit(f)
    return {name: uses[0] for name, uses in users.items() if len(uses) == 1}


def group_bits(clsname: str, fields: list[FieldSpec]) -> list[FieldSpec]:
    """Collapse consecutive ``Bits`` fields into one synthetic ``bitrun``.

    Grouping first keeps the run emitters' invariants intact: they map struct
    return values 1:1 onto targets and sum ``f.size`` for the run width, and a
    bit run maps N fields onto a single ``{N}s`` value with no per-field byte
    size of its own.
    """
    out: list[FieldSpec] = []
    run: list[FieldSpec] = []

    def flush() -> None:
        if not run:
            return
        total = sum(m.width for m in run)
        if total % 8:
            raise LayoutError(
                f"{clsname}: the run of Bits fields starting at {run[0].name!r} is "
                f"{total} bits wide; a bit run must total a whole number of bytes. "
                f"Add a reserved field, e.g. _rsv: Bits[{8 - total % 8}, 0]."
            )
        out.append(FieldSpec(
            name=f"_bits_{run[0].name}", kind="bitrun", stored=False,
            size=total // 8, count=total, members=tuple(run),
        ))
        run.clear()

    for f in fields:
        if f.kind == "bits":
            run.append(f)
        else:
            flush()
            out.append(f)
    flush()
    return out


def _segment(fields: list[FieldSpec]) -> list[object]:
    """Group fields into fixed-width runs (lists) and single dynamic fields."""
    segments: list[object] = []
    run: list[FieldSpec] = []
    for f in fields:
        if f.fixed:
            run.append(f)
        else:
            if run:
                segments.append(run)
                run = []
            segments.append(f)
    if run:
        segments.append(run)
    return segments


def field_width(f: FieldSpec) -> int | None:
    """One field's encoded width, or None if it is variable-length."""
    if f.kind in ("scalar", "const", "padding", "bitrun", "checksum"):
        return f.size
    if f.kind == "bits":
        return None  # only a whole run has a byte width
    if f.kind in ("bytes", "str"):
        return f.count
    if f.kind == "nested":
        return f.nested.__struct_size__
    if f.kind == "array":
        if f.count is None:
            return None
        if issubclass(f.elem, _Scalar):
            return f.count * f.elem.size
        if f.elem.__struct_size__ is None:
            return None
        return f.count * f.elem.__struct_size__
    return None  # pragma: no cover - kinds are exhaustive


def _static_size(fields: list[FieldSpec]) -> int | None:
    """Total encoded width, or None if any field is variable-length."""
    total = 0
    for f in fields:
        width = field_width(f)
        if width is None:
            return None
        total += width
    return total


def compile_codec(cls: type, fields: list[FieldSpec], endian: str) -> None:
    """Generate and attach the codec for ``cls``."""
    ns: dict[str, object] = {
        "_cls": cls,
        "_CLS": cls.__name__,
        "_new": cls.__new__,
        "_len": len,
        "_str": str,
        "_bytes": bytes,
        "_range": range,
        "_M": MISSING,
        "_serror": struct.error,
        "_BuildError": BuildError,
        "_view": as_view,
        "_short": _short,
        "_const_err": _const_err,
        "_len_err": _len_err,
        "_count_err": _count_err,
        "_missing": _missing,
        "_type_err": _type_err,
        "_trailing": _trailing,
        "_if_err": _if_err,
        "_sum_err": _sum_err,
        "_ptr_err": _ptr_err,
        "_ptr_pack_err": _ptr_pack_err,
        "_ptr_null_err": _ptr_null_err,
        "_switch_err": _switch_err,
        "_switch_build_err": _switch_build_err,
        "_switch_none": _switch_none,
        "_switch_extra": _switch_extra,
        "_bits_err": _bits_err,
        "_bitconst_err": _bitconst_err,
        "_r": _readable,
        "_ifb": int.from_bytes,
    }

    referenced = _referenced_names(fields)
    positioned = _positioned_names(fields)
    computed = _computed_offsets(fields)
    has_pointer = any(f.kind == "pointer" for f in fields)
    _check_pointer_layout(cls.__name__, fields, computed)
    # Only pack needs an offset field's position; unpack reads it as a value.
    positioned_pack = positioned | set(computed)
    grouped = group_bits(cls.__name__, fields)
    segments = _segment(grouped)
    _bind_constants(ns, segments, endian)
    needs_view = _needs_view(fields)

    src = _Source()
    _emit_unpack(src, segments, referenced, positioned, has_pointer, copy=False)
    src(0)
    _emit_unpack(src, segments, referenced, positioned, has_pointer, copy=True)
    src(0)
    _emit_entry_points(src, needs_view, strict_trailing=not has_pointer)
    src(0)
    _emit_pack(src, segments, referenced, positioned_pack, computed, has_pointer)
    src(0)
    _emit_init(src, fields, computed)
    src(0)
    _emit_dunders(src, cls.__name__, fields, computed)

    source = src.text()
    exec(compile(source, f"<pybinary:{cls.__name__}>", "exec"), ns)

    cls.__codec_source__ = source
    cls.__struct_size__ = _static_size(grouped)
    cls.__needs_view__ = needs_view
    cls._unpack_view = staticmethod(ns["_unpack_view"])
    cls._unpack_copy = staticmethod(ns["_unpack_copy"])
    cls._pack_into = ns["_pack_into"]
    cls.unpack = staticmethod(ns["unpack"])
    cls.unpack_from = staticmethod(ns["unpack_from"])
    cls.pack = ns["pack"]
    cls.__init__ = ns["__init__"]
    cls.__repr__ = ns["__repr__"]
    cls.__eq__ = ns["__eq__"]
    cls.__hash__ = None  # fields are mutable


def _needs_view(fields: list[FieldSpec]) -> bool:
    """Whether zero-copy parsing requires the buffer to be a memoryview.

    Only raw ``Bytes`` fields slice the buffer and hand the result out; str,
    arrays and scalars copy regardless. Nested structures are consulted too,
    since they parse out of the same buffer object.
    """
    for f in fields:
        if f.kind == "bytes":
            return True
        if f.kind == "nested" and f.nested.__needs_view__:
            return True
        if f.kind == "array" and not issubclass(f.elem, _Scalar) and f.elem.__needs_view__:
            return True
        if f.inner is not None and _needs_view([f.inner]):
            return True
        if f.children() and _needs_view(list(f.children())):
            return True
    return False


def _bind_constants(ns: dict, segments: list[object], endian: str) -> None:
    """Pre-build every Struct, array factory and nested codec reference.

    Namespace keys are strings rather than segment indices so that wrapping
    kinds can derive stable sub-keys for their payloads: ``_s3`` is the
    top-level segment, ``_s3i`` its If/Pointer/Checksum inner, ``_s3v1`` the
    second Switch variant.
    """
    for i, seg in enumerate(segments):
        _bind_field(ns, str(i), seg, endian)


def _bind_field(ns: dict, key: str, seg: object, endian: str) -> None:
    if isinstance(seg, list):
        fmt = endian + "".join(_run_format(f) for f in seg)
        ns[f"_s{key}"] = struct.Struct(fmt)
        return
    if seg.fixed:
        # A lone fixed field emitted inside a wrapper still packs through a
        # Struct, so bind it as a one-element run.
        _bind_field(ns, key, [seg], endian)
        return
    if seg.kind == "array" and issubclass(seg.elem, _Scalar):
        if seg.count is not None:
            ns[f"_s{key}"] = struct.Struct(f"{endian}{seg.count}{seg.elem.char}")
        else:
            ns[f"_af{key}"] = _array_factory(endian, seg.elem.char)
    elif seg.kind == "array":
        _bind_nested(ns, key, seg.elem)
    elif seg.kind == "nested":
        _bind_nested(ns, key, seg.nested)
    elif seg.kind in ("if", "pointer", "checksum"):
        _bind_field(ns, key + "i", seg.inner, endian)
        if seg.kind == "checksum":
            ns[f"_fn{key}"] = seg.fn
        if seg.kind == "pointer" and seg.char:
            ns[f"_sp{key}"] = struct.Struct(endian + seg.char)
    elif seg.kind == "switch":
        for j, (_value, spec) in enumerate(seg.variants):
            if spec is not None:
                _bind_field(ns, f"{key}v{j}", spec, endian)


def _bind_nested(ns: dict, key: str, cls: type) -> None:
    ns[f"_nv{key}"] = cls._unpack_view
    ns[f"_nc{key}"] = cls._unpack_copy
    ns[f"_np{key}"] = cls._pack_into
    ns[f"_nt{key}"] = cls


def _run_format(f: FieldSpec) -> str:
    if f.kind == "padding":
        return f"{f.count}x"
    if f.kind == "const":
        return f"{len(f.const)}s"
    if f.kind == "bitrun":
        # Raw bytes: endian-immune, so bit order stays MSB-first whatever the
        # structure's endian is.
        return f"{f.size}s"
    return f.char


def _run_size(seg: list[FieldSpec]) -> int:
    return sum(f.size for f in seg)


# --------------------------------------------------------------------------
# unpack
# --------------------------------------------------------------------------


class _Ctx:
    """Per-mode emission settings threaded through the unpack emitters."""

    def __init__(self, copy: bool, referenced: set[str], positioned: set[str]) -> None:
        self.copy = copy
        self.referenced = referenced
        self.positioned = positioned
        self.nested_prefix = "_nc" if copy else "_nv"

    def slice_of(self, start: str, stop: str) -> str:
        # bytes(...) of a bytes slice is free, so copy mode works directly on a
        # bytes buffer without paying for a memoryview.
        expr = f"_buf[{start}:{stop}]"
        return f"_bytes({expr})" if self.copy else expr


def _emit_unpack(
    src: _Source, segments: list[object], referenced: set[str],
    positioned: set[str], has_pointer: bool, copy: bool,
) -> None:
    ctx = _Ctx(copy, referenced, positioned)
    src(0, f"def {'_unpack_copy' if copy else '_unpack_view'}(_buf, _off):")
    src(1, "_lim = _len(_buf)")
    src(1, "_o = _new(_cls)")
    if has_pointer:
        src(1, "_base = _off")
        # A pointer target usually sits past the record, so the cursor alone
        # would leave the next record (or the trailing check) reading the
        # target's bytes. Track the furthest byte touched instead.
        src(1, "_hi = _off")

    for i, seg in enumerate(segments):
        _emit_field_unpack(src, str(i), seg, ctx, 1)

    src(1, "return _o, _hi if _hi > _off else _off" if has_pointer
           else "return _o, _off")


def _emit_field_unpack(
    src: _Source, key: str, seg: object, ctx: _Ctx, indent: int
) -> None:
    """Emit one segment: a fixed run, or a single field of any kind."""
    if isinstance(seg, list):
        _emit_run_unpack(src, key, seg, ctx, indent)
    elif seg.fixed:
        _emit_run_unpack(src, key, [seg], ctx, indent)
    else:
        _emit_dynamic_unpack(src, key, seg, ctx, indent)


def _emit_run_unpack(
    src: _Source, key: str, seg: list[FieldSpec], ctx: _Ctx, indent: int
) -> None:
    referenced = ctx.referenced
    size = _run_size(seg)
    first = seg[0].members[0].name if seg[0].kind == "bitrun" else seg[0].name
    targets: list[str] = []
    checks: list[tuple[str, FieldSpec, int]] = []
    stores: list[str] = []

    # Field positions inside a run are static: _normalize_endian never yields
    # '@', and the other prefixes use standard sizes with no alignment padding.
    at = 0
    bitruns: list[tuple[str, FieldSpec]] = []
    for j, f in enumerate(seg):
        pos, at = at, at + (f.size or 0)
        if f.kind == "padding":
            continue
        if f.kind == "bitrun":
            tmp = f"_g{key}_{j}"
            targets.append(tmp)
            bitruns.append((tmp, f))
            continue
        if f.kind == "const":
            tmp = f"_k{key}_{j}"
            targets.append(tmp)
            checks.append((tmp, f, pos))
        elif f.name in referenced:
            targets.append(f.name)
            stores.append(f.name)
        else:
            targets.append(f"_o.{f.name}")

    if size:
        src(indent, f"if _off + {size} > _lim: _short(_CLS, {first!r}, _off, {size}, _lim)")
    at = 0
    for f in seg:
        if f.name in ctx.positioned:
            src(indent, f"_p_{f.name} = _off + {at}" if at else f"_p_{f.name} = _off")
        at += f.size or 0
    if targets:
        lhs = ", ".join(targets) + ("," if len(targets) == 1 else "")
        src(indent, f"{lhs} = _s{key}.unpack_from(_buf, _off)")
    if size:
        src(indent, f"_off += {size}")

    for tmp, f, pos in checks:
        where = f"_off - {size - pos}" if size - pos else "_off"
        src(
            indent,
            f"if {tmp} != {f.const!r}: "
            f"_const_err(_CLS, {f.name!r}, {tmp}, {f.const!r}, {where})",
        )
    for tmp, run in bitruns:
        _emit_bits_unpack(src, tmp, run, referenced, stores, indent)
    for n in stores:
        src(indent, f"_o.{n} = {n}")


def _emit_bits_unpack(
    src: _Source, tmp: str, run: FieldSpec, referenced: set[str],
    stores: list[str], indent: int,
) -> None:
    """Shift and mask each member out of a bit run's raw bytes."""
    src(indent, f"{tmp} = _ifb({tmp}, 'big')")
    shift = run.count  # total bits
    for m in run.members:
        shift -= m.width
        mask = (1 << m.width) - 1
        read = f"{tmp} & {mask}" if shift == 0 else f"({tmp} >> {shift}) & {mask}"
        if not m.stored:
            src(indent, f"if ({read}) != {m.const}: "
                        f"_bitconst_err(_CLS, {m.name!r}, {read}, {m.const})")
        elif m.name in referenced:
            src(indent, f"{m.name} = {read}")
            stores.append(m.name)
        else:
            src(indent, f"_o.{m.name} = {read}")


def _emit_dynamic_unpack(
    src: _Source, key: str, f: FieldSpec, ctx: _Ctx, indent: int
) -> None:
    name = f.name
    slice_of = ctx.slice_of
    np = ctx.nested_prefix
    if name in ctx.positioned:
        src(indent, f"_p_{name} = _off")

    if f.kind in ("bytes", "str"):
        if f.count is None and f.expr is None:  # rest of buffer
            value = (f"_str(_buf[_off:_lim], {f.encoding!r})" if f.kind == "str"
                     else slice_of("_off", "_lim"))
            src(indent, f"_o.{name} = {value}")
            src(indent, "_off = _lim")
            return
        if f.expr is None:
            src(indent, f"_e = _off + {f.count}")
            src(indent, f"if _e > _lim: _short(_CLS, {name!r}, _off, {f.count}, _lim)")
        else:
            src(indent, f"_n = {f.expr}")
            src(indent, "_e = _off + _n")
            src(indent, f"if _n < 0 or _e > _lim: _short(_CLS, {name!r}, _off, _n, _lim)")
        value = (f"_str(_buf[_off:_e], {f.encoding!r})" if f.kind == "str"
                 else slice_of("_off", "_e"))
        src(indent, f"_o.{name} = {value}")
        src(indent, "_off = _e")
        return

    if f.kind == "if":
        src(indent, f"if {f.cond}:")
        _emit_field_unpack(src, key + "i", f.inner, ctx, indent + 1)
        src(indent, "else:")
        src(indent + 1, f"_o.{name} = None")
        return

    if f.kind == "pointer":
        body = indent
        if f.nullable:
            src(indent, f"if ({f.off_expr}) != 0:")
            body = indent + 1
        src(body, f"_q = _base + ({f.off_expr})")
        src(body, f"if _q < 0 or _q > _lim: _ptr_err(_CLS, {name!r}, _q, _lim)")
        src(body, "_sv = _off")
        src(body, "_off = _q")
        _emit_field_unpack(src, key + "i", f.inner, ctx, body)
        src(body, "if _off > _hi: _hi = _off")
        src(body, "_off = _sv")
        if f.nullable:
            src(indent, "else:")
            src(indent + 1, f"_o.{name} = None")
        return

    if f.kind == "checksum":
        mask = (1 << (f.size * 8)) - 1
        # The span ends where the checksum begins, so it is complete right now.
        src(indent, f"_c = _fn{key}(_buf[_p_{f.start}:_off]) & {mask}")
        src(indent, f"if _off + {f.size} > _lim: "
                    f"_short(_CLS, {name!r}, _off, {f.size}, _lim)")
        src(indent, f"_cv, = _s{key}i.unpack_from(_buf, _off)")
        src(indent, f"_off += {f.size}")
        src(indent, f"if _cv != _c: _sum_err(_CLS, {name!r}, _cv, _c)")
        return

    if f.kind == "switch":
        src(indent, f"_d = {f.disc}")
        fallback = False
        for j, (value, spec) in enumerate(f.variants):
            if value is Ellipsis:
                fallback = True
                src(indent, "else:")
            else:
                src(indent, f"{'if' if j == 0 else 'elif'} _d == {value!r}:")
            if spec is None:
                src(indent + 1, f"_o.{name} = None")
            else:
                _emit_field_unpack(src, f"{key}v{j}", spec, ctx, indent + 1)
        if not fallback:
            src(indent, "else:")
            src(indent + 1, f"_switch_err(_CLS, {name!r}, _d, _off)")
        return

    if f.kind == "nested":
        src(indent, f"_o.{name}, _off = {np}{key}(_buf, _off)")
        return

    if f.kind == "array":
        if issubclass(f.elem, _Scalar):
            width = f.elem.size
            if f.expr is None:
                total = f.count * width
                src(indent, f"_e = _off + {total}")
                src(indent, f"if _e > _lim: _short(_CLS, {name!r}, _off, {total}, _lim)")
                src(indent, f"_o.{name} = _s{key}.unpack_from(_buf, _off)")
            else:
                src(indent, f"_n = {f.expr}")
                src(indent, f"_e = _off + _n * {width}")
                src(indent, f"if _n < 0 or _e > _lim: _short(_CLS, {name!r}, _off, _n * {width}, _lim)")
                src(indent, f"_o.{name} = _af{key}(_n).unpack_from(_buf, _off)")
            src(indent, "_off = _e")
        else:
            count = f.count if f.expr is None else None
            src(indent, f"_n = {count if count is not None else f.expr}")
            src(indent, "_l = []")
            src(indent, "_ap = _l.append")
            src(indent, "for _i in _range(_n):")
            src(indent + 1, f"_v, _off = {np}{key}(_buf, _off)")
            src(indent + 1, "_ap(_v)")
            src(indent, f"_o.{name} = _l")
        return

    raise AssertionError(f"unhandled field kind {f.kind!r}")  # pragma: no cover


# --------------------------------------------------------------------------
# Entry points
#
# Generated per class rather than inherited so the codec functions are closure
# constants instead of attribute lookups, and so classes that never slice the
# buffer can skip building a memoryview entirely.
# --------------------------------------------------------------------------


def _emit_prepare(src: _Source, indent: int, needs_view: bool, copy_known: bool) -> None:
    """Emit buffer normalization. ``copy_known`` means copy mode is in effect."""
    if needs_view and not copy_known:
        src(indent, "if type(_buf) is not memoryview:")
        src(indent + 1, "_buf = _view(_buf)")
        src(indent, "elif _buf.itemsize != 1 or _buf.ndim != 1:")
        src(indent + 1, "_buf = _buf.cast('B')")
    else:
        # bytes and bytearray slice and measure correctly on their own.
        src(indent, "_t = type(_buf)")
        src(indent, "if _t is not bytes and _t is not bytearray:")
        src(indent + 1, "_buf = _view(_buf)")


def _emit_entry_points(
    src: _Source, needs_view: bool, strict_trailing: bool = True
) -> None:
    for name in ("unpack", "unpack_from"):
        whole = name == "unpack"
        start = "0" if whole else "_offset"
        args = "_buf, *, copy=True" if whole else "_buf, _offset=0, *, copy=True"

        src(0, f"def {name}({args}):")
        if needs_view:
            src(1, "if copy:")
            _emit_prepare(src, 2, needs_view, copy_known=True)
            src(2, f"_o, _off = _unpack_copy(_buf, {start})")
            src(1, "else:")
            _emit_prepare(src, 2, needs_view, copy_known=False)
            src(2, f"_o, _off = _unpack_view(_buf, {start})")
        else:
            # Nothing in this structure slices the buffer, so both modes are
            # the same code and no memoryview is needed.
            _emit_prepare(src, 1, needs_view, copy_known=True)
            src(1, f"_o, _off = _unpack_view(_buf, {start})")

        if whole:
            if strict_trailing:
                src(1, "_lim = _len(_buf)")
                src(1, "if _off != _lim: _trailing(_CLS, _off, _lim)")
            # With pointers the targets normally sit past the cursor, so
            # "trailing bytes" is not a meaningful notion for this layout.
            src(1, "return _o")
        else:
            src(1, "return _o, _off")
        src(0)

    src(0, "def pack(_self):")
    src(1, "_out = bytearray()")
    src(1, "_pack_into(_self, _out)")
    src(1, "return _bytes(_out)")


# --------------------------------------------------------------------------
# pack
# --------------------------------------------------------------------------


def _emit_pack(
    src: _Source, segments: list[object], referenced: set[str],
    positioned: set[str], computed: dict[str, FieldSpec], has_pointer: bool,
) -> None:
    src(0, "def _pack_into(_self, _out):")
    if has_pointer:
        src(1, "_base = _len(_out)")
    src(1, "try:")
    if not segments:
        src(2, "pass")

    # Pass 1 writes the record; pointer targets go after it, so their segments
    # emit nothing here beyond the offset placeholder in the run above them.
    pointers = []
    for i, seg in enumerate(segments):
        if not isinstance(seg, list) and seg.kind == "pointer":
            pointers.append((str(i), seg))
            continue
        _emit_field_pack(src, str(i), seg, referenced, positioned, 2, computed)

    # Pass 2: append each target and patch the offset that points at it.
    for key, seg in pointers:
        _emit_pointer_pack(src, key, seg, referenced, positioned, computed, 2)

    src(1, "except _serror as _exc:")
    src(2, "raise _BuildError(f'{_CLS}: {_exc}') from None")


def _emit_pointer_pack(
    src: _Source, key: str, f: FieldSpec, referenced: set[str],
    positioned: set[str], computed: dict[str, FieldSpec], indent: int,
) -> None:
    name = f.name
    auto = computed.get(f.off_expr) is f
    width = struct.calcsize(f.char) if f.char else 0
    body = indent

    if f.nullable:
        src(indent, f"if _self.{name} is None:")
        if auto:
            src(indent + 1, f"_out[_p_{f.off_expr}:_p_{f.off_expr} + {width}] "
                            f"= _sp{key}.pack(0)")
        else:
            src(indent + 1, f"if ({f.off_expr}) != 0: "
                            f"_ptr_null_err(_CLS, {name!r}, ({f.off_expr}))")
        src(indent, "else:")
        body = indent + 1

    src(body, "_t = _len(_out) - _base")
    if auto:
        src(body, f"_out[_p_{f.off_expr}:_p_{f.off_expr} + {width}] "
                  f"= _sp{key}.pack(_t)")
    else:
        src(body, f"if ({f.off_expr}) != _t: _ptr_pack_err(_CLS, {name!r}, "
                  f"({f.off_expr}), _t, {f.off_expr!r})")
    _emit_field_pack(src, key + "i", f.inner, referenced, positioned, body)


def _emit_field_pack(
    src: _Source, key: str, seg: object, referenced: set[str],
    positioned: set[str], indent: int, computed: dict | None = None,
) -> None:
    """Emit one segment: a fixed run, or a single field of any kind."""
    computed = computed or {}
    if isinstance(seg, list):
        _emit_run_pack(src, key, seg, referenced, positioned, indent, computed)
    elif seg.fixed:
        _emit_run_pack(src, key, [seg], referenced, positioned, indent, computed)
    else:
        _emit_dynamic_pack(src, key, seg, referenced, positioned, indent)


def _emit_run_pack(
    src: _Source, key: str, seg: list[FieldSpec], referenced: set[str],
    positioned: set[str], indent: int, computed: dict | None = None,
) -> None:
    computed = computed or {}
    at = 0
    for f in seg:
        if f.name in positioned:
            src(indent, f"_p_{f.name} = _len(_out) + {at}" if at
                        else f"_p_{f.name} = _len(_out)")
        at += f.size or 0

    args: list[str] = []
    for j, f in enumerate(seg):
        if f.kind == "padding":
            continue
        if f.kind == "bitrun":
            tmp = f"_b{key}_{j}"
            _emit_bits_pack(src, tmp, f, referenced, indent)
            args.append(f"{tmp}.to_bytes({f.size}, 'big')")
        elif f.kind == "const":
            args.append(repr(f.const))
        elif f.name in computed:
            # Patched in pass 2 once the target's position is known.
            args.append("0")
        elif f.name in referenced:
            src(indent, f"{f.name} = _self.{f.name}")
            args.append(f.name)
        else:
            args.append(f"_self.{f.name}")

    # With no args this is an all-padding run; struct emits the zero bytes.
    src(indent, f"_out += _s{key}.pack({', '.join(args)})")


def _emit_bits_pack(
    src: _Source, tmp: str, run: FieldSpec, referenced: set[str], indent: int
) -> None:
    """OR each member into the run's integer, checking it fits its width."""
    src(indent, f"{tmp} = 0")
    shift = run.count
    for m in run.members:
        shift -= m.width
        if not m.stored:
            if m.const:
                src(indent, f"{tmp} |= {m.const << shift}")
            continue
        # A bit field can size a later field, so it needs the same bare-local
        # hoist an ordinary scalar in a run gets.
        var = m.name if m.name in referenced else "_v"
        src(indent, f"{var} = _self.{m.name}")
        src(indent, f"if {var} < 0 or {var} > {(1 << m.width) - 1}: "
                    f"_bits_err(_CLS, {m.name!r}, {var}, {m.width})")
        src(indent, f"{tmp} |= {var}" if shift == 0 else f"{tmp} |= {var} << {shift}")


def _emit_dynamic_pack(
    src: _Source, key: str, f: FieldSpec, referenced: set[str],
    positioned: set[str], indent: int,
) -> None:
    name = f.name
    if name in positioned:
        src(indent, f"_p_{name} = _len(_out)")

    if f.kind == "if":
        # Bind before the branch: the `elif` tests it, and a str payload would
        # call .encode() on None before any check of ours could run.
        src(indent, f"_v = _self.{name}")
        src(indent, f"if {f.cond}:")
        src(indent + 1, f"if _v is None: _if_err(_CLS, {name!r}, True)")
        _emit_field_pack(src, key + "i", f.inner, referenced, positioned, indent + 1)
        src(indent, "elif _v is not None:")
        src(indent + 1, f"_if_err(_CLS, {name!r}, False)")
        return

    if f.kind == "checksum":
        mask = (1 << (f.size * 8)) - 1
        # A plain slice, not memoryview(_out)[...]: holding a live export
        # across the next `_out +=` raises BufferError on resize.
        src(indent, f"_cv = _fn{key}(_out[_p_{f.start}:]) & {mask}")
        src(indent, f"_out += _s{key}i.pack(_cv)")
        return

    if f.kind == "switch":
        src(indent, f"_d = {f.disc}")
        fallback = False
        for j, (value, spec) in enumerate(f.variants):
            if value is Ellipsis:
                fallback = True
                src(indent, "else:")
            else:
                src(indent, f"{'if' if j == 0 else 'elif'} _d == {value!r}:")
            if spec is None:
                src(indent + 1, f"if _self.{name} is not None: "
                                f"_switch_extra(_CLS, {name!r}, _d)")
            else:
                src(indent + 1, f"if _self.{name} is None: "
                                f"_switch_none(_CLS, {name!r}, _d)")
                _emit_field_pack(src, f"{key}v{j}", spec, referenced, positioned, indent + 1)
        if not fallback:
            src(indent, "else:")
            src(indent + 1, f"_switch_build_err(_CLS, {name!r}, _d)")
        return

    if f.kind == "str":
        src(indent, f"_v = _self.{name}.encode({f.encoding!r})")
    else:
        src(indent, f"_v = _self.{name}")

    if f.kind in ("bytes", "str"):
        want = f.count if f.expr is None else f.expr
        if want is not None:
            src(indent, f"if _len(_v) != ({want}): _len_err(_CLS, {name!r}, _len(_v), ({want}), {str(want)!r})")
        src(indent, "_out += _v")
        return

    if f.kind == "nested":
        # The declared class fixes the layout, so a subclass instance would
        # write fields no reader of this structure will decode.
        src(indent, f"if type(_v) is not _nt{key}: _type_err(_CLS, {name!r}, _v, _nt{key})")
        src(indent, f"_np{key}(_v, _out)")
        return

    if f.kind == "array":
        want = f.count if f.expr is None else f.expr
        src(indent, f"if _len(_v) != ({want}): _count_err(_CLS, {name!r}, _len(_v), ({want}), {str(want)!r})")
        if issubclass(f.elem, _Scalar):
            if f.expr is None:
                src(indent, f"_out += _s{key}.pack(*_v)")
            else:
                src(indent, f"_out += _af{key}(_len(_v)).pack(*_v)")
        else:
            src(indent, "for _x in _v:")
            src(indent + 1, f"if type(_x) is not _nt{key}: _type_err(_CLS, {name!r}, _x, _nt{key})")
            src(indent + 1, f"_np{key}(_x, _out)")
        return

    raise AssertionError(f"unhandled field kind {f.kind!r}")  # pragma: no cover


# --------------------------------------------------------------------------
# __init__ and friends
# --------------------------------------------------------------------------


def _emit_init(
    src: _Source, fields: list[FieldSpec], computed: dict | None = None,
) -> None:
    computed = computed or {}
    stored = [f for f in fields if f.stored]
    derivable = _derivable(fields)

    params = ", ".join(f"{f.name}=_M" for f in stored)
    src(0, f"def __init__(self{', ' + params if params else ''}):")
    if not stored:
        src(1, "pass")
        return

    for f in stored:
        source = derivable.get(f.name)
        if f.name in computed:
            # pack() works this out; a hand-built record starts at zero.
            src(1, f"if {f.name} is _M: {f.name} = 0")
        elif f.optional:
            src(1, f"if {f.name} is _M: {f.name} = None")
        elif source is None:
            src(1, f"if {f.name} is _M: _missing(_CLS, {f.name!r})")
        else:
            # Fill in a length/count field from the field it measures.
            src(1, f"if {f.name} is _M:")
            src(2, f"if {source.name} is _M: _missing(_CLS, {f.name!r})")
            if source.kind == "str":
                src(2, f"{f.name} = _len({source.name}.encode({source.encoding!r}))")
            else:
                src(2, f"{f.name} = _len({source.name})")

    for f in stored:
        src(1, f"self.{f.name} = {f.name}")


def _emit_dunders(
    src: _Source, clsname: str, fields: list[FieldSpec],
    computed: dict | None = None,
) -> None:
    computed = computed or {}
    stored = [f for f in fields if f.stored]

    src(0, "def __repr__(self):")
    if stored:
        parts = ", ".join(f"{f.name}={{_r(self.{f.name})!r}}" for f in stored)
        src(1, f'return f"{clsname}({parts})"')
    else:
        src(1, f'return "{clsname}()"')

    src(0)
    src(0, "def __eq__(self, other):")
    src(1, "if other.__class__ is not self.__class__: return NotImplemented")
    # A pointer offset pack() computes is fixed by the layout, so two packed
    # records with equal content always agree on it. Comparing it would only
    # ever separate a hand-built record from a parsed one, so it is left out.
    compared = [f for f in stored if f.name not in computed]
    if compared:
        cond = " and ".join(f"self.{f.name} == other.{f.name}" for f in compared)
        src(1, f"return {cond}")
    else:
        src(1, "return True")
