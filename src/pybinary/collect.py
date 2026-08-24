"""Turn a class body's annotations into an ordered list of :class:`FieldSpec`.

Annotations are read in ``FORWARDREF`` format, so a reference to an earlier
field (``Bytes[n]``) arrives as a ``ForwardRef`` carrying the source of the
expression rather than raising ``NameError``.
"""

from __future__ import annotations

import ast
import sys
import typing

from ._compat import ForwardRef
from .errors import LayoutError
from .fields import FieldSpec
from .types import (
    ARRAY, BITS, BYTES, CHECKSUM, CONST, IF, PADDING, POINTER, STR, SWITCH,
    _Marker, _Scalar,
)

__all__ = ["build_fields", "STRUCT_MARKER"]

# Attribute that marks a class as a compiled structure. Checked instead of
# importing Binary, which would be circular.
STRUCT_MARKER = "__pybinary_fields__"

# Stored field names become locals in generated code, where everything internal
# is underscore-prefixed.
_RESERVED = frozenset({"self", "cls"})

# Nodes permitted inside a length/count expression. Deliberately narrow: these
# get inlined into generated source, and a tight grammar keeps error messages
# specific.
_ARITH_NODES = (
    ast.Expression, ast.Name, ast.Load, ast.Constant, ast.BinOp, ast.UnaryOp,
    ast.Add, ast.Sub, ast.Mult, ast.FloorDiv, ast.Mod, ast.USub, ast.UAdd,
    ast.BitAnd, ast.BitOr, ast.BitXor, ast.LShift, ast.RShift, ast.Invert,
)

# Conditions may compare and combine bitwise. Kept separate from the arithmetic
# set on purpose: sharing one widened tuple would make `Bytes[n > 2]` a legal
# zero-or-one-byte field, which is correctly rejected today.
_COND_NODES = _ARITH_NODES + (
    ast.Compare, ast.Eq, ast.NotEq, ast.Lt, ast.LtE, ast.Gt, ast.GtE,
    ast.BoolOp, ast.And, ast.Or, ast.Not,
)


def build_fields(
    clsname: str,
    annotations: dict[str, object],
    sources: dict[str, str],
    inherited: list[FieldSpec],
    endian: str = "<",
) -> list[FieldSpec]:
    """Interpret ``annotations`` into fields, appended after ``inherited``.

    ``sources`` is the same mapping in ``Format.STRING`` form, used to catch
    references that silently resolved to something other than the field they
    name.
    """
    fields = list(inherited)
    by_name = {f.name: f for f in fields}

    for name, ann in annotations.items():
        if _is_classvar(ann):
            continue
        spec = _interpret(clsname, name, ann, by_name, endian)
        if spec is None:
            continue
        if spec.name in by_name:
            raise LayoutError(f"{clsname}.{spec.name}: field is declared twice")
        _check_name(clsname, spec)
        _check_shadowing(clsname, spec, sources.get(name), by_name)
        fields.append(spec)
        by_name[spec.name] = spec

    return fields


def _check_shadowing(
    clsname: str, spec: FieldSpec, source: str | None, seen: dict[str, FieldSpec]
) -> None:
    """Reject annotations where a field name was captured by an outer binding.

    Annotations resolve against module globals and builtins first, so
    ``Bytes[len]`` picks up the builtin and ``Bytes[n]`` picks up a module-level
    ``n = 5`` -- the latter silently, as a static length. Comparing the source
    text against what actually resolved catches both.
    """
    if not source:
        return
    try:
        tree = ast.parse(source, mode="eval")
    except SyntaxError:  # pragma: no cover - annotation sources are valid python
        return

    used = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}
    # Every reference the spec actually captured -- lengths, conditions,
    # discriminators, pointer offsets, checksum spans -- recursively. Anything
    # left over appeared in the source but never reached the spec, which means
    # it resolved to something else.
    collisions = (used & seen.keys()) - spec.ref_names()
    if not collisions:
        return

    name = sorted(collisions)[0]
    raise LayoutError(
        f"{clsname}.{spec.name}: {name!r} in {source!r} resolved to a module global or "
        f"builtin instead of the field {name!r} declared above -- annotations see those "
        f"names first. Quote the reference (e.g. Bytes[{name!r}]) or rename the field."
    )


def _check_name(clsname: str, spec: FieldSpec) -> None:
    if not spec.stored:
        return  # padding/const never become locals, so anything goes
    if spec.name.startswith("_"):
        raise LayoutError(
            f"{clsname}.{spec.name}: field names may not start with an underscore "
            f"(reserved for generated code)"
        )
    if spec.name in _RESERVED:
        raise LayoutError(f"{clsname}.{spec.name}: {spec.name!r} is a reserved name")


def _is_classvar(ann: object) -> bool:
    return ann is typing.ClassVar or typing.get_origin(ann) is typing.ClassVar


def _is_struct(obj: object) -> bool:
    return isinstance(obj, type) and STRUCT_MARKER in getattr(obj, "__dict__", {})


def _is_scalar(obj: object) -> bool:
    return isinstance(obj, type) and issubclass(obj, _Scalar) and obj is not _Scalar


def _interpret(
    clsname: str, name: str, ann: object, seen: dict[str, FieldSpec],
    endian: str = "<",
) -> FieldSpec | None:
    where = f"{clsname}.{name}"

    if _is_scalar(ann):
        return FieldSpec(name=name, kind="scalar", char=ann.char, size=ann.size)

    if _is_struct(ann):
        return FieldSpec(name=name, kind="nested", nested=ann,
                         size=ann.__struct_size__)

    if isinstance(ann, tuple) and len(ann) == 2 and isinstance(ann[0], _Marker):
        return _parametric(where, name, ann[0], ann[1], seen, endian)

    if isinstance(ann, ForwardRef):
        raise LayoutError(
            f"{where}: unknown field type {ann.__forward_arg__!r}; it is not defined "
            f"at the point the class is created"
        )

    if isinstance(ann, str):
        # PEP 563 stringifies every annotation, including `ver: u16`, so the
        # marker tuples this module works from never get built. Caught here
        # rather than misparsed: a field type is never legitimately a bare
        # string (quoted references only ever appear *inside* a subscript).
        raise LayoutError(
            f"{where}: the annotation arrived as the string {ann!r}. Remove "
            f"`from __future__ import annotations` from this module; pybinary "
            f"reads annotations itself and cannot work with PEP 563 strings."
        )

    raise LayoutError(
        f"{where}: {ann!r} is not a pybinary field type. Use a scalar (u8/i32/f64/...), "
        f"Const[...], Bytes[...], Str[...], Array[...], Padding[...], or a Binary subclass."
    )


def _parametric(
    where: str, name: str, marker, arg, seen: dict[str, FieldSpec],
    endian: str = "<",
) -> FieldSpec:
    if marker is CONST:
        value = _const_bytes(where, arg, endian)
        return FieldSpec(name=name, kind="const", stored=False,
                         size=len(value), const=value)

    if marker is PADDING:
        if not isinstance(arg, int) or isinstance(arg, bool) or arg < 0:
            raise LayoutError(f"{where}: Padding[...] takes a non-negative int, got {arg!r}")
        return FieldSpec(name=name, kind="padding", stored=False, size=arg, count=arg)

    if marker is BYTES:
        count, expr = _length(where, arg, seen)
        return FieldSpec(name=name, kind="bytes", count=count, expr=expr,
                         size=count if expr is None else None)

    if marker is STR:
        encoding = "utf-8"
        if isinstance(arg, tuple):
            if len(arg) != 2 or not isinstance(arg[1], str):
                raise LayoutError(
                    f"{where}: Str[...] takes a length and an optional encoding name"
                )
            arg, encoding = arg
        count, expr = _length(where, arg, seen)
        return FieldSpec(name=name, kind="str", count=count, expr=expr, encoding=encoding)

    if marker is ARRAY:
        if not isinstance(arg, tuple) or len(arg) != 2:
            raise LayoutError(f"{where}: Array[...] takes an element type and a count")
        elem, raw_count = arg
        if not (_is_scalar(elem) or _is_struct(elem)):
            raise LayoutError(
                f"{where}: Array element must be a scalar or a Binary subclass, got {elem!r}"
            )
        count, expr = _length(where, raw_count, seen, allow_rest=False)
        return FieldSpec(name=name, kind="array", elem=elem, count=count, expr=expr)

    if marker is BITS:
        return _bits(where, name, arg)

    if marker is POINTER:
        if not isinstance(arg, tuple) or len(arg) not in (2, 3):
            raise LayoutError(
                f"{where}: Pointer[...] takes an offset and a field type, "
                f"e.g. Pointer[\"off\", Str[\"n\"]]"
            )
        nullable = False
        if len(arg) == 3:
            raw_off, inner_ann, flag = arg
            if flag != "nullable":
                raise LayoutError(
                    f"{where}: the third Pointer argument may only be \"nullable\", "
                    f"got {flag!r}"
                )
            nullable = True
        else:
            raw_off, inner_ann = arg
        off_expr = _offset_expr(where, raw_off, seen)
        inner = _interpret(clsname_of(where), name, inner_ann, seen, endian)
        if inner is None or not inner.stored:
            raise LayoutError(
                f"{where}: Pointer[...] target must be a stored field type"
            )
        if inner.kind in ("if", "switch", "pointer", "checksum", "bits", "bitrun"):
            raise LayoutError(
                f"{where}: Pointer[...] cannot target a {inner.kind} field"
            )
        # An auto-patched offset is written through the referenced scalar's own
        # struct char, recorded here while `seen` is in scope.
        char = None
        if off_expr.isidentifier():
            ref = seen[off_expr]
            if ref.kind == "bits":
                raise LayoutError(
                    f"{where}: a pointer offset cannot be the bit field "
                    f"{off_expr!r}; it must be a whole scalar field"
                )
            char = ref.char
        return FieldSpec(name=name, kind="pointer", off_expr=off_expr, inner=inner,
                         nullable=nullable, char=char)

    if marker is CHECKSUM:
        if not isinstance(arg, tuple) or len(arg) != 3:
            raise LayoutError(
                f"{where}: Checksum[...] takes an integer type, a function and the "
                f"field its coverage starts at, e.g. "
                f"Checksum[u32, zlib.crc32, magic]"
            )
        ty, fn, start_ref = arg
        if not _is_scalar(ty) or ty.char in "fd":
            raise LayoutError(
                f"{where}: Checksum[...] must store an integer scalar type, got {ty!r}"
            )
        if isinstance(fn, ForwardRef):
            raise LayoutError(
                f"{where}: the checksum function {fn.__forward_arg__!r} is not defined "
                f"at the point the class is created -- import it above the class"
            )
        if not callable(fn):
            raise LayoutError(
                f"{where}: the checksum function must be callable, got {fn!r}"
            )
        start = _reference(where, start_ref, seen, "checksum coverage")
        if seen[start].kind == "bits":
            raise LayoutError(
                f"{where}: checksum coverage cannot start at the bit field {start!r}; "
                f"a span must start on a byte boundary"
            )
        inner = FieldSpec(name=name, kind="scalar", char=ty.char, size=ty.size)
        return FieldSpec(name=name, kind="checksum", stored=False, size=ty.size,
                         char=ty.char, fn=fn, start=start, inner=inner)

    if marker is SWITCH:
        if not isinstance(arg, tuple) or len(arg) != 2:
            raise LayoutError(
                f"{where}: Switch[...] takes a discriminator and a {{value: type}} "
                f"mapping, e.g. Switch[\"kind\", {{1: Ping, 2: Pong}}]"
            )
        raw_disc, mapping = arg
        if isinstance(raw_disc, bool):
            raise LayoutError(
                f"{where}: the discriminator collapsed to the constant {raw_disc!r}; "
                f"a name in it resolved to a module global instead of the field "
                f"declared above. Quote it, e.g. Switch[\"kind\", ...]."
            )
        if isinstance(raw_disc, ForwardRef):
            disc = raw_disc.__forward_arg__
        elif isinstance(raw_disc, str):
            disc = raw_disc
        else:
            raise LayoutError(
                f"{where}: expected a discriminator referring to an earlier field, "
                f"got {raw_disc!r}"
            )
        _validate_expr(where, disc, seen, _ARITH_NODES, "discriminator")
        if not isinstance(mapping, dict) or not mapping:
            raise LayoutError(
                f"{where}: Switch[...] needs a non-empty {{value: type}} mapping, "
                f"got {mapping!r}"
            )
        variants = _variants(where, name, mapping, seen, endian)
        return FieldSpec(name=name, kind="switch", disc=disc, variants=variants)

    if marker is IF:
        if not isinstance(arg, tuple) or len(arg) != 2:
            raise LayoutError(
                f"{where}: If[...] takes a condition and a field type, "
                f"e.g. If[\"version > 1\", u32]"
            )
        raw_cond, inner_ann = arg
        cond = _condition(where, raw_cond, seen)
        inner = _interpret(clsname_of(where), name, inner_ann, seen, endian)
        if inner is None or not inner.stored:
            raise LayoutError(
                f"{where}: If[...] cannot wrap {inner.kind if inner else 'that'}; "
                f"the payload must be a stored field"
            )
        if inner.kind in ("if", "switch", "pointer", "checksum", "bits", "bitrun"):
            raise LayoutError(
                f"{where}: If[...] cannot wrap a {inner.kind} field"
            )
        return FieldSpec(name=name, kind="if", cond=cond, inner=inner)

    raise LayoutError(f"{where}: unsupported field type")


def _bits(where: str, name: str, arg) -> FieldSpec:
    """``Bits[width]`` or ``Bits[width, constant]``."""
    const = None
    if isinstance(arg, tuple):
        if len(arg) != 2:
            raise LayoutError(
                f"{where}: Bits[...] takes a width and an optional constant"
            )
        arg, const = arg
        if not isinstance(const, int) or isinstance(const, bool) or const < 0:
            raise LayoutError(
                f"{where}: Bits[...] constant must be a non-negative int, got {const!r}"
            )
    if not isinstance(arg, int) or isinstance(arg, bool) or arg < 1:
        raise LayoutError(
            f"{where}: Bits[...] takes a width of at least 1 bit, got {arg!r}"
        )
    if const is not None and const >= (1 << arg):
        raise LayoutError(
            f"{where}: Bits[{arg}] constant {const} does not fit in {arg} bits"
        )
    return FieldSpec(name=name, kind="bits", width=arg,
                     stored=const is None, const=const)


_BYTEORDER = {"<": "little", ">": "big", "!": "big", "=": sys.byteorder}


def _offset_expr(where: str, arg, seen: dict[str, FieldSpec]) -> str:
    """Resolve a pointer offset to inlinable source text."""
    if isinstance(arg, bool):
        raise LayoutError(
            f"{where}: the pointer offset collapsed to the constant {arg!r}; a name "
            f"in it resolved to a module global instead of the field declared above"
        )
    if isinstance(arg, ForwardRef):
        expr = arg.__forward_arg__
    elif isinstance(arg, str):
        expr = arg
    else:
        raise LayoutError(
            f"{where}: expected an offset referring to an earlier field, got {arg!r}"
        )
    _validate_expr(where, expr, seen, _ARITH_NODES, "pointer offset")
    return expr


def _reference(where: str, arg, seen: dict[str, FieldSpec], what: str) -> str:
    """Resolve a bare reference to an earlier field's name."""
    if isinstance(arg, ForwardRef):
        ref = arg.__forward_arg__
    elif isinstance(arg, str):
        ref = arg
    else:
        shadowed = getattr(arg, "__name__", None)
        if shadowed in seen:
            raise LayoutError(
                f"{where}: {shadowed!r} resolved to {arg!r} instead of the field "
                f"declared above. Quote the reference, e.g. {shadowed!r}."
            )
        raise LayoutError(
            f"{where}: expected the name of an earlier field for {what}, got {arg!r}"
        )
    if not ref.isidentifier():
        raise LayoutError(
            f"{where}: {what} must name a single earlier field, got {ref!r}"
        )
    if ref not in seen:
        raise LayoutError(
            f"{where}: {what} names {ref!r}, which is not a field declared before "
            f"this one"
        )
    return ref


def _const_bytes(where: str, arg, endian: str) -> bytes:
    """Resolve a ``Const[...]`` argument to the literal bytes on the wire.

    Three spellings, because magic values are written three ways in the wild:

    * ``b"BLOB"`` -- an explicit byte sequence, used as-is.
    * ``"76 2f 31 01"`` -- hex digits, whitespace and ``_`` ignored, used
      as-is. Always hex, never ASCII: write ``b"..."`` for that.
    * ``0x01312F76`` -- an integer, encoded in the structure's byte order.
      Width defaults to the fewest bytes that hold it; ``Const[0x1234, 4]``
      pads to an explicit width.
    """
    width = None
    if isinstance(arg, tuple):
        if len(arg) != 2:
            raise LayoutError(
                f"{where}: Const[...] takes a value and an optional byte width"
            )
        arg, width = arg
        if not isinstance(width, int) or isinstance(width, bool) or width < 1:
            raise LayoutError(
                f"{where}: Const[...] width must be a positive int, got {width!r}"
            )

    if isinstance(arg, (bytes, bytearray)):
        value = bytes(arg)
        if width is not None and len(value) != width:
            raise LayoutError(
                f"{where}: Const[...] is {len(value)} bytes but the declared width "
                f"is {width}"
            )
        return value

    if isinstance(arg, str):
        cleaned = arg.replace(" ", "").replace("_", "").replace(":", "")
        if cleaned[:2].lower() == "0x":
            cleaned = cleaned[2:]
        if not cleaned or len(cleaned) % 2 or not all(
            c in "0123456789abcdefABCDEF" for c in cleaned
        ):
            raise LayoutError(
                f"{where}: Const[{arg!r}] is not an even-length run of hex digits. "
                f"A str constant is always hex -- for ASCII write b{arg!r}."
            )
        value = bytes.fromhex(cleaned)
        if width is not None and len(value) != width:
            raise LayoutError(
                f"{where}: Const[{arg!r}] is {len(value)} bytes but the declared "
                f"width is {width}"
            )
        return value

    if isinstance(arg, int) and not isinstance(arg, bool):
        if arg < 0:
            raise LayoutError(
                f"{where}: Const[...] integer must be non-negative, got {arg}"
            )
        need = max(1, (arg.bit_length() + 7) // 8)
        if width is None:
            width = need
        elif need > width:
            raise LayoutError(
                f"{where}: Const[{arg:#x}] needs {need} bytes but the declared "
                f"width is {width}"
            )
        return arg.to_bytes(width, _BYTEORDER[endian])

    raise LayoutError(
        f"{where}: Const[...] takes bytes (b\"BLOB\"), a hex string "
        f"(\"76 2f 31 01\") or an int (0x01312F76), got {arg!r}"
    )


def _variants(
    where: str, name: str, mapping: dict, seen: dict[str, FieldSpec],
    endian: str = "<",
) -> tuple:
    """Interpret a Switch mapping into ``((value, spec | None), ...)``."""
    out = []
    keys = list(mapping)
    for i, key in enumerate(keys):
        if key is Ellipsis:
            if i != len(keys) - 1:
                raise LayoutError(
                    f"{where}: the '...' fallback case must come last"
                )
        elif not isinstance(key, int) or isinstance(key, bool):
            raise LayoutError(
                f"{where}: Switch case keys must be ints or '...', got {key!r}"
            )
        ann = mapping[key]
        if ann is None:
            out.append((key, None))
            continue
        spec = _interpret(clsname_of(where), name, ann, seen, endian)
        if spec is None or not spec.stored:
            raise LayoutError(
                f"{where}: the Switch case {key!r} must be a stored field type"
            )
        if spec.kind in ("if", "switch", "pointer", "checksum", "bits", "bitrun"):
            raise LayoutError(
                f"{where}: a Switch case may not be a {spec.kind} field"
            )
        out.append((key, spec))
    return tuple(out)


def clsname_of(where: str) -> str:
    return where.rsplit(".", 1)[0]


def _condition(where: str, arg, seen: dict[str, FieldSpec]) -> str:
    """Resolve an ``If`` condition to inlinable source text."""
    if isinstance(arg, bool):
        # `and`/`or`/`not` are evaluated while the annotation is built, and a
        # name that resolved to a global collapses the same way. Both land here
        # as a plain bool, so one check covers both.
        raise LayoutError(
            f"{where}: the condition collapsed to the constant {arg!r} while the "
            f"annotation was being built. Either it uses `and`/`or`/`not` -- Python "
            f"evaluates those eagerly, so pybinary never sees them -- or a name in it "
            f"resolved to a module global instead of the field declared above. "
            f"Use `&`/`|`/`~`, or quote the condition."
        )
    if isinstance(arg, ForwardRef):
        expr = arg.__forward_arg__
    elif isinstance(arg, str):
        expr = arg
    else:
        raise LayoutError(
            f"{where}: expected a condition referring to an earlier field, got {arg!r}"
        )
    _validate_expr(where, expr, seen, _COND_NODES, "condition")
    if not any(isinstance(n, ast.Name) for n in ast.walk(ast.parse(expr, mode="eval"))):
        raise LayoutError(
            f"{where}: the condition {expr!r} refers to no field, so it is constant"
        )
    return expr


def _length(
    where: str, arg, seen: dict[str, FieldSpec], allow_rest: bool = True
) -> tuple[int | None, str | None]:
    """Resolve a length/count argument to ``(static, expression)``.

    ``(None, None)`` means "consume the rest of the buffer".
    """
    if arg is Ellipsis:
        if not allow_rest:
            raise LayoutError(f"{where}: '...' is not allowed here; a count is required")
        return None, None

    if isinstance(arg, int) and not isinstance(arg, bool):
        if arg < 0:
            raise LayoutError(f"{where}: length must be non-negative, got {arg}")
        return arg, None

    if isinstance(arg, ForwardRef):
        expr = arg.__forward_arg__
    elif isinstance(arg, str):
        expr = arg
    else:
        shadowed = getattr(arg, "__name__", None)
        if shadowed in seen:
            raise LayoutError(
                f"{where}: {shadowed!r} resolved to {arg!r} instead of the field "
                f"{shadowed!r} declared above -- annotations see globals and builtins "
                f"first. Quote the reference (e.g. Bytes[{shadowed!r}]) or rename the field."
            )
        raise LayoutError(
            f"{where}: expected an int, '...', or a reference to an earlier field, got {arg!r}"
        )

    _validate_expr(where, expr, seen)
    return None, expr


def _validate_expr(
    where: str, expr: str, seen: dict[str, FieldSpec],
    allowed: tuple = _ARITH_NODES, what: str = "length expression",
) -> None:
    try:
        tree = ast.parse(expr, mode="eval")
    except SyntaxError as exc:  # pragma: no cover - ForwardRef sources are valid python
        raise LayoutError(f"{where}: cannot parse {what} {expr!r}") from exc

    grammar = ("earlier field names, integer literals, comparisons "
               "and + - * // % & | ^ ~ << >>"
               if allowed is _COND_NODES else
               "earlier field names, integer literals and + - * // % & | ^ ~ << >>")
    for node in ast.walk(tree):
        if not isinstance(node, allowed):
            raise LayoutError(
                f"{where}: {what} {expr!r} may only use {grammar}"
            )

    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            ref = seen.get(node.id)
            if ref is None:
                raise LayoutError(
                    f"{where}: {what} {expr!r} refers to {node.id!r}, which is "
                    f"not a field declared before this one"
                )
            if not ref.is_integer:
                raise LayoutError(
                    f"{where}: {what} {expr!r} refers to {node.id!r}, which is "
                    f"not an integer field"
                )
