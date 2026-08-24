import pytest

from pybinary._compat import HAS_LAZY_ANNOTATIONS

if not HAS_LAZY_ANNOTATIONS:  # pragma: no cover - 3.13 has no lazy annotations
    pytest.skip(
        "unquoted field references (Bytes[n]) need PEP 649; see test_quoted_refs.py "
        "for the cross-version coverage",
        allow_module_level=True,
    )


import random
import struct

import pytest

from pybinary import (
    Array, Binary, BuildError, Bytes, Const, Padding, Str,
    f32, f64, i8, i16, i32, i64, u8, u16, u32, u64,
)

SCALARS = [
    (u8, "B", 0, 255), (u16, "H", 0, 65535),
    (u32, "I", 0, 2**32 - 1), (u64, "Q", 0, 2**64 - 1),
    (i8, "b", -128, 127), (i16, "h", -32768, 32767),
    (i32, "i", -(2**31), 2**31 - 1), (i64, "q", -(2**63), 2**63 - 1),
]


@pytest.mark.parametrize("kind,char,lo,hi", SCALARS)
@pytest.mark.parametrize("endian", ["<", ">", "little", "big", "network"])
def test_scalar_matches_struct(kind, char, lo, hi, endian):
    cls = type("S", (Binary,), {"__annotations__": {"v": kind}}, endian=endian)
    prefix = {"little": "<", "big": ">", "network": "!"}.get(endian, endian)
    for value in (lo, hi, 0, 1):
        raw = struct.pack(f"{prefix}{char}", value)
        assert cls.unpack(raw).v == value
        assert cls(v=value).pack() == raw


@pytest.mark.parametrize("kind,char", [(f32, "f"), (f64, "d")])
def test_float_roundtrip(kind, char):
    cls = type("F", (Binary,), {"__annotations__": {"v": kind}})
    raw = struct.pack(f"<{char}", 1.5)
    assert cls.unpack(raw).v == 1.5
    assert cls(v=1.5).pack() == raw


class Flat(Binary, endian="<"):
    magic: Const[b"PB"]
    a: u8
    _pad: Padding[3]
    b: i32


def test_fixed_layout_is_one_struct_call():
    assert Flat.__struct_size__ == 2 + 1 + 3 + 4
    raw = struct.pack("<2sB3xi", b"PB", 5, -9)
    f = Flat.unpack(raw)
    assert (f.a, f.b) == (5, -9)
    assert f.pack() == raw
    assert not hasattr(f, "_pad") and not hasattr(f, "magic")


class Varied(Binary, endian="<"):
    n: u16
    body: Bytes[n]
    doubled: Bytes[n * 2]
    fixed: Bytes[3]
    rest: Bytes[...]


def test_variable_lengths_and_expressions():
    raw = struct.pack("<H", 2) + b"ab" + b"cdef" + b"ghi" + b"trailing"
    v = Varied.unpack(raw)
    assert bytes(v.body) == b"ab"
    assert bytes(v.doubled) == b"cdef"
    assert bytes(v.fixed) == b"ghi"
    assert bytes(v.rest) == b"trailing"
    assert v.pack() == raw


class Text(Binary, endian="<"):
    n: u8
    label: Str[n]
    tag: Str[4, "ascii"]


def test_str_fields():
    raw = b"\x06" + "héllo".encode("utf-8") + b"abcd"
    t = Text.unpack(raw)
    assert t.label == "héllo"
    assert t.tag == "abcd"
    assert t.pack() == raw


class Point(Binary, endian="<"):
    x: i16
    y: i16


class Poly(Binary, endian="<"):
    count: u8
    points: Array[Point, count]
    weights: Array[f32, count]
    three: Array[u8, 3]


def test_arrays_of_scalars_and_structs():
    raw = (b"\x02" + struct.pack("<hhhh", 1, 2, 3, 4)
           + struct.pack("<2f", 0.5, 1.5) + b"\x07\x08\x09")
    p = Poly.unpack(raw)
    assert p.points == [Point(x=1, y=2), Point(x=3, y=4)]
    assert p.weights == (0.5, 1.5)
    assert p.three == (7, 8, 9)
    assert p.pack() == raw


class Inner(Binary, endian=">"):
    a: u16


class Outer(Binary, endian=">"):
    head: u8
    inner: Inner
    also: Inner


def test_nested_structures():
    raw = b"\x01" + struct.pack(">H", 258) + struct.pack(">H", 999)
    o = Outer.unpack(raw)
    assert o.inner.a == 258 and o.also.a == 999
    assert o.pack() == raw
    assert Outer.__struct_size__ == 5


class BaseRec(Binary, endian="<"):
    kind: u8


class SubRec(BaseRec):
    n: u8
    data: Bytes[n]


def test_inheritance_extends_fields():
    assert [f.name for f in SubRec.fields()] == ["kind", "n", "data"]
    raw = b"\x01\x02xy"
    s = SubRec.unpack(raw)
    assert s.kind == 1 and bytes(s.data) == b"xy"
    assert s.pack() == raw
    assert SubRec.__pybinary_endian__ == "<"


class Head(Binary, endian="<"):
    kind: u8
    n: u16


class Body(Head):
    data: Bytes[n]      # 'n' comes from the base class body
    tail: u8


BODY_RAW = b"\x01" + struct.pack("<H", 3) + b"abc" + b"\x09"


def test_length_reference_crosses_the_inheritance_boundary():
    b = Body.unpack(BODY_RAW)
    assert [f.name for f in Body.fields()] == ["kind", "n", "data", "tail"]
    assert (b.kind, b.n, bytes(b.data), b.tail) == (1, 3, b"abc", 9)
    assert b.pack() == BODY_RAW


def test_inherited_fields_still_merge_into_one_struct_call():
    unpack_src = Body.__codec_source__.split("def _unpack_copy")[0]
    calls = [ln for ln in unpack_src.splitlines() if "unpack_from" in ln]
    # kind (base) and n (base) share one call; tail needs its own after data.
    assert "_o.kind, n = _s0.unpack_from(_buf, _off)" in calls[0]
    assert len(calls) == 2


class Leaf(Binary, endian="<"):
    v: u16


class Mid(Binary, endian="<"):
    leaf: Leaf
    blob: Bytes[2]


class Top(Binary, endian="<"):
    mid: Mid
    also: Body          # a subclass used as a field type


def test_deep_nesting_and_subclass_as_field_type():
    raw = struct.pack("<H", 7) + b"xy" + BODY_RAW
    t = Top.unpack(raw)
    assert t.mid.leaf.v == 7
    assert bytes(t.mid.blob) == b"xy"
    assert t.also == Body.unpack(BODY_RAW)
    assert t.pack() == raw
    assert Top.__needs_view__ is True


class Pair(Binary, endian="<"):
    a: u8
    inner: Leaf


class Bag(Binary, endian="<"):
    n: u8
    pairs: Array[Pair, n]


def test_array_elements_may_themselves_nest():
    raw = b"\x02" + b"\x01" + struct.pack("<H", 10) + b"\x02" + struct.pack("<H", 20)
    g = Bag.unpack(raw)
    assert g.pairs == [Pair(a=1, inner=Leaf(v=10)), Pair(a=2, inner=Leaf(v=20))]
    assert g.pack() == raw


def test_subclass_in_a_base_typed_slot_is_rejected():
    class Holder(Binary, endian="<"):
        h: Head

    assert Holder(h=Head(kind=1, n=3)).pack() == b"\x01\x03\x00"

    # Body would encode 4 extra bytes that a Holder reader never decodes.
    with pytest.raises(BuildError, match="expected a Head instance, got Body"):
        Holder(h=Body(kind=1, n=3, data=b"abc", tail=9)).pack()


def test_wrong_type_in_a_nested_array_is_rejected():
    class Bad(Binary, endian="<"):
        items: Array[Leaf, 1]

    with pytest.raises(BuildError, match="expected a Leaf instance, got Pair"):
        Bad(items=[Pair(a=1, inner=Leaf(v=2))]).pack()


def test_length_field_is_derived_when_unambiguous():
    class Msg(Binary, endian="<"):
        ln: u16
        text: Str[ln]

    m = Msg(text="hé")            # 3 encoded bytes
    assert m.ln == 3
    assert m.pack() == struct.pack("<H", 3) + "hé".encode()
    assert Msg(ln=3, text="hé") == m


def test_length_field_stays_required_when_shared():
    class Two(Binary, endian="<"):
        n: u8
        a: Bytes[n]
        b: Bytes[n]

    with pytest.raises(TypeError, match="missing required field 'n'"):
        Two(a=b"xy", b=b"zw")
    assert Two(n=2, a=b"xy", b=b"zw").pack() == b"\x02xyzw"


def test_unpack_from_returns_offset():
    raw = struct.pack("<hh", 1, 2) + struct.pack("<hh", 3, 4)
    first, off = Point.unpack_from(raw)
    second, end = Point.unpack_from(raw, off)
    assert (first.x, second.y, end) == (1, 4, 8)


def test_pack_into_appends():
    out = bytearray(b"head")
    Point(x=1, y=2).pack_into(out)
    assert out == b"head" + struct.pack("<hh", 1, 2)


def test_repr_and_equality():
    p = Point(x=1, y=2)
    assert repr(p) == "Point(x=1, y=2)"
    assert p == Point(x=1, y=2) and p != Point(x=1, y=3)
    assert (p == "not a point") is False
    with pytest.raises(TypeError):
        hash(p)


def test_no_instance_dict():
    p = Point(x=1, y=2)
    assert not hasattr(p, "__dict__")
    with pytest.raises(AttributeError):
        p.z = 1


def test_randomized_roundtrip():
    rng = random.Random(20260810)

    class Rec(Binary, endian="<"):
        kind: u8
        n: u16
        payload: Bytes[n]
        vals: Array[i32, n]
        note: Str[...]

    for _ in range(200):
        n = rng.randrange(0, 8)
        rec = Rec(
            kind=rng.randrange(256),
            n=n,
            payload=bytes(rng.randrange(256) for _ in range(n)),
            vals=tuple(rng.randrange(-(2**31), 2**31) for _ in range(n)),
            note="".join(rng.choice("abc€") for _ in range(rng.randrange(5))),
        )
        raw = rec.pack()
        assert Rec.unpack(raw, copy=True) == rec
        assert Rec.unpack(raw).pack() == raw


def test_randomized_roundtrip_across_the_new_field_types():
    """Bits, If, Switch, Pointer and Checksum, exercised together."""
    import zlib

    from pybinary import Bits, Checksum, If, Pointer, Switch

    rng = random.Random(20260824)

    class Ping(Binary, endian="<"):
        seq: u16

    class Pong(Binary, endian="<"):
        seq: u16
        ok:  u8

    class Rec(Binary, endian="<"):
        ver:   Bits[4]
        kind:  Bits[4]
        off:   u16          # before the checksum span: it is patched last
        n:     u8
        extra: If["ver > 1", u32]
        body:  Switch["kind", {1: Ping, 2: Pong, 3: None, ...: u16}]
        blob:  Pointer["off", Bytes["n"]]
        crc:   Checksum[u32, zlib.crc32, "n"]

    variants = {
        1: lambda: Ping(seq=rng.randrange(1 << 16)),
        2: lambda: Pong(seq=rng.randrange(1 << 16), ok=rng.randrange(256)),
        3: lambda: None,
    }

    for _ in range(200):
        ver = rng.randrange(16)
        kind = rng.randrange(16)
        n = rng.randrange(0, 8)
        rec = Rec(
            ver=ver,
            kind=kind,
            n=n,
            extra=rng.randrange(1 << 32) if ver > 1 else None,
            body=variants.get(kind, lambda: rng.randrange(1 << 16))(),
            blob=bytes(rng.randrange(256) for _ in range(n)),
        )
        raw = rec.pack()
        assert Rec.unpack(raw, copy=True) == rec
        assert Rec.unpack(raw).pack() == raw
