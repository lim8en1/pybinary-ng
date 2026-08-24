import pytest

from pybinary._compat import HAS_LAZY_ANNOTATIONS

if not HAS_LAZY_ANNOTATIONS:  # pragma: no cover - 3.13 has no lazy annotations
    pytest.skip(
        "unquoted field references (Bytes[n]) need PEP 649; see test_quoted_refs.py "
        "for the cross-version coverage",
        allow_module_level=True,
    )


import struct

import pytest

from pybinary import (
    Array, Binary, BuildError, Bytes, Const, LayoutError, Padding,
    ParseError, Str, f64, i16, u8, u16, u32,
)


# --------------------------------------------------------------------------
# Layout errors: raised while the class is being created.
# --------------------------------------------------------------------------


def test_forward_reference_is_rejected():
    with pytest.raises(LayoutError, match="not a field declared before this one"):
        class Bad(Binary):
            items: Bytes[n]
            n: u32


def test_unknown_name_in_length_expression():
    with pytest.raises(LayoutError, match="refers to 'nope'"):
        class Bad(Binary):
            items: Bytes[nope]


def test_length_must_reference_an_integer_field():
    with pytest.raises(LayoutError, match="not an integer field"):
        class Bad(Binary):
            size: f64
            items: Bytes[size]


def test_length_expression_grammar_is_restricted():
    with pytest.raises(LayoutError, match="may only use earlier field names"):
        class Bad(Binary):
            n: u16
            items: Bytes[n.real]


def test_length_expression_arithmetic_is_allowed():
    class Ok(Binary, endian="<"):
        n: u16
        items: Bytes[n * 2 + 1]

    raw = struct.pack("<H", 2) + b"abcde"
    assert bytes(Ok.unpack(raw).items) == b"abcde"


def test_unsupported_annotation():
    with pytest.raises(LayoutError, match="is not a pybinary field type"):
        class Bad(Binary):
            v: int


def test_unknown_field_type_name():
    with pytest.raises(LayoutError, match="unknown field type 'Missing'"):
        class Bad(Binary):
            v: Missing


def test_stored_field_may_not_start_with_underscore():
    with pytest.raises(LayoutError, match="may not start with an underscore"):
        class Bad(Binary):
            _v: u8


def test_padding_and_const_may_use_private_names():
    class Ok(Binary, endian="<"):
        _pad: Padding[2]
        _magic: Const[b"AB"]
        v: u8

    assert Ok.__struct_size__ == 5
    assert Ok(v=1).pack() == b"\x00\x00AB\x01"


def test_duplicate_field_name():
    with pytest.raises(LayoutError, match="declared twice"):
        class Base(Binary):
            v: u8

        class Sub(Base):
            v: u16


def test_explicit_slots_are_rejected():
    with pytest.raises(LayoutError, match="do not declare __slots__"):
        class Bad(Binary):
            __slots__ = ("v",)
            v: u8


def test_bad_endian():
    with pytest.raises(LayoutError, match="is not valid"):
        class Bad(Binary, endian="middle"):
            v: u8


def test_fields_from_two_structures():
    class A(Binary):
        a: u8

    class B(Binary):
        b: u8

    with pytest.raises(LayoutError, match="more than one structure"):
        class C(A, B):
            pass


def test_bad_parametric_arguments():
    with pytest.raises(LayoutError, match="Const"):
        class BadConst(Binary):
            v: Const["not bytes"]

    with pytest.raises(LayoutError, match="Padding"):
        class BadPad(Binary):
            v: Padding[-1]

    with pytest.raises(LayoutError, match="Array element"):
        class BadArray(Binary):
            v: Array[int, 2]

    with pytest.raises(LayoutError, match="not a pybinary field type"):
        class BadFieldType(Binary):
            v: int

    with pytest.raises(LayoutError, match="not allowed here"):
        class BadCount(Binary):
            v: Array[u8, ...]


SHADOW = 5  # module global that collides with a field name below


def test_module_global_shadowing_a_field_is_caught():
    with pytest.raises(LayoutError, match="resolved to a module global or builtin"):
        class Trap(Binary, endian="<"):
            SHADOW: u8
            data: Bytes[SHADOW]


def test_builtin_shadowing_a_field_is_caught():
    with pytest.raises(LayoutError, match="instead of the field 'len'"):
        class Trap(Binary, endian="<"):
            len: u8
            data: Bytes[len]


def test_quoted_reference_escapes_shadowing():
    class Quoted(Binary, endian="<"):
        len: u8
        data: Bytes["len"]

    q = Quoted.unpack(b"\x03abc")
    assert bytes(q.data) == b"abc"
    assert q.pack() == b"\x03abc"


def test_classvar_is_not_a_field():
    import typing

    class WithClassVar(Binary, endian="<"):
        TAG: typing.ClassVar[str] = "x"
        v: u8

    assert [f.name for f in WithClassVar.fields()] == ["v"]
    assert WithClassVar.TAG == "x"


# --------------------------------------------------------------------------
# Parse errors
# --------------------------------------------------------------------------


class Rec(Binary, endian="<"):
    magic: Const[b"PB"]
    n: u16
    body: Bytes[n]
    tags: Array[u16, n]


GOOD = b"PB" + struct.pack("<H", 2) + b"xy" + struct.pack("<2H", 1, 2)


def test_truncated_buffers_name_the_field():
    with pytest.raises(ParseError, match=r"Rec\.magic: needs 4 bytes"):
        Rec.unpack(GOOD[:3])
    with pytest.raises(ParseError, match=r"Rec\.body: needs 2 bytes"):
        Rec.unpack(GOOD[:5])
    with pytest.raises(ParseError, match=r"Rec\.tags: needs 4 bytes"):
        Rec.unpack(GOOD[:7])


def test_const_mismatch():
    with pytest.raises(ParseError, match=r"Rec\.magic: expected b'PB', got b'XX'"):
        Rec.unpack(b"XX" + GOOD[2:])


def test_trailing_bytes_are_rejected_by_unpack():
    with pytest.raises(ParseError, match="1 trailing byte"):
        Rec.unpack(GOOD + b"\x00")
    obj, off = Rec.unpack_from(GOOD + b"\x00")
    assert off == len(GOOD)


def test_negative_length_is_rejected():
    class Signed(Binary, endian="<"):
        n: i16
        body: Bytes[n]

    raw = struct.pack("<h", -1) + b"abc"
    with pytest.raises(ParseError, match=r"Signed\.body"):
        Signed.unpack(raw)


def test_nested_error_names_the_inner_field():
    class Inner(Binary, endian="<"):
        deep: u32

    class Outer(Binary, endian="<"):
        head: u8
        inner: Inner

    with pytest.raises(ParseError, match=r"Inner\.deep"):
        Outer.unpack(b"\x01\x02")


# --------------------------------------------------------------------------
# Build errors
# --------------------------------------------------------------------------


def test_declared_length_must_match():
    r = Rec(n=2, body=b"xyz", tags=(1, 2))
    with pytest.raises(BuildError, match=r"Rec\.body: holds 3 bytes .* 'n' evaluates to 2"):
        r.pack()


def test_declared_count_must_match():
    r = Rec(n=2, body=b"xy", tags=(1, 2, 3))
    with pytest.raises(BuildError, match=r"Rec\.tags: holds 3 elements"):
        r.pack()


def test_out_of_range_value():
    class Small(Binary, endian="<"):
        v: u8

    with pytest.raises(BuildError, match="Small:"):
        Small(v=300).pack()


def test_missing_field():
    with pytest.raises(TypeError, match=r"Rec\(\) missing required field 'tags'"):
        Rec(n=1, body=b"x")


def test_str_length_counts_encoded_bytes():
    class Msg(Binary, endian="<"):
        n: u8
        text: Str[n]

    with pytest.raises(BuildError, match=r"Msg\.text: holds 3 bytes"):
        Msg(n=2, text="hé").pack()
