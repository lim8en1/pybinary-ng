"""``If`` -- fields that are only present when a condition holds."""

import struct

import pytest

from pybinary import (
    Array, Binary, Bits, BuildError, Bytes, Const, If, LayoutError, ParseError,
    Str, u8, u16, u32,
)
from pybinary._compat import HAS_LAZY_ANNOTATIONS

SHADOW = 7  # deliberately collides with a field name below


class Versioned(Binary, endian="<"):
    ver:   u8
    extra: If["ver > 1", u32]
    tail:  If["ver > 5", Bytes[2]]


def test_absent_when_the_condition_is_false():
    v = Versioned(ver=1)
    assert v.extra is None and v.tail is None
    assert v.pack() == b"\x01"


def test_present_when_the_condition_is_true():
    v = Versioned(ver=2, extra=7)
    assert v.pack() == b"\x02" + struct.pack("<I", 7)
    assert v.tail is None


def test_several_conditions_are_independent():
    v = Versioned(ver=9, extra=1, tail=b"hi")
    assert v.pack() == b"\x09" + struct.pack("<I", 1) + b"hi"


@pytest.mark.parametrize("kwargs", [
    {"ver": 1},
    {"ver": 2, "extra": 7},
    {"ver": 9, "extra": 1, "tail": b"hi"},
])
def test_round_trips(kwargs):
    v = Versioned(**kwargs)
    assert Versioned.unpack(v.pack()) == v


def test_optional_fields_default_to_none():
    # no explicit value needed for either conditional field
    assert Versioned(ver=1) == Versioned(ver=1, extra=None, tail=None)


def test_a_conditional_structure_has_no_static_size():
    assert Versioned.__struct_size__ is None


# --------------------------------------------------------------------------
# payload kinds
# --------------------------------------------------------------------------


def test_conditional_str_and_array():
    class S(Binary, endian="<"):
        n:    u8
        text: If["n > 0", Str["n"]]
        nums: If["n > 0", Array[u16, "n"]]

    s = S(n=2, text="ab", nums=(1, 2))
    assert S.unpack(s.pack()) == s
    empty = S(n=0)
    assert empty.pack() == b"\x00"
    assert S.unpack(b"\x00").text is None


def test_conditional_nested_structure():
    class Inner(Binary, endian="<"):
        a: u16

    class Outer(Binary, endian="<"):
        flag: u8
        maybe: If["flag & 1", Inner]

    o = Outer(flag=1, maybe=Inner(a=5))
    assert Outer.unpack(o.pack()) == o
    assert Outer.unpack(b"\x00").maybe is None


def test_a_bit_field_can_drive_a_condition():
    class B(Binary, endian=">"):
        flag: Bits[1]
        rest: Bits[7]
        opt:  If["flag", u8]

    on = B(flag=1, rest=0, opt=9)
    assert on.pack() == b"\x80\x09"
    assert B.unpack(on.pack()) == on
    assert B.unpack(b"\x00").opt is None


# --------------------------------------------------------------------------
# conditions
# --------------------------------------------------------------------------


def test_bitwise_operators_survive_unquoted():
    class D(Binary, endian="<"):
        v: u8
        x: If["(v > 1) & (v < 9)", u32]

    assert D(v=5, x=1).pack() == b"\x05" + struct.pack("<I", 1)
    assert D(v=90).pack() == b"\x5a"


@pytest.mark.skipif(not HAS_LAZY_ANNOTATIONS, reason="needs PEP 649")
def test_unquoted_conditions_work():
    ns = {"Binary": Binary, "If": If, "u8": u8, "u32": u32}
    exec("class T(Binary):\n    v: u8\n    x: If[v > 1, u32]\n", ns)
    assert ns["T"](v=1).pack() == b"\x01"


@pytest.mark.skipif(not HAS_LAZY_ANNOTATIONS, reason="needs PEP 649")
def test_a_condition_captured_by_a_global_is_rejected():
    ns = {"Binary": Binary, "If": If, "u8": u8, "u32": u32, "SHADOW": SHADOW}
    with pytest.raises(LayoutError, match="collapsed to the constant"):
        exec("class T(Binary):\n    SHADOW: u8\n    x: If[SHADOW > 1, u32]\n", ns)


@pytest.mark.skipif(not HAS_LAZY_ANNOTATIONS, reason="needs PEP 649")
def test_not_in_a_condition_is_rejected():
    ns = {"Binary": Binary, "If": If, "u8": u8, "u32": u32}
    with pytest.raises(LayoutError, match="collapsed to the constant"):
        exec("class T(Binary):\n    v: u8\n    x: If[not v, u32]\n", ns)


def test_a_constant_condition_is_rejected():
    with pytest.raises(LayoutError, match="refers to no field"):
        class T(Binary):
            v: u8
            x: If["1 > 0", u32]


def test_a_condition_naming_an_unknown_field_is_rejected():
    with pytest.raises(LayoutError, match="not a field declared before this one"):
        class T(Binary):
            v: u8
            x: If["nope > 1", u32]


def test_a_condition_on_a_later_field_is_rejected():
    with pytest.raises(LayoutError, match="not a field declared before this one"):
        class T(Binary):
            x:     If["v > 1", u32]
            v:     u8


# --------------------------------------------------------------------------
# errors
# --------------------------------------------------------------------------


def test_missing_value_when_the_condition_holds():
    with pytest.raises(BuildError, match="condition is true.*required"):
        Versioned(ver=2).pack()


def test_value_present_when_the_condition_does_not_hold():
    with pytest.raises(BuildError, match="condition is false.*must be None"):
        Versioned(ver=1, extra=5).pack()


def test_truncated_conditional_payload():
    with pytest.raises(ParseError, match=r"Versioned\.extra"):
        Versioned.unpack(b"\x02\x00")


def test_if_needs_two_arguments():
    with pytest.raises(LayoutError, match="takes a condition and a field type"):
        class T(Binary):
            v: u8
            x: If[u32]


def test_if_cannot_wrap_a_non_stored_field():
    with pytest.raises(LayoutError, match="stored field|cannot wrap"):
        class T(Binary):
            v: u8
            x: If["v", Const[b"AB"]]


def test_if_cannot_nest():
    with pytest.raises(LayoutError, match="cannot wrap a if field"):
        class T(Binary):
            v: u8
            x: If["v", If["v", u32]]
