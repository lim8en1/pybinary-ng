"""Coverage that runs on every supported Python.

The quoted reference form (``Bytes["n"]``) produces byte-identical field specs
on 3.13 and 3.14, so this module is the cross-version contract. The unquoted
form needs PEP 649 and is covered by the 3.14-only modules.
"""

import struct

import pytest

from pybinary import (
    Array, Binary, BuildError, Bytes, Const, LayoutError, Padding,
    ParseError, Str, f64, i16, u8, u16, u32,
)
from pybinary._compat import HAS_LAZY_ANNOTATIONS


class Point(Binary, endian="<"):
    kind: u8
    val:  f64


class Record(Binary, endian="<"):
    magic: Const[b"REC0"]
    n:     u16
    body:  Bytes["n"]
    text:  Str["n"]
    tags:  Array[u16, "n"]
    point: Point


def make(n=2, body=b"ab", text="cd", tags=(7, 8)):
    return (
        struct.pack("<4sH", b"REC0", n)
        + body
        + text.encode()
        + struct.pack(f"<{len(tags)}H", *tags)
        + struct.pack("<Bd", 1, 2.5)
    )


# --------------------------------------------------------------------------
# round trip
# --------------------------------------------------------------------------


def test_parse_matches_hand_written_struct():
    raw = make()
    r = Record.unpack(raw)
    assert r.n == 2
    assert r.body == b"ab"
    assert r.text == "cd"
    assert r.tags == (7, 8)
    assert r.point == Point(kind=1, val=2.5)


def test_pack_reproduces_the_input_bytes():
    raw = make()
    assert Record.unpack(raw).pack() == raw


def test_zero_copy_mode_agrees_with_copy_mode():
    raw = make()
    assert Record.unpack(raw, copy=False) == Record.unpack(raw, copy=True)


def test_bytes_are_copies_by_default():
    buf = bytearray(make())
    r = Record.unpack(buf)
    assert isinstance(r.body, bytes)
    buf[6:8] = b"ZZ"
    assert r.body == b"ab"


@pytest.mark.parametrize("endian", ["<", ">", "!", "=", "little", "big"])
def test_endianness_round_trips(endian):
    class S(Binary, endian=endian):
        a: u32
        b: i16

    s = S(a=70000, b=-3)
    assert S.unpack(s.pack()) == s


def test_length_is_derived_from_the_single_referencing_field():
    class Msg(Binary, endian="<"):
        ln:   u16
        text: Str["ln"]

    m = Msg(text="hello")
    assert m.ln == 5
    assert Msg.unpack(m.pack()) == m


def test_expression_lengths_work_quoted():
    class Doubled(Binary, endian="<"):
        n:    u8
        data: Bytes["n * 2 + 1"]

    d = Doubled(n=2, data=b"abcde")
    assert Doubled.unpack(d.pack()).data == b"abcde"


def test_rest_of_buffer():
    class Tail(Binary, endian="<"):
        a:    u8
        rest: Bytes[...]

    assert Tail.unpack(b"\x01rest of it").rest == b"rest of it"


def test_padding_is_skipped_and_zero_filled():
    class P(Binary, endian="<"):
        a:    u8
        _pad: Padding[3]
        b:    u8

    assert P(a=1, b=2).pack() == b"\x01\x00\x00\x00\x02"
    assert P.unpack(b"\x01\xff\xff\xff\x02") == P(a=1, b=2)


def test_a_structure_class_is_used_directly_as_a_field():
    class A(Binary, endian="<"):
        p: Point

    raw = struct.pack("<Bd", 3, 1.5)
    a = A.unpack(raw)
    assert a.p == Point(kind=3, val=1.5)
    assert a.pack() == raw


def test_inheritance_appends_and_references_reach_the_base():
    class Head(Binary, endian="<"):
        kind: u8
        n:    u16

    class Body(Head):
        data: Bytes["n"]
        tail: u8

    b = Body(kind=1, n=3, data=b"xyz", tail=9)
    assert Body.unpack(b.pack()) == b
    # base and subclass fields still collapse into a shared struct call
    head_src = Body.__codec_source__.split("def _unpack_copy")[0]
    assert "kind, n = _s0.unpack_from" in head_src
    # one call for the merged head run, one for the trailing scalar
    assert head_src.count("unpack_from") == 2


# --------------------------------------------------------------------------
# errors
# --------------------------------------------------------------------------


def test_truncated_buffer_names_the_field_and_offset():
    # a short fixed run is reported against the field the run starts at
    with pytest.raises(ParseError, match=r"Record\.magic: needs 6 bytes at offset 0"):
        Record.unpack(b"REC0\x02")


def test_truncated_variable_field_names_that_field():
    with pytest.raises(ParseError, match=r"Record\.body"):
        Record.unpack(struct.pack("<4sH", b"REC0", 40) + b"ab")


def test_const_mismatch_reports_the_offset():
    raw = bytearray(make())
    raw[0:4] = b"XXXX"
    with pytest.raises(ParseError, match=r"magic.*at offset 0"):
        Record.unpack(bytes(raw))


def test_trailing_bytes_are_rejected():
    with pytest.raises(ParseError, match="trailing"):
        Record.unpack(make() + b"\x00")


def test_declared_length_mismatch_is_a_build_error():
    class Msg(Binary, endian="<"):
        a:    u8
        ln:   u16
        text: Str["ln"]

    with pytest.raises(BuildError, match="ln"):
        Msg(a=1, ln=99, text="hi").pack()


def test_out_of_range_scalar_is_a_build_error():
    class S(Binary, endian="<"):
        a: u8

    with pytest.raises(BuildError):
        S(a=300).pack()


def test_reference_to_an_undeclared_field_is_a_layout_error():
    with pytest.raises(LayoutError, match="nope"):
        class Bad(Binary, endian="<"):
            n:    u8
            data: Bytes["nope"]


def test_reference_to_a_float_field_is_a_layout_error():
    with pytest.raises(LayoutError):
        class Bad(Binary, endian="<"):
            n:    f64
            data: Bytes["n"]


def test_pep563_is_rejected_with_a_pointed_message():
    src = "from __future__ import annotations\nclass T(Binary):\n    n: u16\n"
    with pytest.raises(LayoutError, match="PEP 563|__future__"):
        exec(src, {"Binary": Binary, "u16": u16})


# --------------------------------------------------------------------------
# introspection
# --------------------------------------------------------------------------


def test_describe_reports_offsets_until_the_first_variable_field():
    rows = [ln.split() for ln in Record.describe().splitlines()[2:]]
    by_name = {r[2]: r for r in rows}
    assert by_name["magic"][:2] == ["0", "4"]
    assert by_name["n"][:2] == ["4", "2"]
    assert by_name["body"][0] == "6"
    assert by_name["tags"][0] == "?"


def test_struct_size_is_known_for_fixed_layouts():
    assert Point.__struct_size__ == 9
    assert Record.__struct_size__ is None


def test_unquoted_references_only_exist_on_314():
    """The version split this module exists to pin down."""
    ns = {"Binary": Binary, "Bytes": Bytes, "u8": u8}
    src = "class T(Binary):\n    n: u8\n    d: Bytes[n]\n"
    if HAS_LAZY_ANNOTATIONS:
        exec(src, ns)
        assert ns["T"](n=1, d=b"x").pack() == b"\x01x"
    else:
        with pytest.raises(NameError):
            exec(src, ns)
