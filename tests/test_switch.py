"""``Switch`` -- a variant chosen by an earlier field."""

import struct

import pytest

from pybinary import (
    Array, Binary, Bits, BuildError, Bytes, Const, If, LayoutError, ParseError,
    Str, Switch, u8, u16, u32,
)


class Ping(Binary, endian="<"):
    seq: u16


class Pong(Binary, endian="<"):
    seq: u16
    ok:  u8


class Msg(Binary, endian="<"):
    kind: u8
    body: Switch["kind", {1: Ping, 2: Pong, 3: u32, 4: None}]


@pytest.mark.parametrize("msg, raw", [
    (Msg(kind=1, body=Ping(seq=5)), b"\x01\x05\x00"),
    (Msg(kind=2, body=Pong(seq=5, ok=1)), b"\x02\x05\x00\x01"),
    (Msg(kind=3, body=9), b"\x03" + struct.pack("<I", 9)),
    (Msg(kind=4), b"\x04"),
])
def test_each_variant_round_trips(msg, raw):
    assert msg.pack() == raw
    assert Msg.unpack(raw) == msg


def test_dispatch_is_an_if_elif_chain():
    src = Msg.__codec_source__
    assert "_d = kind" in src
    assert "if _d == 1:" in src and "elif _d == 2:" in src


def test_a_none_variant_occupies_no_bytes():
    assert Msg(kind=4).body is None
    assert len(Msg(kind=4).pack()) == 1


def test_a_switch_has_no_static_size():
    assert Msg.__struct_size__ is None


# --------------------------------------------------------------------------
# fallback
# --------------------------------------------------------------------------


class WithFallback(Binary, endian="<"):
    kind: u8
    body: Switch["kind", {1: Ping, ...: u32}]


def test_the_fallback_case_catches_unlisted_values():
    m = WithFallback(kind=99, body=7)
    assert m.pack() == b"\x63" + struct.pack("<I", 7)
    assert WithFallback.unpack(m.pack()) == m


def test_listed_cases_still_win_over_the_fallback():
    m = WithFallback(kind=1, body=Ping(seq=2))
    assert WithFallback.unpack(m.pack()).body == Ping(seq=2)


def test_an_unknown_discriminator_without_a_fallback_is_a_parse_error():
    with pytest.raises(ParseError, match=r"no case for discriminator 9.*no '\.\.\.' fallback"):
        Msg.unpack(b"\x09")


def test_an_unknown_discriminator_without_a_fallback_is_a_build_error():
    with pytest.raises(BuildError, match="no case for discriminator 9"):
        Msg(kind=9, body=1).pack()


# --------------------------------------------------------------------------
# payload kinds and discriminators
# --------------------------------------------------------------------------


def test_variable_length_variants():
    class V(Binary, endian="<"):
        kind: u8
        n:    u8
        body: Switch["kind", {1: Bytes["n"], 2: Str["n"], 3: Array[u16, "n"]}]

    for kind, value in [(1, b"ab"), (2, "cd"), (3, (7, 8))]:
        v = V(kind=kind, n=2, body=value)
        assert V.unpack(v.pack()) == v


def test_an_expression_discriminator():
    class E(Binary, endian="<"):
        kind: u8
        body: Switch["kind & 3", {1: u8, 2: u16}]

    e = E(kind=0b0101, body=9)   # 5 & 3 == 1
    assert e.pack() == b"\x05\x09"
    assert E.unpack(e.pack()) == e


def test_a_bit_field_can_be_the_discriminator():
    class B(Binary, endian=">"):
        kind: Bits[4]
        rest: Bits[4]
        body: Switch["kind", {1: u8, 2: u16}]

    b = B(kind=2, rest=0, body=0x1234)
    assert b.pack() == b"\x20\x12\x34"
    assert B.unpack(b.pack()) == b


def test_nested_variants_keep_the_strict_type_check():
    # Pong is not a Ping; the declared variant fixes the layout
    with pytest.raises(BuildError, match="expected a Ping instance"):
        Msg(kind=1, body=Pong(seq=1, ok=1)).pack()


def test_a_switch_may_not_be_wrapped_in_a_conditional():
    with pytest.raises(LayoutError, match="cannot wrap a switch field"):
        class C(Binary, endian="<"):
            flag: u8
            kind: u8
            body: If["flag", Switch["kind", {1: u8}]]


# --------------------------------------------------------------------------
# build-side consistency
# --------------------------------------------------------------------------


def test_a_selected_payload_may_not_be_none():
    with pytest.raises(BuildError, match="selects a payload, but the field is None"):
        Msg(kind=1).pack()


def test_a_none_variant_must_hold_none():
    with pytest.raises(BuildError, match="selects no payload"):
        Msg(kind=4, body=5).pack()


def test_truncated_variant_payload():
    with pytest.raises(ParseError, match=r"Msg\.body|Msg\.seq"):
        Msg.unpack(b"\x03\x00")


# --------------------------------------------------------------------------
# layout errors
# --------------------------------------------------------------------------


def test_switch_needs_a_discriminator_and_a_mapping():
    with pytest.raises(LayoutError, match="takes a discriminator"):
        class T(Binary):
            k: u8
            b: Switch["k"]


def test_the_mapping_must_be_a_non_empty_dict():
    with pytest.raises(LayoutError, match="non-empty"):
        class T(Binary):
            k: u8
            b: Switch["k", {}]


def test_case_keys_must_be_ints_or_ellipsis():
    with pytest.raises(LayoutError, match="case keys must be ints"):
        class T(Binary):
            k: u8
            b: Switch["k", {"a": u8}]


def test_the_fallback_must_come_last():
    with pytest.raises(LayoutError, match="must come last"):
        class T(Binary):
            k: u8
            b: Switch["k", {...: u8, 1: u16}]


def test_a_case_may_not_be_a_non_stored_field():
    with pytest.raises(LayoutError, match="stored field"):
        class T(Binary):
            k: u8
            b: Switch["k", {1: Const[b"AB"]}]


def test_a_discriminator_naming_an_unknown_field_is_rejected():
    with pytest.raises(LayoutError, match="not a field declared before this one"):
        class T(Binary):
            k: u8
            b: Switch["nope", {1: u8}]


def test_a_float_discriminator_is_rejected():
    from pybinary import f64

    with pytest.raises(LayoutError, match="not an integer field"):
        class T(Binary):
            k: f64
            b: Switch["k", {1: u8}]
