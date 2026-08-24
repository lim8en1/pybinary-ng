"""``Pointer`` -- a field stored elsewhere, addressed from the record's start."""

import zlib

import pytest

from pybinary import (
    Array, Binary, Bits, BuildError, Bytes, Checksum, LayoutError, ParseError,
    Pointer, Str, u8, u16, u32,
)


class Rec(Binary, endian="<"):
    n:    u8
    off:  u32
    name: Pointer["off", Str["n"]]
    tail: u8


def test_the_target_is_appended_after_the_record():
    r = Rec(n=3, name="abc", tail=9)
    assert r.pack() == b"\x03\x06\x00\x00\x00\x09abc"


def test_the_offset_is_patched_automatically():
    raw = Rec(n=3, name="abc", tail=9).pack()
    assert int.from_bytes(raw[1:5], "little") == 6


def test_round_trips():
    r = Rec(n=3, name="abc", tail=9)
    assert Rec.unpack(r.pack()) == r


def test_the_cursor_does_not_advance_past_the_pointer():
    # `tail` reads from immediately after `off`, not after the target
    assert Rec.unpack(b"\x03\x06\x00\x00\x00\x09abc").tail == 9


def test_pack_does_not_mutate_the_instance():
    r = Rec(n=3, name="abc", tail=9)
    r.pack()
    assert r.off == 0


def test_a_computed_offset_is_excluded_from_equality():
    hand_built = Rec(n=3, name="abc", tail=9)
    parsed = Rec.unpack(hand_built.pack())
    assert parsed.off == 6 and hand_built.off == 0
    assert parsed == hand_built


def test_a_pointer_structure_has_no_static_size():
    assert Rec.__struct_size__ is None


# --------------------------------------------------------------------------
# nullable
# --------------------------------------------------------------------------


class Maybe(Binary, endian="<"):
    off:  u32
    tail: u8
    name: Pointer["off", Str[3], "nullable"]


def test_a_null_pointer_writes_a_zero_offset():
    m = Maybe(tail=1, name=None)
    assert m.pack() == b"\x00\x00\x00\x00\x01"
    assert Maybe.unpack(m.pack()).name is None


def test_a_set_nullable_pointer_behaves_normally():
    m = Maybe(tail=1, name="abc")
    assert m.pack() == b"\x05\x00\x00\x00\x01abc"
    assert Maybe.unpack(m.pack()) == m


def test_a_nullable_pointer_defaults_to_none():
    assert Maybe(tail=1).name is None


# --------------------------------------------------------------------------
# offsets that are expressions
# --------------------------------------------------------------------------


class Scaled(Binary, endian="<"):
    off:  u16
    pad:  u16
    data: Pointer["off * 2", Bytes[2]]


def test_an_expression_offset_is_validated_not_computed():
    s = Scaled(off=2, pad=0, data=b"hi")   # 2 * 2 == 4, where the target lands
    assert s.pack() == b"\x02\x00\x00\x00hi"
    assert Scaled.unpack(s.pack()) == s


def test_an_expression_offset_that_disagrees_is_a_build_error():
    with pytest.raises(BuildError, match=r"'off \* 2' evaluates to 18, but the "
                                         r"target lands at 4"):
        Scaled(off=9, pad=0, data=b"hi").pack()


# --------------------------------------------------------------------------
# composition
# --------------------------------------------------------------------------


def test_offsets_are_record_relative_inside_an_array():
    class Item(Binary, endian="<"):
        n:    u8
        off:  u16
        name: Pointer["off", Str["n"]]

    class Table(Binary, endian="<"):
        cnt:  u8
        recs: Array[Item, "cnt"]

    t = Table(cnt=2, recs=[Item(n=1, name="a"), Item(n=2, name="bc")])
    raw = t.pack()
    # each record carries the same relative offset despite different positions
    assert raw == b"\x02" + b"\x01\x03\x00a" + b"\x02\x03\x00bc"
    assert Table.unpack(raw) == t


def test_pointer_targets_of_every_payload_kind():
    class P(Binary, endian="<"):
        n:    u8
        off:  u16
        body: Pointer["off", Array[u16, "n"]]

    p = P(n=2, body=(7, 8))
    assert P.unpack(p.pack()) == p


def test_pack_into_a_prepopulated_buffer_keeps_offsets_relative():
    r = Rec(n=3, name="abc", tail=9)
    out = bytearray(b"JUNK")
    r.pack_into(out)
    assert bytes(out) == b"JUNK" + r.pack()
    assert Rec.unpack_from(out, 4)[0] == r


# --------------------------------------------------------------------------
# errors
# --------------------------------------------------------------------------


def test_a_target_outside_the_buffer_is_a_parse_error():
    with pytest.raises(ParseError, match="outside the buffer"):
        Maybe.unpack(b"\xff\x00\x00\x00\x01")


def test_a_truncated_target_is_a_parse_error():
    with pytest.raises(ParseError, match=r"Rec\.name"):
        Rec.unpack(b"\x03\x06\x00\x00\x00\x09ab")


def test_pointer_needs_an_offset_and_a_type():
    with pytest.raises(LayoutError, match="takes an offset and a field type"):
        class T(Binary):
            off: u16
            p:   Pointer["off"]


def test_an_unknown_third_argument_is_rejected():
    with pytest.raises(LayoutError, match='may only be "nullable"'):
        class T(Binary):
            off: u16
            p:   Pointer["off", u8, "weird"]


def test_a_bit_field_offset_is_rejected():
    with pytest.raises(LayoutError, match="cannot be the bit field"):
        class T(Binary):
            a:   Bits[4]
            off: Bits[4]
            p:   Pointer["off", u8]


def test_a_rest_of_buffer_field_cannot_coexist_with_a_pointer():
    with pytest.raises(LayoutError, match="rest-of-buffer field cannot coexist"):
        class T(Binary):
            a:   u8
            off: u16
            p:   Pointer["off", Bytes[2]]
            r:   Bytes[...]


def test_a_field_cannot_be_both_a_length_and_an_offset():
    with pytest.raises(LayoutError, match="both a length and a pointer offset"):
        class T(Binary):
            n: u8
            d: Bytes["n"]
            p: Pointer["n", Bytes[2]]


def test_a_checksum_may_not_cover_a_patched_offset():
    with pytest.raises(LayoutError, match="patched after the record is built"):
        class T(Binary):
            a:   u8
            off: u16
            p:   Pointer["off", Bytes[2]]
            ck:  Checksum[u32, zlib.crc32, "a"]


def test_pointer_cannot_target_a_pointer():
    with pytest.raises(LayoutError, match="cannot target a pointer field"):
        class T(Binary):
            a:   u16
            off: u16
            p:   Pointer["off", Pointer["a", u8]]
