"""``Bits`` -- sub-byte fields packed MSB-first into a whole number of bytes."""

import pytest

from pybinary import (
    Binary, Bits, BuildError, Bytes, LayoutError, ParseError, u8, u16,
)


class IPv4(Binary, endian=">"):
    ver:   Bits[4]
    ihl:   Bits[4]
    tos:   u8
    total: u16


def test_bit_order_is_msb_first():
    p = IPv4(ver=4, ihl=5, tos=0, total=1500)
    assert p.pack() == bytes.fromhex("450005dc")


def test_round_trips():
    p = IPv4(ver=4, ihl=5, tos=3, total=1500)
    assert IPv4.unpack(p.pack()) == p


def test_a_bit_run_joins_the_surrounding_struct_call():
    src = IPv4.__codec_source__.split("def _unpack_copy")[0]
    assert src.count("unpack_from") == 1
    assert "_g0_0, _o.tos, _o.total = _s0.unpack_from" in src


def test_bit_order_ignores_the_structure_endian():
    class Little(Binary, endian="<"):
        a: Bits[12]
        b: Bits[4]

    class Big(Binary, endian=">"):
        a: Bits[12]
        b: Bits[4]

    assert Little(a=0xABC, b=0xD).pack() == b"\xab\xcd"
    assert Big(a=0xABC, b=0xD).pack() == b"\xab\xcd"


def test_multi_byte_runs_round_trip():
    class M(Binary, endian="<"):
        a: Bits[12]
        b: Bits[4]

    m = M(a=0xABC, b=0xD)
    assert M.unpack(m.pack()) == m


def test_a_run_must_total_whole_bytes():
    with pytest.raises(LayoutError, match=r"5 bits wide.*Bits\[3, 0\]"):
        class Bad(Binary):
            a: Bits[3]
            b: Bits[2]


def test_runs_are_delimited_by_ordinary_fields():
    # 4 + 4 and 4 + 4, not one 16-bit run
    class Split(Binary, endian=">"):
        a: Bits[4]
        b: Bits[4]
        m: u8
        c: Bits[4]
        d: Bits[4]

    s = Split(a=1, b=2, m=9, c=3, d=4)
    assert s.pack() == b"\x12\x09\x34"

    # ...and an incomplete run is still caught when a field interrupts it
    with pytest.raises(LayoutError, match="4 bits wide"):
        class Bad(Binary, endian=">"):
            a: Bits[4]
            m: u8


# --------------------------------------------------------------------------
# reserved constants
# --------------------------------------------------------------------------


class Reserved(Binary, endian=">"):
    a:    Bits[4]
    _rsv: Bits[4, 0]


def test_constant_bits_are_written_and_not_stored():
    r = Reserved(a=9)
    assert r.pack() == b"\x90"
    assert not hasattr(r, "_rsv")
    assert "_rsv" not in repr(r)


def test_constant_bits_are_verified_on_parse():
    assert Reserved.unpack(b"\x90").a == 9
    with pytest.raises(ParseError, match=r"_rsv: expected the 0 bit pattern, got 15"):
        Reserved.unpack(b"\x9f")


def test_a_nonzero_constant_round_trips():
    class Tagged(Binary, endian=">"):
        _tag: Bits[4, 0b1010]
        v:    Bits[4]

    assert Tagged(v=3).pack() == b"\xa3"
    with pytest.raises(ParseError):
        Tagged.unpack(b"\x53")


# --------------------------------------------------------------------------
# as a reference
# --------------------------------------------------------------------------


class Sized(Binary, endian=">"):
    n:    Bits[4]
    flag: Bits[4]
    data: Bytes["n"]


def test_a_bit_field_can_size_a_later_field():
    s = Sized(n=3, flag=1, data=b"abc")
    assert s.pack() == b"\x31abc"
    assert Sized.unpack(s.pack()) == s


def test_a_bit_length_is_auto_derived():
    s = Sized(flag=1, data=b"wxyz")
    assert s.n == 4
    assert Sized.unpack(s.pack()).data == b"wxyz"


def test_a_declared_bit_length_is_still_validated():
    with pytest.raises(BuildError, match="n"):
        Sized(n=7, flag=1, data=b"ab").pack()


# --------------------------------------------------------------------------
# errors
# --------------------------------------------------------------------------


@pytest.mark.parametrize("value", [16, -1, 255])
def test_a_value_that_does_not_fit_its_width_is_a_build_error(value):
    with pytest.raises(BuildError, match=r"does not fit in 4 bit"):
        Reserved(a=value).pack()


@pytest.mark.parametrize("bad", [0, -1, "4", 1.5, True])
def test_bad_widths_are_layout_errors(bad):
    with pytest.raises(LayoutError, match="width of at least 1 bit"):
        class B(Binary):
            a: Bits[bad]


def test_a_constant_wider_than_its_field_is_a_layout_error():
    with pytest.raises(LayoutError, match="does not fit in 4 bits"):
        class B(Binary):
            a: Bits[4, 99]
            b: Bits[4]


def test_a_negative_constant_is_a_layout_error():
    with pytest.raises(LayoutError, match="non-negative"):
        class B(Binary):
            a: Bits[8, -1]


def test_truncation_names_the_first_field_of_the_run():
    with pytest.raises(ParseError, match=r"IPv4\.ver"):
        IPv4.unpack(b"\x45")


# --------------------------------------------------------------------------
# introspection
# --------------------------------------------------------------------------


def test_describe_reports_bit_widths_against_the_run_offset():
    rows = [ln.split() for ln in IPv4.describe().splitlines()[2:]]
    assert rows[0] == ["0", "4b", "ver", "Bits[4]"]
    assert rows[1] == ["0", "4b", "ihl", "Bits[4]"]
    assert rows[2] == ["1", "1", "tos", "u8"]
    assert rows[3] == ["2", "2", "total", "u16"]


def test_struct_size_counts_a_bit_run_as_its_bytes():
    assert IPv4.__struct_size__ == 4
    assert Reserved.__struct_size__ == 1
