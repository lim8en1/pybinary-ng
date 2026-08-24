"""``Const`` accepts bytes, a hex string, or an integer in the struct's order."""

import pytest

from pybinary import Binary, Const, LayoutError, ParseError, u16, u32


def test_bytes_are_used_as_written():
    class A(Binary, endian="<"):
        magic: Const[b"BLOB"]

    assert A.fields()[0].const == b"BLOB"
    assert A().pack() == b"BLOB"


def test_an_integer_uses_the_structure_byte_order():
    class LE(Binary, endian="<"):
        magic: Const[0x01312F76]

    class BE(Binary, endian=">"):
        magic: Const[0x01312F76]

    assert LE().pack() == b"\x76\x2f\x31\x01"
    assert BE().pack() == b"\x01\x31\x2f\x76"


def test_an_integer_width_defaults_to_the_bytes_it_needs():
    class A(Binary, endian="<"):
        magic: Const[0x01312F76]

    assert A.__struct_size__ == 4

    class B(Binary, endian="<"):
        magic: Const[0xFF]

    assert B.__struct_size__ == 1


def test_an_explicit_width_pads_the_integer():
    class A(Binary, endian="<"):
        magic: Const[0x1234, 4]

    assert A().pack() == b"\x34\x12\x00\x00"

    class B(Binary, endian=">"):
        magic: Const[0x1234, 4]

    assert B().pack() == b"\x00\x00\x12\x34"


def test_zero_still_occupies_a_byte():
    class A(Binary, endian="<"):
        magic: Const[0]

    assert A().pack() == b"\x00"


@pytest.mark.parametrize("text", [
    "762f3101", "76 2f 31 01", "76_2f_31_01", "76:2f:31:01", "0x762f3101",
])
def test_hex_strings_are_read_as_bytes(text):
    cls = type("H", (Binary,), {"__annotations__": {"magic": Const[text]}})
    assert cls.fields()[0].const == b"\x76\x2f\x31\x01"


def test_a_hex_string_matches_the_equivalent_integer():
    class Hex(Binary, endian="<"):
        magic: Const["76 2f 31 01"]

    class Int(Binary, endian="<"):
        magic: Const[0x01312F76]

    assert Hex.fields()[0].const == Int.fields()[0].const


def test_all_three_forms_verify_on_parse():
    class A(Binary, endian="<"):
        magic: Const[0x01312F76]
        n:     u16

    assert A.unpack(b"\x76\x2f\x31\x01\x05\x00").n == 5
    with pytest.raises(ParseError, match="expected"):
        A.unpack(b"\x00\x00\x00\x00\x05\x00")


def test_a_const_is_not_stored():
    class A(Binary, endian="<"):
        magic: Const[0x1234, 2]
        n:     u16

    a = A(n=1)
    assert not hasattr(a, "magic")
    assert "magic" not in repr(a)


# --------------------------------------------------------------------------
# rendering
# --------------------------------------------------------------------------


def test_describe_shows_ascii_magics_as_bytes():
    class A(Binary, endian="<"):
        magic: Const[b"BLOB"]

    assert "Const[b'BLOB']" in A.describe()


def test_describe_shows_binary_magics_as_hex():
    class PNG(Binary, endian=">"):
        magic: Const[0x89504E470D0A1A0A]
        ln:    u32

    assert 'Const["89 50 4e 47 0d 0a 1a 0a"]' in PNG.describe()


# --------------------------------------------------------------------------
# errors
# --------------------------------------------------------------------------


def test_an_ascii_string_is_rejected_with_a_pointer_to_the_bytes_form():
    with pytest.raises(LayoutError, match=r"always hex.*for ASCII write b'MAGC'"):
        type("B", (Binary,), {"__annotations__": {"m": Const["MAGC"]}})


def test_an_odd_length_hex_string_is_rejected():
    with pytest.raises(LayoutError, match="even-length run of hex digits"):
        type("B", (Binary,), {"__annotations__": {"m": Const["123"]}})


def test_an_empty_string_is_rejected():
    with pytest.raises(LayoutError, match="even-length run of hex digits"):
        type("B", (Binary,), {"__annotations__": {"m": Const[""]}})


def test_a_negative_integer_is_rejected():
    with pytest.raises(LayoutError, match="non-negative"):
        type("B", (Binary,), {"__annotations__": {"m": Const[-5]}})


def test_an_integer_wider_than_its_declared_width_is_rejected():
    with pytest.raises(LayoutError, match=r"needs 3 bytes but the declared width is 2"):
        type("B", (Binary,), {"__annotations__": {"m": Const[0x123456, 2]}})


def test_bytes_disagreeing_with_the_declared_width_are_rejected():
    with pytest.raises(LayoutError, match="is 4 bytes but the declared width is 2"):
        type("B", (Binary,), {"__annotations__": {"m": Const[b"BLOB", 2]}})


def test_a_bad_width_is_rejected():
    with pytest.raises(LayoutError, match="width must be a positive int"):
        type("B", (Binary,), {"__annotations__": {"m": Const[0x12, 0]}})


@pytest.mark.parametrize("bad", [1.5, None, True, [1, 2]])
def test_unsupported_const_types_are_rejected(bad):
    with pytest.raises(LayoutError, match="takes bytes"):
        type("B", (Binary,), {"__annotations__": {"m": Const[bad]}})
