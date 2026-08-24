import pytest

from pybinary._compat import HAS_LAZY_ANNOTATIONS

if not HAS_LAZY_ANNOTATIONS:  # pragma: no cover - 3.13 has no lazy annotations
    pytest.skip(
        "unquoted field references (Bytes[n]) need PEP 649; see test_quoted_refs.py "
        "for the cross-version coverage",
        allow_module_level=True,
    )


import struct

from pybinary import Array, Binary, Bytes, Str, u8, u16


class Inner(Binary, endian="<"):
    k: u8
    blob: Bytes[2]


class Rec(Binary, endian="<"):
    n: u16
    body: Bytes[n]
    text: Str[2]
    inner: Inner
    tags: Array[u8, 2]


def make() -> bytearray:
    return bytearray(
        struct.pack("<H", 3) + b"abc" + b"hi" + b"\x01" + b"zz" + b"\x07\x08"
    )


def test_view_mode_aliases_the_source_buffer():
    buf = make()
    r = Rec.unpack(buf, copy=False)
    assert isinstance(r.body, memoryview)
    assert bytes(r.body) == b"abc"

    buf[2:5] = b"ABC"
    assert bytes(r.body) == b"ABC"


def test_copy_mode_detaches_from_the_source_buffer():
    buf = make()
    r = Rec.unpack(buf, copy=True)
    assert isinstance(r.body, bytes)

    buf[2:5] = b"ABC"
    assert r.body == b"abc"


def test_nested_bytes_follow_the_same_mode():
    buf = make()
    assert isinstance(Rec.unpack(buf, copy=False).inner.blob, memoryview)
    assert isinstance(Rec.unpack(buf, copy=True).inner.blob, bytes)


def test_text_and_scalar_arrays_are_always_copies():
    buf = make()
    r = Rec.unpack(buf)
    assert r.text == "hi" and isinstance(r.text, str)
    assert r.tags == (7, 8)

    buf[5:7] = b"XY"
    assert r.text == "hi"


def test_views_pack_correctly():
    buf = make()
    r = Rec.unpack(buf, copy=False)
    assert r.pack() == bytes(buf)


def test_both_modes_compare_equal():
    buf = make()
    assert Rec.unpack(buf, copy=False) == Rec.unpack(buf, copy=True)


def test_accepts_any_buffer():
    raw = make()
    expected = Rec.unpack(bytes(raw), copy=True)
    for buf in (bytes(raw), bytearray(raw), memoryview(raw)):
        assert Rec.unpack(buf, copy=True) == expected
        assert Rec.unpack(buf, copy=False) == expected


def test_accepts_buffers_with_a_wider_itemsize():
    import array

    raw = make()
    raw.extend(b"\x00" * (-len(raw) % 4))
    words = array.array("I")
    words.frombytes(bytes(raw))
    obj, off = Rec.unpack_from(words)
    assert obj == Rec.unpack_from(bytes(raw))[0]
    assert off == len(make())


def test_bytes_input_still_yields_views_in_zero_copy_mode():
    r = Rec.unpack(bytes(make()), copy=False)
    assert isinstance(r.body, memoryview)


def test_needs_view_is_computed_through_nesting():
    class NoSlice(Binary, endian="<"):
        a: u8
        b: u16

    class HoldsNoSlice(Binary, endian="<"):
        inner: NoSlice

    assert NoSlice.__needs_view__ is False
    assert HoldsNoSlice.__needs_view__ is False
    assert Inner.__needs_view__ is True   # has a Bytes field
    assert Rec.__needs_view__ is True     # only through its nested Inner


def test_unpack_defaults_to_copy_mode():
    buf = make()
    r = Rec.unpack(buf)
    assert isinstance(r.body, bytes)
    assert isinstance(r.inner.blob, bytes)

    buf[2:5] = b"ABC"
    assert r.body == b"abc"


def test_unpack_from_defaults_to_copy_mode():
    buf = make()
    obj, _ = Rec.unpack_from(buf)
    assert isinstance(obj.body, bytes)
